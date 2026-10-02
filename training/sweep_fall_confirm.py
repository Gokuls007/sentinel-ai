"""Sweep the fall confirmation time (how long a person must lie still before the alert).

    python training/sweep_fall_confirm.py                       # 0.5, 1, 2, 3, 5 s
    python training/sweep_fall_confirm.py --values 0.5,1,1.5 --write docs/BENCHMARKS.md

Detection, tracking and pose run **once** per video. The per-frame poses are cached, and only
the fall state machine is replayed at each setting, so every row sees exactly the same
frames and poses. Everything else stays at the production thresholds in config/settings.py,
with one exception: the "gave up waiting" timeout is raised to at least confirmation + 2 s.
Otherwise a 5 s confirmation could never fire before the 5 s timeout reset the state.

Footage:
- **Falls:** URFD's 30 fall sequences (camera 0).
- **No falls:** URFD's 40 ADL sequences, the committed sample clips
  (training/nonfall_videos.json), and anything in data/recordings/ whose sidecar JSON
  doesn't say it contains falls (``"contains_falls": true`` excludes a clip).

Columns per setting:
- **Recall:** confirmed alerts after fall onset (eval_fall.py scoring).
- **Confirmable:** fall clips with at least that much video left after reaching FALLEN.
  This is the most recall could be on URFD, whose clips are trimmed shortly after the fall.
- **False alarms per hour:** over all no-fall footage, with the hours stated.
- **"On the ground" stage recall:** fall clips that reached FALLEN at all. It doesn't depend
  on the confirmation time; it is listed to show where alerts are lost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import eval_fall as ef
import eval_false_alarms as efa

SECTION_START = "<!-- benchmark:fall-sweep:start -->"
SECTION_END = "<!-- benchmark:fall-sweep:end -->"
DEFAULT_VALUES = (0.5, 1.0, 2.0, 3.0, 5.0)


@dataclass
class Cached:
    """Per-frame inputs to the fall state machine for one video."""
    name: str
    kind: str  # "fall" | "adl" | "sample" | "recording"
    fps: float
    onset_frame: int | None = None
    # [(ts, active_ids, [(tid, PoseResult, standing_h)], {tid: {chain: PoseResult | None}})]
    frames: list = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.frames) / self.fps


@dataclass(frozen=True)
class Fixes:
    """The four 'lost on the floor' fixes (config.settings.FallDetectorConfig names)."""
    lost_hold_seconds: float = 0.0      # 1. hold a lost falling/fallen track
    recovery_low: bool = False          # 2. region-local low-threshold retry
    recovery_rotated: bool = False      # 3. rotated retry
    upright_hold_seconds: float = 0.0   # 4. hysteresis before leaving the ground state
    recovery_mode: str = "pose"         # how a re-found person is used: "pose" or "presence"
    ground_mode: str = "torso"          # "torso" or "combined" (direction-independent signals)
    box_calibration: bool = False       # standing height from upright boxes if no skeleton

    @property
    def chain(self) -> str | None:
        if self.recovery_low and self.recovery_rotated:
            return "both"
        return "low" if self.recovery_low else "rot" if self.recovery_rotated else None


BASELINE = Fixes()
FIX_VARIANTS = (
    ("Baseline (no fixes)", BASELINE),
    ("1. Hold lost track 5 s (last seen lying)", Fixes(lost_hold_seconds=5.0)),
    ("2. Region-local low threshold, as pose", Fixes(recovery_low=True)),
    ("2. Region-local low threshold, as presence", Fixes(recovery_low=True, recovery_mode="presence")),
    ("3. Rotated fallback, as pose", Fixes(recovery_rotated=True)),
    ("3. Rotated fallback, as presence", Fixes(recovery_rotated=True, recovery_mode="presence")),
    ("4. Ground-state hysteresis 0.5 s", Fixes(upright_hold_seconds=0.5)),
    ("2 + 3 + 4, as pose (no hold)", Fixes(0.0, True, True, 0.5)),
    ("3 + 4, as presence (no hold)", Fixes(0.0, False, True, 0.5, "presence")),
    ("2 + 3 + 4, as presence (no hold)", Fixes(0.0, True, True, 0.5, "presence")),
    ("All four, as pose", Fixes(5.0, True, True, 0.5)),
    ("All four, as presence", Fixes(5.0, True, True, 0.5, "presence")),
)
# Frozen 2026-10-02 after the URFD fix table, before CAUCAFall was opened.
CANDIDATE = ("Candidate: rotated retry + hysteresis 0.5 s, as presence",
             Fixes(0.0, False, True, 0.5, "presence"))
RECOVERY_LOW_CONF = 0.15
RECOVERY_WINDOW_S = 3.0
CACHE_RECOVERY_S = 5.0  # cached retries cover this long after a track's last sighting


# --- caching (runs the models) ------------------------------------------------------------------

def code_stamp() -> str:
    """Hash of the code and thresholds that produce the cached poses."""
    h = hashlib.sha256()
    for rel in ("backend/core/pose_estimator.py", "backend/core/detector.py", "backend/config/settings.py",
                "backend/config/trackers/sentinel_bytetrack.yaml", "backend/anomaly/fall_recovery.py"):
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]

CHAINS = {  # cached retry chains: (try_low, try_rotated, confidence)
    "low": (True, False, RECOVERY_LOW_CONF),
    "rot": (False, True, None),  # rotated only: the normal pose threshold, as in production
    "both": (True, True, RECOVERY_LOW_CONF),
}


def cache_frames(models, frames_iter, name, kind, fps, onset_frame=None, resize=None) -> Cached:
    """Run the models once. Besides the tracked poses, for every tracked person missing this
    frame (seen in the last CACHE_RECOVERY_S), run each retry chain of
    anomaly/fall_recovery.py around that chain's last box. This is the same call production
    makes; there it only runs for people who are down, which the replay applies."""
    from anomaly.fall_recovery import recover_pose
    from core.pose_estimator import PoseResult

    out = Cached(name=name, kind=kind, fps=fps, onset_frame=onset_frame)
    models.fresh()  # reset tracker and pose history
    det, pose = models.detector, models.pose
    normal_conf = models.cfg.pose.confidence_threshold
    last: dict[str, dict[int, tuple[float, np.ndarray]]] = {c: {} for c in CHAINS}  # chain -> tid -> (ts, bbox)
    for idx, frame in frames_iter:
        if resize and (frame.shape[1], frame.shape[0]) != resize:
            frame = cv2.resize(frame, resize)
        ts = (idx - 1) / fps
        dets = det.detect_and_track(frame)
        people = [d for d in dets.detections if d.class_name == "person" and d.track_id is not None]
        poses = pose.estimate(frame, [d.track_id for d in people], [d.bbox for d in people], ts)
        feats = pose.get_all_features()
        rows = [(tid, PoseResult(tid, p.keypoints.copy(), np.asarray(p.bbox).copy()),
                 feats[tid].initial_standing_height)
                for tid, p in poses.items() if tid in feats]
        recov: dict[int, dict] = {}
        for chain, (try_low, try_rot, conf) in CHAINS.items():
            seen = last[chain]
            for tid, p, _h in rows:
                seen[tid] = (ts, p.bbox)
            for tid in feats:
                if tid in poses or tid not in seen or ts - seen[tid][0] > CACHE_RECOVERY_S:
                    continue
                found, _how = recover_pose(pose.model, frame, tid, seen[tid][1], low_conf=conf or normal_conf,
                                           try_low=try_low, try_rotated=try_rot)
                recov.setdefault(tid, {})[chain] = found
                if found is not None:
                    seen[tid] = (ts, found.bbox)
        out.frames.append((ts, tuple(feats.keys()), rows, recov))
    return out


CAUCAFALL_DIR = os.path.join(ROOT, "data", "datasets", "caucafall")


def caucafall_videos() -> list[tuple[str, str, str]]:
    """(video path, labels.csv path, 'Subject.N/Activity') for every CAUCAFall video present."""
    out = []
    for dirpath, _dirs, files in os.walk(CAUCAFALL_DIR):
        avis = [f for f in files if f.lower().endswith(".avi")]
        if avis and "labels.csv" in files:
            rel = "/".join(os.path.relpath(dirpath, CAUCAFALL_DIR).replace("\\", "/").split("/")[-2:])
            out.append((os.path.join(dirpath, avis[0]), os.path.join(dirpath, "labels.csv"), rel))
    return sorted(out, key=lambda x: x[2])


def caucafall_onset(labels_csv: str) -> int | None:
    """First frame (1-based) labelled 1 = fall."""
    with open(labels_csv, encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            frame, label = line.split(",")
            if label.strip() == "1":
                return int(frame)
    return None


def video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 240 else 25.0

    def gen():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            yield i, frame
        cap.release()

    return gen(), fps


# --- replay (pure; unit-tested) -------------------------------------------------------------------

def make_detector(fall_cfg, confirm_s: float, fixes: Fixes = BASELINE):
    from anomaly.fall_detector import FallDetector

    return FallDetector(
        descent_speed_threshold=fall_cfg.descent_speed_threshold,
        aspect_ratio_threshold=fall_cfg.aspect_ratio_threshold,
        head_drop_ratio=fall_cfg.head_drop_ratio,
        stillness_seconds=confirm_s,
        stillness_speed_threshold=fall_cfg.stillness_speed_threshold,
        fallen_timeout_seconds=max(fall_cfg.fallen_timeout_seconds, confirm_s + 2.0),
        cooldown_seconds=fall_cfg.cooldown_seconds,
        lost_hold_seconds=fixes.lost_hold_seconds,
        upright_hold_seconds=fixes.upright_hold_seconds,
        ground_mode=fixes.ground_mode,
        box_calibration=fixes.box_calibration,
    )


def replay(video: Cached, detector, fixes: Fixes = BASELINE) -> dict:
    """Run the state machine over cached frames, in the order AnomalyEngine.process uses:
    first the people who are down but have no pose this frame (a recovered pose, if that fix
    is on and the pipeline would have retried, else held), then the tracked poses. It also
    follows the first track that reached the ground, to explain a missing alert."""
    alerts: list[float] = []
    possible: list[float] = []
    fallen_s = fallen_tid = None
    frames_after = seen_after = 0
    left_ground = False
    standing: dict[int, float] = {}
    onset_s = (video.onset_frame - 1) / video.fps if video.onset_frame else None
    chain = fixes.chain
    for frame in video.frames:
        ts, active, rows, recov = frame if len(frame) == 4 else (*frame, {})
        detector.prune(active)
        seen = False
        posed = {tid for tid, _p, _h in rows}
        for tid in [t for t in active if t not in posed and detector.is_down(t)]:
            st = detector.tracks[tid]
            pose = None
            # The same gate as SentinelPipeline._recover_fallen.
            if (chain and ts - st.falling_since <= RECOVERY_WINDOW_S and st.last_seen is not None
                    and ts - st.last_seen <= max(fixes.lost_hold_seconds, 1.0)):
                pose = recov.get(tid, {}).get(chain)
            if pose is not None:
                event = detector.check_recovered(
                    tid, pose, SimpleNamespace(initial_standing_height=standing.get(tid, 0.0)), ts,
                    mode=fixes.recovery_mode)
                seen = seen or tid == fallen_tid
            else:
                event = detector.check_missing(tid, ts)
            if event:
                alerts.append(round(ts, 3))
        for tid, pose, standing_h in rows:
            standing[tid] = standing_h
            if detector.check(tid, pose, SimpleNamespace(initial_standing_height=standing_h), ts):
                alerts.append(round(ts, 3))
            on_ground = detector.state_of(tid) in (detector.FALLEN, detector.CONFIRMED)
            if fallen_s is None and on_ground and (onset_s is None or ts >= onset_s - 2.0):
                fallen_s, fallen_tid = ts, tid
            elif tid == fallen_tid:
                seen = True
                left_ground = left_ground or not on_ground
        possible += [round(e.timestamp, 3) for e in detector.drain_possible()]
        if fallen_s is None:
            for tid in active:  # a held/recovered track can reach the ground without a pose row
                if tid not in posed and detector.state_of(tid) in (detector.FALLEN, detector.CONFIRMED) \
                        and (onset_s is None or ts >= onset_s - 2.0):
                    fallen_s, fallen_tid = ts, tid
                    break
        if fallen_s is not None and ts > fallen_s:
            frames_after += 1
            seen_after += seen
    return {"alerts": alerts, "possible": possible, "fallen_s": fallen_s, "onset_s": onset_s,
            "end_s": (len(video.frames) - 1) / video.fps if video.frames else 0.0,
            "pose_after_ground": seen_after / frames_after if frames_after else None,
            "left_ground": left_ground}


def why_unconfirmed(r: dict) -> str:
    """Why a fall clip that reached the ground and had enough video still got no alert."""
    if r["pose_after_ground"] is not None and r["pose_after_ground"] < 0.5:
        return "lost from view"  # the person's pose was missing in most later frames
    if r["left_ground"]:
        return "left the ground state"  # pose looked upright again, or the wait timed out
    return "not still long enough"


def activity_of(v: Cached) -> str:
    """'Subject.3/Kneel' -> 'Kneel'; other footage is grouped by kind."""
    return v.name.split("/")[-1] if "/" in v.name else v.kind


def score_setting(videos: list[Cached], fall_cfg, confirm_s: float, tolerance_s: float = 2.0,
                  fixes: Fixes = BASELINE) -> dict:
    falls = [v for v in videos if v.kind == "fall"]
    clean = [v for v in videos if v.kind != "fall"]
    tp = reached = confirmable = lost = possible_tp = 0
    false_alarms: dict[str, int] = {}
    missed: dict[str, int] = {}
    for v in falls:
        r = replay(v, make_detector(fall_cfg, confirm_s, fixes), fixes)
        hit = any(t >= r["onset_s"] - tolerance_s for t in r["alerts"])
        tp += hit
        possible_tp += any(t >= r["onset_s"] - tolerance_s for t in r["possible"])
        if r["fallen_s"] is not None:
            reached += 1
            lost += r["pose_after_ground"] is not None and r["pose_after_ground"] < 0.5
            # Enough video left after reaching the ground for this confirmation time.
            enough = r["end_s"] - r["fallen_s"] >= confirm_s
            confirmable += enough
            if enough and not hit:
                reason = why_unconfirmed(r)
                missed[reason] = missed.get(reason, 0) + 1
        false_alarms["fall clips (early/extra)"] = false_alarms.get("fall clips (early/extra)", 0) + \
            len(r["alerts"]) - int(hit)
    hours: dict[str, float] = {}
    clean_reached = possible_fa = 0
    by_activity: dict[str, list[int]] = {}  # activity -> [confirmed, possible] false alarms
    for v in clean:
        r = replay(v, make_detector(fall_cfg, confirm_s, fixes), fixes)
        false_alarms[v.kind] = false_alarms.get(v.kind, 0) + len(r["alerts"])
        hours[v.kind] = hours.get(v.kind, 0.0) + v.seconds / 3600
        clean_reached += r["fallen_s"] is not None
        possible_fa += len(r["possible"])
        act = by_activity.setdefault(activity_of(v), [0, 0])
        act[0] += len(r["alerts"])
        act[1] += len(r["possible"])
    clean_hours = sum(hours.values())
    clean_fa = sum(n for k, n in false_alarms.items() if k in hours)
    return {
        "possible_tp": possible_tp, "possible_recall": possible_tp / len(falls) if falls else None,
        "possible_false_alarms": possible_fa,
        "possible_false_alarms_per_hour": possible_fa / clean_hours if clean_hours else None,
        "false_alarms_by_activity": by_activity,
        "confirm_s": confirm_s, "falls": len(falls), "tp": tp, "recall": tp / len(falls) if falls else None,
        "confirmable": confirmable, "on_ground": reached, "on_ground_recall": reached / len(falls) if falls else None,
        "false_alarms": false_alarms, "clean_hours": clean_hours, "hours_by_kind": hours,
        "false_alarms_clean": clean_fa, "false_alarms_per_hour": clean_fa / clean_hours if clean_hours else None,
        "clean_videos_reaching_ground": clean_reached, "clean_videos": len(clean),
        "confirmable_but_missed": missed, "lost_from_view": lost, "fixes": fixes.__dict__,
    }


# --- report -------------------------------------------------------------------------------------

KIND_LABEL = {"adl": "URFD ADL", "sample": "sample clips", "recording": "your recordings",
              "caucafall_adl": "CAUCAFall daily activities"}


def _rate(x) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def markdown_heldout(rows: list[tuple[str, dict]], confirm_s: float) -> list[str]:
    """CAUCAFall, held out: never used to design or tune anything. Baseline and the frozen
    candidate only, at the default confirmation time (a table of settings here would turn it
    into tuning data)."""
    r0 = rows[0][1]
    acts = sorted({a for _l, r in rows for a in r["false_alarms_by_activity"]})
    lines = [
        "",
        "### Held-out test: CAUCAFall",
        "",
        f"CAUCAFall (CC BY 4.0) has {r0['falls']} falls (5 types x 10 subjects) and {r0['clean_videos']} daily "
        f"activities, including sitting down and kneeling: {r0['clean_hours'] * 60:.1f} min of no-fall video. "
        "Fall onset is the first frame labelled \"fall\". It was **not used** to design or tune any rule. "
        f"Only the baseline and the candidate frozen beforehand are reported, at the default {confirm_s:g} s, "
        "run once.",
        "",
        "| Setting | Confirmed recall | Confirmed false alarms / h (count) | Possible recall "
        "| Possible false alarms / h (count) | Lost from view |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in rows:
        lines.append(f"| {label} | {r['tp']} / {r['falls']} ({r['recall']:.0%}) | "
                     f"{_rate(r['false_alarms_per_hour'])} ({r['false_alarms_clean']}) | "
                     f"{r['possible_tp']} / {r['falls']} ({r['possible_recall']:.0%}) | "
                     f"{_rate(r['possible_false_alarms_per_hour'])} ({r['possible_false_alarms']}) | "
                     f"{r['lost_from_view']} / {r['on_ground']} |")
    lines += ["", "False alarms by daily activity (confirmed / possible):", "",
              "| Setting | " + " | ".join(acts) + " |", "|---|" + "---|" * len(acts)]
    for label, r in rows:
        counts = [r["false_alarms_by_activity"].get(a, [0, 0]) for a in acts]
        cells = [f"{c} / {p}" for c, p in counts]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Any further tuning will split CAUCAFall by video into a tuning half and a test half, and "
              "report test-half numbers only."]
    return lines


def markdown_fixes(fix_rows: list[tuple[str, dict]], confirm_s: float) -> list[str]:
    """One row per 'lost on the floor' fix, each alone, plus all four together."""
    base = fix_rows[0][1]
    lines = [
        "",
        f"**Fixes for losing the person on the floor**, each alone, at the default {confirm_s:g} s confirmation. "
        "These are replayed from the same pose cache. The region-local and rotated retries were run once per "
        "missing person and are used only where production would use them (falling or on the ground, within "
        f"{RECOVERY_WINDOW_S:g} s of the fall). The retry threshold is {RECOVERY_LOW_CONF:g}; normal is the "
        "pose model's threshold.",
        "",
        "| Fix | Confirmed catches | Confirmed false alarms (/ h) | Possible catches | Possible false alarms (/ h) "
        "| Lost from view (of falls reaching the ground) |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in fix_rows:
        delta = r["false_alarms_clean"] - base["false_alarms_clean"]
        flag = f" {delta:+d}" if delta else ""
        lines.append(f"| {label} | {r['tp']} / {r['falls']} | {r['false_alarms_clean']}{flag} "
                     f"({_rate(r['false_alarms_per_hour'])}) | {r['possible_tp']} / {r['falls']} | "
                     f"{r['possible_false_alarms']} ({_rate(r['possible_false_alarms_per_hour'])}) | "
                     f"{r['lost_from_view']} / {r['on_ground']} |")
    lines += [
        "",
        "*Lost from view*: the person's pose (tracked or recovered) was missing in most frames after reaching "
        "the ground. Fix 1 doesn't find the person; it keeps the fall alive while they are missing, so it "
        "raises catches without lowering this count.",
        "",
        "*As pose*: a re-found person goes through the normal check. *As presence*: a re-found person who was "
        "last seen lying counts as still in place, unless clearly upright or moved more than half a body height. "
        "The re-found keypoints jitter too much to measure stillness directly.",
        "",
        "**Caveat: these rows are optimistic.** The \"last seen lying\" gate and the presence mode were designed "
        "after inspecting these same URFD clips. Before choosing defaults they need confirming on footage not "
        "used here: your own recordings and a held-out dataset (CAUCAFall). All fixes stay off by default until "
        "one is chosen.",
    ]
    return lines


def markdown(rows: list[dict], default_s: float, device: str, fix_rows: list[tuple[str, dict]] | None = None,
             heldout_rows: list[tuple[str, dict]] | None = None) -> str:
    r0 = rows[0]
    basis = ", ".join(f"{KIND_LABEL.get(k, k)} {h * 60:.1f} min" for k, h in r0["hours_by_kind"].items())
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python training/sweep_fall_confirm.py` on {device}._ "
        f"The confirmation time is how long a person must lie still on the ground before the alert. "
        f"The current default is **{default_s:g} s**. Detection and pose ran once per video, and only the "
        "fall state machine was replayed at each setting. The \"gave up\" timeout is at least "
        "confirmation + 2 s. These rows use the earlier torso-only ground rule (`ground_mode=\"torso\"`, the "
        "default before 2026-10-02). The current default is compared in \"Fall detection: rules vs learned "
        "model (unseen subjects)\" above.",
        "",
        f"No-fall footage: **{r0['clean_hours'] * 60:.1f} min** ({basis}).",
        "",
        "Two alert levels: a **possible fall** (yellow, dashboard only) when the person reaches the ground, "
        "and a **confirmed fall** (red, notifies) after the stillness check.",
        "",
        "| Confirmation time | Confirmed recall | Confirmed false alarms / h (count) | Possible recall "
        "| Possible false alarms / h (count) | Confirmable on URFD | Confirmable but missed (why) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " (default)" if abs(r["confirm_s"] - default_s) < 1e-9 else ""
        why = ", ".join(f"{n} {k}" for k, n in sorted(r["confirmable_but_missed"].items(), key=lambda x: -x[1]))
        lines.append(
            f"| {r['confirm_s']:g} s{mark} | {r['tp']} / {r['falls']} ({r['recall']:.0%}) | "
            f"{_rate(r['false_alarms_per_hour'])} ({r['false_alarms_clean']}) | "
            f"{r['possible_tp']} / {r['falls']} ({r['possible_recall']:.0%}) | "
            f"{_rate(r['possible_false_alarms_per_hour'])} ({r['possible_false_alarms']}) | "
            f"{r['confirmable']} / {r['falls']} | {why or 'none'} |")
    lines += [
        "",
        f"**\"On the ground\" stage recall** (reached FALLEN, the step before confirmation; it is the same at "
        f"every setting): **{r0['on_ground']} / {r0['falls']} ({r0['on_ground_recall']:.0%})** of URFD falls. "
        f"{r0['clean_videos_reaching_ground']} of {r0['clean_videos']} no-fall videos also reached that stage "
        "without confirming.",
        *(markdown_fixes(fix_rows, default_s) if fix_rows else []),
        "",
        "How to read it:",
        "- URFD trims each fall clip about 1–2 s after the fall. So \"Confirmable\" caps recall, and at 2 s "
        "and above URFD can't tell you anything about recall.",
        "- \"Confirmable but missed\": the clip had enough video after reaching the ground, but no alert. "
        "*Lost from view*: the person's pose was missing in most later frames. *Left the ground state*: "
        "the pose looked upright again, or the wait timed out. *Not still long enough*: the person kept "
        "moving on the ground.",
        "- The false-alarm rate rests on only a few minutes of no-fall video. Treat the rows as a comparison "
        "between settings, not a field rate.",
        "- Your own recordings (`data/recordings/`) are added automatically when this is re-run.",
        *(markdown_heldout(heldout_rows, default_s) if heldout_rows else []),
        SECTION_END,
    ]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        marker = "## Fall detection accuracy (UR Fall Detection dataset)"
        block = "## Fall confirmation time sweep\n\n" + section + "\n\n"
        text = text.replace(marker, block + marker) if marker in text else text.rstrip() + "\n\n" + block
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    # Pickle the cache under the module's import name, so it also loads when this file is
    # imported (tests, notebooks) rather than run as __main__.
    sys.modules.setdefault("sweep_fall_confirm", sys.modules[__name__])
    Cached.__module__ = Fixes.__module__ = "sweep_fall_confirm"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", default=",".join(f"{v:g}" for v in DEFAULT_VALUES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", metavar="MD")
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs", "sweep_fall_cache.pkl"),
                    help="reuse cached poses (delete it, or pass --refresh, after changing the models or pose code)")
    ap.add_argument("--refresh", action="store_true", help="re-run the models even if a cache exists")
    args = ap.parse_args()
    values = [float(v) for v in args.values.split(",") if v.strip()]

    labels = os.path.join(ef.DATA_DIR, "urfall-cam0-falls.csv")
    if not os.path.isfile(labels):
        print("No URFD data. Run: python training/eval_fall.py --download")
        return 1
    onsets = ef.fall_onsets(labels)
    models = ef.Models(args.device)
    size = (models.cfg.frame_width, models.cfg.frame_height)

    # Cached poses per video, keyed by the file's size and mtime plus a hash of the code that
    # produces them; a new recording or a changed pose/detector file is recomputed.
    stamp = code_stamp()
    cache: dict = {}
    if os.path.isfile(args.cache) and not args.refresh:
        with open(args.cache, "rb") as f:
            loaded = pickle.load(f)
        cache = loaded.get("videos", {}) if loaded.get("stamp") == stamp else {}

    def get(path, make):
        st = os.stat(path)
        key = (path, st.st_size, int(st.st_mtime))
        if key not in cache:
            cache[key] = make()
        return cache[key]

    videos: list[Cached] = []
    t0 = time.time()
    for i in range(1, ef.N_FALLS + 1):
        name = f"fall-{i:02d}"
        zip_path = os.path.join(ef.DATA_DIR, f"{name}-cam0-rgb.zip")
        if os.path.isfile(zip_path) and onsets.get(name):
            videos.append(get(zip_path, lambda zp=zip_path, n=name: cache_frames(
                models, ef.frames_from_zip(zp), n, "fall", ef.FPS, onsets[n])))
    for i in range(1, ef.N_ADLS + 1):
        zip_path = os.path.join(ef.DATA_DIR, f"adl-{i:02d}-cam0-rgb.zip")
        if os.path.isfile(zip_path):
            videos.append(get(zip_path, lambda zp=zip_path, n=f"adl-{i:02d}": cache_frames(
                models, ef.frames_from_zip(zp), n, "adl", ef.FPS)))
    clips = [(v, "sample") for v in efa.load_manifest()] + [(v, "recording") for v in efa.load_recordings()]
    for v, kind in clips:
        meta_path = os.path.splitext(efa._abs(v.path))[0] + ".json"
        if kind == "recording" and os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                if json.load(f).get("contains_falls"):
                    continue
        path = efa._abs(v.path)
        if not os.path.isfile(path):
            continue
        def make(p=path, k=kind):
            frames, fps = video_frames(p)
            return cache_frames(models, frames, os.path.basename(p), k, fps, resize=size)

        videos.append(get(path, make))

    # CAUCAFall: the held-out set (training/fetch_caucafall.py). Kept apart from everything above.
    heldout: list[Cached] = []
    for video, labels, rel in caucafall_videos():
        onset = caucafall_onset(labels)
        kind = "fall" if os.path.basename(os.path.dirname(video)).lower().startswith("fall") else "caucafall_adl"
        if kind == "fall" and onset is None:
            print(f"skip {rel}: no frame labelled fall")
            continue

        def make(p=video, k=kind, n=rel, o=onset):
            frames, fps = video_frames(p)
            return cache_frames(models, frames, n, k, fps, onset_frame=o if k == "fall" else None, resize=size)

        heldout.append(get(video, make))
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    with open(args.cache, "wb") as f:
        pickle.dump({"stamp": stamp, "videos": cache}, f)
    print(f"{len(videos)} videos ready in {time.time() - t0:.0f}s (cache: {args.cache})", flush=True)

    rows = [score_setting(videos, models.cfg.fall, v) for v in values]
    for r in rows:
        print(f"{r['confirm_s']:>4g}s  recall {r['tp']}/{r['falls']}  confirmable {r['confirmable']}  "
              f"FA {r['false_alarms_clean']} ({r['false_alarms_per_hour']:.1f}/h over "
              f"{r['clean_hours'] * 60:.1f} min)  missed-when-confirmable {r['confirmable_but_missed']}  "
              f"on-ground {r['on_ground']}/{r['falls']}", flush=True)
    default_s = models.cfg.fall.stillness_seconds
    fix_rows = [(label, score_setting(videos, models.cfg.fall, default_s, fixes=fx)) for label, fx in FIX_VARIANTS]
    print(f"\nfixes at {default_s:g}s:")
    for label, r in fix_rows:
        print(f"  {label:<34} catches {r['tp']}/{r['falls']}  lost {r['lost_from_view']}/{r['on_ground']}  "
              f"FA {r['false_alarms_clean']}  on-ground {r['on_ground']}", flush=True)
    out = os.path.join(ROOT, "outputs", "sweep_fall_confirm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"confirmation": rows, "fixes": [{"label": lb, **r} for lb, r in fix_rows]}, f, indent=2)
    heldout_rows = None
    if heldout:
        heldout_rows = [(label, score_setting(heldout, models.cfg.fall, default_s, fixes=fx))
                        for label, fx in (("Baseline (no fixes)", BASELINE), CANDIDATE)]
        print("\nheld-out CAUCAFall:")
        for label, r in heldout_rows:
            print(f"  {label:<58} confirmed {r['tp']}/{r['falls']}  FA {r['false_alarms_clean']}  "
                  f"possible {r['possible_tp']}/{r['falls']}  possible FA {r['possible_false_alarms']}  "
                  f"over {r['clean_hours'] * 60:.1f} min  by activity {r['false_alarms_by_activity']}", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"confirmation": rows, "fixes": [{"label": lb, **r} for lb, r in fix_rows],
                       "heldout_caucafall": [{"label": lb, **r} for lb, r in heldout_rows]}, f, indent=2)
    device = models.detector.device if hasattr(models.detector, "device") else args.device
    section = markdown(rows, default_s, str(device), fix_rows, heldout_rows)
    print("\n" + section)
    if args.write:
        write_section(args.write, section)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
