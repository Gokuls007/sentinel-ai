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
    frames: list = field(default_factory=list)  # [(ts, active_ids, [(tid, PoseResult, standing_h)])]

    @property
    def seconds(self) -> float:
        return len(self.frames) / self.fps


# --- caching (runs the models) ------------------------------------------------------------------

def code_stamp() -> str:
    """Hash of the code and thresholds that produce the cached poses."""
    h = hashlib.sha256()
    for rel in ("backend/core/pose_estimator.py", "backend/core/detector.py", "backend/config/settings.py",
                "backend/config/trackers/sentinel_bytetrack.yaml"):
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]

def cache_frames(models, frames_iter, name, kind, fps, onset_frame=None, resize=None) -> Cached:
    from core.pose_estimator import PoseResult

    out = Cached(name=name, kind=kind, fps=fps, onset_frame=onset_frame)
    models.fresh()  # reset tracker and pose history
    det, pose = models.detector, models.pose
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
        out.frames.append((ts, tuple(feats.keys()), rows))
    return out


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

def make_detector(fall_cfg, confirm_s: float):
    from anomaly.fall_detector import FallDetector

    return FallDetector(
        descent_speed_threshold=fall_cfg.descent_speed_threshold,
        aspect_ratio_threshold=fall_cfg.aspect_ratio_threshold,
        head_drop_ratio=fall_cfg.head_drop_ratio,
        stillness_seconds=confirm_s,
        stillness_speed_threshold=fall_cfg.stillness_speed_threshold,
        fallen_timeout_seconds=max(fall_cfg.fallen_timeout_seconds, confirm_s + 2.0),
        cooldown_seconds=fall_cfg.cooldown_seconds,
    )


def replay(video: Cached, detector) -> dict:
    """Run the state machine over cached frames. Besides the alerts, it follows the first track
    that reached the ground, to explain a missing alert (see ``why_unconfirmed``)."""
    alerts: list[float] = []
    fallen_s = fallen_tid = None
    frames_after = seen_after = 0
    left_ground = False
    onset_s = (video.onset_frame - 1) / video.fps if video.onset_frame else None
    for ts, active, rows in video.frames:
        detector.prune(active)
        seen = False
        for tid, pose, standing_h in rows:
            if detector.check(tid, pose, SimpleNamespace(initial_standing_height=standing_h), ts):
                alerts.append(round(ts, 3))
            on_ground = detector.state_of(tid) in (detector.FALLEN, detector.CONFIRMED)
            if fallen_s is None and on_ground and (onset_s is None or ts >= onset_s - 2.0):
                fallen_s, fallen_tid = ts, tid
            elif tid == fallen_tid:
                seen = True
                left_ground = left_ground or not on_ground
        if fallen_s is not None and ts > fallen_s:
            frames_after += 1
            seen_after += seen
    return {"alerts": alerts, "fallen_s": fallen_s, "onset_s": onset_s,
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


def score_setting(videos: list[Cached], fall_cfg, confirm_s: float, tolerance_s: float = 2.0) -> dict:
    falls = [v for v in videos if v.kind == "fall"]
    clean = [v for v in videos if v.kind != "fall"]
    tp = reached = confirmable = 0
    false_alarms: dict[str, int] = {}
    missed: dict[str, int] = {}
    for v in falls:
        r = replay(v, make_detector(fall_cfg, confirm_s))
        hit = any(t >= r["onset_s"] - tolerance_s for t in r["alerts"])
        tp += hit
        if r["fallen_s"] is not None:
            reached += 1
            # Enough video left after reaching the ground for this confirmation time.
            enough = r["end_s"] - r["fallen_s"] >= confirm_s
            confirmable += enough
            if enough and not hit:
                reason = why_unconfirmed(r)
                missed[reason] = missed.get(reason, 0) + 1
        false_alarms["fall clips (early/extra)"] = false_alarms.get("fall clips (early/extra)", 0) + \
            len(r["alerts"]) - int(hit)
    hours: dict[str, float] = {}
    clean_reached = 0
    for v in clean:
        r = replay(v, make_detector(fall_cfg, confirm_s))
        false_alarms[v.kind] = false_alarms.get(v.kind, 0) + len(r["alerts"])
        hours[v.kind] = hours.get(v.kind, 0.0) + v.seconds / 3600
        clean_reached += r["fallen_s"] is not None
    clean_hours = sum(hours.values())
    clean_fa = sum(n for k, n in false_alarms.items() if k in hours)
    return {
        "confirm_s": confirm_s, "falls": len(falls), "tp": tp, "recall": tp / len(falls) if falls else None,
        "confirmable": confirmable, "on_ground": reached, "on_ground_recall": reached / len(falls) if falls else None,
        "false_alarms": false_alarms, "clean_hours": clean_hours, "hours_by_kind": hours,
        "false_alarms_clean": clean_fa, "false_alarms_per_hour": clean_fa / clean_hours if clean_hours else None,
        "clean_videos_reaching_ground": clean_reached, "clean_videos": len(clean),
        "confirmable_but_missed": missed,
    }


# --- report -------------------------------------------------------------------------------------

KIND_LABEL = {"adl": "URFD ADL", "sample": "sample clips", "recording": "your recordings"}


def markdown(rows: list[dict], default_s: float, device: str) -> str:
    r0 = rows[0]
    basis = ", ".join(f"{KIND_LABEL.get(k, k)} {h * 60:.1f} min" for k, h in r0["hours_by_kind"].items())
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python training/sweep_fall_confirm.py` on {device}._ "
        f"The confirmation time is how long a person must lie still on the ground before the alert. "
        f"The current default is **{default_s:g} s**. Detection and pose ran once per video, and only the "
        "fall state machine was replayed at each setting. The \"gave up\" timeout is at least "
        "confirmation + 2 s.",
        "",
        f"No-fall footage: **{r0['clean_hours'] * 60:.1f} min** ({basis}).",
        "",
        "| Confirmation time | URFD recall (confirmed alerts) | Confirmable on URFD | Confirmable but missed (why) "
        "| False alarms / hour (no-fall footage) | False alarms (count) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " (default)" if abs(r["confirm_s"] - default_s) < 1e-9 else ""
        fa = r["false_alarms_per_hour"]
        why = ", ".join(f"{n} {k}" for k, n in sorted(r["confirmable_but_missed"].items(), key=lambda x: -x[1]))
        lines.append(
            f"| {r['confirm_s']:g} s{mark} | {r['tp']} / {r['falls']} ({r['recall']:.0%}) | "
            f"{r['confirmable']} / {r['falls']} | {why or 'none'} | {'n/a' if fa is None else f'{fa:.1f}'} | "
            f"{r['false_alarms_clean']} |")
    lines += [
        "",
        f"**\"On the ground\" stage recall** (reached FALLEN, the step before confirmation; it is the same at "
        f"every setting): **{r0['on_ground']} / {r0['falls']} ({r0['on_ground_recall']:.0%})** of URFD falls. "
        f"{r0['clean_videos_reaching_ground']} of {r0['clean_videos']} no-fall videos also reached that stage "
        "without confirming.",
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
    out = os.path.join(ROOT, "outputs", "sweep_fall_confirm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    device = models.detector.device if hasattr(models.detector, "device") else args.device
    section = markdown(rows, models.cfg.fall.stillness_seconds, str(device))
    print("\n" + section)
    if args.write:
        write_section(args.write, section)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
