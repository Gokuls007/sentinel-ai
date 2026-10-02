"""Last fall-detection round: diagnosis, direction-independent signals, learned model.

    python training/sweep_fall_confirm.py            # builds the pose cache first
    python training/fall_round3.py diagnose          # tuning subjects only
    python training/fall_round3.py final --write docs/BENCHMARKS.md

The split is **by subject**, so no person appears on both sides:
- **Tuning / training:** URFD (all of it) plus CAUCAFall subjects 1-5.
- **Test:** CAUCAFall subjects 6-10, used only for the final table.

Everything replays the cached poses from sweep_fall_confirm.py; no model inference runs here.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import re
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sweep_fall_confirm as sw

TUNE_SUBJECTS = {1, 2, 3, 4, 5}
TEST_SUBJECTS = {6, 7, 8, 9, 10}
CACHE = os.path.join(ROOT, "outputs", "sweep_fall_cache.pkl")


def subject_of(v) -> int | None:
    m = re.match(r"Subject\.(\d+)/", v.name)
    return int(m.group(1)) if m else None


def load_videos(path: str = CACHE):
    sys.modules.setdefault("sweep_fall_confirm", sw)
    with open(path, "rb") as f:
        videos = list(pickle.load(f)["videos"].values())
    urfd = [v for v in videos if subject_of(v) is None and v.kind in ("fall", "adl")]
    clips = [v for v in videos if v.kind in ("sample", "recording")]
    cauca = [v for v in videos if subject_of(v) is not None]
    tune = [v for v in cauca if subject_of(v) in TUNE_SUBJECTS]
    test = [v for v in cauca if subject_of(v) in TEST_SUBJECTS]
    return urfd, clips, tune, test


# --- 1. diagnosis --------------------------------------------------------------------------------

def track_series(v, after_s: float | None = None):
    """The main person's per-frame poses: the track with the most frames (falls have one subject)."""
    counts = Counter(tid for _ts, _a, rows, *_ in v.frames for tid, _p, _h in rows)
    if not counts:
        return None, []
    tid = counts.most_common(1)[0][0]
    out = []
    for ts, _active, rows, *_ in v.frames:
        pose = next(((p, h) for t, p, h in rows if t == tid), None)
        out.append((ts, pose))
    return tid, out


def on_screen_direction(series, onset_s: float) -> str:
    """How the fall looks on screen: the torso rotates (sideways to the camera) or the body just
    gets shorter (toward / away from the camera)."""
    before = [p.bbox[3] - p.bbox[1] for ts, ph in series if ph and onset_s - 1.5 <= ts < onset_s for p in [ph[0]]]
    after = [(p.bbox[3] - p.bbox[1], p.torso_angle) for ts, ph in series if ph and onset_s <= ts <= onset_s + 3.0
             for p in [ph[0]]]
    if not before or not after:
        return "not visible"
    h0 = float(np.median(before))
    max_torso = max((a for _h, a in after if a is not None), default=None)
    min_h = min(h for h, _a in after)
    if max_torso is not None and max_torso >= 60:
        return "sideways (rotates)"
    if min_h <= 0.65 * h0:
        return "toward/away (shortens)"
    return "unclear"


def diagnose_clip(v, fall_cfg) -> dict:
    """Why a fall clip never reached the on-the-ground stage (baseline rules)."""
    det = sw.make_detector(fall_cfg, 1.0)
    r = sw.replay(v, det)
    onset = r["onset_s"]
    _tid, series = track_series(v)
    window = [(ts, ph) for ts, ph in series if onset - 0.5 <= ts <= onset + 3.0]
    seen = sum(1 for _ts, ph in window if ph)
    calibrated = any(ph and ph[1] > 0 for ts, ph in series if ts <= onset + 0.5)
    # Replay once more, recording the signals the state machine saw.
    det2 = sw.make_detector(fall_cfg, 1.0)
    max_descent, entered_falling, horiz, head_drop = 0.0, False, False, False
    for ts, active, rows, *_ in v.frames:
        det2.prune(active)
        for t, p, h in rows:
            det2.check(t, p, SimpleNamespace(initial_standing_height=h), ts)
            st = det2.tracks.get(t)
            if st and st.last_signals and onset - 0.5 <= ts <= onset + 3.0:
                max_descent = max(max_descent, st.last_signals["descent_speed"])
                horiz = horiz or bool(st.last_signals["horizontal_pose"])
                head_drop = head_drop or bool(st.last_signals["head_dropped"])
            if det2.state_of(t) == det2.FALLING and onset - 0.5 <= ts:
                entered_falling = True
    if r["fallen_s"] is not None:
        reason = "reached the ground"
    elif not calibrated:
        reason = "never calibrated (no standing height before the fall)"
    elif seen < 0.5 * max(1, len(window)):
        reason = "person not tracked during the fall"
    elif not entered_falling:
        reason = f"descent too slow (peak {max_descent:.2f} < {fall_cfg.descent_speed_threshold} body heights/s)"
    elif not horiz:
        reason = "descended, but the torso never looked horizontal"
    elif not head_drop:
        reason = "descended and horizontal, but the head never dropped enough"
    else:
        reason = "descended, but not horizontal and head-down at the same time within 1.5 s"
    return {"name": v.name, "type": v.name.split("/")[-1], "direction": on_screen_direction(series, onset),
            "reason": reason, "peak_descent": round(max_descent, 2)}


def diagnose(write: str | None = None) -> str:
    from config.settings import FallDetectorConfig

    _urfd, _clips, tune, _test = load_videos()
    cfg = FallDetectorConfig()
    rows = [diagnose_clip(v, cfg) for v in tune if v.kind == "fall"]
    missed = [r for r in rows if r["reason"] != "reached the ground"]
    reasons = Counter(re.sub(r" \(peak.*", "", r["reason"]) for r in missed)
    lines = [
        f"CAUCAFall tuning subjects 1-5: {len(rows)} falls, {len(rows) - len(missed)} reached the ground, "
        f"{len(missed)} did not.", "", "Why (missed falls):", ""]
    lines += [f"- {n} x {k}" for k, n in reasons.most_common()]
    lines += ["", "By on-screen direction (all falls: reached / total):", ""]
    for d in sorted({r["direction"] for r in rows}):
        group = [r for r in rows if r["direction"] == d]
        lines.append(f"- {d}: {sum(r['reason'] == 'reached the ground' for r in group)} / {len(group)}")
    lines += ["", "By fall type (folder label):", ""]
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        lines.append(f"- {t}: {sum(r['reason'] == 'reached the ground' for r in group)} / {len(group)}")
    lines += ["", "Per clip:", ""] + [f"- {r['name']}: {r['direction']}; {r['reason']}" for r in rows]
    text = "\n".join(lines)
    if write:
        with open(write, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


# --- 2. rule variants ---------------------------------------------------------------------------

RULES = (
    ("Rules: baseline", sw.BASELINE),
    ("Rules: frozen candidate (rotated retry + hysteresis, presence)", sw.CANDIDATE[1]),
    ("Rules: + direction-independent signals", sw.Fixes(ground_mode="combined")),
    ("Rules: + direction-independent signals + box calibration", sw.Fixes(ground_mode="combined",
                                                                          box_calibration=True)),
    ("Rules: candidate + direction-independent + box calibration",
     sw.Fixes(0.0, False, True, 0.5, "presence", "combined", True)),
)


def rule_rows(videos, confirm_s: float = 1.0):
    from config.settings import FallDetectorConfig

    cfg = FallDetectorConfig()
    return [(label, sw.score_setting(videos, cfg, confirm_s, fixes=fx)) for label, fx in RULES]


# --- 3. learned model -----------------------------------------------------------------------------

FEATURES = ("descent_now", "descent_max_1s", "drop_from_start", "shrink", "shrink_min_1s", "aspect",
            "torso", "spread", "hip_height", "head_drop", "hip_std_05s", "visible_kps")
WINDOW_S = 1.0


def video_features(v) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (frame, track) features from the cached poses. Returns X, labels, timestamps, track ids.
    Label 1 = a fall clip at or after the labelled onset; 0 = anything else; -1 = the 0.5 s just
    before onset (ambiguous, excluded from training)."""
    onset = (v.onset_frame - 1) / v.fps if v.kind == "fall" and v.onset_frame else None
    hist: dict[int, list] = {}
    xs, ys, ts_out, tids = [], [], [], []
    for ts, _active, rows, *_ in v.frames:
        for tid, p, standing_h in rows:
            h = hist.setdefault(tid, [])
            box_h = float(p.bbox[3] - p.bbox[1])
            hip = p._mean_visible(11, 12)
            hip_y = float(hip[1]) if hip is not None else float((p.bbox[1] + p.bbox[3]) / 2)
            nose_y = float(p.keypoints[0, 1]) if p.keypoints[0, 2] >= 0.3 else None
            h.append((ts, box_h, hip_y, nose_y, p, standing_h))
            ref_box = max(x[1] for x in h)  # tallest box seen = standing reference
            scale = standing_h if standing_h > 0 else 0.9 * ref_box
            if scale <= 0:
                continue
            recent = [x for x in h if ts - x[0] <= WINDOW_S]
            half = [x for x in h if ts - x[0] <= 0.5]
            first = h[: max(1, min(len(h), 10))]
            start_hip = float(np.median([x[2] for x in first]))
            descents = []
            for a, b in itertools.pairwise(recent):
                if b[0] > a[0]:
                    descents.append((b[2] - a[2]) / (b[0] - a[0]) / scale)
            upright_noses = [x[3] for x in first if x[3] is not None]
            kp = p.keypoints
            vis = kp[kp[:, 2] >= 0.3]
            ankles = kp[[15, 16]]
            ankles = ankles[ankles[:, 2] >= 0.3]
            torso = p.torso_angle
            feats = {
                "descent_now": descents[-1] if descents else 0.0,
                "descent_max_1s": max(descents) if descents else 0.0,
                "drop_from_start": (hip_y - start_hip) / scale,
                "shrink": box_h / ref_box,
                "shrink_min_1s": min(x[1] for x in recent) / ref_box,
                "aspect": float(p.bbox[2] - p.bbox[0]) / box_h if box_h > 0 else 0.0,
                "torso": torso if torso is not None else -1.0,
                "spread": float(vis[:, 1].max() - vis[:, 1].min()) / scale if len(vis) >= 3 else -1.0,
                "hip_height": (float(ankles[:, 1].mean()) - hip_y) / scale if len(ankles) and hip is not None
                else -1.0,
                "head_drop": (nose_y - float(np.median(upright_noses))) / scale
                if nose_y is not None and upright_noses else -9.0,
                "hip_std_05s": float(np.std([x[2] for x in half])) / scale if len(half) >= 3 else -1.0,
                "visible_kps": float(len(vis)),
            }
            xs.append([feats[k] for k in FEATURES])
            if onset is None:
                ys.append(0)
            else:
                ys.append(1 if ts >= onset else (-1 if ts >= onset - 0.5 else 0))
            ts_out.append(ts)
            tids.append(tid)
    return (np.array(xs, float).reshape(-1, len(FEATURES)), np.array(ys, int), np.array(ts_out, float),
            np.array(tids, int))


def train_model(videos, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = [], []
    for v in videos:
        xv, yv, _t, _i = video_features(v)
        keep = yv >= 0
        X.append(xv[keep])
        y.append(yv[keep])
    X, y = np.vstack(X), np.concatenate(y)
    pos = max(1, int(y.sum()))
    weights = np.where(y == 1, (len(y) - pos) / pos, 1.0)
    model = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(X, y, sample_weight=weights)
    return model


def model_events(model, v, threshold: float, possible_s: float = 0.3, confirm_s: float = 1.0):
    """Alerts from per-frame probabilities: "possible" once a track stays above the threshold
    for ``possible_s``, "confirmed" once it stays above for ``possible_s + confirm_s``. Any
    frame below the threshold resets that track's streak; frames where the track has no pose
    are tolerated up to 0.5 s."""
    X, _y, ts, tids = video_features(v)
    if not len(X):
        return [], []
    prob = model.predict_proba(X)[:, 1]
    possible, confirmed = [], []
    streak: dict[int, tuple[float, float]] = {}  # tid -> (streak start, last above time)
    fired: dict[int, set] = {}
    last_p: dict[str, float] = {"possible": -1e9, "confirmed": -1e9}
    for p, t, tid in zip(prob, ts, tids, strict=True):
        start, last = streak.get(tid, (None, None))
        if p >= threshold:
            if start is None or (last is not None and t - last > 0.5):
                start = t
                fired[tid] = set()
            streak[tid] = (start, t)
            dur = t - start
            if dur >= possible_s and "p" not in fired[tid] and t - last_p["possible"] >= 10:
                fired[tid].add("p")
                last_p["possible"] = t
                possible.append(t)
            if dur >= possible_s + confirm_s and "c" not in fired[tid] and t - last_p["confirmed"] >= 30:
                fired[tid].add("c")
                last_p["confirmed"] = t
                confirmed.append(t)
        else:
            streak[tid] = (None, None)  # below the threshold: the streak is broken
    return possible, confirmed


def score_model(model, videos, threshold: float, tolerance_s: float = 2.0) -> dict:
    falls = [v for v in videos if v.kind == "fall"]
    clean = [v for v in videos if v.kind != "fall"]
    tp = ptp = fa = pfa = 0
    by_activity: dict[str, list[int]] = {}
    for v in falls:
        poss, conf = model_events(model, v, threshold)
        onset = (v.onset_frame - 1) / v.fps
        tp += any(t >= onset - tolerance_s for t in conf)
        ptp += any(t >= onset - tolerance_s for t in poss)
    hours = sum(v.seconds for v in clean) / 3600
    for v in clean:
        poss, conf = model_events(model, v, threshold)
        fa += len(conf)
        pfa += len(poss)
        act = by_activity.setdefault(sw.activity_of(v), [0, 0])
        act[0] += len(conf)
        act[1] += len(poss)
    n = len(falls)
    return {"falls": n, "tp": tp, "recall": tp / n if n else None, "possible_tp": ptp,
            "possible_recall": ptp / n if n else None, "false_alarms_clean": fa, "possible_false_alarms": pfa,
            "clean_hours": hours, "false_alarms_per_hour": fa / hours if hours else None,
            "possible_false_alarms_per_hour": pfa / hours if hours else None, "false_alarms_by_activity": by_activity,
            "clean_videos": len(clean), "threshold": threshold}


def choose_threshold(videos, folds: int = 5, seed: int = 0) -> tuple[float, list]:
    """Grouped cross-validation on the training videos only (a video never in both its train
    and validation fold). Picks the threshold that maximises confirmed recall minus confirmed
    false alarms per no-fall video."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    grid = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    totals = {th: [0, 0, 0, 0] for th in grid}  # tp, falls, fa, clean videos
    for k in range(folds):
        val = [videos[i] for i in order[k::folds]]
        train = [videos[i] for i in order if i not in set(order[k::folds])]
        model = train_model(train, seed)
        for th in grid:
            s = score_model(model, val, th)
            totals[th][0] += s["tp"]
            totals[th][1] += s["falls"]
            totals[th][2] += s["false_alarms_clean"]
            totals[th][3] += s["clean_videos"]
    table = [(th, t[0] / max(1, t[1]) - t[2] / max(1, t[3]), t) for th, t in totals.items()]
    best = max(table, key=lambda x: x[1])[0]
    return best, table


# --- 4. final table ------------------------------------------------------------------------------

def final(write: str | None = None) -> str:
    urfd, clips, tune, test = load_videos()
    train = urfd + clips + tune
    threshold, cv = choose_threshold(train)
    model = train_model(train)
    learned = (f"Learned: gradient boosting on pose features (threshold {threshold:g}, "
               "chosen by cross-validation on training videos)")
    rows = [*rule_rows(test), (learned, score_model(model, test, threshold))]
    tune_rows = rule_rows(tune)
    summary = diagnose().split("\n\nPer clip:")[0]
    lines = [SECTION_START, *final_markdown(rows, tune_rows, test, train, cv, threshold), "",
             "**Diagnosis (tuning subjects only): why falls never reached the on-the-ground stage** "
             "(baseline rules):", "", summary, "",
             "Most misses are visibility problems (the person isn't tracked during the fall, or never "
             "calibrated), not fall direction: sideways falls reach the ground in most clips, and only one clip "
             "is clearly toward or away from the camera.", SECTION_END]
    text = "\n".join(lines)
    if write:
        sw_write(write, text)
    return text


SECTION_START = "<!-- benchmark:fall-final:start -->"
SECTION_END = "<!-- benchmark:fall-final:end -->"


def _row(label, r) -> str:
    rate = sw._rate
    return (f"| {label} | {r['tp']} / {r['falls']} ({r['recall']:.0%}) | {rate(r['false_alarms_per_hour'])} "
            f"({r['false_alarms_clean']}) | {r['possible_tp']} / {r['falls']} ({r['possible_recall']:.0%}) | "
            f"{rate(r['possible_false_alarms_per_hour'])} ({r['possible_false_alarms']}) |")


def final_markdown(rows, tune_rows, test, train, cv, threshold) -> list[str]:
    from datetime import datetime

    hours = rows[0][1]["clean_hours"] * 60
    falls_train = sum(v.kind == "fall" for v in train)
    acts = sorted({a for _l, r in rows for a in r["false_alarms_by_activity"]})
    lines = [
        f"_Measured {datetime.now():%Y-%m-%d} with `python training/fall_round3.py final`._ "
        "**Test set: CAUCAFall subjects 6-10 only** (25 falls, 25 daily activities, "
        f"{hours:.1f} min of no-fall video). These people were never used to design, tune or train anything. "
        f"Training and tuning used URFD, the sample clips and CAUCAFall subjects 1-5 ({falls_train} falls). "
        "Default 1 s confirmation.",
        "",
        "| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall "
        "| Possible false alarms / h (count) |",
        "|---|---|---|---|---|",
        *[_row(label, r) for label, r in rows],
        "",
        "False alarms on the test subjects by activity (confirmed / possible):",
        "",
        "| Method | " + " | ".join(acts) + " |",
        "|---|" + "---|" * len(acts),
    ]
    for label, r in rows:
        counts = [r["false_alarms_by_activity"].get(a, [0, 0]) for a in acts]
        lines.append(f"| {label.split(' (')[0]} | " + " | ".join(f"{c} / {p}" for c, p in counts) + " |")
    lines += [
        "",
        "Tuning half (CAUCAFall subjects 1-5), for reference. Rules only; the learned model was trained on "
        "these subjects:",
        "",
        "| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall "
        "| Possible false alarms / h (count) |",
        "|---|---|---|---|---|",
        *[_row(label, r) for label, r in tune_rows],
        "",
        f"Learned model: `HistGradientBoostingClassifier` (200 trees, depth 4) on {len(FEATURES)} per-frame "
        f"pose features over a {WINDOW_S:g} s window ({', '.join(FEATURES)}). "
        "Possible = probability above the threshold for 0.3 s; confirmed = above it for a further 1 s. "
        "Threshold grid with grouped 5-fold cross-validation on training videos (recall minus false alarms per "
        "no-fall video): " + ", ".join(f"{th:g} → {score:+.2f}" for th, score, _t in cv) + ".",
        "",
        "Notes on the learned model:",
        "- Its \"confirmed\" level means the probability stayed above the threshold for 1.3 s. It has no "
        "separate stillness check like the rules, so the two confirmed levels are not identical in meaning.",
        "- It runs offline only (this script). It isn't wired into the live pipeline.",
        "- An earlier run had a bug: a dip below the threshold shorter than 0.5 s didn't break a streak. "
        "These numbers are from the corrected code; the rule rows were unaffected.",
        "",
        f"With only {sum(v.kind == 'fall' for v in test)} test falls and {hours:.1f} min of no-fall video, "
        "one fall is 4 points of recall and one false alarm is about "
        f"{60 / max(hours, 1e-9):.0f} per hour. Differences of one or two events are noise.",
    ]
    return lines


def sw_write(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        marker = "## Fall confirmation time sweep"
        block = "## Fall detection: rules vs learned model (unseen subjects)\n\n" + section + "\n\n"
        text = text.replace(marker, block + marker) if marker in text else text.rstrip() + "\n\n" + block
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["diagnose", "tune", "final"])
    ap.add_argument("--write")
    args = ap.parse_args()
    if args.what == "diagnose":
        print(diagnose(args.write))
    elif args.what == "tune":
        urfd, clips, tune, _test = load_videos()
        for name, vids in (("URFD + clips", urfd + clips), ("CAUCAFall subjects 1-5", tune)):
            print(f"\n== {name} (tuning data)")
            for label, r in rule_rows(vids):
                print(f"  {label:<62} conf {r['tp']}/{r['falls']} FA {r['false_alarms_clean']}  "
                      f"poss {r['possible_tp']}/{r['falls']} pFA {r['possible_false_alarms']}  "
                      f"{r['false_alarms_by_activity']}")
    else:
        print(final(args.write))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
