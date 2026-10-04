"""Evaluate the live activity labels on CAUCAFall clips.

    python scripts/eval_activity.py --subjects 1-5            # tuning set (the only one used to tune)
    python scripts/eval_activity.py --subjects 6-10           # held out (also the fall test set)
    python scripts/eval_activity.py --subjects 1-5 --write docs/BENCHMARKS.md

Each clip runs through the production path (YOLOv8n + ByteTrack, YOLOv8n-pose, the fall
detector, the activity tracker); the main (largest) person's smoothed label is recorded per
frame. A clip passes when its expected label is shown for at least ``--min-s`` seconds:

| Clip            | Expected                                   |
|-----------------|--------------------------------------------|
| Walk            | Walking                                    |
| Sit down        | Sitting                                    |
| Pick up object  | Bending, and Lifting afterwards            |
| Fall *          | Fallen or Lying down                       |
| Hop, Kneel      | (reported) never Fallen                    |

CAUCAFall is filmed from the front/side at room scale; real warehouse cameras differ, so this
checks the rules work end to end, not field accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from itertools import pairwise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.join(ROOT, "training"))

SECTION_START = "<!-- benchmark:activity:start -->"
SECTION_END = "<!-- benchmark:activity:end -->"


def expected(activity: str) -> tuple[list[list[str]], list[str]]:
    """(label groups that must each appear, labels that must never appear)."""
    a = activity.lower()
    if a.startswith("fall"):
        return [["Fallen", "Lying down"]], []
    if a == "walk":
        return [["Walking"]], ["Fallen"]
    if a == "sit down":
        return [["Sitting"]], ["Fallen"]
    if a == "pick up object":
        return [["Bending"], ["Lifting"]], ["Fallen"]
    return [], ["Fallen"]  # hop, kneel: only "no false fall"


def grade(labels: list[tuple[float, str]], activity: str, min_s: float) -> tuple[bool, list[str]]:
    must, never = expected(activity)
    secs = Counter()
    for (t0, lbl), (t1, _n) in pairwise(labels):
        secs[lbl] += t1 - t0
    problems = [f"no {' or '.join(g)}" for g in must if sum(secs[x] for x in g) < min_s]
    problems += [f"showed {x}" for x in never if secs[x] >= min_s]
    return not problems, problems


def extract(models, path: str) -> list[dict]:
    """Per frame: the main (largest) person's keypoints, box, body height, fall state, and the
    objects a person could carry. Cached, so rule changes can be re-scored without the models."""
    import cv2

    from activity import CARRY_CLASSES

    models.detector.reset_tracker()
    models.pose.track_features.clear()
    falls = models.fresh()
    models.detector.classes = sorted({0} | {cid for cid, n in models.detector.COCO_NAMES.items() if n in CARRY_CLASSES})
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = models.cfg.frame_width, models.cfg.frame_height
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        ts = idx / fps
        idx += 1
        dets = models.detector.detect_and_track(frame)
        people = [d for d in dets.detections if d.class_name == "person" and d.track_id is not None]
        poses = models.pose.estimate(frame, [d.track_id for d in people], [d.bbox for d in people], ts)
        feats = models.pose.get_all_features()
        falls.prune(feats.keys())
        for tid, p in poses.items():
            if tid in feats:
                falls.check(tid, p, feats[tid], ts)
        if not poses:
            continue
        tid, p = max(poses.items(), key=lambda kv: (kv[1].bbox[2] - kv[1].bbox[0]) * (kv[1].bbox[3] - kv[1].bbox[1]))
        objects = [(d.class_name, [float(v) for v in d.bbox]) for d in dets.detections if d.class_name in CARRY_CLASSES]
        frames.append({"ts": ts, "tid": int(tid), "kp": p.keypoints.tolist(), "bh": float(p.body_height),
                       "box": [float(v) for v in p.bbox], "fall": falls.state_of(tid), "objects": objects})
    cap.release()
    return frames


def score_frames(frames: list[dict]) -> list[tuple[float, str]]:
    import numpy as np

    from activity import ActivityTracker

    act = ActivityTracker()
    labels = []
    for f in frames:
        out = act.update(f["tid"], np.array(f["kp"]), f["ts"], f["bh"], fallen=f["fall"] == "confirmed",
                         objects=[(n, tuple(b)) for n, b in f["objects"]], fall_state=f["fall"],
                         box=tuple(f["box"]))
        labels.append((f["ts"], out["label"]))
    return labels


def parse_subjects(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="1-5")
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", metavar="MD")
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs", "activity_cache"),
                    help="per-clip keypoint cache (re-score rule changes without running the models)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from eval_fall import Models
    from sweep_fall_confirm import caucafall_videos

    subjects = set(parse_subjects(args.subjects))
    clips = [v for v in caucafall_videos() if int(v[2].split("/")[0].split(".")[1]) in subjects]
    import json

    os.makedirs(args.cache, exist_ok=True)
    models = None
    rows = []
    for path, _labels, rel in clips:
        activity = rel.split("/")[1]
        cache = os.path.join(args.cache, rel.replace("/", "__").replace(" ", "_") + ".json")
        if os.path.isfile(cache):
            with open(cache, encoding="utf-8") as f:
                frames = json.load(f)
        else:
            models = models or Models(args.device)
            frames = extract(models, path)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(frames, f)
        labels = score_frames(frames)
        ok, problems = grade(labels, activity, args.min_s)
        shown = Counter(lbl for _t, lbl in labels).most_common(4)
        lift_s = sum(t1 - t0 for (t0, lbl), (t1, _n) in pairwise(labels) if lbl == "Lifting")
        false_lift = activity.lower() in ("walk", "sit down", "hop", "kneel") and lift_s >= args.min_s
        rows.append({"clip": rel, "activity": activity, "ok": ok, "problems": problems, "shown": shown,
                     "false_lift": false_lift})
        shown_text = ", ".join(f"{k} {v}" for k, v in shown)
        print(f"{'PASS' if ok else 'FAIL'} {rel}: {', '.join(problems) or 'ok'}  [{shown_text}]", flush=True)

    by_act: dict[str, list[int]] = {}
    for r in rows:
        key = "Fall (any direction)" if r["activity"].lower().startswith("fall") else r["activity"]
        by_act.setdefault(key, [0, 0])
        by_act[key][0] += r["ok"]
        by_act[key][1] += 1
    start = SECTION_START.replace("activity", f"activity-{args.subjects}")
    end = SECTION_END.replace("activity", f"activity-{args.subjects}")
    tuned = parse_subjects(args.subjects) == [1, 2, 3, 4, 5]
    lines = [start,
             f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/eval_activity.py --subjects {args.subjects}`._ "
             f"CAUCAFall subjects {args.subjects} "
             f"({'the tuning set' if tuned else 'held out: never used for tuning'}), "
             f"{len(rows)} clips. The rules were tuned **only on subjects 1-5**; subjects 6-10 are held out (they are "
             f"also the fall test set). A clip passes when its expected label shows for at least {args.min_s:g} s "
             "(main person, smoothed labels).",
             "", "| Clip type | Expected | Passed |", "|---|---|---|"]
    expect_text = {"Walk": "Walking", "Sit down": "Sitting", "Pick up object": "Bending, then Lifting",
                   "Fall (any direction)": "Fallen or Lying down", "Hop": "no false Fallen",
                   "Kneel": "no false Fallen"}
    for act, (k, n) in sorted(by_act.items()):
        lines.append(f"| {act} | {expect_text.get(act, '-')} | {k} / {n} |")
    others = [r for r in rows if r["activity"].lower() in ("walk", "sit down", "hop", "kneel")]
    lines += ["", f"False \"Lifting\" (shown for {args.min_s:g} s or more in Walk, Sit down, Hop or Kneel clips): "
                  f"{sum(r['false_lift'] for r in others)} / {len(others)} clips."]
    fails = [r for r in rows if not r["ok"]]
    if fails:
        lines += ["", "Failures: " + "; ".join(f"`{r['clip']}` ({', '.join(r['problems'])})" for r in fails)]
    lines += ["", "Front-on room-scale clips, not warehouse footage: this checks the rules end to end, not field "
                  "accuracy.", end]
    md = "\n".join(lines)
    print(md)
    if args.write:
        with open(args.write, encoding="utf-8") as f:
            text = f.read()
        if start in text:
            before, rest = text.split(start, 1)
            text = before + md + rest.split(end, 1)[1]
        else:
            title = "tuning set" if tuned else "held out"
            heading = f"## Activity labels (CAUCAFall subjects {args.subjects}, {title})"
            text = text.rstrip() + f"\n\n{heading}\n\n" + md + "\n"
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
