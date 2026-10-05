"""Check the seated rule on public footage: CAUCAFall "Sit down" (hand-marked seated / standing
intervals) and "Walk" (no sitting at all), old rules vs current.

    python scripts/eval_activity.py --subjects 1-10     # builds the keypoint cache first
    python scripts/validate_seated.py

The seated intervals below were marked from 16-frame contact sheets of each clip (~0.6 s apart), with
the moves in and out of the chair left out; "standing" is upright before the person reaches the
chair (and after standing up again). Chairs come from YOLO11m (the home-mode COCO model, chair floor
0.25), cached in outputs/seat_cache. Subjects 6-10 are reported only: nothing is tuned on them.

Per frame: the activity label, the shin/thigh ratio (``_shank_ratio``) and whether a seat was seen.
``--old REV`` scores the rules from that git revision too (default: before the shin/thigh cue).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

CAUCA = os.path.join(ROOT, "data", "datasets", "caucafall", "CAUCAFall")
SEAT_CACHE = os.path.join(ROOT, "outputs", "seat_cache")
SEATS = {"chair": 0.25, "couch": 0.3, "bed": 0.3, "bench": 0.3, "toilet": 0.3}
# subject: ((seated start, end), [(standing start, end), ...]) in seconds
GT = {1: ((5.8, 9.3), [(0, 0.9)]), 2: ((3.2, 5.0), [(0, 1.2)]), 3: ((5.5, 12.8), [(0, 2.0)]),
      4: ((6.2, 9.5), [(0, 1.0), (2.4, 3.6)]), 5: ((7.0, 11.4), [(0, 1.0)]), 6: ((5.6, 9.0), [(0, 1.4)]),
      7: ((3.4, 8.5), [(0, 1.6), (9.5, 99)]), 8: ((4.2, 99), [(0, 2.0)]), 9: ((5.0, 11.0), [(0, 1.8)]),
      10: ((4.7, 10.0), [(0, 1.0)])}


def seat_boxes(video: str, det) -> list:
    os.makedirs(SEAT_CACHE, exist_ok=True)
    rel = os.path.relpath(video, CAUCA).replace(os.sep, "__").replace(" ", "_")
    path = os.path.join(SEAT_CACHE, rel + ".json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    import cv2

    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = det.predict(cv2.resize(frame, (640, 480)), conf=0.25, verbose=False)[0]
        boxes = zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf, strict=True)
        out.append([(r.names[int(c)], [float(v) for v in b]) for c, b, s in boxes
                    if r.names[int(c)] in SEATS and float(s) >= SEATS[r.names[int(c)]]])
    cap.release()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def old_tracker(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:backend/activity/rules.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "rules_old.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("rules_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rules_old"] = mod
    spec.loader.exec_module(mod)
    return mod.ActivityTracker


def frames_of(cache: str, subject: int, activity: str):
    with open(os.path.join(cache, f"Subject.{subject}__{activity.replace(' ', '_')}.json"), encoding="utf-8") as f:
        return json.load(f)


def score(tracker_cls, frames, seats, fps=20.0):
    from activity.rules import _shank_ratio

    a, rows = tracker_cls(), []
    for f in frames:
        objs = [(n, tuple(b)) for n, b in f["objects"]]
        i = round(f["ts"] * fps)
        if i < len(seats):
            objs += [(n, tuple(b)) for n, b in seats[i]]
        kp = np.array(f["kp"])
        out = a.update(f["tid"], kp, f["ts"], f["bh"], fallen=f["fall"] == "confirmed", objects=objs,
                       fall_state=f["fall"], box=tuple(f["box"]))
        rows.append((f["ts"], out["label"], _shank_ratio(kp, 0.3)))
    return rows


def main() -> int:
    from activity import ActivityTracker
    from eval_fall import setup_tag

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="0689fb9", help="git revision of the old rules (before the shin/thigh cue)")
    ap.add_argument("--cache", default=None, help="keypoint cache of scripts/eval_activity.py")
    args = ap.parse_args()
    cache = args.cache or os.path.join(ROOT, "outputs", "activity_cache", setup_tag())
    from ultralytics import YOLO

    det = YOLO(os.path.join(ROOT, "yolo11m.pt"))
    trackers = {"old": old_tracker(args.old), "current": ActivityTracker}
    for half, subjects in (("subjects 1-5 (tuning)", range(1, 6)), ("subjects 6-10 (held out)", range(6, 11))):
        print(f"\n{half}")
        for name, cls in trackers.items():
            tot = {"seated": [0, 0], "standing": [0, 0], "walking": [0, 0]}
            ratios_sit, ratios_walk = [], []
            for s in subjects:
                sit_iv, stand_ivs = GT[s]
                for activity in ("Sit down", "Walk"):
                    d = os.path.join(CAUCA, f"Subject.{s}", activity)
                    video = os.path.join(d, next(f for f in os.listdir(d) if f.endswith(".avi")))
                    for ts, label, ratio in score(cls, frames_of(cache, s, activity), seat_boxes(video, det)):
                        if activity == "Walk":
                            key = "walking"
                            if ratio is not None:
                                ratios_walk.append(ratio)
                        elif sit_iv[0] <= ts <= sit_iv[1]:
                            key = "seated"
                            if ratio is not None:
                                ratios_sit.append(ratio)
                        elif any(a <= ts <= b for a, b in stand_ivs):
                            key = "standing"
                        else:
                            continue
                        tot[key][0] += label == "Sitting"
                        tot[key][1] += 1
            print(f"  {name:8} labelled Sitting: " + ", ".join(
                f"{k} {100 * v[0] / max(v[1], 1):.0f}% of {v[1]}" for k, v in tot.items()))
            if name == "current":
                rs, rw = np.array(ratios_sit), np.array(ratios_walk)
                print(f"  shin/thigh while seated: median {np.median(rs):.2f}, p10 {np.percentile(rs, 10):.2f}, "
                      f">=1.3 {np.mean(rs >= 1.3):.0%}, >=1.5 {np.mean(rs >= 1.5):.0%}; while walking: "
                      f">=1.3 {np.mean(rw >= 1.3):.0%}, >=1.5 {np.mean(rw >= 1.5):.0%}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "training"))
    raise SystemExit(main())
