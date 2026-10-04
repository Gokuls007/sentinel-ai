"""Smoke test for the object detectors (YOLO11m + YOLO-World) on your own clips: is each class detected consistently?

    python scripts/smoke_yolo_world.py data/recordings/*.mp4
    python scripts/smoke_yolo_world.py clip.mp4 --classes "cardboard box,chair" --every 2

Not an accuracy evaluation (there are no labels). It runs the production detector (several
prompts per class, per-class confidence floors, voting per object) and reports per clip and
class:
- **raw**: share of frames with a detection above the class floor (per-frame guesses);
- **steady**: share of frames where the voted, shown objects include the class;
- the median confidence, the longest run of frames without a steady detection, and
- **flips**: how often a shown object's label changed (should be ~0 with voting).
Good enough to see whether the classes the rules depend on (cardboard box, chair) are found
reliably, and to pick the per-class floors, before building on them.
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))


def main() -> int:
    from config.settings import COCO_OBJECTS, DEFAULT_OBJECT_CLASSES, ObjectsConfig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--classes", default=",".join(DEFAULT_OBJECT_CLASSES))
    ap.add_argument("--model", default=os.path.join(ROOT, "yolov8s-worldv2.pt"))
    ap.add_argument("--coco-model", default=os.path.join(ROOT, "yolo11m.pt"))
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--floor", type=float, default=None, help="one floor for every class (default: the config's)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import cv2

    from core.object_detector import ObjectDetector

    cfg = ObjectsConfig()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    floors = {c: args.floor for c in classes} if args.floor is not None else cfg.floors
    for clip in args.clips:
        det = ObjectDetector(classes, args.model, args.coco_model, COCO_OBJECTS,
                             confidence=args.floor or cfg.confidence,
                             cache_dir=os.path.join(ROOT, cfg.cache_dir), synonyms=cfg.synonyms, floors=floors,
                             vote_window=cfg.vote_window, min_hits=cfg.min_hits)
        if not det.available:
            print(det.error)
            return 1
        for e in det.errors:
            print("warning:", e)
        cap = cv2.VideoCapture(clip)
        raw = {c: [] for c in classes}
        steady = {c: [] for c in classes}
        labels: dict[int, str] = {}
        flips = dict.fromkeys(classes, 0)
        prompts = {c: {} for c in classes}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % args.every:
                continue
            shown = det.detect(frame)
            for c in classes:
                raw[c].append(max((o.confidence for o in det.last_raw if o.class_name == c), default=0.0))
                steady[c].append(max((o.confidence for o in shown if o.class_name == c), default=0.0))
            for o in shown:
                if o.track_id in labels and labels[o.track_id] != o.class_name:
                    flips[o.class_name] += 1
                labels[o.track_id] = o.class_name
                prompts[o.class_name][o.prompt] = prompts[o.class_name].get(o.prompt, 0) + 1
        cap.release()
        n = len(raw[classes[0]]) if classes else 0
        print(f"\n{os.path.basename(clip)}: {n} frames")
        print(f"  {'class':<15} {'floor':>5}  {'raw':>5}  {'steady':>6}  median conf  longest gap  flips  best prompt")
        for c in classes:
            seen = [v for v in steady[c] if v > 0]
            gap = run = 0
            for v in steady[c]:
                run = 0 if v > 0 else run + 1
                gap = max(gap, run)
            best = max(prompts[c], key=prompts[c].get) if prompts[c] else "--"
            print(f"  {c:<15} {det.floor(c):>5.2f}  {sum(v > 0 for v in raw[c]) / max(n, 1):>5.0%}  "
                  f"{len(seen) / max(n, 1):>6.0%}  {median(seen) if seen else 0:>11.2f}  {gap:>6} frames  "
                  f"{flips[c]:>5}  {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
