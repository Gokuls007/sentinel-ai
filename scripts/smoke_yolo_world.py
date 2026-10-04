"""Smoke test for YOLO-World on your own clips: is each class detected consistently?

    python scripts/smoke_yolo_world.py data/recordings/*.mp4
    python scripts/smoke_yolo_world.py clip.mp4 --classes "cardboard box,chair" --every 2

Not an accuracy evaluation (there are no labels): it reports, per clip and class, the share of
frames with at least one detection at a few confidence thresholds, the median confidence, and
the longest run of frames without one. Good enough to see whether the classes the rules
depend on (cardboard box, chair) are found reliably before building on them.
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

DEFAULT = "cardboard box,chair,ladder,backpack,hard hat,safety vest,forklift"
THRESHOLDS = (0.05, 0.15, 0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--classes", default=DEFAULT)
    ap.add_argument("--model", default=os.path.join(ROOT, "yolov8s-worldv2.pt"))
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import cv2
    from ultralytics import YOLOWorld

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    model = YOLOWorld(args.model)
    model.set_classes(classes)
    for clip in args.clips:
        cap = cv2.VideoCapture(clip)
        per: dict[str, list[float]] = {c: [] for c in classes}  # best confidence per frame (0 = none)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % args.every:
                continue
            res = model.predict(frame, conf=min(THRESHOLDS), imgsz=args.imgsz, verbose=False)[0]
            best = dict.fromkeys(classes, 0.0)
            for cls, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=True):
                name = classes[int(cls)]
                best[name] = max(best[name], float(conf))
            for c in classes:
                per[c].append(best[c])
        cap.release()
        n = len(next(iter(per.values()), []))
        print(f"\n{os.path.basename(clip)}: {n} frames")
        print(f"  {'class':<15}" + "".join(f"  seen@{t:<5}" for t in THRESHOLDS) + "  median conf  longest gap@0.15")
        for c in classes:
            seen = [v for v in per[c] if v > 0]
            shares = "".join(f"  {sum(v >= t for v in per[c]) / max(n, 1):>8.0%}" for t in THRESHOLDS)
            gap = run = 0
            for v in per[c]:
                run = 0 if v >= 0.15 else run + 1
                gap = max(gap, run)
            med = f"{median(seen):.2f}" if seen else "--"
            print(f"  {c:<15}{shares}  {med:>11}  {gap:>6} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
