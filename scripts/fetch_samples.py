"""Download and verify the sample videos (and YOLO weights) so Sentinel AI runs without a camera.

Usage:
    python scripts/fetch_samples.py            # sample videos only
    python scripts/fetch_samples.py --models   # also pre-download yolov8n / yolov8n-pose weights
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from core.samples import LICENSE, SAMPLES, ensure_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true", help="also download the YOLO weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for name, path in ensure_all().items():
        print(f"  {name:10s} {path}  ({SAMPLES[name].description})")
    print(f"  License: {LICENSE}")

    if args.models:
        from ultralytics import YOLO
        for weights in ("yolov8n.pt", "yolov8n-pose.pt"):
            YOLO(weights)
            print(f"  model      {weights}")


if __name__ == "__main__":
    main()
