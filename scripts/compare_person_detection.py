"""How often is a person found at all? Person detectors compared per segment, on the same frames.

    python scripts/compare_person_detection.py data/recordings/<clip>.mp4 [more clips...]
    python scripts/compare_person_detection.py clip.mp4 --write docs/BENCHMARKS.md

Segments come from ``<clip>.labels.json`` next to each video. Top-down pose (RTMPose) needs a
person box, so a frame with no person found has no pose, no activity and no fall check. Lying
down is where detectors lose people. Compared, per frame, without a tracker (the tracker can only
bridge short gaps; it can't invent a person the detector never saw):

- YOLOv8n (the current person detector) at 0.5 (current) and 0.25;
- YOLO11m as the person detector at 0.5 and 0.25;
- RTMO-m and RTMO-l (one-stage: people and their poses together, no boxes needed) at their
  default score threshold (0.7) and at 0.4.

Reported: "no person" (share of frames with nobody found) per segment, and ms per frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

RTMO = {
    "rtmo-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
              "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "rtmo-l": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
              "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
}


def build():
    import torch  # noqa: F401  (first, so onnxruntime reuses PyTorch's CUDA libraries)
    from rtmlib import RTMO as RTMOModel
    from ultralytics import YOLO

    v8, v11 = YOLO(os.path.join(ROOT, "yolov8n.pt")), YOLO(os.path.join(ROOT, "yolo11m.pt"))

    def yolo(model, conf):
        return lambda f: len(model.predict(f, classes=[0], conf=conf, verbose=False)[0].boxes)

    dets = {"yolov8n @0.5 (current)": yolo(v8, 0.5), "yolov8n @0.25": yolo(v8, 0.25),
            "yolo11m @0.5": yolo(v11, 0.5), "yolo11m @0.25": yolo(v11, 0.25)}
    for name, url in RTMO.items():
        m = RTMOModel(onnx_model=url, backend="onnxruntime", device="cuda")
        for thr in (0.7, 0.4):
            dets[f"{name} @{thr}"] = (lambda mm, t: lambda f: len(mm(f, score_thr=t)[0]))(m, thr)
    return dets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--write", metavar="MD")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import cv2

    dets = build()
    lines = []
    for clip in args.clips:
        with open(os.path.splitext(clip)[0] + ".labels.json", encoding="utf-8") as f:
            segments = json.load(f)["segments"]
        order = list(dict.fromkeys(s["action"] for s in segments))
        found = {d: defaultdict(lambda: [0, 0]) for d in dets}  # det -> segment -> [frames, no person]
        ms = {d: [] for d in dets}
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = idx / fps
            idx += 1
            seg = next((s["action"] for s in segments if s["start"] <= ts < s["end"]), None)
            if seg is None:
                continue
            for name, fn in dets.items():
                t0 = time.perf_counter()
                n = fn(frame)
                ms[name].append((time.perf_counter() - t0) * 1000)
                found[name][seg][0] += 1
                found[name][seg][1] += n == 0
        cap.release()
        lines += [f"**{os.path.basename(clip)}**: share of frames with no person found, per segment.", "",
                  "| Detector | " + " | ".join(order) + " | ms/frame |", "|---|" + "---|" * (len(order) + 1)]
        for name in dets:
            cells = [f"{found[name][s][1] / max(found[name][s][0], 1):.0%}" for s in order]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.median(ms[name][5:]):.1f} |")
        lines.append("")
    md = "\n".join(lines).rstrip()
    print(md)
    if args.write:
        start, end = "<!-- benchmark:person-detection:start -->", "<!-- benchmark:person-detection:end -->"
        with open(args.write, encoding="utf-8") as f:
            text = f.read()
        block = f"{start}\n{md}\n{end}"
        if start in text:
            before, rest = text.split(start, 1)
            text = before + block + rest.split(end, 1)[1]
        else:
            heading = "## Person detection in hard poses (lying, sliding, getting up)"
            text = text.rstrip() + f"\n\n{heading}\n\n" + block + "\n"
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
