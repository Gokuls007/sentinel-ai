"""Compare pose models (YOLOv8n-pose vs RTMPose-m) on the same frames and the same person boxes.

    python scripts/compare_pose.py data/recordings/<clip>.mp4
    python scripts/compare_pose.py clip.mp4 --write docs/BENCHMARKS.md

Segments come from ``<clip>.labels.json`` next to the video, e.g.
``{"segments": [{"action": "standing", "start": 0, "end": 20}, {"action": "lying", ...}]}``.
Both models see each frame with the person tracker's boxes (YOLOv8n + ByteTrack), the
production setup; the main (largest) person is scored. Per segment and model:

- **box missing**: frames where the tracker had no person box (top-down RTMPose then has
  nothing to work on; this is the same for both models);
- **no pose**: frames with a box but no pose from the model (the YOLO pose model detects on the
  whole frame and must match the box; RTMPose always answers for a box);
- **mean conf** and **low conf** (share of keypoints under 0.3), plus the weakest keypoints;
- **jitter**: median frame-to-frame movement of confident keypoints, in % of box height. Standing
  still is the baseline: higher in a still pose means less stable keypoints;
- **ms**: model time per frame (pose only, on this GPU).
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

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def segment_of(ts, segments):
    for s in segments:
        if s["start"] <= ts < s["end"]:
            return s["action"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--write", metavar="MD")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import cv2

    from config.settings import SentinelConfig
    from core.detector import Detector
    from core.pose_estimator import PoseEstimator
    from core.rtmpose_estimator import RTMPoseEstimator

    with open(os.path.splitext(args.clip)[0] + ".labels.json", encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    cfg = SentinelConfig()
    d = cfg.detector
    det = Detector(d.model_path, d.confidence_threshold, d.iou_threshold, device=d.device)
    models = {"yolov8n-pose": PoseEstimator(d.pose_model_path, confidence_threshold=cfg.pose.confidence_threshold),
              "rtmpose-m": RTMPoseEstimator(cfg.pose.rtmpose_model)}
    stats = {m: defaultdict(lambda: {"frames": 0, "box_missing": 0, "no_pose": 0, "conf": [], "jitter": [], "ms": []})
             for m in models}
    prev = {m: None for m in models}
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        idx += 1
        seg = segment_of(ts, segments)
        dets = det.detect_and_track(frame)
        people = [x for x in dets.detections if x.class_name == "person" and x.track_id is not None]
        main = max(people, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), default=None)
        for name, model in models.items():
            s = stats[name][seg] if seg else None
            if s is not None:
                s["frames"] += 1
            if main is None:
                prev[name] = None
                if s is not None:
                    s["box_missing"] += 1
                continue
            t0 = time.perf_counter()
            poses = model.estimate(frame, [main.track_id], [main.bbox], ts)
            ms = (time.perf_counter() - t0) * 1000
            pose = poses.get(main.track_id)
            if s is None:
                continue
            s["ms"].append(ms)
            if pose is None:
                s["no_pose"] += 1
                prev[name] = None
                continue
            kp = np.asarray(pose.keypoints, float)
            s["conf"].append(kp[:, 2])
            h = max(float(main.bbox[3] - main.bbox[1]), 1.0)
            if prev[name] is not None:
                ok_kp = (kp[:, 2] >= 0.3) & (prev[name][:, 2] >= 0.3)
                if ok_kp.any():
                    move = np.hypot(*(kp[ok_kp, :2] - prev[name][ok_kp, :2]).T) / h * 100
                    s["jitter"].append(float(np.median(move)))
            prev[name] = kp
    cap.release()

    order = [s["action"] for s in segments]
    lines = [f"Clip `{os.path.basename(args.clip)}`, {idx} frames at {fps:.0f} fps; same frames and tracker boxes for "
             "both models; main (largest) person.", "",
             "| Segment | Model | Frames | Box missing | No pose | Mean conf | Low conf (<0.3) "
             "| Jitter (% box h) | ms/frame |",
             "|---|---|---|---|---|---|---|---|---|"]
    weakest = []
    for seg in order:
        for name in models:
            s = stats[name][seg]
            n = max(s["frames"], 1)
            conf = np.array(s["conf"]) if s["conf"] else np.zeros((0, 17))
            mean = f"{conf.mean():.2f}" if conf.size else "--"
            low = f"{(conf < 0.3).mean():.0%}" if conf.size else "--"
            jit = f"{np.median(s['jitter']):.2f}" if s["jitter"] else "--"
            ms = f"{np.median(s['ms']):.1f}" if s["ms"] else "--"
            lines.append(f"| {seg} | {name} | {s['frames']} | {s['box_missing'] / n:.0%} | {s['no_pose'] / n:.0%} | "
                         f"{mean} | {low} | {jit} | {ms} |")
            if conf.size:
                per = conf.mean(axis=0)
                worst = ", ".join(f"{KP_NAMES[i]} {per[i]:.2f}" for i in np.argsort(per)[:4])
                weakest.append(f"- {seg}, {name}: weakest {worst}")
    lines += ["", "Weakest keypoints (mean confidence):", *weakest]
    md = "\n".join(lines)
    print(md)
    if args.write:
        start, end = "<!-- benchmark:pose-compare:start -->", "<!-- benchmark:pose-compare:end -->"
        block = f"{start}\n{md}\n{end}"
        with open(args.write, encoding="utf-8") as f:
            text = f.read()
        if start in text:
            before, rest = text.split(start, 1)
            text = before + block + rest.split(end, 1)[1]
        else:
            text = text.rstrip() + "\n\n## Pose models: YOLOv8n-pose vs RTMPose-m\n\n" + block + "\n"
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
