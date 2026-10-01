"""Agreement between Sentinel's 2D REBA and hand labels (from scripts/label_reba.py).

    python scripts/eval_ergo.py my_clip.mp4 --labels my_clip.reba_labels.json
    python scripts/eval_ergo.py my_clip.mp4 --labels ... --write docs/BENCHMARKS.md

The whole clip runs through the production path (YOLOv8n + ByteTrack, then YOLOv8n-pose, then
the ergonomics tracker with its 1 s smoothing), so tracking and smoothing behave as they do
live. At each labelled frame the system's level for the main person (largest box) is
compared with the label.

Reported:
- exact agreement and within-one-level agreement on the 5 REBA risk levels;
- quadratic-weighted Cohen's kappa (chance-corrected, penalising big disagreements more);
- a confusion matrix;
- coverage: the share of labelled frames where the system had a *reliable* (confident)
  score. Agreement is reported for reliable frames and for all scored frames.
This is a 2D approximation of REBA checked against one person's judgement, not a validation
against a certified assessment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

LEVELS = [1, 2, 3, 4, 5]
NAMES = {1: "negligible", 2: "low", 3: "medium", 4: "high", 5: "very high"}
SECTION_START = "<!-- benchmark:ergonomics:start -->"
SECTION_END = "<!-- benchmark:ergonomics:end -->"


def weighted_kappa(pairs: list[tuple[int, int]], k: int = 5) -> float | None:
    """Quadratic-weighted Cohen's kappa for ordinal labels 1..k."""
    n = len(pairs)
    if n == 0:
        return None
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        obs[a - 1][b - 1] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * row[i] * col[j] / n
    return 1.0 - num / den if den else None


def agreement(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    if not n:
        return {"n": 0, "exact": None, "within_one": None, "kappa": None}
    return {
        "n": n,
        "exact": sum(a == b for a, b in pairs) / n,
        "within_one": sum(abs(a - b) <= 1 for a, b in pairs) / n,
        "kappa": weighted_kappa(pairs),
    }


def confusion(pairs: list[tuple[int, int]]) -> list[list[int]]:
    m = [[0] * 5 for _ in range(5)]
    for label, pred in pairs:
        m[label - 1][pred - 1] += 1
    return m


def run_system(video: str, wanted: set[int], device: str) -> dict[int, dict]:
    """System output (level, reliable, score) for the main person at each wanted frame."""
    from config.settings import SentinelConfig
    from core.detector import Detector
    from core.pose_estimator import PoseEstimator
    from ergonomics import ErgoTracker

    cfg = SentinelConfig()
    d = cfg.detector
    detector = Detector(d.model_path, d.confidence_threshold, d.iou_threshold, device=device)
    pose = PoseEstimator(d.pose_model_path, confidence_threshold=cfg.pose.confidence_threshold, device=device)
    ergo = ErgoTracker(cfg.ergonomics)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: dict[int, dict] = {}
    idx = 0
    last = max(wanted) if wanted else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        dets = detector.detect_and_track(frame)
        people = [x for x in dets.detections if x.class_name == "person" and x.track_id is not None]
        poses = pose.estimate(frame, [p.track_id for p in people], [p.bbox for p in people], ts)
        views = {tid: ergo.update(tid, p.keypoints, ts)[0] for tid, p in poses.items()}
        ergo.prune(views.keys())
        if idx in wanted:
            main = max(people, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]), default=None)
            v = views.get(main.track_id) if main is not None else None
            out[idx] = {"level": v.level if v else None, "reliable": bool(v and v.reliable),
                        "score": v.score if v else None, "confidence": round(v.confidence, 2) if v else 0.0}
        idx += 1
    cap.release()
    return out


def report(video: str, labels: dict, system: dict[int, dict], device: str) -> tuple[str, dict]:
    labelled = {int(k): v["level"] for k, v in labels.items() if v.get("level") in LEVELS}
    unsure = sum(1 for v in labels.values() if v.get("level") is None)
    scored = [(labelled[f], system[f]["level"]) for f in labelled if system.get(f, {}).get("level")]
    reliable = [(labelled[f], system[f]["level"]) for f in labelled
                if system.get(f, {}).get("level") and system[f]["reliable"]]
    all_m, rel_m = agreement(scored), agreement(reliable)
    coverage = len(reliable) / len(labelled) if labelled else None
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    num = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
    cm = confusion(reliable)
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/eval_ergo.py` on `{os.path.basename(video)}` "
        f"({device}). {len(labelled)} labelled frames ({unsure} marked unsure and excluded)._",
        "",
        "| | Reliable scores only | All scored frames |",
        "|---|---|---|",
        f"| Frames compared | {rel_m['n']} | {all_m['n']} |",
        f"| Exact level agreement | {pct(rel_m['exact'])} | {pct(all_m['exact'])} |",
        f"| Within one level | {pct(rel_m['within_one'])} | {pct(all_m['within_one'])} |",
        f"| Quadratic-weighted kappa | {num(rel_m['kappa'])} | {num(all_m['kappa'])} |",
        "",
        f"Coverage: the system had a reliable (side-view) score on **{pct(coverage)}** of labelled frames.",
        "",
        "Confusion matrix, reliable frames (rows = hand label, columns = system):",
        "",
        "| label \\ system | " + " | ".join(NAMES[c] for c in LEVELS) + " |",
        "|---|" + "---|" * 5,
        *[f"| {NAMES[r]} | " + " | ".join(str(cm[r - 1][c - 1]) for c in LEVELS) + " |" for r in LEVELS],
        "",
        "This is a **2D approximation of REBA** compared with one person's judgement of risk level, "
        "not a certified ergonomic assessment. Wrist angle, twisting and load are not measured "
        "(see `backend/ergonomics/reba.py`).",
        SECTION_END,
    ]
    summary = {"labelled": len(labelled), "unsure": unsure, "coverage": coverage,
               "reliable": rel_m, "all_scored": all_m, "confusion_reliable": cm}
    return "\n".join(lines), summary


def write_section(path: str, section: str) -> None:
    text = "# Benchmarks\n\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Ergonomic risk (2D REBA) agreement\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", metavar="MD", help="replace the ergonomics section of this markdown file")
    args = ap.parse_args()
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    wanted = {int(k) for k, v in labels.items() if v.get("level") in LEVELS}
    if not wanted:
        print("no labelled frames in", args.labels)
        return 1
    system = run_system(args.video, wanted, args.device)
    section, summary = report(args.video, labels, system, args.device)
    print(section)
    out = os.path.join(ROOT, "outputs", "eval_ergo.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": {str(k): v for k, v in system.items()}}, f, indent=2)
    if args.write:
        write_section(args.write, section)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
