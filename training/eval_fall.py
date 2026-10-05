"""Evaluate Sentinel AI's fall detector on the UR Fall Detection dataset (URFD).

    python training/eval_fall.py --download            # fetch the official files (asks nothing; ~4.5 GB)
    python training/eval_fall.py --download --falls-only   # ~1 GB: recall/latency only, no false-alarm rate
    python training/eval_fall.py                       # evaluate what is in data/datasets/urfd
    python training/eval_fall.py --write docs/BENCHMARKS.md

Dataset: Kwolek & Kepski, "Human fall detection on embedded platform using depth maps and
wireless accelerometer", Computer Methods and Programs in Biomedicine 117(3), 2014.
Official page: https://fenix.ur.edu.pl/~mkepski/ds/uf.html. The license is CC BY-NC-SA 4.0,
for non-commercial academic use, so the data is downloaded here, never committed.

What runs is the production path: YOLOv8n + ByteTrack, then YOLOv8n-pose, then the fall state
machine in anomaly/fall_detector.py, with the thresholds from config/settings.py. Zones and
loitering are off. Frames come from camera 0 (front view, RGB, 640x480, 30 fps).

Scoring (one fall per URFD fall sequence):
- **Fall sequences.** Onset is the first frame labelled 0 ("falling") in urfall-cam0-falls.csv.
  The first fall alert at or after ``onset - tolerance`` is a true positive, and
  latency = alert time - onset. A sequence with no such alert is a false negative.
  Alerts before ``onset - tolerance``, or later extra alerts, are false positives.
- **ADL sequences (no fall).** Every fall alert is a false positive.
- **False alarms per hour** = ADL false positives / hours of ADL video. The hours are
  computed from the frames actually evaluated, and are always reported next to the rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
DATA_DIR = os.path.join(ROOT, "data", "datasets", "urfd")
FPS = 30.0
N_FALLS, N_ADLS = 30, 40
SECTION_START = "<!-- benchmark:falls:start -->"
SECTION_END = "<!-- benchmark:falls:end -->"


# --- download ---------------------------------------------------------------------------

def urfd_files(falls_only: bool) -> list[str]:
    files = ["urfall-cam0-falls.csv"] + [f"fall-{i:02d}-cam0-rgb.zip" for i in range(1, N_FALLS + 1)]
    if not falls_only:
        files += ["urfall-cam0-adls.csv"] + [f"adl-{i:02d}-cam0-rgb.zip" for i in range(1, N_ADLS + 1)]
    return files


def download(falls_only: bool) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in urfd_files(falls_only):
        dest = os.path.join(DATA_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        print(f"downloading {name} ...", flush=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        os.replace(tmp, dest)
    print(f"URFD files are in {DATA_DIR}")


# --- labels -----------------------------------------------------------------------------

def fall_onsets(labels_csv: str) -> dict[str, int]:
    """sequence name ("fall-01") -> first frame labelled 0 (falling). Lenient about format:
    each row starts with sequence name, frame number, label."""
    onsets: dict[str, int] = {}
    with open(labels_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 3 or not re.match(r"^\s*fall-\d+", row[0]):
                continue
            seq = re.match(r"^\s*(fall-\d+)", row[0]).group(1)
            try:
                frame, label = int(float(row[1])), int(float(row[2]))
            except ValueError:
                continue
            if label == 0 and (seq not in onsets or frame < onsets[seq]):
                onsets[seq] = frame
    return onsets


def frames_from_zip(path: str):
    """Yield (frame_index starting at 1, BGR image) from a URFD PNG-sequence zip, in order."""
    with zipfile.ZipFile(path) as zf:
        names = sorted(
            (n for n in zf.namelist() if n.lower().endswith(".png")),
            key=lambda n: int(re.findall(r"(\d+)\.png$", n, re.IGNORECASE)[0]),
        )
        for i, name in enumerate(names, start=1):
            img = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield i, img


# --- running the detector -----------------------------------------------------------------

@dataclass
class SequenceResult:
    name: str
    is_fall: bool
    frames: int
    alert_times_s: list[float] = field(default_factory=list)
    onset_s: float | None = None
    tp: int = 0
    fp: int = 0
    fn: int = 0
    latency_s: float | None = None
    # Stage diagnostics: when the state machine first reached FALLEN (on the ground, waiting
    # for stillness to confirm), and how much video was left after that.
    fallen_s: float | None = None
    video_after_fallen_s: float | None = None
    video_after_onset_s: float | None = None
    reached_fallen: bool = False


def setup_tag() -> str:
    """Detector + pose setup as a short tag (for keypoint caches), from the production config."""
    from config.settings import SentinelConfig

    cfg = SentinelConfig.from_env(env_file=None)
    d = cfg.detector
    return f"{os.path.splitext(os.path.basename(d.model_path))[0]}-{d.confidence_threshold:g}-{cfg.pose.backend}"


class Models:
    """Detector + pose model loaded once; tracker, pose history and fall state reset per sequence."""

    def __init__(self, device: str):
        from config.settings import SentinelConfig
        from core.detector import Detector
        from core.pipeline import SentinelPipeline

        self.cfg = SentinelConfig.from_env(env_file=None)  # production defaults (+ POSE_BACKEND etc. if set)
        self.cfg.detector.device = device
        d = self.cfg.detector
        self.detector = Detector(d.model_path, d.confidence_threshold, d.iou_threshold, device=device)
        # The production pose model (RTMPose-m by default, YOLO pose if it can't load).
        self.pose, self.pose_status = SentinelPipeline._make_pose_estimator(self.cfg)
        print(f"Models: {d.model_path} @ {d.confidence_threshold}, {self.pose_status}", flush=True)

    @property
    def recovery_model(self):
        """YOLO pose for recovery retries on crops (RTMPose needs a box)."""
        if getattr(self.pose, "name", None) != "rtmpose-m":
            return self.pose.model
        if getattr(self, "_recovery", None) is None:
            from ultralytics import YOLO

            self._recovery = YOLO(self.cfg.detector.pose_model_path)
        return self._recovery



    def fresh(self):
        """Independent sequences: forget tracks, pose history and fall state."""
        from anomaly.fall_detector import FallDetector

        self.detector.reset_tracker()
        self.pose.track_features.clear()
        f = self.cfg.fall
        return FallDetector(
            descent_speed_threshold=f.descent_speed_threshold, aspect_ratio_threshold=f.aspect_ratio_threshold,
            head_drop_ratio=f.head_drop_ratio, stillness_seconds=f.stillness_seconds,
            stillness_speed_threshold=f.stillness_speed_threshold,
            fallen_timeout_seconds=f.fallen_timeout_seconds, cooldown_seconds=f.cooldown_seconds,
            lost_hold_seconds=f.lost_hold_seconds, upright_hold_seconds=f.upright_hold_seconds,
            ground_mode=f.ground_mode, box_calibration=f.box_calibration,
        )


def score(result: SequenceResult, onset_frame: int | None, tolerance_s: float) -> SequenceResult:
    """Fill tp/fp/fn/latency from the alert times (see the module docstring), and the FALLEN
    stage diagnostics from ``fallen_s``."""
    end_s = max(0, result.frames - 1) / FPS
    if result.fallen_s is not None:
        result.video_after_fallen_s = round(end_s - result.fallen_s, 3)
    if onset_frame is not None:
        result.onset_s = (onset_frame - 1) / FPS
        result.video_after_onset_s = round(end_s - result.onset_s, 3)
        result.reached_fallen = result.fallen_s is not None and result.fallen_s >= result.onset_s - tolerance_s
    else:
        result.reached_fallen = result.fallen_s is not None
    if onset_frame is not None:
        valid = [t for t in result.alert_times_s if t >= result.onset_s - tolerance_s]
        if valid:
            result.tp = 1
            result.latency_s = round(valid[0] - result.onset_s, 3)
        else:
            result.fn = 1
        result.fp = len(result.alert_times_s) - result.tp
    else:
        result.fp = len(result.alert_times_s)
    return result


def run_sequence(models: Models, name: str, zip_path: str, onset_frame: int | None,
                 tolerance_s: float) -> SequenceResult:
    detector, pose, falls = models.detector, models.pose, models.fresh()
    result = SequenceResult(name=name, is_fall=onset_frame is not None, frames=0)
    for idx, frame in frames_from_zip(zip_path):
        ts = (idx - 1) / FPS
        dets = detector.detect_and_track(frame)
        people = [d for d in dets.detections if d.class_name == "person" and d.track_id is not None]
        poses = pose.estimate(frame, [d.track_id for d in people], [d.bbox for d in people], ts)
        feats = pose.get_all_features()
        falls.prune(feats.keys())
        for tid, p in poses.items():
            if tid in feats and falls.check(tid, p, feats[tid], ts):
                result.alert_times_s.append(round(ts, 3))
            on_ground = falls.state_of(tid) in (falls.FALLEN, falls.CONFIRMED)
            after_onset = onset_frame is None or ts >= (onset_frame - 1) / FPS - tolerance_s
            if result.fallen_s is None and on_ground and after_onset:
                result.fallen_s = round(ts, 3)
        result.frames = idx
    return score(result, onset_frame, tolerance_s)


# --- report --------------------------------------------------------------------------------

def summarize(results: list[SequenceResult]) -> dict:
    falls = [r for r in results if r.is_fall]
    adls = [r for r in results if not r.is_fall]
    tp, fn = sum(r.tp for r in falls), sum(r.fn for r in falls)
    fp_fall, fp_adl = sum(r.fp for r in falls), sum(r.fp for r in adls)
    fp = fp_fall + fp_adl
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    adl_hours = sum(r.frames for r in adls) / FPS / 3600
    latencies = [r.latency_s for r in falls if r.latency_s is not None]
    reached = [r for r in falls if r.reached_fallen]
    after_onset = [r.video_after_onset_s for r in falls if r.video_after_onset_s is not None]
    after_fallen = [r.video_after_fallen_s for r in reached if r.video_after_fallen_s is not None]
    return {
        "fallen_stage_reached": len(reached),
        "fallen_stage_recall": len(reached) / len(falls) if falls else None,
        "video_after_onset_median_s": float(np.median(after_onset)) if after_onset else None,
        "video_after_fallen_median_s": float(np.median(after_fallen)) if after_fallen else None,
        "adl_reaching_fallen": sum(r.reached_fallen for r in adls),
        "fall_sequences": len(falls), "adl_sequences": len(adls),
        "tp": tp, "fn": fn, "fp_in_fall_sequences": fp_fall, "fp_in_adl_sequences": fp_adl,
        "precision": precision, "recall": recall, "f1": f1,
        "adl_hours": adl_hours, "adl_frames": sum(r.frames for r in adls),
        "false_alarms_per_hour": (fp_adl / adl_hours) if adl_hours else None,
        "latency_median_s": float(np.median(latencies)) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
    }


def fmt(x, pct=False, digits=2):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.{digits}f}"


def markdown(s: dict, device: str, tolerance_s: float, stillness_s: float = 1.0) -> str:
    minutes = s["adl_hours"] * 60
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python training/eval_fall.py` on {device}._ "
        "URFD camera 0 (RGB 640x480, 30 fps); production thresholds; onset tolerance "
        f"{tolerance_s:.1f} s.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Fall sequences / detected (recall) | {s['fall_sequences']} / {s['tp']} ({fmt(s['recall'], True)}) |",
        f"| Precision | {fmt(s['precision'], True)} |",
        f"| F1 | {fmt(s['f1'])} |",
        f"| False positives in fall sequences (early or repeated alerts) | {s['fp_in_fall_sequences']} |",
        f"| False positives in ADL sequences | {s['fp_in_adl_sequences']} over {s['adl_sequences']} sequences |",
        f"| **False alarms per hour of non-fall video** | **{fmt(s['false_alarms_per_hour'])}**, based on "
        f"**{s['adl_hours']:.3f} h ({minutes:.1f} min, {s['adl_frames']} frames)** of ADL video |",
        f"| Alert latency after fall onset (median / max) | {fmt(s['latency_median_s'])} s / "
        f"{fmt(s['latency_max_s'])} s |",
        "",
        f"The false-alarm rate rests on only {minutes:.1f} minutes of non-fall video (all that URFD "
        "provides), so treat it as a rough indicator, not a measured field rate. A reliable figure "
        "needs hours of normal-activity footage from the target site.",
        "",
        "**Why alerts are missed here: stage diagnostics.** An alert is raised only after the state "
        "machine reaches FALLEN (a fast descent, then a lying pose) and the person then stays still "
        f"for {stillness_s:.1f} s.",
        "",
        "| Stage | Value |",
        "|---|---|",
        f"| Fall sequences that reached FALLEN (on the ground, before confirmation) | "
        f"{s['fallen_stage_reached']} / {s['fall_sequences']} ({fmt(s['fallen_stage_recall'], True)}) |",
        f"| Video left after fall onset (median) | {fmt(s['video_after_onset_median_s'])} s |",
        f"| Video left after reaching FALLEN (median) | {fmt(s['video_after_fallen_median_s'])} s |",
        f"| ADL sequences that reached FALLEN (not confirmed) | {s['adl_reaching_fallen']} / {s['adl_sequences']} |",
        "",
        "URFD trims each fall clip shortly after the fall. Where the detector reaches FALLEN, the "
        f"median video left after that ({fmt(s['video_after_fallen_median_s'])} s) is "
        f"{'shorter than' if (s['video_after_fallen_median_s'] or 0) < stillness_s else 'barely longer than'} "
        f"the {stillness_s:.1f} s of stillness the detector waits for, so many clips end before an "
        "alert could fire. The confirmation-time sweep above breaks down the clips that had enough "
        "video but still got no alert (mostly the person is lost from view once on the floor). "
        "These numbers describe how the detector behaves on "
        "short, trimmed clips. They are not the recall you would see on continuous video, "
        "which needs longer fall recordings to measure.",
        SECTION_END,
    ]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    text = "# Benchmarks\n\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Fall detection accuracy (UR Fall Detection dataset)\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the official URFD files first")
    ap.add_argument("--falls-only", action="store_true", help="with --download: skip ADL sequences")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds before onset still counted as a hit")
    ap.add_argument("--limit", type=int, default=None, help="only the first N sequences of each kind (smoke test)")
    ap.add_argument("--write", metavar="MD", help="replace the fall-accuracy section of this markdown file")
    args = ap.parse_args()

    if args.download:
        download(args.falls_only)
    labels = os.path.join(DATA_DIR, "urfall-cam0-falls.csv")
    if not os.path.isfile(labels):
        print(f"No URFD data in {DATA_DIR}. Run with --download (about 4.5 GB, or 1 GB with --falls-only).")
        return 1
    onsets = fall_onsets(labels)
    seqs = [(f"fall-{i:02d}", onsets.get(f"fall-{i:02d}")) for i in range(1, N_FALLS + 1)]
    seqs += [(f"adl-{i:02d}", None) for i in range(1, N_ADLS + 1)]
    if args.limit:
        seqs = [s for s in seqs if s[0].startswith("fall")][: args.limit] + \
               [s for s in seqs if s[0].startswith("adl")][: args.limit]

    models = Models(args.device)
    results = []
    for name, onset in seqs:
        zip_path = os.path.join(DATA_DIR, f"{name}-cam0-rgb.zip")
        if not os.path.isfile(zip_path):
            continue
        if name.startswith("fall") and onset is None:
            print(f"skip {name}: no onset label in {os.path.basename(labels)}")
            continue
        t0 = time.time()
        r = run_sequence(models, name, zip_path, onset, args.tolerance)
        results.append(r)
        print(f"{name}: {r.frames} frames, alerts at {r.alert_times_s} "
              f"(onset {r.onset_s}) tp={r.tp} fp={r.fp} fn={r.fn} [{time.time() - t0:.0f}s]", flush=True)
    if not results:
        print("no sequences evaluated")
        return 1
    summary = summarize(results)
    out = os.path.join(ROOT, "outputs", "eval_fall_urfd.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sequences": [asdict(r) for r in results]}, f, indent=2)
    section = markdown(summary, args.device, args.tolerance, models.cfg.fall.stillness_seconds)
    print("\n" + section + f"\n\nper-sequence results: {out}")
    if args.write:
        write_section(args.write, section)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
