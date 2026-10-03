"""Per-frame cost of the rules: building the scene and checking 0, 3 and 10 rules.

    python scripts/benchmark_rules.py
    python scripts/benchmark_rules.py --people 5 --frames 3000 --write docs/BENCHMARKS.md

The detector and pose models run with or without rules (a rule about phones adds those
classes to the same detector pass, whose cost isn't measured here), so the rules' cost is the
Python work per frame measured here, on synthetic poses through the real pipeline code (scene building, pose
signals, rule checks). It is reported next to the pipeline's measured total per frame.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.join(ROOT, "tests"))

SECTION_START = "<!-- benchmark:rules-cost:start -->"
SECTION_END = "<!-- benchmark:rules-cost:end -->"

RULES = [
    {"name": "Dock dwell", "conditions": [{"type": "in_zone", "zone": "dock_1"}], "duration_s": 30},
    {"name": "Fallen", "conditions": [{"type": "fallen"}]},
    {"name": "Phone in lane", "conditions": [{"type": "holding_object", "object": "cell phone"},
                                             {"type": "in_zone", "zone": "dock_1"}], "duration_s": 5},
    {"name": "Motionless", "conditions": [{"type": "stationary_for", "seconds": 120}]},
    {"name": "Posture", "conditions": [{"type": "posture_risk_at_least", "level": 4}], "duration_s": 30},
    {"name": "Crowd", "conditions": [{"type": "count_greater_than", "n": 8}]},
    {"name": "Dock crowd", "conditions": [{"type": "count_in_zone_greater_than", "zone": "dock_1", "n": 3}]},
    {"name": "Night", "conditions": [{"type": "time_window", "start": "22:00", "end": "06:00"},
                                     {"type": "count_greater_than", "n": 0}]},
    {"name": "Head turned", "conditions": [{"type": "head_turned", "direction": "either"}], "duration_s": 5},
    {"name": "Looking down", "conditions": [{"type": "looking_down"}], "duration_s": 10},
]


def build(tmp_cfg, n_rules: int):
    from anomaly.engine import AnomalyEngine
    from core.pipeline import SentinelPipeline
    from rules.dsl import Rule
    from rules.engine import RuleEngine

    p = SentinelPipeline.__new__(SentinelPipeline)
    p.config = tmp_cfg
    p.anomaly_engine = AnomalyEngine(tmp_cfg)
    p.rules = RuleEngine([Rule(id=f"r{i}", **r) for i, r in enumerate(RULES[:n_rules])])
    p.detector = SimpleNamespace(classes=[0])
    return p


def measure(p, people: int, frames: int) -> float:
    from conftest import make_pose

    rng = np.random.default_rng(0)
    poses = {i: make_pose(track_id=i, cx=100 + 120 * i) for i in range(1, people + 1)}
    for pose in poses.values():
        pose.keypoints[[1, 2, 3, 4, 9, 10], 2] = 0.9  # eyes, ears, wrists visible
    dets = SimpleNamespace(detections=[SimpleNamespace(class_name="cell phone", bbox=(90, 300, 110, 330))])
    times = []
    for f in range(frames):
        for pose in poses.values():
            pose.keypoints[:, :2] += rng.normal(0, 0.5, (17, 2)).astype(np.float32)
        t0 = time.perf_counter()
        if p.rules.rules:
            p._evaluate_rules(poses, {}, dets, 1_790_000_000 + f / 25)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times[frames // 10:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", type=int, default=5)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--pipeline-ms", type=float, default=18.42, help="measured total per frame (BENCHMARKS.md)")
    ap.add_argument("--write", metavar="MD")
    args = ap.parse_args()

    import tempfile

    from config.settings import SentinelConfig

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SentinelConfig()
        cfg.zone.zones_file = os.path.join(tmp, "zones.json")
        cfg.output.db_path = os.path.join(tmp, "events.db")
        cfg.anomaly.lstm_model_path = os.path.join(tmp, "none.pt")
        for n in (0, 3, 10):
            ms = measure(build(cfg, n), args.people, args.frames)
            fps = 1000 / (args.pipeline_ms + ms)
            rows.append((n, ms, fps))
            print(f"{n:>2} rules: {ms:.3f} ms/frame -> {fps:.1f} FPS at {args.pipeline_ms} ms pipeline")
    base = 1000 / args.pipeline_ms
    lines = [SECTION_START,
             f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/benchmark_rules.py`_, {args.people} people in "
             f"view, {args.frames} frames, real pipeline code on synthetic poses. FPS combines this with the measured "
             f"pipeline total of {args.pipeline_ms} ms per frame (above).",
             "", "| Rules | Rules cost per frame | Pipeline FPS |", "|---|---|---|"]
    lines += [f"| {n} | {ms:.2f} ms | {fps:.1f} ({(fps / base - 1) * 100:+.1f}%) |" for n, ms, fps in rows]
    lines += ["", "Rules that need objects (phone, laptop, book) add those classes to the detector's existing pass; "
              "any change in detector time from that is not included here.", SECTION_END]
    md = "\n".join(lines)
    print(md)
    if args.write:
        with open(args.write, encoding="utf-8") as f:
            text = f.read()
        if SECTION_START in text:
            before, rest = text.split(SECTION_START, 1)
            text = before + md + rest.split(SECTION_END, 1)[1]
        else:
            text = text.rstrip() + "\n\n## Cost of rules\n\n" + md + "\n"
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
