"""Replay a real movement-coach session through the movement tracker.

    python scripts/replay_movement_log.py                                   # the fixture session
    python scripts/replay_movement_log.py data/posture_movement_log_laptop.jsonl

The log has one row per second: whether shoulders were measured, and the change vs the
tracker's position at the time (shift in shoulder widths, width change, lean change). The
replay turns each measured row into a pose with that offset from a fixed reference, and each
unmeasured row into "nobody visible". Approximate (the original tracker's reference moved
after each detected movement), but close enough to compare settings and behaviour.
Prints the state, still time and events at the moments that matter.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "real_session_movement_log.json")
T0 = 1_790_000_000.0
WIDTH = 400.0


def load(path: str) -> list[list]:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["rows"]
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    t0 = rows[0]["t"]
    return [[r["t"] - t0, bool(r.get("measured")), r.get("shift"), r.get("width"), r.get("lean")] for r in rows]


def pose(shift: float, width: float, lean: float) -> np.ndarray:
    w = WIDTH * (1 + width)
    x, y = 640 + shift * WIDTH, 500.0
    dx, dy = w / 2 * math.cos(math.radians(lean)), w / 2 * math.sin(math.radians(lean))
    k = np.zeros((17, 3), np.float32)
    k[0] = (x, y - 180, 0.9)
    k[1], k[2] = (x - 30, y - 200, 0.9), (x + 30, y - 200, 0.9)
    k[5], k[6] = (x - dx, y - dy, 0.9), (x + dx, y + dy, 0.9)
    return k


def replay(rows: list[list], tracker) -> list[dict]:
    """Feed the rows (at 10 fps between logged seconds, unless there was a gap) and return
    one record per logged row."""
    out = []
    prev = None
    for t, measured, shift, width, lean in rows:
        kp = pose(shift or 0.0, width or 0.0, lean or 0.0) if measured else None
        # Fill the second with frames, like the camera did, but never across a real gap.
        steps = [t] if prev is None or t - prev > 1.5 else [prev + (t - prev) * (i + 1) / 10 for i in range(10)]
        for ts in steps:
            tracker.update(kp, T0 + ts)
        prev = t
        snap = tracker.snapshot(T0 + t)
        out.append({"t": t, "measured": measured, "state": snap["state"], "still_s": snap["still_s"],
                    "breaks": snap["breaks_today"]})
    return out


def summary(records: list[dict]) -> list[str]:
    lines = []
    prev = None
    for r in records:
        key = (r["state"], r["breaks"])
        if key != prev:
            lines.append(f"{r['t']:7.0f}s  {r['state']:<11} still {r['still_s']:7.1f}s  breaks {r['breaks']}"
                         f"  {'(shoulders measured)' if r['measured'] else '(nobody measured)'}")
            prev = key
    return lines


def main() -> int:
    from posture.movement import MovementTracker

    path = sys.argv[1] if len(sys.argv) > 1 else FIXTURE
    rows = load(path)
    records = replay(rows, MovementTracker(now=T0))
    print("\n".join(summary(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
