"""Per-seat calibration: each seat's own "normal" head yaw, head pitch and hand height.

Students sit at different angles to the camera, so later signals compare a seat with its
own baseline, never a global threshold. A seat calibrates over ``duration_s`` from the first
moment someone sits in it after the exam starts: seats taken at the start finish with the
session's calibration; late arrivals calibrate on their own over their first ``duration_s``.

Spread is half the interquartile range (robust to the odd glance), with a floor so a very
still student doesn't get a hair-trigger baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
MIN_CONF = 0.3


@dataclass
class Baseline:
    yaw_med: float
    yaw_spread: float
    pitch_med: float | None
    pitch_spread: float | None
    hand_height_med: float | None  # wrists below the shoulders, in shoulder widths
    samples: int
    calibrated_at: float

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def hand_height(kp: np.ndarray) -> float | None:
    """Mean wrist height below the shoulder line, in shoulder widths (+ = lower). None when no
    wrist is visible (hands under the desk look like this too)."""
    kp = np.asarray(kp, float)
    if kp[L_SH, 2] < MIN_CONF or kp[R_SH, 2] < MIN_CONF:
        return None
    sw = abs(kp[L_SH, 0] - kp[R_SH, 0])
    if sw < 4:
        return None
    sh_y = (kp[L_SH, 1] + kp[R_SH, 1]) / 2
    wrists = [kp[i, 1] for i in (L_WR, R_WR) if kp[i, 2] >= MIN_CONF]
    return float((np.mean(wrists) - sh_y) / sw) if wrists else None


def _med_spread(values: list[float], floor: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), max(float(q3 - q1) / 2, floor)


class SeatCalibration:
    def __init__(self, duration_s: float = 120.0, min_samples: int = 30, yaw_floor: float = 6.0,
                 pitch_floor: float = 5.0):
        self.duration_s = duration_s
        self.min_samples = min_samples
        self.yaw_floor = yaw_floor
        self.pitch_floor = pitch_floor
        self.started_at: float | None = None  # first frame with someone in the seat
        self.yaw: list[float] = []
        self.pitch: list[float] = []
        self.hands: list[float] = []
        self.baseline: Baseline | None = None

    def add(self, ts: float, head, hands: float | None, min_conf: float = 0.5) -> Baseline | None:
        """One frame with someone in the seat. Returns the baseline the moment it's ready."""
        if self.baseline is not None:
            return None
        if self.started_at is None:
            self.started_at = ts
        if head is not None and head.confidence >= min_conf:
            self.yaw.append(head.yaw)
            if head.pitch is not None:
                self.pitch.append(head.pitch)
        if hands is not None:
            self.hands.append(hands)
        if ts - self.started_at >= self.duration_s and len(self.yaw) >= self.min_samples:
            yaw_med, yaw_spread = _med_spread(self.yaw, self.yaw_floor)
            pitch_med, pitch_spread = _med_spread(self.pitch, self.pitch_floor)
            hand_med, _ = _med_spread(self.hands, 0.0)
            self.baseline = Baseline(yaw_med, yaw_spread, pitch_med, pitch_spread, hand_med, len(self.yaw), ts)
            return self.baseline
        return None

    def progress(self, ts: float) -> float:
        if self.baseline is not None:
            return 1.0
        if self.started_at is None:
            return 0.0
        return min(0.99, (ts - self.started_at) / self.duration_s)
