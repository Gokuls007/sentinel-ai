"""Guided stretch breaks, checked by the camera.

The movement reminder (posture/movement.py) offers a 1-minute break after you've been still
too long: neck tilts, shoulder shrugs and standing up. The pose is used to
count the reps and confirm each step. Gentle, general movements only: not medical advice.

Shoulder *shrugs* rather than rolls: from the front, 2D keypoints can't tell a roll from a
shrug, so what is counted is the shoulders rising toward the ears and dropping back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOSE, L_EYE, R_EYE, L_SH, R_SH = 0, 1, 2, 5, 6

SAFETY_NOTE = ("Gentle, general movements, not medical advice. Stop if anything hurts, and skip any "
               "movement that's uncomfortable.")

STEPS = (
    {"key": "neck_tilts", "label": "Neck tilts",
     "instruction": "Slowly tilt one ear toward its shoulder, back to centre, then the other side. "
                    "Keep it gentle."},
    {"key": "shoulder_shrugs", "label": "Shoulder shrugs",
     "instruction": "Lift both shoulders toward your ears, then let them drop and relax."},
    {"key": "stand_up", "label": "Stand up",
     "instruction": "Stand up and stretch tall for a moment (or press \"I stood up\")."},
)


@dataclass
class BreakConfig:
    sit_minutes: float = 50.0
    away_reset_s: float = 300.0     # this long away from the desk counts as a break
    snooze_s: float = 600.0
    get_ready_s: float = 3.0        # sit facing the screen: the reference for tilts and shrugs
    step_max_s: float = 25.0        # a step not finished by then is left partial
    tilt_enter_deg: float = 15.0    # head roll from centre to count one tilt...
    tilt_exit_deg: float = 6.0      # ...once it comes back within this
    tilts_per_side: int = 3
    shrug_enter: float = 0.15       # neck (nose to shoulder line) shorter by this share...
    shrug_exit: float = 0.06        # ...then back within this
    shrugs: int = 5
    stand_rise_widths: float = 0.8  # shoulders up by this many shoulder widths = standing
    stand_gone_s: float = 2.0       # or nobody in view this long (stood up out of the picture)


def pose_values(keypoints, min_conf: float = 0.4) -> dict | None:
    """Head roll (eye line angle), neck length (/ shoulder width), shoulder height and width."""
    if keypoints is None:
        return None
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or min(kp[i, 2] for i in (NOSE, L_EYE, R_EYE, L_SH, R_SH)) < min_conf:
        return None
    width = float(np.hypot(*(kp[L_SH, :2] - kp[R_SH, :2])))
    if width < 10:
        return None
    a, b = (kp[L_EYE, :2], kp[R_EYE, :2]) if kp[L_EYE, 0] <= kp[R_EYE, 0] else (kp[R_EYE, :2], kp[L_EYE, :2])
    mid_y = float((kp[L_SH, 1] + kp[R_SH, 1]) / 2)
    return {"roll": math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
            "neck": (mid_y - float(kp[NOSE, 1])) / width, "mid_y": mid_y, "width": width}


class RepCounter:
    """Counts a rep each time ``value`` goes past ``enter`` and then back within ``exit`` (the gap
    stops keypoint jitter around one threshold from counting twice)."""

    def __init__(self, enter: float, exit: float):
        self.enter, self.exit = enter, exit
        self.count = 0
        self._out = False

    def update(self, value: float) -> bool:
        if not self._out and value >= self.enter:
            self._out = True
        elif self._out and value <= self.exit:
            self._out = False
            self.count += 1
            return True
        return False


class BreakRoutine:
    """One guided break: get ready, then each step until done or ``step_max_s``."""

    def __init__(self, ts: float, cfg: BreakConfig | None = None, min_conf: float = 0.4):
        self.cfg = cfg or BreakConfig()
        self.min_conf = min_conf
        self.started = ts
        self.ready_until = ts + self.cfg.get_ready_s
        self.step = 0
        self.step_start = self.ready_until
        self._ref_samples: list[dict] = []
        self.ref: dict | None = None
        c = self.cfg
        self.tilt_a = RepCounter(c.tilt_enter_deg, c.tilt_exit_deg)  # one side (roll increases)
        self.tilt_b = RepCounter(c.tilt_enter_deg, c.tilt_exit_deg)  # the other side
        self.shrug = RepCounter(c.shrug_enter, c.shrug_exit)
        self.stood = False
        self._gone_since: float | None = None
        self.completed: dict[str, bool] = {s["key"]: False for s in STEPS}
        self.done = False

    def _targets(self) -> dict:
        c = self.cfg
        return {"neck_tilts": {"side_a": (self.tilt_a.count, c.tilts_per_side),
                               "side_b": (self.tilt_b.count, c.tilts_per_side)},
                "shoulder_shrugs": {"reps": (self.shrug.count, c.shrugs)},
                "stand_up": {"stood": (int(self.stood), 1)}}

    def _step_done(self) -> bool:
        return all(n >= target for n, target in self._targets()[STEPS[self.step]["key"]].values())

    def _next(self, ts: float) -> None:
        self.completed[STEPS[self.step]["key"]] = self._step_done()
        self.step += 1
        self.step_start = ts
        if self.step >= len(STEPS):
            self.done = True

    def stood_up(self, ts: float) -> None:
        """The manual fallback when the camera can't confirm standing."""
        if not self.done and STEPS[self.step]["key"] == "stand_up":
            self.stood = True
            self._next(ts)

    def update(self, keypoints, ts: float) -> None:
        if self.done:
            return
        v = pose_values(keypoints, self.min_conf)
        if ts < self.ready_until:
            if v is not None:
                self._ref_samples.append(v)
            return
        if self.ref is None:
            samples = self._ref_samples or ([v] if v is not None else [])
            if not samples:
                return  # nobody visible yet: wait for a reference
            self.ref = {k: float(np.median([s[k] for s in samples])) for k in samples[0]}
        key = STEPS[self.step]["key"]
        if key == "neck_tilts" and v is not None:
            d = v["roll"] - self.ref["roll"]
            self.tilt_a.update(d)
            self.tilt_b.update(-d)
        elif key == "shoulder_shrugs" and v is not None and self.ref["neck"] > 0:
            self.shrug.update((self.ref["neck"] - v["neck"]) / self.ref["neck"])
        elif key == "stand_up":
            if v is not None:
                self._gone_since = None
                if (self.ref["mid_y"] - v["mid_y"]) / self.ref["width"] >= self.cfg.stand_rise_widths:
                    self.stood = True
            else:
                self._gone_since = ts if self._gone_since is None else self._gone_since
                if ts - self._gone_since >= self.cfg.stand_gone_s:
                    self.stood = True
        if self._step_done() or ts - self.step_start >= self.cfg.step_max_s:
            self._next(ts)

    def result(self) -> dict:
        done = sum(self.completed.values())
        return {"outcome": "completed" if done == len(STEPS) else "partial",
                "steps": dict(self.completed),
                "reps": {"tilts": [self.tilt_a.count, self.tilt_b.count], "shrugs": self.shrug.count,
                         "stood": self.stood}}

    def snapshot(self, ts: float) -> dict:
        if ts < self.ready_until:
            return {"phase": "get_ready", "left_s": round(self.ready_until - ts, 1), "step": 0,
                    "steps": len(STEPS), "label": "Get ready",
                    "instruction": "Sit tall, facing the screen. The break starts in a moment.",
                    "counts": {}, "note": SAFETY_NOTE}
        s = STEPS[min(self.step, len(STEPS) - 1)]
        return {"phase": "step", "key": s["key"], "label": s["label"], "instruction": s["instruction"],
                "step": self.step + 1, "steps": len(STEPS),
                "left_s": round(max(0.0, self.cfg.step_max_s - (ts - self.step_start)), 1),
                "counts": {k: list(v) for k, v in self._targets()[s["key"]].items()},
                "waiting_for_reference": self.ref is None, "note": SAFETY_NOTE}
