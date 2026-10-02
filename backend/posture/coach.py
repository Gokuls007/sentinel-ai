"""Desk posture coach: compares a seated person, seen from the front by a laptop webcam, with
their own upright baseline.

REBA isn't used here: a frontal webcam can't measure trunk or neck flexion reliably (see
ergonomics/). Instead the coach tracks ratios that a frontal view *can* see, all normalised by
shoulder width, so moving a little closer or further doesn't by itself change them:

- head height: nose above the shoulder line. It shrinks when the head drops or juts
  forward (slouching).
- face-to-shoulder ratio: eye span / shoulder width. It grows when the shoulders roll in
  (hunching).
- shoulder tilt and the nose's sideways offset: leaning to one side.
- face size vs baseline: sitting too close to the screen.

Per-frame values are smoothed (median over ``smooth_s``), and the displayed status changes only
after a new status has held for ``switch_s``, so it doesn't flicker. Everything runs locally;
nothing is sent anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger("sentinel.posture")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6

GOOD, SLOUCHING, LEANING, TOO_CLOSE = "good", "slouching", "leaning", "too_close"
AWAY, NO_BASELINE, CALIBRATING = "away", "no_baseline", "calibrating"
BAD = (SLOUCHING, LEANING, TOO_CLOSE)
LABELS = {GOOD: "Good", SLOUCHING: "Slouching", LEANING: "Leaning", TOO_CLOSE: "Too close",
          AWAY: "Away", NO_BASELINE: "Set your baseline", CALIBRATING: "Hold still..."}


@dataclass
class PostureConfig:
    min_kp_conf: float = 0.4
    baseline_seconds: float = 3.0
    baseline_min_frames: int = 10
    smooth_s: float = 1.0          # median window for per-frame measurements
    switch_s: float = 2.0          # a new status must hold this long before it is shown
    away_after_s: float = 2.0      # no usable pose this long: "away"
    # Deviations from baseline that count as poor posture.
    head_drop_ratio: float = 0.80  # head height falls below 80% of baseline
    hunch_ratio: float = 1.15      # face/shoulder ratio grows 15%
    tilt_deg: float = 8.0          # shoulder line tilts 8 degrees more than baseline
    lateral_shift: float = 0.20    # nose moves sideways by 20% of shoulder width
    too_close_ratio: float = 1.25  # face 25% bigger than at baseline
    timeline_bucket_s: float = 10.0


@dataclass
class Measurement:
    shoulder_width: float
    head_ratio: float              # (shoulder line y - nose y) / shoulder width
    tilt_deg: float                # shoulder line angle from horizontal (signed)
    lateral: float                 # (nose x - shoulder midpoint x) / shoulder width
    face_size: float | None        # eye span in px
    face_ratio: float | None       # eye span / shoulder width


@dataclass
class Baseline:
    shoulder_width: float
    head_ratio: float
    tilt_deg: float
    lateral: float
    face_size: float | None
    face_ratio: float | None
    recorded_at: float = 0.0
    frames: int = 0


def measure(keypoints: np.ndarray, min_conf: float = 0.4) -> Measurement | None:
    """Posture measurements from 17 COCO keypoints (x, y, conf), or None if the shoulders and
    nose aren't all visible."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or min(kp[NOSE, 2], kp[L_SH, 2], kp[R_SH, 2]) < min_conf:
        return None
    ls, rs, nose = kp[L_SH, :2], kp[R_SH, :2], kp[NOSE, :2]
    width = float(np.hypot(*(ls - rs)))
    if width < 10:
        return None
    mid = (ls + rs) / 2
    # Image left/right: order the shoulders by x so the sign doesn't depend on mirroring.
    a, b = (ls, rs) if ls[0] <= rs[0] else (rs, ls)
    tilt = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    face = None
    if min(kp[L_EYE, 2], kp[R_EYE, 2]) >= min_conf:
        face = float(np.hypot(*(kp[L_EYE, :2] - kp[R_EYE, :2])))
    return Measurement(
        shoulder_width=width,
        head_ratio=float(mid[1] - nose[1]) / width,
        tilt_deg=tilt,
        lateral=float(nose[0] - mid[0]) / width,
        face_size=face,
        face_ratio=face / width if face else None,
    )


def classify(m: Measurement, base: Baseline, cfg: PostureConfig) -> tuple[str, list[str]]:
    """Status and the reasons for it, from (smoothed) measurements vs the baseline."""
    reasons: dict[str, list[str]] = {TOO_CLOSE: [], SLOUCHING: [], LEANING: []}
    if m.face_size and base.face_size and m.face_size / base.face_size > cfg.too_close_ratio:
        bigger = m.face_size / base.face_size
        reasons[TOO_CLOSE].append(f"Face {bigger:.0%} of baseline size: move back from the screen")
    elif not m.face_size and m.shoulder_width / base.shoulder_width > cfg.too_close_ratio:
        reasons[TOO_CLOSE].append("Shoulders much wider than baseline: move back from the screen")
    if base.head_ratio > 0 and m.head_ratio / base.head_ratio < cfg.head_drop_ratio:
        reasons[SLOUCHING].append("Head has dropped toward the shoulders: sit tall, chin back")
    if m.face_ratio and base.face_ratio and m.face_ratio / base.face_ratio > cfg.hunch_ratio:
        reasons[SLOUCHING].append("Shoulders are hunched forward: roll them back and down")
    tilt = m.tilt_deg - base.tilt_deg
    if abs(tilt) > cfg.tilt_deg:
        reasons[LEANING].append(f"Shoulders tilted {abs(tilt):.0f}°: level your shoulders")
    shift = m.lateral - base.lateral
    if abs(shift) > cfg.lateral_shift:
        side = "right" if shift > 0 else "left"  # image side
        reasons[LEANING].append(f"Head shifted to the {side} of the screen: centre yourself")
    for status in (TOO_CLOSE, SLOUCHING, LEANING):
        if reasons[status]:
            return status, reasons[status]
    return GOOD, []


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    seconds: dict = field(default_factory=lambda: {s: 0.0 for s in (GOOD, SLOUCHING, LEANING, TOO_CLOSE, AWAY)})
    timeline: list = field(default_factory=list)  # [{"t": offset_s, "status": ...}] one per bucket
    reminders: int = 0


class PostureCoach:
    def __init__(self, cfg: PostureConfig | None = None, baseline_path: str | None = None,
                 clock=time.time):
        self.cfg = cfg or PostureConfig()
        self.baseline_path = baseline_path
        self.clock = clock
        self.baseline: Baseline | None = self._load()
        self.session = Session(started_at=clock())
        self._window: deque[tuple[float, Measurement]] = deque()
        self._calib: list[Measurement] | None = None
        self._calib_until = 0.0
        self._calib_error: str | None = None
        self.status = NO_BASELINE if self.baseline is None else AWAY
        self.reasons: list[str] = []
        self._candidate = self.status
        self._candidate_since = 0.0
        self._status_since = 0.0
        self.episode = 0  # increments every time the shown status changes
        self._last_ts: float | None = None
        self._last_seen = -1e9
        self._bucket: dict[str, float] = {}
        self._bucket_start: float | None = None
        self._origin: float | None = None  # first frame time of the session (timeline offsets)
        self.last: Measurement | None = None

    # --- baseline -------------------------------------------------------------------------

    def start_baseline(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self._calib = []
        self._calib_until = now + self.cfg.baseline_seconds
        self._calib_error = None

    def _finish_baseline(self) -> None:
        samples, self._calib = self._calib or [], None
        if len(samples) < self.cfg.baseline_min_frames:
            self._calib_error = ("Couldn't see your face and shoulders clearly enough. Sit facing the camera "
                                 "with both shoulders in view, then try again.")
            logger.info("posture baseline failed: %d usable frames", len(samples))
            return

        def med(values):
            values = [v for v in values if v is not None]
            return float(np.median(values)) if values else None

        self.baseline = Baseline(
            shoulder_width=med([s.shoulder_width for s in samples]),
            head_ratio=med([s.head_ratio for s in samples]),
            tilt_deg=med([s.tilt_deg for s in samples]),
            lateral=med([s.lateral for s in samples]),
            face_size=med([s.face_size for s in samples]),
            face_ratio=med([s.face_ratio for s in samples]),
            recorded_at=self.clock(),
            frames=len(samples),
        )
        self._save()
        self._window.clear()
        self._set_status(GOOD, [], self._last_ts or self.clock())
        logger.info("posture baseline recorded from %d frames", len(samples))

    def clear_baseline(self) -> None:
        self.baseline = None
        if self.baseline_path and os.path.isfile(self.baseline_path):
            os.remove(self.baseline_path)
        self._set_status(NO_BASELINE, [], self._last_ts or self.clock())

    # --- per frame ------------------------------------------------------------------------

    def update(self, keypoints: np.ndarray | None, ts: float) -> None:
        """Feed the main person's keypoints for one frame (None if nobody is visible)."""
        dt = 0.0 if self._last_ts is None else min(1.0, max(0.0, ts - self._last_ts))
        self._last_ts = ts
        m = measure(keypoints, self.cfg.min_kp_conf) if keypoints is not None else None
        if m is not None:
            self._last_seen = ts
            self.last = m

        if self._calib is not None:
            if m is not None:
                self._calib.append(m)
            if ts >= self._calib_until:
                self._finish_baseline()
            self._account(dt, ts, count=False)
            return

        if self.baseline is None:
            self._set_status(NO_BASELINE, [], ts)
        elif ts - self._last_seen > self.cfg.away_after_s:
            self._set_status(AWAY, ["Nobody in front of the camera"], ts)
            self._window.clear()
        elif m is not None:
            self._window.append((ts, m))
            while self._window and ts - self._window[0][0] > self.cfg.smooth_s:
                self._window.popleft()
            status, reasons = classify(self._smoothed(), self.baseline, self.cfg)
            self._propose(status, reasons, ts)
        self._account(dt, ts)

    def _smoothed(self) -> Measurement:
        ms = [m for _t, m in self._window]

        def med(attr):
            values = [getattr(m, attr) for m in ms if getattr(m, attr) is not None]
            return float(np.median(values)) if values else None

        return Measurement(med("shoulder_width"), med("head_ratio"), med("tilt_deg"), med("lateral"),
                           med("face_size"), med("face_ratio"))

    def _propose(self, status: str, reasons: list[str], ts: float) -> None:
        if status == self.status:
            self.reasons = reasons
            self._candidate = status
            return
        if status != self._candidate:
            self._candidate, self._candidate_since = status, ts
        if ts - self._candidate_since >= self.cfg.switch_s:
            self._set_status(status, reasons, ts)

    def _set_status(self, status: str, reasons: list[str], ts: float) -> None:
        if status != self.status:
            self.status = status
            self._status_since = ts
            self.episode += 1
        self.reasons = reasons
        self._candidate, self._candidate_since = status, ts

    def _account(self, dt: float, ts: float, count: bool = True) -> None:
        key = self.status if self.status in self.session.seconds else None
        if count and key and dt > 0:
            self.session.seconds[key] += dt
        # Timeline: the most common status in each bucket.
        if self._bucket_start is None:
            self._bucket_start = ts
        if self._origin is None:
            self._origin = ts
        if key and dt > 0:
            self._bucket[key] = self._bucket.get(key, 0.0) + dt
        if ts - self._bucket_start >= self.cfg.timeline_bucket_s:
            if self._bucket:
                self.session.timeline.append({
                    "t": round(self._bucket_start - self._origin, 1),
                    "status": max(self._bucket, key=self._bucket.get),
                })
                self.session.timeline = self.session.timeline[-720:]  # 2 h at 10 s
            self._bucket, self._bucket_start = {}, ts

    def note_reminder(self) -> None:
        self.session.reminders += 1

    def reset_session(self) -> None:
        self.session = Session(started_at=self.clock())
        self._bucket, self._bucket_start = {}, None
        self._origin = None

    # --- output ---------------------------------------------------------------------------

    def snapshot(self) -> dict:
        ts = self._last_ts or 0.0
        calibrating = self._calib is not None
        status = CALIBRATING if calibrating else self.status
        held = ts - self._status_since if self.status in BAD and not calibrating else 0.0
        good = self.session.seconds[GOOD]
        poor = sum(self.session.seconds[s] for s in BAD)
        return {
            "status": status,
            "label": LABELS.get(status, status),
            "reasons": self.reasons if not calibrating else ["Sit upright and look at the screen"],
            "held_s": round(held, 1),
            "episode": self.episode,
            "has_baseline": self.baseline is not None,
            "calibrating": calibrating,
            "calibration_left_s": round(max(0.0, self._calib_until - ts), 1) if calibrating else 0.0,
            "calibration_error": self._calib_error,
            "baseline_recorded_at": self.baseline.recorded_at if self.baseline else None,
            "session": {
                "seconds": {k: round(v, 1) for k, v in self.session.seconds.items()},
                "good_s": round(good, 1),
                "poor_s": round(poor, 1),
                "good_fraction": round(good / (good + poor), 3) if good + poor > 0 else None,
                "started_at": self.session.started_at,
                "reminders": self.session.reminders,
            },
            "measurements": asdict(self.last) if self.last else None,
        }

    def timeline(self) -> list[dict]:
        return list(self.session.timeline)

    # --- persistence ----------------------------------------------------------------------

    def _load(self) -> Baseline | None:
        if not self.baseline_path or not os.path.isfile(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, encoding="utf-8") as f:
                return Baseline(**json.load(f))
        except (OSError, ValueError, TypeError):
            logger.warning("ignoring unreadable posture baseline %s", self.baseline_path)
            return None

    def _save(self) -> None:
        if not self.baseline_path or self.baseline is None:
            return
        os.makedirs(os.path.dirname(self.baseline_path) or ".", exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.baseline), f, indent=2)
