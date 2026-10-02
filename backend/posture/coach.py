"""Desk posture coach: compares a seated person, seen from the front by a laptop webcam, with
their own upright baseline.

REBA isn't used here: a frontal webcam can't measure trunk or neck flexion reliably (see
ergonomics/). Every posture measurement is **relative to your own shoulders**, never to where you
sit in the frame, so moving your chair or sitting off-centre isn't a posture problem:

- head height: nose above the shoulder line / shoulder width. It shrinks when the head drops
  or juts forward (slouching).
- face-to-shoulder ratio: eye span / shoulder width. It grows when the shoulders roll in
  (hunching).
- head offset: ear midpoint (else eye midpoint; never the nose, which swings sideways whenever
  you turn your head) minus shoulder midpoint, / shoulder width. With shoulder tilt, it shows
  leaning.
- face size vs baseline: sitting too close to the screen (distance is the point here).

If your whole body has moved a lot since the baseline (shoulders shifted by more than a shoulder
width, or shoulder width changed by over 25% without your face growing to match), no posture is
judged; a "You've moved" hint suggests resetting the baseline instead.

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
from dataclasses import asdict, dataclass, field, fields

import numpy as np

logger = logging.getLogger("sentinel.posture")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6

GOOD, SLOUCHING, LEANING, TOO_CLOSE = "good", "slouching", "leaning", "too_close"
AWAY, NO_BASELINE, CALIBRATING, MOVED = "away", "no_baseline", "calibrating", "moved"
BAD = (SLOUCHING, LEANING, TOO_CLOSE)
LABELS = {GOOD: "Good", SLOUCHING: "Slouching", LEANING: "Leaning", TOO_CLOSE: "Too close",
          AWAY: "Away", NO_BASELINE: "Set your baseline", CALIBRATING: "Hold still...",
          MOVED: "You've moved"}


@dataclass
class PostureConfig:
    min_kp_conf: float = 0.4
    baseline_delay_s: float = 3.0   # "get ready" countdown after pressing Set baseline
    baseline_seconds: float = 3.0   # then recording
    baseline_min_frames: int = 10
    # A baseline is rejected if you moved while it recorded (spread of the measurements) or it
    # doesn't look like sitting upright.
    baseline_max_std_head: float = 0.06
    baseline_max_std_lateral: float = 0.08
    baseline_max_std_tilt: float = 4.0
    baseline_max_abs_tilt: float = 12.0
    baseline_min_head_ratio: float = 0.2
    smooth_s: float = 1.0          # median window for per-frame measurements
    switch_s: float = 2.0          # a new status must hold this long before it is shown
    away_after_s: float = 2.0      # no usable pose this long: "away"
    # Deviations from baseline that count as poor posture.
    head_drop_ratio: float = 0.80  # head height falls below 80% of baseline
    hunch_ratio: float = 1.15      # face/shoulder ratio grows 15%
    tilt_deg: float = 8.0          # shoulder line tilts 8 degrees more than baseline
    lateral_shift: float = 0.30    # head offset changes by 30% of shoulder width
    too_close_ratio: float = 1.25  # face 25% bigger than at baseline
    # "You've moved": shoulder midpoint shifted by more than this many baseline shoulder widths,
    # or shoulder width changed by more than this fraction (unless the face grew to match:
    # that's sitting too close, which *is* judged).
    moved_shift_widths: float = 1.0
    moved_width_change: float = 0.25
    timeline_bucket_s: float = 10.0


@dataclass
class Measurement:
    shoulder_width: float
    head_ratio: float              # (shoulder line y - nose y) / shoulder width
    tilt_deg: float                # shoulder line angle from horizontal (signed)
    lateral: float | None          # (ear midpoint, else eye midpoint, x - shoulder midpoint x) / shoulder width
    face_size: float | None        # eye span in px
    face_ratio: float | None       # eye span / shoulder width
    mid_x: float | None = None     # shoulder midpoint in the frame (px): only for "you've moved"
    mid_y: float | None = None
    head_ref: str | None = None    # "ears" | "eyes": what the head offset was measured from


@dataclass
class Baseline:
    shoulder_width: float
    head_ratio: float
    tilt_deg: float
    lateral: float | None
    face_size: float | None
    face_ratio: float | None
    recorded_at: float = 0.0
    frames: int = 0
    mid_x: float | None = None
    mid_y: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Baseline:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _visible(kp: np.ndarray, *idx: int, conf: float) -> bool:
    return all(kp[i, 2] >= conf for i in idx)


def measure(keypoints: np.ndarray, min_conf: float = 0.4) -> Measurement | None:
    """Posture measurements from 17 COCO keypoints (x, y, conf), or None if the shoulders and
    nose aren't all visible."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or not _visible(kp, NOSE, L_SH, R_SH, conf=min_conf):
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
    if _visible(kp, L_EYE, R_EYE, conf=min_conf):
        face = float(np.hypot(*(kp[L_EYE, :2] - kp[R_EYE, :2])))
    # Head offset from the ear midpoint (barely moves when you turn your head), else the eye
    # midpoint. Never the nose.
    head_x, head_ref = None, None
    if _visible(kp, L_EAR, R_EAR, conf=min_conf):
        head_x, head_ref = float((kp[L_EAR, 0] + kp[R_EAR, 0]) / 2), "ears"
    elif _visible(kp, L_EYE, R_EYE, conf=min_conf):
        head_x, head_ref = float((kp[L_EYE, 0] + kp[R_EYE, 0]) / 2), "eyes"
    return Measurement(
        shoulder_width=width,
        head_ratio=float(mid[1] - nose[1]) / width,
        tilt_deg=tilt,
        lateral=(head_x - float(mid[0])) / width if head_x is not None else None,
        face_size=face,
        face_ratio=face / width if face else None,
        mid_x=float(mid[0]),
        mid_y=float(mid[1]),
        head_ref=head_ref,
    )


def _ratio(a, b):
    return a / b if a is not None and b else None


def explain(m: Measurement, base: Baseline, cfg: PostureConfig) -> list[dict]:
    """Every measurement next to its baseline and limit (the debug panel and ``classify``)."""
    rows = []

    def row(key, label, now, baseline, change, limit, triggered, status):
        rows.append({"key": key, "label": label, "now": now, "baseline": baseline, "change": change,
                     "limit": limit, "triggered": bool(triggered), "status": status})

    face_change = _ratio(m.face_size, base.face_size)
    width_change = _ratio(m.shoulder_width, base.shoulder_width)
    head_change = _ratio(m.head_ratio, base.head_ratio)
    hunch_change = _ratio(m.face_ratio, base.face_ratio)
    tilt_change = m.tilt_deg - base.tilt_deg
    offset_change = m.lateral - base.lateral if m.lateral is not None and base.lateral is not None else None
    shift = None
    if None not in (m.mid_x, m.mid_y, base.mid_x, base.mid_y):
        shift = math.hypot(m.mid_x - base.mid_x, m.mid_y - base.mid_y) / base.shoulder_width

    too_close = (face_change is not None and face_change > cfg.too_close_ratio) or (
        face_change is None and width_change is not None and width_change > cfg.too_close_ratio)
    row("face_size", "Face size vs baseline", m.face_size, base.face_size, face_change,
        f"> {cfg.too_close_ratio:.2f}x", too_close, TOO_CLOSE)
    row("head_ratio", "Head height (/ shoulder width)", m.head_ratio, base.head_ratio, head_change,
        f"< {cfg.head_drop_ratio:.2f}x", head_change is not None and head_change < cfg.head_drop_ratio, SLOUCHING)
    row("face_ratio", "Face / shoulder width", m.face_ratio, base.face_ratio, hunch_change,
        f"> {cfg.hunch_ratio:.2f}x", hunch_change is not None and hunch_change > cfg.hunch_ratio, SLOUCHING)
    row("tilt_deg", "Shoulder tilt (deg)", m.tilt_deg, base.tilt_deg, tilt_change,
        f"> {cfg.tilt_deg:g} deg", abs(tilt_change) > cfg.tilt_deg, LEANING)
    row("lateral", f"Head offset ({m.head_ref or 'n/a'}, / shoulder width)", m.lateral, base.lateral,
        offset_change, f"> {cfg.lateral_shift:.2f}",
        offset_change is not None and abs(offset_change) > cfg.lateral_shift, LEANING)
    moved_width = width_change is not None and abs(width_change - 1) > cfg.moved_width_change and not too_close
    moved = (shift is not None and shift > cfg.moved_shift_widths) or moved_width
    row("moved", "Body moved (shoulder shift / width change)", shift, None, width_change,
        f"> {cfg.moved_shift_widths:g} widths or {cfg.moved_width_change:.0%}", moved, MOVED)
    return rows


def classify(m: Measurement, base: Baseline, cfg: PostureConfig) -> tuple[str, list[str]]:
    """Status and the reasons for it, from (smoothed) measurements vs the baseline."""
    rows = {r["key"]: r for r in explain(m, base, cfg)}
    if rows["moved"]["triggered"]:
        return MOVED, ["You've moved since your baseline. Reset baseline?"]
    reasons: dict[str, list[str]] = {TOO_CLOSE: [], SLOUCHING: [], LEANING: []}
    if rows["face_size"]["triggered"]:
        change = rows["face_size"]["change"]
        reasons[TOO_CLOSE].append(f"Face {change:.0%} of its baseline size: move back from the screen"
                                  if change else "Much closer than at baseline: move back from the screen")
    if rows["head_ratio"]["triggered"]:
        reasons[SLOUCHING].append("Head has dropped toward the shoulders: sit tall, chin back")
    if rows["face_ratio"]["triggered"]:
        reasons[SLOUCHING].append("Shoulders are hunched forward: roll them back and down")
    if rows["tilt_deg"]["triggered"]:
        reasons[LEANING].append(f"Shoulders tilted {abs(rows['tilt_deg']['change']):.0f}° more than at baseline: "
                                "level your shoulders")
    if rows["lateral"]["triggered"]:
        side = "right" if rows["lateral"]["change"] > 0 else "left"  # image side
        reasons[LEANING].append(f"Head leaning to the {side} of your shoulders: bring it back over them")
    for status in (TOO_CLOSE, SLOUCHING, LEANING):
        if reasons[status]:
            return status, reasons[status]
    return GOOD, []


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    seconds: dict = field(default_factory=lambda: {s: 0.0 for s in (GOOD, SLOUCHING, LEANING, TOO_CLOSE, AWAY,
                                                                     MOVED)})
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
        self._calib_start = 0.0
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
        self.smoothed: Measurement | None = None  # what the last classification used

    # --- baseline -------------------------------------------------------------------------

    def start_baseline(self, now: float | None = None, delay: float | None = None) -> None:
        """Count down ``baseline_delay_s`` (time to sit back after clicking), then record for
        ``baseline_seconds``."""
        now = self.clock() if now is None else now
        delay = self.cfg.baseline_delay_s if delay is None else delay
        self._calib = []
        self._calib_start = now + delay
        self._calib_until = self._calib_start + self.cfg.baseline_seconds
        self._calib_error = None

    def baseline_problem(self, samples: list[Measurement]) -> str | None:
        """Why these samples don't make a usable upright baseline, or None if they do."""
        cfg = self.cfg
        if len(samples) < cfg.baseline_min_frames:
            return ("Couldn't see your face and shoulders clearly enough. Sit facing the camera with both "
                    "shoulders in view, then try again.")
        head = np.array([s.head_ratio for s in samples])
        lateral = np.array([s.lateral for s in samples if s.lateral is not None] or [0.0])
        tilt = np.array([s.tilt_deg for s in samples])
        if (head.std() > cfg.baseline_max_std_head or lateral.std() > cfg.baseline_max_std_lateral
                or tilt.std() > cfg.baseline_max_std_tilt):
            return "You moved while the baseline was recording. Sit upright, keep still, and try again."
        if abs(float(np.median(tilt))) > cfg.baseline_max_abs_tilt:
            return ("Your shoulders looked tilted, so this doesn't look like an upright baseline. Sit level "
                    "and try again.")
        if float(np.median(head)) < cfg.baseline_min_head_ratio:
            return ("Your head looked very low (slouched or looking down). Sit tall, look at the screen, "
                    "and try again.")
        return None

    def _finish_baseline(self) -> None:
        samples, self._calib = self._calib or [], None
        problem = self.baseline_problem(samples)
        if problem:
            self._calib_error = problem
            logger.info("posture baseline rejected (%d frames): %s", len(samples), problem)
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
            mid_x=med([s.mid_x for s in samples]),
            mid_y=med([s.mid_y for s in samples]),
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
            if m is not None and ts >= self._calib_start:  # not during the get-ready countdown
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
            self.smoothed = self._smoothed()
            status, reasons = classify(self.smoothed, self.baseline, self.cfg)
            self._propose(status, reasons, ts)
        self._account(dt, ts)

    def _smoothed(self) -> Measurement:
        ms = [m for _t, m in self._window]

        def med(attr):
            values = [getattr(m, attr) for m in ms if getattr(m, attr) is not None]
            return float(np.median(values)) if values else None

        refs = [m.head_ref for m in ms if m.head_ref]
        return Measurement(med("shoulder_width"), med("head_ratio"), med("tilt_deg"), med("lateral"),
                           med("face_size"), med("face_ratio"), med("mid_x"), med("mid_y"),
                           max(set(refs), key=refs.count) if refs else None)

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
        getting_ready = calibrating and ts < self._calib_start
        status = CALIBRATING if calibrating else self.status
        held = ts - self._status_since if self.status in BAD and not calibrating else 0.0
        good = self.session.seconds[GOOD]
        poor = sum(self.session.seconds[s] for s in BAD)
        if getting_ready:
            label, reasons = "Get ready", ["Sit upright, shoulders relaxed and level, and look at the screen"]
            left = self._calib_start - ts
        elif calibrating:
            label, reasons = LABELS[CALIBRATING], ["Recording your baseline: keep still"]
            left = self._calib_until - ts
        else:
            label, reasons, left = LABELS.get(status, status), self.reasons, 0.0
        return {
            "status": status,
            "label": label,
            "reasons": reasons,
            "held_s": round(held, 1),
            "episode": self.episode,
            "has_baseline": self.baseline is not None,
            "calibrating": calibrating,
            "calibration_phase": ("get_ready" if getting_ready else "recording") if calibrating else None,
            "calibration_left_s": round(max(0.0, left), 1),
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
            # Every value behind the status (smoothed, vs baseline, with its limit): the debug panel.
            "debug": explain(self.smoothed, self.baseline, self.cfg)
            if self.smoothed is not None and self.baseline is not None and not calibrating else None,
        }

    def timeline(self) -> list[dict]:
        return list(self.session.timeline)

    # --- persistence ----------------------------------------------------------------------

    def _load(self) -> Baseline | None:
        if not self.baseline_path or not os.path.isfile(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, encoding="utf-8") as f:
                return Baseline.from_dict(json.load(f))
        except (OSError, ValueError, TypeError):
            logger.warning("ignoring unreadable posture baseline %s", self.baseline_path)
            return None

    def _save(self) -> None:
        if not self.baseline_path or self.baseline is None:
            return
        os.makedirs(os.path.dirname(self.baseline_path) or ".", exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.baseline), f, indent=2)
