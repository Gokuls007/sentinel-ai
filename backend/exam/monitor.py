"""Exam Hall monitor (stage E1): seats, setup check and per-seat calibration, every frame.

Runs only in Exam Hall mode. People are keyed by seat (shoulder midpoint inside the seat);
anyone in no seat (e.g. the invigilator walking around) is ignored by everything exam-related.
No flags yet: the signal engine is stage E2 (docs/plans/exam-hall.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from exam.calibration import Baseline, SeatCalibration, hand_height
from exam.head_pose import HeadPoseEstimator, default_estimator
from exam.seats import Seat, assign, label_seats, seat_rect_for, shoulders
from exam.setup_check import SetupCheck

ACTIVE = ("calibrating", "live")


@dataclass
class ExamConfig:
    calibration_s: float = 120.0
    stable_s: float = 5.0          # seated and still this long to be offered as a seat by "Detect seats"
    stable_motion: float = 0.25    # shoulder midpoint moved less than this (x shoulder width)
    head_min_conf: float = 0.5


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ExamMonitor:
    def __init__(self, cfg: ExamConfig | None = None, estimator: HeadPoseEstimator | None = None):
        self.cfg = cfg or ExamConfig()
        self.estimator = estimator or default_estimator()
        self.session: dict | None = None
        self.seats: list[Seat] = []
        self.calib: dict[str, SeatCalibration] = {}
        self.setup = SetupCheck()
        self._recent: dict[int, deque] = {}  # track id -> (ts, shoulder mid, shoulder width)
        self._latest_kp: dict[int, np.ndarray] = {}
        self._size = (1, 1)
        self._calib_start: float | None = None
        self.on_baseline = None  # callback(session_id, seat label, Baseline)
        self.on_status = None    # callback(session_id, status)
        self._snapshot: dict = {}

    # --- session -----------------------------------------------------------------------------

    def load(self, session: dict | None, seats: list[Seat] | None = None,
             baselines: dict[str, Baseline] | None = None) -> None:
        """Watch this session (None: none). Keeps calibration already done for its seats."""
        same = session is not None and self.session is not None and session["id"] == self.session["id"]
        self.session = dict(session) if session else None
        if not same:
            self.calib, self._calib_start = {}, None
        self.set_seats(seats or [], baselines)

    def set_seats(self, seats: list[Seat], baselines: dict[str, Baseline] | None = None) -> None:
        duration = float((self.session or {}).get("calibration_s") or self.cfg.calibration_s)
        old = self.calib
        self.seats = list(seats)
        self.calib = {}
        for s in self.seats:
            cal = old.get(s.label) or SeatCalibration(duration)
            if baselines and s.label in baselines:
                cal.baseline = baselines[s.label]
            self.calib[s.label] = cal

    def set_status(self, status: str) -> None:
        if self.session is not None:
            if status == "calibrating" and self.session.get("status") != "calibrating":
                self._calib_start = None  # starts at the next frame
            self.session["status"] = status

    # --- every frame -------------------------------------------------------------------------

    def update(self, poses: dict, ts: float, width: int, height: int, frame=None) -> dict:
        self._size = (width, height)
        kps = {tid: np.asarray(p.keypoints, float) for tid, p in poses.items()}
        self.setup.update(list(kps.values()), ts)
        self._track_stability(kps, ts)
        seated = assign(kps, self.seats, width, height)
        staff = sum(1 for tid in kps if tid not in seated.values())
        status = (self.session or {}).get("status")
        remaining = None
        if status in ACTIVE:
            if self._calib_start is None:
                self._calib_start = ts
            for label, tid in seated.items():
                cal = self.calib.get(label)
                if cal is None or cal.baseline is not None:
                    continue
                head = self.estimator.estimate(kps[tid], frame, getattr(poses[tid], "bbox", None))
                b = cal.add(ts, head, hand_height(kps[tid]), self.cfg.head_min_conf)
                if b is not None and self.on_baseline:
                    self.on_baseline(self.session["id"], label, b)
            duration = float(self.session.get("calibration_s") or self.cfg.calibration_s)
            remaining = max(0.0, duration - (ts - self._calib_start))
            if status == "calibrating" and remaining <= 0:
                self.set_status("live")
                status = "live"
                if self.on_status:
                    self.on_status(self.session["id"], "live")
        seats = []
        for s in self.seats:
            cal = self.calib.get(s.label)
            seats.append({**s.to_dict(), "occupied": s.label in seated,
                          "calibrated": bool(cal and cal.baseline),
                          "calibration_progress": round(cal.progress(ts), 2) if cal and status in ACTIVE else 0.0,
                          "baseline": cal.baseline.to_dict() if cal and cal.baseline else None})
        self._snapshot = {
            "session_id": (self.session or {}).get("id"),
            "status": status,
            "setup": self.setup.snapshot(),
            "seats": seats,
            "staff": staff,  # in view but in no seat: ignored by exam logic
            "seated_candidates": len(self.stable_tracks(ts)),
            "calibration_remaining_s": round(remaining, 1) if status == "calibrating" else None,
        }
        return self._snapshot

    def snapshot(self) -> dict:
        return self._snapshot

    # --- seat detection ----------------------------------------------------------------------

    def _track_stability(self, kps: dict, ts: float) -> None:
        for tid, kp in kps.items():
            s = shoulders(kp)
            if s is None:
                continue
            q = self._recent.setdefault(tid, deque())
            q.append((ts, s[0], s[1]))
            while q and ts - q[0][0] > self.cfg.stable_s + 1.0:
                q.popleft()
            self._latest_kp[tid] = kp
        for tid in [t for t in self._recent if t not in kps]:
            del self._recent[tid]
            self._latest_kp.pop(tid, None)

    def stable_tracks(self, ts: float) -> list[int]:
        """People sitting still (shoulders steady) for ``stable_s``: seated, not walking."""
        out = []
        for tid, q in self._recent.items():
            if not q or q[-1][0] - q[0][0] < self.cfg.stable_s - 0.2:
                continue
            last, sw = q[-1][1], max(q[-1][2], 1.0)
            if max(float(np.hypot(*(m - last))) for _t, m, _w in q) <= self.cfg.stable_motion * sw:
                out.append(tid)
        return out

    def detect_seats(self, ts: float | None = None) -> list[Seat]:
        """One seat per person sitting still, labelled in reading order (not saved)."""
        if ts is None:
            ts = max((q[-1][0] for q in self._recent.values() if q), default=0.0)
        w, h = self._size
        rects = []
        for tid in self.stable_tracks(ts):
            r = seat_rect_for(self._latest_kp[tid], w, h)
            if r is None:
                continue
            # Two detections of one person (or touching seats): keep the first.
            if all(_iou(r, o) < 0.3 for o in rects):
                rects.append(r)
        return label_seats(rects)
