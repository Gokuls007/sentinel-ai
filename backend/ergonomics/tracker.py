"""Per-track REBA over time: smoothing, sustained-risk alerts, static postures, time at risk.

Scores are tracked **per track ID**, not per person: when the tracker loses someone and
re-finds them under a new ID, their time is split across both IDs.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from activity.visibility import ergo_reason
from ergonomics.angles import PostureAngles, compute_angles
from ergonomics.config import ErgonomicsConfig
from ergonomics.reba import RISK_LEVELS, RebaResult, assess, risk_level

MAX_FRAME_GAP_S = 1.0  # longer gaps between sightings don't count as time at risk


def _nested():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(float)))


@dataclass
class TrackErgo:
    """What the dashboard shows for one person right now."""

    track_id: int
    score: int | None = None         # smoothed REBA (median over the window)
    level: int | None = None
    level_name: str | None = None
    confidence: float = 0.0
    reliable: bool = False           # confident enough to show normally and to alert
    reason: str | None = None        # why not reliable, in plain words (e.g. "legs not visible")
    dominant: str | None = None
    raw: RebaResult | None = None
    angles: PostureAngles | None = None

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id, "score": self.score, "level": self.level,
            "level_name": self.level_name, "confidence": round(self.confidence, 2),
            "reliable": self.reliable, "dominant": self.dominant, "reason": self.reason,
            "angles": self.angles.as_dict() if self.angles else None,
            "estimated_parts": self.raw.estimated_parts if self.raw else [],
        }


@dataclass
class ErgoAlert:
    track_id: int
    timestamp: float
    peak_score: int
    level: int
    level_name: str
    dominant: str
    duration_s: float
    confidence: float
    angles: dict
    zone_ids: list[str]


@dataclass
class _State:
    history: deque = field(default_factory=deque)       # (ts, score, confidence, dominant)
    trunk_history: deque = field(default_factory=deque)  # (ts, trunk angle) for static detection
    high_since: float | None = None
    peak: int = 0
    peak_dominant: str = ""
    peak_angles: dict = field(default_factory=dict)
    last_alert: float = -1e18
    last_ts: float | None = None


class ErgoTracker:
    def __init__(self, cfg: ErgonomicsConfig | None = None):
        self.cfg = cfg or ErgonomicsConfig()
        self.tracks: dict[int, _State] = {}
        self.current: dict[int, TrackErgo] = {}
        # Seconds at each risk level (1..5), by local date. Unreliable frames are counted
        # under level 0 ("unknown") so totals still add up to time observed.
        self.by_zone: dict = _nested()   # day -> zone_id ("" = no zone) -> level -> seconds
        self.by_hour: dict = _nested()   # day -> hour -> level -> seconds

    def _static(self, st: _State, ts: float, trunk: float | None) -> bool:
        """Posture held: trunk angle within a few degrees for static_after_s."""
        if trunk is None:
            st.trunk_history.clear()
            return False
        st.trunk_history.append((ts, trunk))
        while st.trunk_history and ts - st.trunk_history[0][0] > self.cfg.static_after_s:
            st.trunk_history.popleft()
        span = ts - st.trunk_history[0][0]
        angles = [a for _, a in st.trunk_history]
        steady = max(angles) - min(angles) <= self.cfg.static_max_trunk_change_deg
        return span >= self.cfg.static_after_s * 0.95 and steady

    def update(self, track_id: int, keypoints, ts: float, *, load: int = 0,
               zone_ids: list[str] | None = None,
               box_height_frac: float | None = None) -> tuple[TrackErgo, ErgoAlert | None]:
        cfg = self.cfg
        st = self.tracks.setdefault(track_id, _State())
        angles = compute_angles(keypoints, cfg)
        static = self._static(st, ts, angles.trunk)
        result = assess(angles, load=load, static=static, wrist_score=cfg.wrist_default_score)
        view = TrackErgo(track_id=track_id, angles=angles, raw=result, confidence=angles.confidence)
        alert = None

        if result is not None:
            st.history.append((ts, result.score, angles.confidence, result.dominant))
        while st.history and ts - st.history[0][0] > cfg.smoothing_window_s:
            st.history.popleft()
        confident = [h for h in st.history if h[2] >= cfg.min_confidence]
        window = confident or list(st.history)
        if window:
            score = round(statistics.median(h[1] for h in window))
            level, name = risk_level(score)
            view.score, view.level, view.level_name = score, level, name
            view.reliable = bool(confident) and angles.confidence >= cfg.min_confidence
            view.dominant = max((h[3] for h in window), key=[h[3] for h in window].count)
        if not view.reliable:
            view.reason = ergo_reason(keypoints, angles.confidence, cfg.min_confidence, box_height_frac,
                                      min_conf=cfg.keypoint_min_conf)

        # Time at risk (per zone, per hour; never per person), using the gap since the last frame.
        if st.last_ts is not None:
            dt = ts - st.last_ts
            if 0 < dt <= MAX_FRAME_GAP_S:
                day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                hour = datetime.fromtimestamp(ts).hour
                lvl = view.level if (view.reliable and view.level) else 0
                self.by_hour[day][hour][lvl] += dt
                for z in zone_ids or [""]:
                    self.by_zone[day][z][lvl] += dt
        st.last_ts = ts

        # Sustained high risk -> one event, then cooldown.
        if view.reliable and view.level is not None and view.level >= cfg.alert_level:
            if st.high_since is None:
                st.high_since, st.peak = ts, 0
            if view.score >= st.peak:
                st.peak, st.peak_dominant, st.peak_angles = view.score, view.dominant or "", angles.as_dict()
            lasted = ts - st.high_since
            if lasted >= cfg.alert_after_s and ts - st.last_alert >= cfg.alert_cooldown_s:
                level, name = risk_level(st.peak)
                alert = ErgoAlert(
                    track_id=track_id, timestamp=ts, peak_score=st.peak, level=level, level_name=name,
                    dominant=st.peak_dominant, duration_s=round(lasted, 1),
                    confidence=round(view.confidence, 2), angles=st.peak_angles,
                    zone_ids=list(zone_ids or []),
                )
                st.last_alert = ts
        else:
            st.high_since = None

        self.current[track_id] = view
        return view, alert

    def prune(self, active_ids) -> None:
        active = set(active_ids)
        for tid in [t for t in self.tracks if t not in active]:
            del self.tracks[tid]
            self.current.pop(tid, None)

    def drain_time(self) -> list[tuple[str, str, str, int, float]]:
        """Accumulated (day, kind, key, level, seconds) rows since the last drain, then reset.
        kind is "zone" or "hour" (never per person); used to persist time-at-risk."""
        rows = []
        for kind, data in (("zone", self.by_zone), ("hour", self.by_hour)):
            for day, keys in data.items():
                for key, levels in keys.items():
                    for lvl, secs in levels.items():
                        if secs > 0:
                            rows.append((day, kind, str(key), int(lvl), float(secs)))
            data.clear()
        return rows


LEVEL_NAMES = {0: "unknown", **{i + 1: n for i, n in enumerate(RISK_LEVELS)}}
