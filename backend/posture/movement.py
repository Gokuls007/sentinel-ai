"""Movement coach: how long since you last moved, breaks, and static time.

Nobody holds one posture all day, and shifting is normal; the real problem is staying static
too long. So the coach tracks *movement*, not instant posture:

- **Moved** = a large position change for ``hold_s`` in total within ``hold_window_s`` (held,
  or moving around in view: walking crosses back near the old spot now and then; the torso shifts by
  ``shift_widths`` shoulder widths, the shoulder line turns by ``lean_change_deg``, or the
  shoulder width changes by ``width_change``, i.e. leaning far in or back), getting up (away
  for ``away_s``), or finishing a stretch break. It is measured against your own position
  since the last movement, smoothed over ``smooth_s``, so it needs no calibration.
- **Not movement:** typing, fidgeting, glancing around, and ordinary seated posture changes
  (upright, slouching, leaning). From a laptop webcam a small torso turn changes the
  shoulders' apparent width and angle a lot: in a real session, sitting "naturally" in one
  posture swung the 1 s-smoothed shoulder width by up to 40% and the tilt by up to 24 degrees.
  With the first thresholds (1 s smoothing, 3 s hold, 15% / 10 degrees) two minutes of
  seated posture changes registered five "movements", so the still timer never got far.
  The defaults below register none on that recording (tests/data/real_seated_shoulders.json).
- **Breaks** = away for ``break_min_s`` or more, or a completed stretch break.
- **Static time** = only still stretches longer than ``static_min_s`` (10 min) count, so short
  pauses between movements don't inflate it.
- **Reminder** once you've been still for ``reminder_s`` (30 min), whatever your posture; it
  offers the guided stretch break. Moving resets it; snoozing delays it.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

L_SH, R_SH = 5, 6
MOVING, STILL, LONG_STILL, AWAY, BREAK = "moving", "still", "long_still", "away", "on_break"


@dataclass
class MovementConfig:
    shift_widths: float = 0.5
    lean_change_deg: float = 20.0
    width_change: float = 0.35
    hold_s: float = 8.0
    hold_window_s: float = 12.0   # ...counted in total within this window
    smooth_s: float = 3.0
    away_s: float = 20.0
    break_min_s: float = 60.0
    static_min_s: float = 600.0
    reminder_s: float = 1800.0
    snooze_s: float = 600.0
    moving_shown_s: float = 10.0   # "moving" for this long after a real movement, then "still"
    log_every_s: float = 1.0       # the movement log (for tuning): one line per second
    log_max_bytes: int = 4_000_000
    timeline_bucket_s: float = 10.0


@dataclass
class _Pose:
    x: float
    y: float
    width: float
    tilt: float


def pose_of(keypoints, min_conf: float = 0.4) -> _Pose | None:
    if keypoints is None:
        return None
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or min(kp[L_SH, 2], kp[R_SH, 2]) < min_conf:
        return None
    a, b = (kp[L_SH, :2], kp[R_SH, :2]) if kp[L_SH, 0] <= kp[R_SH, 0] else (kp[R_SH, :2], kp[L_SH, :2])
    width = float(np.hypot(*(b - a)))
    if width < 10:
        return None
    return _Pose(float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2), width,
                 math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


@dataclass
class Today:
    day: str
    breaks: int = 0
    static_s: float = 0.0
    longest_still_s: float = 0.0
    timeline: list = field(default_factory=list)  # [{"t": unix ts, "state": ...}] one per bucket


class MovementTracker:
    def __init__(self, cfg: MovementConfig | None = None, history=None, now: float | None = None,
                 log_path: str | None = None):
        self.cfg = cfg or MovementConfig()
        self.history = history
        self._window: deque[tuple[float, _Pose]] = deque()
        self.anchor: _Pose | None = None
        self.last_moved: float | None = None
        self._changed_since: float | None = None
        self._big: deque[tuple[float, bool]] = deque()  # (ts, large change?) over hold_window_s
        self._last_seen: float | None = None
        self._away_since: float | None = None
        self._away_ended = False  # away long enough that the still stretch was closed
        self._skip_from: float | None = None
        self.on_break = False
        self.reminder_offered = False
        self.reminder_id = 0
        self.snooze_until = 0.0
        self.last_change: dict | None = None  # what the last movement was, for the details view
        self._bucket: dict[str, float] = {}
        self._bucket_start: float | None = None
        self._last_ts: float | None = None
        self.today = self._load_today(now)
        self._moved_at: float | None = None  # the last *real* movement (not the session start)
        self.log_path = log_path
        self._last_log = 0.0
        self._last_measure: dict | None = None
        self._measured_at: float | None = None

    # --- today's totals (persisted in the history) ---------------------------------------------

    @staticmethod
    def _day(ts: float | None) -> str:
        return datetime.fromtimestamp(ts if ts is not None else datetime.now().timestamp()).date().isoformat()

    def _load_today(self, now: float | None) -> Today:
        day = self._day(now)
        today = Today(day=day)
        if self.history is not None:
            midnight = datetime.fromisoformat(day).timestamp()
            for ev in self.history.events(midnight, kinds=("move_break", "move_static")):
                if ev["kind"] == "move_break":
                    today.breaks += 1
                else:
                    s = float(ev["data"].get("seconds", 0))
                    today.static_s += s
                    today.longest_still_s = max(today.longest_still_s, s)
        return today

    def _roll_day(self, ts: float) -> None:
        if self._day(ts) != self.today.day:
            self.today = Today(day=self._day(ts))

    def _log(self, ts: float, kind: str, data: dict) -> None:
        if self.history is not None:
            self.history.add_event(ts, kind, data)

    # --- per frame -----------------------------------------------------------------------------

    def _smoothed(self) -> _Pose:
        ps = [p for _t, p in self._window]
        return _Pose(*(float(np.median([getattr(p, k) for p in ps])) for k in ("x", "y", "width", "tilt")))

    def _change(self, p: _Pose) -> dict:
        a = self.anchor
        return {"shift": math.hypot(p.x - a.x, p.y - a.y) / a.width, "width": abs(p.width / a.width - 1),
                "lean": abs(p.tilt - a.tilt)}

    def _moved(self, ts: float, pose: _Pose | None, why: str) -> None:
        """A movement: end the still stretch, reset the reminder, re-anchor."""
        self._end_stretch(ts)
        self.last_moved = ts
        self.anchor = pose
        self._changed_since = None
        self.reminder_offered = False
        self.last_change = {"at": ts, "why": why}
        self._moved_at = ts

    def _end_stretch(self, ts: float) -> None:
        if self.last_moved is None:
            return
        length = ts - self.last_moved
        self.today.longest_still_s = max(self.today.longest_still_s, length)
        if length > self.cfg.static_min_s:
            self.today.static_s += length
            self._log(ts, "move_static", {"seconds": round(length, 1)})

    def count_break(self, ts: float, kind: str, seconds: float = 0.0) -> None:
        self.today.breaks += 1
        self._log(ts, "move_break", {"kind": kind, "seconds": round(seconds, 1)})

    def update(self, keypoints, ts: float, min_conf: float = 0.4, seen: bool | None = None) -> None:
        """``seen``: someone is there even if their shoulders can't be measured this frame (face
        visible, dim light). Then nothing changes: it's neither movement nor being away."""
        dt = 0.0 if self._last_ts is None else min(1.0, max(0.0, ts - self._last_ts))
        self._last_ts = ts
        self._roll_day(ts)
        pose = pose_of(keypoints, min_conf)
        if pose is not None:
            self._present(pose, ts)
        elif seen:
            if self._away_since is not None and not self._away_ended:
                self._away_since = None  # back before it counted as getting up
        else:
            self._away(ts)
        self._account(dt, ts)
        self._log_measures(ts)
        if (not self.reminder_offered and not self.on_break and self.state(ts) not in (AWAY, BREAK)
                and self.still_s(ts) >= self.cfg.reminder_s and ts >= self.snooze_until):
            self.reminder_offered = True
            self.reminder_id += 1
            self._log(ts, "move_reminder", {"still_s": round(self.still_s(ts), 1)})

    def _away(self, ts: float) -> None:
        if self._last_seen is None:
            return  # nobody has sat down yet
        if self._away_since is None:
            self._away_since = self._last_seen
        if not self._away_ended and ts - self._away_since >= self.cfg.away_s:
            # Really gone (got up), not a detection dropout: close the still stretch.
            self._away_ended = True
            self._end_stretch(self._away_since)
            self.last_moved = None
            self._window.clear()

    def _present(self, pose: _Pose, ts: float) -> None:
        if self._away_since is not None:
            gone = ts - self._away_since
            self._away_since, self._away_ended = None, False
            if gone >= self.cfg.away_s:  # got up: that's movement (and a break if long enough)
                if gone >= self.cfg.break_min_s:
                    self.count_break(ts, "away", gone)
                self._window.clear()
                self._window.append((ts, pose))
                self._moved(ts, pose, f"away for {gone:.0f} s")
                self._last_seen = ts
                return
        self._last_seen = ts
        self._window.append((ts, pose))
        while self._window and ts - self._window[0][0] > self.cfg.smooth_s:
            self._window.popleft()
        if self.anchor is None or self.last_moved is None:
            self.anchor, self.last_moved = self._smoothed(), ts
            return
        c = self._change(self._smoothed())
        self._last_measure = c
        self._measured_at = ts
        big = (c["shift"] >= self.cfg.shift_widths or c["lean"] >= self.cfg.lean_change_deg
               or c["width"] >= self.cfg.width_change)
        # A large change for ``hold_s`` in total within the last ``hold_window_s``: a new position
        # held, or moving around in view (walking crosses back near the old spot now and then).
        self._big.append((ts, big))
        while self._big and ts - self._big[0][0] > self.cfg.hold_window_s:
            self._big.popleft()
        if big and self._changed_since is None:
            self._changed_since = ts
        if not any(b for _t, b in self._big):
            self._changed_since = None
            return
        frames = list(self._big)
        big_s = sum(min(1.0, frames[i + 1][0] - frames[i][0]) for i in range(len(frames) - 1) if frames[i][1])
        if big_s >= self.cfg.hold_s:
            why = max(("shift", c["shift"] / self.cfg.shift_widths), ("lean", c["lean"] / self.cfg.lean_change_deg),
                      ("width", c["width"] / self.cfg.width_change), key=lambda kv: kv[1])[0]
            # Dated from when the change started, not when it was confirmed.
            self._moved(self._changed_since, self._smoothed(), {"shift": "changed position", "lean": "changed lean",
                                                                "width": "leaned in or back"}[why])
            self._moved_at = ts  # show "moving" from when it was confirmed
            self._big.clear()

    def _log_measures(self, ts: float) -> None:
        """One JSON line per second: the change measures vs the anchor and the state, so the
        thresholds can be tuned on real sessions (issue #4). Local only; trimmed when large."""
        if not self.log_path or ts - self._last_log < self.cfg.log_every_s:
            return
        self._last_log = ts
        c = self._last_measure or {}
        fresh = self._measured_at is not None and ts - self._measured_at <= self.cfg.log_every_s
        row = {"t": round(ts, 2), "state": self.state(ts), "still_s": round(self.still_s(ts), 1),
               "measured": fresh, **({k: round(v, 3) for k, v in c.items()} if fresh else {})}
        try:
            if os.path.isfile(self.log_path) and os.path.getsize(self.log_path) > self.cfg.log_max_bytes:
                with open(self.log_path, encoding="utf-8") as f:
                    lines = f.readlines()
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.writelines(lines[len(lines) // 2:])
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            self.log_path = None  # can't write: stop trying

    # --- breaks and the reminder ---------------------------------------------------------------

    def start_break(self) -> None:
        self.on_break = True
        self.reminder_offered = False

    def finish_break(self, ts: float, completed: bool) -> None:
        self.on_break = False
        if completed:
            self.count_break(ts, "stretch")
        self._window.clear()
        self.anchor = None  # re-anchor on the next frame
        self._end_stretch(ts)
        self.last_moved = ts
        self.last_change = {"at": ts, "why": "stretch break"}
        self._moved_at = ts

    def snooze(self, ts: float) -> None:
        self.reminder_offered = False
        self.snooze_until = ts + self.cfg.snooze_s

    def dismiss(self, ts: float) -> None:
        """Skip: no reminder again until you've moved and then been still for the full time."""
        self.reminder_offered = False
        self.snooze_until = float("inf")
        self._skip_from = self.last_moved

    # --- output --------------------------------------------------------------------------------

    def still_s(self, ts: float) -> float:
        if self.last_moved is None or self._away_ended:
            return 0.0
        return max(0.0, ts - self.last_moved)

    def state(self, ts: float) -> str:
        if self.on_break:
            return BREAK
        if self._away_ended:
            return AWAY
        if self.last_moved is None:
            return AWAY
        s = self.still_s(ts)
        if self._moved_at is not None and ts - self._moved_at < self.cfg.moving_shown_s:
            return MOVING
        return LONG_STILL if s > self.cfg.static_min_s else STILL

    def _account(self, dt: float, ts: float) -> None:
        if self._skip_from is not None and self.last_moved != self._skip_from:
            self.snooze_until, self._skip_from = 0.0, None  # moved since skipping: reminders resume
        state = self.state(ts)
        if self._bucket_start is None:
            self._bucket_start = ts
        if dt > 0:
            self._bucket[state] = self._bucket.get(state, 0.0) + dt
        if ts - self._bucket_start >= self.cfg.timeline_bucket_s:
            if self._bucket:
                self.today.timeline.append({"t": round(self._bucket_start, 1),
                                            "state": max(self._bucket, key=self._bucket.get)})
                self.today.timeline = self.today.timeline[-8640:]  # a day at 10 s
                if self.history is not None:
                    for s, secs in self._bucket.items():
                        self.history.add_seconds(self._bucket_start, f"mv:{s}", secs)
            self._bucket, self._bucket_start = {}, ts

    def snapshot(self, ts: float) -> dict:
        still = self.still_s(ts)
        ongoing = still if still > self.cfg.static_min_s else 0.0
        return {
            "state": self.state(ts),
            "still_s": round(still, 1),
            "last_moved_at": self.last_moved,
            "last_change": self.last_change,
            "breaks_today": self.today.breaks,
            "static_today_s": round(self.today.static_s + ongoing, 1),
            "longest_still_s": round(max(self.today.longest_still_s, still), 1),
            "reminder": {"offered": self.reminder_offered, "id": self.reminder_id,
                         "after_s": self.cfg.reminder_s, "snoozed": ts < self.snooze_until != float("inf"),
                         "skipped": self.snooze_until == float("inf"), "snooze_s": self.cfg.snooze_s},
            "static_min_s": self.cfg.static_min_s,
        }

    def timeline(self) -> list[dict]:
        return list(self.today.timeline)
