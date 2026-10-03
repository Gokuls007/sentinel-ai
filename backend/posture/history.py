"""Local posture history (SQLite) and root-cause tips.

Stored per minute: seconds in each status (leaning split by side when known), plus
corrections, reminders and break outcomes. No images or keypoints. The tips look at the last
7 days for patterns that usually come from the desk setup rather than willpower.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime

POOR_KEYS = ("slouching", "leaning", "leaning_left", "leaning_right", "too_close", "slumped")
NOT_TRACKED = ("away", "no_baseline", "calibrating", "checking")


class PostureHistory:
    def __init__(self, path: str, clock=time.time):
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS minutes (
                    minute INTEGER NOT NULL, status TEXT NOT NULL, seconds REAL NOT NULL,
                    PRIMARY KEY (minute, status));
                CREATE TABLE IF NOT EXISTS corrections (
                    ts REAL NOT NULL, posture TEXT NOT NULL, seconds REAL NOT NULL, after_reminder INTEGER);
                CREATE TABLE IF NOT EXISTS events (ts REAL NOT NULL, kind TEXT NOT NULL, data TEXT);
                CREATE TABLE IF NOT EXISTS dismissed (rule TEXT PRIMARY KEY, until REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
        self._pending: dict[tuple[int, str], float] = {}
        self._last_flush = clock()

    # --- writing ------------------------------------------------------------------------------

    def add_seconds(self, ts: float, status: str, seconds: float) -> None:
        if status in NOT_TRACKED or seconds <= 0:
            return
        key = (int(ts // 60), status)
        self._pending[key] = self._pending.get(key, 0.0) + seconds
        if ts - self._last_flush >= 30 or len(self._pending) > 50:
            self.flush(ts)

    def flush(self, ts: float | None = None) -> None:
        pending, self._pending = self._pending, {}
        self._last_flush = self.clock() if ts is None else ts
        if not pending:
            return
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO minutes (minute, status, seconds) VALUES (?, ?, ?) "
                "ON CONFLICT(minute, status) DO UPDATE SET seconds = seconds + excluded.seconds",
                [(m, s, v) for (m, s), v in pending.items()])

    def add_correction(self, ts: float, posture: str, seconds: float, after_reminder: bool) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO corrections VALUES (?, ?, ?, ?)",
                             (ts, posture, seconds, int(after_reminder)))

    def add_event(self, ts: float, kind: str, data: dict | None = None) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO events VALUES (?, ?, ?)", (ts, kind, json.dumps(data or {})))

    def dismiss(self, rule: str, until: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO dismissed VALUES (?, ?)", (rule, until))

    def get_setting(self, key: str, default=None):
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))

    def close(self) -> None:
        self.flush()
        self._db.close()

    # --- reading ------------------------------------------------------------------------------

    def _rows(self, since: float, movement: bool = False) -> list[tuple[int, str, float]]:
        """Posture rows by default; movement-coach rows ("mv:" statuses) with ``movement``."""
        self.flush()
        with self._lock:
            rows = self._db.execute("SELECT minute, status, seconds FROM minutes WHERE minute >= ?",
                                    (int(since // 60),)).fetchall()
        return [r for r in rows if r[1].startswith("mv:") == movement]

    def events(self, since: float, kinds: tuple[str, ...] | None = None) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT ts, kind, data FROM events WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
        return [{"ts": ts, "kind": k, "data": json.loads(d or "{}")} for ts, k, d in rows
                if kinds is None or k in kinds]

    def totals(self, since: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for _m, s, v in self._rows(since):
            out[s] = out.get(s, 0.0) + v
        return out

    def by_hour(self, since: float) -> dict[int, dict]:
        """Local hour of day -> {"seconds": {status: s}, "days": number of days with data}."""
        out: dict[int, dict] = {}
        for minute, s, v in self._rows(since):
            t = datetime.fromtimestamp(minute * 60)
            h = out.setdefault(t.hour, {"seconds": {}, "days": set()})
            h["seconds"][s] = h["seconds"].get(s, 0.0) + v
            h["days"].add(t.date())
        return {k: {"seconds": v["seconds"], "days": len(v["days"])} for k, v in out.items()}

    def daily(self, days: int = 7, now: float | None = None) -> list[dict]:
        now = self.clock() if now is None else now
        per_day: dict[str, dict] = {}
        for minute, s, v in self._rows(now - days * 86400):
            d = datetime.fromtimestamp(minute * 60).date().isoformat()
            per_day.setdefault(d, {})[s] = per_day.setdefault(d, {}).get(s, 0.0) + v
        out = []
        for d, sec in sorted(per_day.items()):
            good = sec.get("good", 0.0)
            poor = sum(sec.get(k, 0.0) for k in POOR_KEYS)
            out.append({"date": d, "seconds": {k: round(v, 1) for k, v in sec.items()},
                        "good_fraction": round(good / (good + poor), 3) if good + poor else None})
        return out

    def corrections(self, since: float) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT ts, posture, seconds, after_reminder FROM corrections WHERE ts >= ?",
                                    (since,)).fetchall()
        return [{"ts": t, "posture": p, "seconds": s, "after_reminder": bool(r)} for t, p, s, r in rows]

    def dismissed(self, now: float) -> set[str]:
        with self._lock:
            return {r for (r,) in self._db.execute("SELECT rule FROM dismissed WHERE until > ?", (now,))}


# --- tips ---------------------------------------------------------------------------------------

TIP_WINDOW_DAYS = 7
MIN_TRACKED_S = 2 * 3600
SCREEN_LOW_SHARE = 0.30
TOO_CLOSE_SHARE = 0.15
LEAN_SHARE = 0.15
LEAN_ONE_SIDE = 0.75
HOUR_MIN_S = 20 * 60
HOUR_MIN_DAYS = 3
HOUR_RATIO = 1.5
HOUR_MIN_SHARE = 0.20


def _hours(s: float) -> str:
    return f"{s / 3600:.1f} h"


def _clock(hour: int) -> str:
    h = hour % 12 or 12
    return f"{h}{'am' if hour < 12 or hour == 24 else 'pm'}"


def find_tips(totals: dict[str, float], by_hour: dict[int, dict]) -> list[dict]:
    """Every tip whose evidence is strong enough, strongest first."""
    tracked = sum(totals.values())
    if tracked < MIN_TRACKED_S:
        return []
    tips = []

    def share(*keys):
        return sum(totals.get(k, 0.0) for k in keys) / tracked

    low = share("looking_down", "slouching")
    if low >= SCREEN_LOW_SHARE:
        tips.append({"rule": "screen_low", "strength": low / SCREEN_LOW_SHARE,
                     "title": "Your screen is probably too low",
                     "advice": "Raise the laptop (a stand or a stack of books) and use an external keyboard, so "
                               "the top of the screen is near eye level.",
                     "evidence": f"Looking down or slouching {low:.0%} of {_hours(tracked)} at the desk this week."})
    close = share("too_close")
    if close >= TOO_CLOSE_SHARE:
        tips.append({"rule": "too_close", "strength": close / TOO_CLOSE_SHARE,
                     "title": "You lean in to read",
                     "advice": "Try a larger font size or zoom (Ctrl +), or move the screen a little closer.",
                     "evidence": f"Too close to the screen {close:.0%} of {_hours(tracked)} this week."})
    left, right = totals.get("leaning_left", 0.0), totals.get("leaning_right", 0.0)
    lean = share("leaning", "leaning_left", "leaning_right")
    if lean >= LEAN_SHARE and left + right > 0:
        side, side_s = ("left", left) if left >= right else ("right", right)
        one_side = side_s / (left + right)
        if one_side >= LEAN_ONE_SIDE:
            tips.append({"rule": "lean_side", "strength": lean / LEAN_SHARE,
                         "title": f"You lean to your {side} a lot",
                         "advice": f"Your screen or mouse may be off to the {side}. Centre the screen in front of "
                                   "you and keep the mouse close to the keyboard.",
                         "evidence": f"Leaning {lean:.0%} of {_hours(tracked)} this week, {one_side:.0%} of it to "
                                     f"your {side}."})
    poor_keys = POOR_KEYS
    overall = share(*poor_keys)
    if overall > 0:
        bad_hours = []
        for hour, h in by_hour.items():
            t = sum(h["seconds"].values())
            if t < HOUR_MIN_S or h["days"] < HOUR_MIN_DAYS:
                continue
            s = sum(h["seconds"].get(k, 0.0) for k in poor_keys) / t
            if s >= HOUR_RATIO * overall and s >= HOUR_MIN_SHARE:
                bad_hours.append((hour, s))
        if bad_hours:
            bad_hours.sort()
            # The longest run of consecutive qualifying hours, e.g. 15:00-17:00.
            runs, run = [], [bad_hours[0]]
            for h in bad_hours[1:]:
                if h[0] == run[-1][0] + 1:
                    run.append(h)
                else:
                    runs.append(run)
                    run = [h]
            runs.append(run)
            run = max(runs, key=lambda r: (len(r), sum(s for _h, s in r)))
            start, end = run[0][0], run[-1][0] + 1
            worst = sum(s for _h, s in run) / len(run)
            tips.append({"rule": "time_of_day", "strength": worst / overall / HOUR_RATIO,
                         "title": f"Your posture slips most between {_clock(start)} and {_clock(end)}",
                         "advice": f"Plan a short break or a walk around {_clock(start)}: tiredness shows up as "
                                   "slouching.",
                         "evidence": f"Poor posture {worst:.0%} of the time then, vs {overall:.0%} on average this "
                                     "week."})
    return sorted(tips, key=lambda t: -t["strength"])


def current_tip(history: PostureHistory, now: float | None = None) -> dict | None:
    """The strongest tip that hasn't been dismissed (one at a time)."""
    now = history.clock() if now is None else now
    since = now - TIP_WINDOW_DAYS * 86400
    hidden = history.dismissed(now)
    for tip in find_tips(history.totals(since), history.by_hour(since)):
        if tip["rule"] not in hidden:
            return {k: v for k, v in tip.items() if k != "strength"}
    return None
