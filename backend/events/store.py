"""SQLite-backed event store: the single place events are written and read.

All queries are parameterized. Filter and group-by column names come from fixed allow-lists,
never from caller input, so the store is safe to expose to search tools (Phase 2).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterable
from typing import Any

from events.migrate import migrate
from events.schema import Event

logger = logging.getLogger("sentinel.events.store")

MAX_LIMIT = 1000

_COLUMNS = (
    "id, type, severity, camera_id, track_id, zone_id, start_ts, end_ts, attributes, "
    "clip_path, thumbnail_path, verified, alert_id, message, confidence"
)
# group_by name -> SQL expression (fixed; callers pick a key, never pass SQL).
_GROUPS = {
    "type": "type",
    "severity": "severity",
    "zone": "COALESCE(zone_id, '')",
    "camera": "camera_id",
    # Hour of day in local time, "00".."23".
    "hour": "strftime('%H', start_ts, 'unixepoch', 'localtime')",
    # Calendar hour bucket, e.g. "2026-09-30 14:00" (for events-per-hour charts).
    "hour_bucket": "strftime('%Y-%m-%d %H:00', start_ts, 'unixepoch', 'localtime')",
    "day": "strftime('%Y-%m-%d', start_ts, 'unixepoch', 'localtime')",
}
GROUP_BY_KEYS = tuple(_GROUPS)


class EventStore:
    def __init__(self, db_path: str = "data/events.db"):
        if os.path.isdir(db_path):
            raise RuntimeError(f"DB_PATH {db_path!r} is a directory; it must be a file path")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()  # one writer at a time; reads use their own connection
        migrate(db_path)
        logger.info("EventStore ready: %s", db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # --- writing -------------------------------------------------------------------

    def emit(self, event: Event) -> Event:
        """Store ``event`` and return it with ``id`` set. A duplicate alert_id is ignored
        (the stored copy is returned)."""
        row = (
            event.type, event.severity, event.camera_id, event.track_id, event.zone_id,
            event.start_ts, event.end_ts, json.dumps(event.attributes, default=str),
            event.clip_path, event.thumbnail_path, event.verified, event.alert_id,
            event.message, event.confidence,
        )
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO events (type, severity, camera_id, track_id, zone_id, "
                "start_ts, end_ts, attributes, clip_path, thumbnail_path, verified, alert_id, "
                "message, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            if cur.rowcount:
                event.id = cur.lastrowid
                return event
        existing = self.get_by_alert_id(event.alert_id) if event.alert_id else None
        return existing or event

    def add_ergo_time(self, camera_id: str, rows) -> None:
        """Add (day, kind, key, level, seconds) rows to the persisted time at risk."""
        rows = [(day, camera_id, kind, str(key), int(level), float(secs)) for day, kind, key, level, secs in rows]
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO ergo_time (day, camera_id, kind, key, level, seconds) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(day, camera_id, kind, key, level) DO UPDATE SET seconds = seconds + excluded.seconds",
                rows,
            )

    def ergo_time(self, kind: str, day_from: str, day_to: str, camera_id: str | None = None) -> dict:
        """{key: {level: seconds}} for kind "zone", "hour" or "track" over [day_from, day_to]."""
        if kind not in ("zone", "hour", "track"):
            raise ValueError(f"kind must be zone, hour or track, got {kind!r}")
        sql = "SELECT key, level, SUM(seconds) FROM ergo_time WHERE kind = ? AND day >= ? AND day <= ?"
        params: list = [kind, day_from, day_to]
        if camera_id:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        out: dict = {}
        with self._connect() as conn:
            for key, level, secs in conn.execute(sql + " GROUP BY key, level", params):
                out.setdefault(key, {})[int(level)] = float(secs)
        return out

    def set_verified(self, event_id: int, verified: bool | None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE events SET verified = ? WHERE id = ?",
                (None if verified is None else int(verified), event_id),
            )

    # --- reading -------------------------------------------------------------------

    @staticmethod
    def _where(
        types: Iterable[str] | None = None,
        severity: Iterable[str] | str | None = None,
        camera_id: str | None = None,
        zone_id: str | None = None,
        track_id: int | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def any_of(column: str, values: Iterable[str] | str | None) -> None:
            if values is None:
                return
            values = [values] if isinstance(values, str) else [v for v in values if v]
            if values:
                clauses.append(f"{column} IN ({', '.join('?' * len(values))})")
                params.extend(values)

        any_of("type", types)
        any_of("severity", severity)
        for column, value in (("camera_id", camera_id), ("zone_id", zone_id), ("track_id", track_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        # An event matches a window if it overlaps it.
        if start is not None:
            clauses.append("end_ts >= ?")
            params.append(float(start))
        if end is not None:
            clauses.append("start_ts <= ?")
            params.append(float(end))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query(self, *, limit: int = 50, offset: int = 0, **filters: Any) -> list[Event]:
        """Events matching ``filters`` (see ``_where``), newest first."""
        where, params = self._where(**filters)
        limit = max(1, min(int(limit), MAX_LIMIT))
        offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM events{where} ORDER BY end_ts DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_to_event(r) for r in rows]

    def total(self, **filters: Any) -> int:
        where, params = self._where(**filters)
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]

    def count(self, group_by: str, **filters: Any) -> dict[str, int]:
        """Event counts grouped by one of ``GROUP_BY_KEYS``."""
        if group_by not in _GROUPS:
            raise ValueError(f"group_by must be one of {GROUP_BY_KEYS}, got {group_by!r}")
        expr = _GROUPS[group_by]
        where, params = self._where(**filters)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {expr} AS k, COUNT(*) FROM events{where} GROUP BY k ORDER BY k", params
            ).fetchall()
        return {str(k): n for k, n in rows}

    def get(self, event_id: int) -> Event | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM events WHERE id = ?", (event_id,)).fetchone()
        return _to_event(row) if row else None

    def get_by_alert_id(self, alert_id: str) -> Event | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM events WHERE alert_id = ?", (alert_id,)
            ).fetchone()
        return _to_event(row) if row else None

    def distinct(self, column: str) -> list[str]:
        """Distinct non-null values of camera_id / zone_id / type (for filter dropdowns)."""
        if column not in ("camera_id", "zone_id", "type"):
            raise ValueError(f"unsupported column {column!r}")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM events WHERE {column} IS NOT NULL ORDER BY {column}"
            ).fetchall()
        return [r[0] for r in rows]


def _to_event(row: sqlite3.Row) -> Event:
    try:
        attributes = json.loads(row["attributes"] or "{}")
    except ValueError:
        attributes = {"raw_attributes": row["attributes"]}
    return Event(
        id=row["id"],
        type=row["type"],
        severity=row["severity"],
        camera_id=row["camera_id"],
        track_id=row["track_id"],
        zone_id=row["zone_id"],
        start_ts=row["start_ts"],
        end_ts=row["end_ts"],
        attributes=attributes if isinstance(attributes, dict) else {"value": attributes},
        clip_path=row["clip_path"],
        thumbnail_path=row["thumbnail_path"],
        verified=row["verified"],
        alert_id=row["alert_id"],
        message=row["message"],
        confidence=row["confidence"],
    )
