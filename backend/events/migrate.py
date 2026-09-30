"""Versioned SQLite schema migrations for the event store (``PRAGMA user_version``).

Version 0 is the original ``events`` table written by ``output.event_logger`` (columns
``alert_type``, ``timestamp``, ``details``...). Version 1 is the unified schema in
``events.schema``. Before migrating a database that already holds data, the file is
copied to ``<db>.bak-YYYYmmdd-HHMMSS``; the old table is also kept as ``events_v0`` until
it is dropped by hand, so nothing is lost either way.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("sentinel.events.migrate")

SCHEMA_VERSION = 1

_V1_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    camera_id TEXT NOT NULL DEFAULT 'cam-0',
    track_id INTEGER,
    zone_id TEXT,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    clip_path TEXT,
    thumbnail_path TEXT,
    verified INTEGER,
    alert_id TEXT UNIQUE,
    message TEXT NOT NULL DEFAULT '',
    confidence REAL,
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""
_V1_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_events_start ON events (start_ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events (type, start_ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_zone ON events (zone_id, start_ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_camera ON events (camera_id, start_ts)",
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _is_legacy(conn: sqlite3.Connection) -> bool:
    cols = _columns(conn, "events")
    return bool(cols) and "alert_type" in cols and "type" not in cols


def backup(db_path: str | os.PathLike) -> Path:
    """Copy the database to a timestamped file next to it and return that path."""
    src = Path(db_path)
    dest = src.with_name(f"{src.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    n = 1
    while dest.exists():  # two migrations in the same second
        dest = src.with_name(f"{src.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}-{n}")
        n += 1
    shutil.copy2(src, dest)
    return dest


def _snapshot_for(clip_path: str | None) -> str | None:
    """Clips live at <dir>/clip_<id>.mp4 with the alert frame at <dir>/snapshot_<id>.jpg."""
    if not clip_path:
        return None
    p = Path(clip_path)
    name = p.name.replace("clip_", "snapshot_", 1).rsplit(".", 1)[0] + ".jpg"
    return str(p.with_name(name)).replace("\\", "/")


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> int:
    """Rename the legacy table, create v1, copy every row. Returns rows copied."""
    conn.execute("ALTER TABLE events RENAME TO events_v0")
    conn.execute(_V1_TABLE)
    rows = conn.execute(
        "SELECT alert_id, alert_type, track_id, timestamp, severity, confidence, message, "
        "details, clip_path FROM events_v0 ORDER BY id"
    ).fetchall()
    copied = 0
    for alert_id, alert_type, track_id, ts, severity, confidence, message, details, clip in rows:
        try:
            attrs = json.loads(details or "{}")
        except ValueError:
            attrs = {"raw_details": details}
        if not isinstance(attrs, dict):
            attrs = {"raw_details": attrs}
        zone_id = attrs.pop("zone_id", None)
        attrs.pop("clip_path", None)
        lasted = attrs.get("dwell_seconds") or attrs.get("duration") or 0.0
        try:
            lasted = max(0.0, float(lasted))
        except (TypeError, ValueError):
            lasted = 0.0
        end = float(ts or 0.0)
        conn.execute(
            "INSERT OR IGNORE INTO events (type, severity, camera_id, track_id, zone_id, "
            "start_ts, end_ts, attributes, clip_path, thumbnail_path, alert_id, message, "
            "confidence) VALUES (?, ?, 'cam-0', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                alert_type or "action",
                severity if severity in ("low", "medium", "high", "critical") else "low",
                track_id,
                zone_id,
                end - lasted,
                end,
                json.dumps(attrs),
                clip or None,
                _snapshot_for(clip),
                alert_id,
                message or "",
                confidence,
            ),
        )
        copied += 1
    return copied


def migrate(db_path: str | os.PathLike) -> int:
    """Bring the database at ``db_path`` to ``SCHEMA_VERSION``. Returns the version.

    Safe to call on every start: a current database is left untouched.
    """
    db_path = str(db_path)
    exists_with_data = os.path.isfile(db_path) and os.path.getsize(db_path) > 0
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        legacy = version == 0 and _is_legacy(conn)
        if version >= SCHEMA_VERSION:
            return version
    if legacy and exists_with_data:
        saved = backup(db_path)
        logger.info("Backed up %s to %s before migrating the event schema", db_path, saved)
    with sqlite3.connect(db_path) as conn:
        if legacy:
            copied = _migrate_v0_to_v1(conn)
            logger.info("Migrated %d event(s) to schema v1 (old table kept as events_v0)", copied)
        else:
            conn.execute(_V1_TABLE)
        for statement in _V1_INDEXES:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return SCHEMA_VERSION
