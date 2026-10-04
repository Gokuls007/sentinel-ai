"""Versioned SQLite schema migrations for the event store (``PRAGMA user_version``).

Version 0 is the original ``events`` table written by ``output.event_logger`` (columns
``alert_type``, ``timestamp``, ``details``...). Version 1 is the unified schema in
``events.schema``. Before migrating a database that already holds data, the file is
copied to ``<db>.bak-YYYYmmdd-HHMMSS``; the old table is also kept as ``events_v0`` until
it is dropped by hand, so nothing is lost either way.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("sentinel.events.migrate")

SCHEMA_VERSION = 5

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
# v2: seconds at each REBA risk level, accumulated per day / camera / grouping.
# kind is "zone" (key = zone id, "" = no zone) or "hour" (key = "0".."23"). Before v4 there was also
# "track" (per tracker identity); v4 deletes those rows (privacy: no per-person rankings).
_V2_ERGO_TABLE = """
CREATE TABLE IF NOT EXISTS ergo_time (
    day TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    level INTEGER NOT NULL,
    seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, camera_id, kind, key, level)
)
"""

# v3: confirmed plain-English rules (rules/dsl.py), one JSON document per rule.
_V3_RULES_TABLE = """
CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""

# v5: Exam Hall (docs/plans/exam-hall.md section 8). Seats are keyed by label within a session;
# nothing here identifies a person (no names, no student ids, no face data).
_V5_EXAM_TABLES = (
    """CREATE TABLE IF NOT EXISTS exam_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        room TEXT NOT NULL DEFAULT '',
        camera_id TEXT NOT NULL,
        preset TEXT NOT NULL DEFAULT 'medium',
        status TEXT NOT NULL DEFAULT 'setup',
        calibration_s REAL NOT NULL DEFAULT 120,
        started_at REAL,
        ended_at REAL,
        created_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS exam_seats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
        label TEXT NOT NULL,
        polygon_json TEXT NOT NULL,
        row INTEGER NOT NULL DEFAULT 0,
        col INTEGER NOT NULL DEFAULT 0,
        neighbours_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE (session_id, label)
    )""",
    """CREATE TABLE IF NOT EXISTS exam_seat_baselines (
        seat_id INTEGER PRIMARY KEY REFERENCES exam_seats(id) ON DELETE CASCADE,
        yaw_med REAL, yaw_spread REAL, pitch_med REAL, pitch_spread REAL,
        hand_height_med REAL, samples INTEGER NOT NULL DEFAULT 0, calibrated_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS exam_flags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
        seat_id INTEGER REFERENCES exam_seats(id) ON DELETE SET NULL,
        signal TEXT NOT NULL,
        start_ts REAL NOT NULL,
        end_ts REAL NOT NULL,
        confidence REAL,
        priority TEXT NOT NULL DEFAULT 'medium',
        measurements_json TEXT NOT NULL DEFAULT '{}',
        clip_path TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        reviewer_note TEXT,
        dismissed_reason TEXT,
        reviewed_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS exam_unblur_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flag_id INTEGER NOT NULL REFERENCES exam_flags(id) ON DELETE CASCADE,
        reason TEXT NOT NULL,
        actor TEXT NOT NULL DEFAULT '',
        at REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_exam_flags_session ON exam_flags (session_id, start_ts)",
)

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
    with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        legacy = version == 0 and _is_legacy(conn)
        if version >= SCHEMA_VERSION:
            return version
    if version >= 1:
        # v2, v3 and v5 add tables, v4 deletes per-track rows: always back up first (promised).
        if exists_with_data:
            saved = backup(db_path)
            logger.info("Backed up %s to %s before migrating to schema v%d", db_path, saved, SCHEMA_VERSION)
        with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(_V2_ERGO_TABLE)
            conn.execute(_V3_RULES_TABLE)
            for statement in _V5_EXAM_TABLES:
                conn.execute(statement)
            dropped = conn.execute("DELETE FROM ergo_time WHERE kind = 'track'").rowcount
            if dropped:
                logger.info("Deleted %d per-track time-at-risk row(s) (schema v4)", dropped)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return SCHEMA_VERSION
    if legacy and exists_with_data:
        saved = backup(db_path)
        logger.info("Backed up %s to %s before migrating the event schema", db_path, saved)
    with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
        if legacy:
            copied = _migrate_v0_to_v1(conn)
            logger.info("Migrated %d event(s) to schema v1 (old table kept as events_v0)", copied)
        else:
            conn.execute(_V1_TABLE)
        for statement in _V1_INDEXES:
            conn.execute(statement)
        conn.execute(_V2_ERGO_TABLE)
        conn.execute(_V3_RULES_TABLE)
        for statement in _V5_EXAM_TABLES:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return SCHEMA_VERSION
