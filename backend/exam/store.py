"""Exam sessions, seats and per-seat baselines in the events database (schema v5)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator

from events.migrate import migrate
from exam.calibration import Baseline
from exam.seats import Seat

STATUSES = ("setup", "calibrating", "live", "ended", "reviewed")
OPEN = ("setup", "calibrating", "live")
_SESSION_COLS = "id, name, room, camera_id, preset, status, calibration_s, started_at, ended_at, created_at"


class ExamStore:
    def __init__(self, db_path: str, clock=time.time):
        self.db_path = db_path
        self.clock = clock
        self._lock = threading.Lock()
        migrate(db_path)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- sessions -------------------------------------------------------------------------------

    def create_session(self, name: str, camera_id: str, room: str = "", preset: str = "medium",
                       calibration_s: float = 120.0) -> dict:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO exam_sessions (name, room, camera_id, preset, status, calibration_s, created_at) "
                "VALUES (?, ?, ?, ?, 'setup', ?, ?)", (name, room, camera_id, preset, calibration_s, self.clock()))
            return self._session(conn, cur.lastrowid)

    def _session(self, conn, session_id: int) -> dict | None:
        row = conn.execute(f"SELECT {_SESSION_COLS} FROM exam_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def get_session(self, session_id: int) -> dict | None:
        with self._connect() as conn:
            return self._session(conn, session_id)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {_SESSION_COLS} FROM exam_sessions ORDER BY created_at DESC, id DESC "
                                "LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def update_session(self, session_id: int, **fields) -> dict | None:
        allowed = {"name", "room", "preset", "status", "calibration_s", "started_at", "ended_at"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if fields.get("status") is not None and fields["status"] not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock, self._connect() as conn:
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields)
                conn.execute(f"UPDATE exam_sessions SET {sets} WHERE id = ?", (*fields.values(), session_id))
            return self._session(conn, session_id)

    def open_session_for(self, camera_id: str) -> dict | None:
        """The newest session on this camera that hasn't ended (the one the camera shows)."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_SESSION_COLS} FROM exam_sessions WHERE camera_id = ? AND status IN "
                f"({', '.join('?' * len(OPEN))}) ORDER BY created_at DESC, id DESC LIMIT 1",
                (camera_id, *OPEN)).fetchone()
        return dict(row) if row else None

    # --- seats ----------------------------------------------------------------------------------

    def seats(self, session_id: int) -> list[Seat]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, label, polygon_json, row, col, neighbours_json FROM exam_seats "
                                "WHERE session_id = ? ORDER BY row, col, label", (session_id,)).fetchall()
        out = []
        for r in rows:
            poly = json.loads(r["polygon_json"])
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(Seat(label=r["label"], rect=(min(xs), min(ys), max(xs), max(ys)), row=r["row"], col=r["col"],
                            neighbours=json.loads(r["neighbours_json"]), id=r["id"]))
        return out

    def save_seats(self, session_id: int, seats: list[Seat]) -> list[Seat]:
        """Replace the session's seats (their baselines go with them). Rectangles are stored as
        4-point polygons, so other shapes can come later without a migration."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM exam_seats WHERE session_id = ?", (session_id,))
            for s in seats:
                x1, y1, x2, y2 = s.rect
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                cur = conn.execute(
                    "INSERT INTO exam_seats (session_id, label, polygon_json, row, col, neighbours_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, s.label, json.dumps([[round(v, 4) for v in p] for p in poly]), s.row, s.col,
                     json.dumps(s.neighbours)))
                s.id = cur.lastrowid
        return seats

    # --- baselines ------------------------------------------------------------------------------

    def save_baseline(self, session_id: int, label: str, b: Baseline) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM exam_seats WHERE session_id = ? AND label = ?",
                               (session_id, label)).fetchone()
            if row is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO exam_seat_baselines (seat_id, yaw_med, yaw_spread, pitch_med, pitch_spread, "
                "hand_height_med, samples, calibrated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], b.yaw_med, b.yaw_spread, b.pitch_med, b.pitch_spread, b.hand_height_med, b.samples,
                 b.calibrated_at))

    def baselines(self, session_id: int) -> dict[str, Baseline]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.label, b.yaw_med, b.yaw_spread, b.pitch_med, b.pitch_spread, b.hand_height_med, b.samples, "
                "b.calibrated_at FROM exam_seat_baselines b JOIN exam_seats s ON s.id = b.seat_id "
                "WHERE s.session_id = ?", (session_id,)).fetchall()
        return {r["label"]: Baseline(r["yaw_med"], r["yaw_spread"], r["pitch_med"], r["pitch_spread"],
                                     r["hand_height_med"], r["samples"], r["calibrated_at"]) for r in rows}
