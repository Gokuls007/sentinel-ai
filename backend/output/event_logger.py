import contextlib
import json
import logging
import os
import sqlite3

from anomaly.engine import AnomalyAlert
from core.utils import to_serializable

logger = logging.getLogger("sentinel.output.logger")

class EventLogger:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        if os.path.isdir(db_path):
            raise RuntimeError(f"DB_PATH {db_path!r} is a directory; it must be a file path")

        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    alert_type TEXT,
                    track_id INTEGER,
                    timestamp REAL,
                    severity TEXT,
                    confidence REAL,
                    message TEXT,
                    details TEXT,
                    clip_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info(f"EventLogger initialized with DB: {self.db_path}")

    def log_event(self, alert: AnomalyAlert, clip_path: str | None = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (
                        alert_id, alert_type, track_id, timestamp, 
                        severity, confidence, message, details, clip_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.alert_type,
                    alert.track_id,
                    alert.timestamp,
                    alert.severity,
                    alert.confidence,
                    alert.message,
                    json.dumps(to_serializable(alert.details)),
                    clip_path
                ))
                conn.commit()
                logger.info(f"Alert {alert.alert_id} logged to database.")
        except sqlite3.IntegrityError:
            # Skip if already logged
            pass
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def get_events(self, limit=50, severity=None, alert_type=None) -> list:
        query = "SELECT * FROM events"
        where, params = [], []
        if severity:
            where.append("severity = ?")
            params.append(severity)
        if alert_type:
            where.append("alert_type = ?")
            params.append(alert_type)
        if where:
            query += " WHERE " + " AND ".join(where)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = []
                for row in cursor.fetchall():
                    item = dict(row)
                    with contextlib.suppress(ValueError):
                        item["details"] = json.loads(item.get("details") or "{}")
                    rows.append(item)
                return rows
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_event_count(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM events")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting events: {e}")
            return 0
