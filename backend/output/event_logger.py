import sqlite3
import os
import json
import logging
from datetime import datetime
from anomaly.engine import AnomalyAlert
from core.utils import to_serializable

logger = logging.getLogger("sentinel.output.logger")

class EventLogger:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
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

    def log_event(self, alert: AnomalyAlert, clip_path: str = None):
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

    def get_events(self, limit=50, severity=None) -> list:
        query = "SELECT * FROM events"
        params = []
        if severity:
            query += " WHERE severity = ?"
            params.append(severity)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
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
