"""Confirmed rules, kept in the event database (schema v3 ``rules`` table)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time

from events.migrate import migrate
from rules.dsl import Rule


class RuleStore:
    def __init__(self, db_path: str, clock=time.time):
        self.db_path = db_path
        self.clock = clock
        self._lock = threading.Lock()
        migrate(db_path)

    def _conn(self):
        return contextlib.closing(sqlite3.connect(self.db_path))

    def list(self) -> list[Rule]:
        with self._lock, self._conn() as conn:
            rows = conn.execute("SELECT body FROM rules ORDER BY created_at").fetchall()
        out = []
        for (body,) in rows:
            try:
                out.append(Rule.model_validate_json(body))
            except ValueError:
                continue  # a rule from a newer version of the format: skip, don't crash
        return out

    def get(self, rule_id: str) -> Rule | None:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT body FROM rules WHERE id = ?", (rule_id,)).fetchone()
        return Rule.model_validate_json(row[0]) if row else None

    def ids(self) -> set[str]:
        with self._lock, self._conn() as conn:
            return {r for (r,) in conn.execute("SELECT id FROM rules")}

    def save(self, rule: Rule) -> Rule:
        now = self.clock()
        with self._lock, self._conn() as conn, conn:
            conn.execute(
                "INSERT INTO rules (id, body, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET body = excluded.body, updated_at = excluded.updated_at",
                (rule.id, rule.model_dump_json(), now, now))
        return rule

    def delete(self, rule_id: str) -> bool:
        with self._lock, self._conn() as conn, conn:
            return conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,)).rowcount > 0

    def replace_preset(self, preset: str, rules: list[Rule]) -> None:
        """Remove every rule that came from a preset, then add ``rules`` (one transaction)."""
        now = self.clock()
        with self._lock, self._conn() as conn, conn:
            for (rid, body) in conn.execute("SELECT id, body FROM rules").fetchall():
                if json.loads(body).get("preset"):
                    conn.execute("DELETE FROM rules WHERE id = ?", (rid,))
            for r in rules:
                conn.execute("INSERT OR REPLACE INTO rules (id, body, created_at, updated_at) VALUES (?, ?, ?, ?)",
                             (r.id, r.model_dump_json(), now, now))
