"""Data retention: delete what's older than N days, everywhere Sentinel stores something.

Covered (each older than the cutoff):
- events (rows) with their clip / snapshot / skeleton folders;
- orphan clip folders (no event points at them, e.g. test footage before it stopped saving);
- camera recordings (``data/recordings/*.mp4`` and their ``.labels.json``);
- time at risk (``ergo_time`` rows, by day);
- posture history (per-minute status, corrections, reminders and breaks);
- the movement-coach log (``posture_movement_log_*.jsonl`` lines);
- database backups (``*.bak-*``).

Not covered (settings, not history): zones, rules, app settings, posture calibration.

``plan()`` only looks; ``apply()`` deletes. The server's worker runs a dry run until the
Privacy panel has been opened once, then deletes for real (startup, then hourly).
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sentinel.privacy.retention")

CATEGORIES = {
    "events": "Events (alerts) with their clips, snapshots and skeletons",
    "orphan_clips": "Clip folders with no event",
    "recordings": "Camera recordings",
    "ergo_time": "Time-at-risk totals (per day)",
    "posture": "Posture history (minutes, corrections, reminders, breaks)",
    "movement_log": "Movement-coach log lines",
    "backups": "Database backups",
}


@dataclass
class Item:
    category: str
    count: int = 0
    bytes: int = 0
    _do: list = field(default_factory=list, repr=False)  # deletions to run in apply()

    def to_dict(self) -> dict:
        return {"category": self.category, "label": CATEGORIES[self.category], "count": self.count,
                "bytes": self.bytes}


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _newest_mtime(path: Path) -> float:
    times = [path.stat().st_mtime] + [f.stat().st_mtime for f in path.rglob("*")]
    return max(times)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        with contextlib.suppress(OSError):
            path.unlink()


class Retention:
    def __init__(self, db_path: str, clips_dir: str, recordings_dir: str, clock=time.time):
        self.db_path = Path(db_path)
        self.data_dir = self.db_path.resolve().parent
        self.clips_dir = Path(clips_dir).resolve()
        self.recordings_dir = Path(recordings_dir).resolve()
        self.clock = clock

    # --- what would go --------------------------------------------------------------------------

    def cutoff(self, days: float) -> float:
        return self.clock() - days * 86400

    def plan(self, days: float | None = None, cutoff: float | None = None) -> list[Item]:
        """What ``apply`` would delete for a retention of ``days`` (or everything before ``cutoff``)."""
        cut = cutoff if cutoff is not None else self.cutoff(days)
        items = [self._events(cut), self._orphans(cut), self._recordings(cut), self._ergo_time(cut),
                 self._posture(cut), self._movement_log(cut), self._backups(cut)]
        return items

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite3.connect(str(path), timeout=10)

    def _clip_dir_for(self, path: str | None) -> Path | None:
        """The event's own folder under clips_dir (clip, snapshot and skeleton live there)."""
        if not path:
            return None
        p = Path(path)
        p = (p if p.is_absolute() else Path.cwd() / p).resolve()
        folder = p.parent
        return folder if folder.parent == self.clips_dir else None

    def _events(self, cut: float) -> Item:
        item = Item("events")
        if not self.db_path.is_file():
            return item
        with contextlib.closing(self._connect(self.db_path)) as conn:
            try:
                rows = conn.execute("SELECT id, alert_id, clip_path, thumbnail_path FROM events WHERE end_ts < ?",
                                    (cut,)).fetchall()
            except sqlite3.Error:
                return item
        folders = set()
        for _id, alert_id, clip, thumb in rows:
            for p in (clip, thumb):
                folder = self._clip_dir_for(p)
                if folder is not None:
                    folders.add(folder)
            if alert_id and (self.clips_dir / alert_id).is_dir():
                folders.add(self.clips_dir / alert_id)
        item.count = len(rows)
        item.bytes = sum(_size(f) for f in folders if f.exists())
        ids = [r[0] for r in rows]

        def delete(ids=ids, folders=folders):
            with contextlib.closing(self._connect(self.db_path)) as conn, conn:
                conn.executemany("DELETE FROM events WHERE id = ?", [(i,) for i in ids])
            for f in folders:
                _remove(f)

        item._do.append(delete)
        return item

    def _orphans(self, cut: float) -> Item:
        item = Item("orphan_clips")
        # Only when the clips folder belongs to this database's data folder: a second server with
        # its own database but a shared clips folder would otherwise see every clip as an orphan.
        if not self.clips_dir.is_dir() or self.data_dir not in self.clips_dir.parents:
            return item
        # Known to ANY events database in the data folder (servers may use different ones, e.g.
        # DB_PATH=data/integration.db vs the default), not just this one.
        known: set[str] = set()
        for db in {self.db_path.resolve(), *self.data_dir.glob("*.db")}:
            if not db.is_file():
                continue
            with contextlib.closing(self._connect(db)) as conn, contextlib.suppress(sqlite3.Error):
                known |= {r[0] for r in conn.execute("SELECT alert_id FROM events WHERE alert_id IS NOT NULL")}
        for folder in self.clips_dir.iterdir():
            if folder.is_dir() and folder.name not in known and _newest_mtime(folder) < cut:
                item.count += 1
                item.bytes += _size(folder)
                item._do.append(lambda f=folder: _remove(f))
        return item

    def _recordings(self, cut: float) -> Item:
        item = Item("recordings")
        if not self.recordings_dir.is_dir():
            return item
        for f in self.recordings_dir.glob("*.mp4"):
            if f.stat().st_mtime < cut:
                sidecar = f.with_suffix(".labels.json")
                item.count += 1
                item.bytes += f.stat().st_size + (sidecar.stat().st_size if sidecar.exists() else 0)
                item._do.append(lambda f=f, s=sidecar: (_remove(f), _remove(s)))
        return item

    def _ergo_time(self, cut: float) -> Item:
        item = Item("ergo_time")
        if not self.db_path.is_file():
            return item
        day = datetime.fromtimestamp(cut).strftime("%Y-%m-%d") if cut != float("inf") else "9999-12-31"
        with contextlib.closing(self._connect(self.db_path)) as conn:
            try:
                item.count = conn.execute("SELECT COUNT(*) FROM ergo_time WHERE day < ?", (day,)).fetchone()[0]
            except sqlite3.Error:
                return item

        def delete(day=day):
            with contextlib.closing(self._connect(self.db_path)) as conn, conn:
                conn.execute("DELETE FROM ergo_time WHERE day < ?", (day,))

        if item.count:
            item._do.append(delete)
        return item

    def _posture(self, cut: float) -> Item:
        item = Item("posture")
        minute_cut = cut / 60 if cut != float("inf") else float("inf")
        queries = (("minutes", "minute < ?", minute_cut), ("corrections", "ts < ?", cut), ("events", "ts < ?", cut))
        for path in sorted(self.data_dir.glob("posture_history_*.db")):
            with contextlib.closing(self._connect(path)) as conn:
                for table, where, value in queries:
                    with contextlib.suppress(sqlite3.Error):
                        n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", (value,)).fetchone()[0]
                        if n:
                            item.count += n

                            def delete(path=path, table=table, where=where, value=value):
                                with contextlib.closing(self._connect(path)) as c, c:
                                    c.execute(f"DELETE FROM {table} WHERE {where}", (value,))

                            item._do.append(delete)
        return item

    def _movement_log(self, cut: float) -> Item:
        item = Item("movement_log")
        for path in sorted(self.data_dir.glob("posture_movement_log_*.jsonl")):
            keep, old = [], 0
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            ts = float(json.loads(line).get("t", 0))
                        except (ValueError, AttributeError):
                            ts = 0.0
                        if ts < cut:
                            old += 1
                            item.bytes += len(line.encode("utf-8"))
                        else:
                            keep.append(line)
            except OSError:
                continue
            if old:
                item.count += old

                def rewrite(path=path, cut=cut):
                    # Re-read at apply time: the coach may have appended since the plan.
                    with open(path, encoding="utf-8") as f:
                        lines = f.readlines()
                    kept = []
                    for line in lines:
                        try:
                            if float(json.loads(line).get("t", 0)) >= cut:
                                kept.append(line)
                        except (ValueError, AttributeError):
                            pass
                    with open(path, "w", encoding="utf-8") as f:
                        f.writelines(kept)

                item._do.append(rewrite)
        return item

    def _backups(self, cut: float) -> Item:
        item = Item("backups")
        for name in glob.glob(str(self.data_dir / "*.bak-*")):
            f = Path(name)
            if f.is_file() and f.stat().st_mtime < cut:
                item.count += 1
                item.bytes += f.stat().st_size
                item._do.append(lambda f=f: _remove(f))
        return item

    # --- deleting -------------------------------------------------------------------------------

    def apply(self, days: float) -> list[Item]:
        items = self.plan(days)
        for item in items:
            for do in item._do:
                try:
                    do()
                except (OSError, sqlite3.Error) as e:
                    logger.warning("retention: could not delete part of %s: %s", item.category, e)
        return items


def summary(items: list[Item]) -> dict:
    rows = [i.to_dict() for i in items]
    return {"items": rows, "count": sum(r["count"] for r in rows), "bytes": sum(r["bytes"] for r in rows)}


class RetentionWorker:
    """Runs retention at startup and then every ``interval_s``.

    ``settings()`` returns ``{"retention_days": int, "privacy_seen": bool}``. Until the Privacy
    panel has been opened once (``privacy_seen``), runs are dry runs: logged, nothing deleted.
    """

    def __init__(self, retention: Retention, settings, interval_s: float = 3600):
        self.retention = retention
        self.settings = settings
        self.interval_s = interval_s
        self.last_run: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run_once(self) -> dict | None:
        s = self.settings()
        days = int(s.get("retention_days") or 0)
        if days <= 0:
            self.last_run = {"at": time.time(), "days": 0, "dry_run": False, "kept_forever": True,
                             **summary([])}
            return self.last_run
        dry = not s.get("privacy_seen")
        with self._lock:
            items = self.retention.plan(days) if dry else self.retention.apply(days)
        result = {"at": time.time(), "days": days, "dry_run": dry, "kept_forever": False, **summary(items)}
        self.last_run = result
        if result["count"]:
            what = ", ".join(f"{r['count']} {r['category']}" for r in result["items"] if r["count"])
            if dry:
                logger.info("Retention (dry run until the Privacy panel is opened once): would delete %s "
                            "older than %d days", what, days)
            else:
                logger.info("Retention: deleted %s older than %d days", what, days)
        return result

    def delete_now(self, days: int) -> dict:
        with self._lock:
            items = self.retention.apply(days)
        self.last_run = {"at": time.time(), "days": days, "dry_run": False, "kept_forever": False,
                         "manual": True, **summary(items)}
        return self.last_run

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:  # never take the server down over housekeeping
                    logger.exception("retention run failed")
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True, name="retention")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
