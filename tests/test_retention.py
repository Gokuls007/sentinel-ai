"""Retention: everything older than N days goes, everywhere history is kept; the first runs are
dry runs until the Privacy panel has been opened."""

import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from events.schema import Event
from events.store import EventStore
from posture.history import PostureHistory
from privacy import Retention, RetentionWorker

NOW = 1_800_000_000.0
DAY = 86400.0
OLD = NOW - 40 * DAY  # older than 30 days
NEW = NOW - 2 * DAY


def age(path, ts):
    os.utime(path, (ts, ts))


@pytest.fixture
def data(tmp_path):
    """A data dir with one old and one recent item of every kind."""
    db = tmp_path / "events.db"
    clips = tmp_path / "clips"
    rec = tmp_path / "recordings"
    rec.mkdir()
    store = EventStore(str(db))
    for alert_id, ts in (("ALT-OLD", OLD), ("ALT-NEW", NEW)):
        folder = clips / alert_id
        folder.mkdir(parents=True)
        (folder / f"clip_{alert_id}.mp4").write_bytes(b"v" * 100)
        (folder / f"snapshot_{alert_id}.jpg").write_bytes(b"j" * 10)
        store.emit(Event("fall", "critical", ts, ts, alert_id=alert_id,
                         clip_path=str(folder / f"clip_{alert_id}.mp4"),
                         thumbnail_path=str(folder / f"snapshot_{alert_id}.jpg")))
        for f in folder.iterdir():
            age(f, ts)
        age(folder, ts)
    for name, ts in (("ALT-ORPHAN-OLD", OLD), ("ALT-ORPHAN-NEW", NEW)):  # e.g. test footage
        (clips / name).mkdir()
        (clips / name / "clip.mp4").write_bytes(b"o" * 50)
        age(clips / name / "clip.mp4", ts)
        age(clips / name, ts)
    for name, ts in (("laptop_old.mp4", OLD), ("laptop_new.mp4", NEW)):
        (rec / name).write_bytes(b"r" * 1000)
        age(rec / name, ts)
    (rec / "laptop_old.labels.json").write_text("{}")
    store.add_ergo_time("laptop", [("2020-01-01", "zone", "", 4, 10.0), ("2099-01-01", "zone", "", 4, 5.0)])
    hist = PostureHistory(str(tmp_path / "posture_history_laptop.db"))
    hist.add_seconds(OLD, "slouching", 30)
    hist.add_seconds(NEW, "slouching", 20)
    hist.flush()
    hist.add_correction(OLD, "slouching", 4.0, True)
    hist.add_event(NEW, "break_taken", {})
    with open(tmp_path / "posture_movement_log_laptop.jsonl", "w", encoding="utf-8") as f:
        for ts in (OLD, OLD + 1, NEW):
            f.write(json.dumps({"t": ts, "state": "still"}) + "\n")
    for name, ts in (("events.db.bak-20200101-000000", OLD), ("events.db.bak-20990101-000000", NEW)):
        (tmp_path / name).write_bytes(b"b" * 20)
        age(tmp_path / name, ts)
    return SimpleNamespace(root=tmp_path, db=db, clips=clips, rec=rec,
                           retention=Retention(str(db), str(clips), str(rec), clock=lambda: NOW))


def counts(items):
    return {i.category: i.count for i in items}


def test_plan_finds_everything_older_than_the_cutoff_and_deletes_nothing(data):
    plan = counts(data.retention.plan(30))
    assert plan == {"events": 1, "orphan_clips": 1, "recordings": 1, "ergo_time": 1, "posture": 2,
                    "movement_log": 2, "backups": 1}
    assert (data.clips / "ALT-OLD").is_dir() and (data.rec / "laptop_old.mp4").exists()  # only looked
    stored = counts(data.retention.plan(cutoff=float("inf")))  # "what's stored": everything
    assert stored["events"] == 2 and stored["recordings"] == 2 and stored["backups"] == 2


def test_apply_deletes_old_items_everywhere_and_keeps_recent_ones(data):
    data.retention.apply(30)
    assert [e.alert_id for e in EventStore(str(data.db)).query()] == ["ALT-NEW"]
    assert sorted(p.name for p in data.clips.iterdir()) == ["ALT-NEW", "ALT-ORPHAN-NEW"]
    assert sorted(p.name for p in data.rec.iterdir()) == ["laptop_new.mp4"]  # sidecar went too
    with sqlite3.connect(data.db) as conn:
        assert [r[0] for r in conn.execute("SELECT day FROM ergo_time")] == ["2099-01-01"]
    with sqlite3.connect(data.root / "posture_history_laptop.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM minutes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    lines = (data.root / "posture_movement_log_laptop.jsonl").read_text().splitlines()
    assert [json.loads(x)["t"] for x in lines] == [NEW]
    assert sorted(p.name for p in data.root.glob("*.bak-*")) == ["events.db.bak-20990101-000000"]
    assert all(i.count == 0 for i in data.retention.plan(30))  # nothing left to do


def test_a_clips_folder_outside_the_data_folder_is_never_treated_as_orphans(data, tmp_path):
    other = tmp_path / "other_server" / "events.db"  # another database sharing these clips
    other.parent.mkdir()
    EventStore(str(other))
    r = Retention(str(other), str(data.clips), str(data.rec), clock=lambda: NOW)
    assert counts(r.plan(30))["orphan_clips"] == 0
    r.apply(30)
    assert (data.clips / "ALT-ORPHAN-OLD").is_dir() and (data.clips / "ALT-OLD").is_dir()


def test_clips_of_another_database_in_the_data_folder_are_not_orphans(data):
    other = EventStore(str(data.root / "integration.db"))  # e.g. another server's DB_PATH
    other.emit(Event("fall", "critical", OLD, NEW, alert_id="ALT-ORPHAN-OLD"))
    assert counts(data.retention.plan(30))["orphan_clips"] == 0


def test_worker_dry_runs_until_the_privacy_panel_is_opened(data):
    settings = {"retention_days": 30, "privacy_seen": False}
    w = RetentionWorker(data.retention, lambda: settings)
    run = w.run_once()
    assert run["dry_run"] and run["count"] == 9
    assert (data.clips / "ALT-OLD").is_dir()  # nothing deleted
    settings["privacy_seen"] = True
    run = w.run_once()
    assert not run["dry_run"] and run["count"] == 9
    assert not (data.clips / "ALT-OLD").exists()
    assert w.run_once()["count"] == 0


def test_zero_days_keeps_everything(data):
    w = RetentionWorker(data.retention, lambda: {"retention_days": 0, "privacy_seen": True})
    assert w.run_once()["kept_forever"]
    assert (data.clips / "ALT-OLD").is_dir()


# --- API --------------------------------------------------------------------------------------


@pytest.fixture
def privacy_api(tmp_config, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True  # the test client isn't 127.0.0.1
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "pipeline", None)
    monkeypatch.setattr(server, "_retention_worker", None)
    monkeypatch.setattr(server, "_recordings_dir", lambda: tmp_path / "recordings")
    with TestClient(server.app) as c:
        yield c


def test_privacy_api_defaults_seen_and_settings(privacy_api):
    body = privacy_api.get("/api/privacy").json()
    assert body["retention_days"] == 30 and body["dry_run"] and not body["skeleton_only"]
    assert body["stored"]["count"] == 0 and body["would_delete"]["count"] == 0
    assert privacy_api.post("/api/privacy/seen", json={}).json() == {"privacy_seen": True}
    assert privacy_api.get("/api/privacy").json()["dry_run"] is False
    body = privacy_api.put("/api/privacy", json={"retention_days": 0}).json()
    assert body["retention_days"] == 0 and body["would_delete"] is None
    assert privacy_api.put("/api/privacy", json={"retention_days": -1}).status_code == 422
    assert privacy_api.get("/api/app").json().keys() == {"mode", "demo_footage"}


def test_delete_now_must_match_the_saved_setting(privacy_api):
    privacy_api.put("/api/privacy", json={"retention_days": 7})
    assert privacy_api.post("/api/privacy/delete-now", json={"retention_days": 30}).status_code == 409
    r = privacy_api.post("/api/privacy/delete-now", json={"retention_days": 7})
    assert r.status_code == 200 and r.json()["manual"] and r.json()["count"] == 0
