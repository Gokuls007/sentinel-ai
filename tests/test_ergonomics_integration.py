"""Ergonomics wired into the engine, the store (schema v2) and the API."""

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anomaly.engine import AnomalyEngine
from anomaly.zone_monitor import Zone
from api import server
from core.pose_estimator import PoseResult, TrackFeatures
from events import Event, EventStore
from events.migrate import SCHEMA_VERSION
from test_ergonomics import skeleton

T0 = 1_790_000_000.0


def pose(kp, track_id=1):
    xs, ys = kp[:, 0], kp[:, 1]
    bbox = np.array([xs.min() - 10, ys.min() - 10, xs.max() + 10, ys.max() + 10], dtype=float)
    return PoseResult(track_id=track_id, keypoints=kp.astype(np.float32), bbox=bbox)


def feed_engine(engine, kp, seconds, fps=10, track_id=1, start=T0):
    feats = {track_id: TrackFeatures(track_id=track_id, first_seen=start)}
    alerts, t = [], start
    for _ in range(int(seconds * fps)):
        alerts += engine.process({track_id: pose(kp, track_id)}, feats, t)
        t += 1 / fps
    return alerts


@pytest.fixture
def engine(tmp_config):
    tmp_config.loiter.time_threshold = 10_000  # keep loitering out of these tests
    eng = AnomalyEngine(tmp_config)
    eng.zone_monitor.set_zones([])
    return eng


def test_engine_raises_one_ergo_risk_event_for_sustained_high_risk(engine):
    engine.zone_monitor.set_frame_size(800, 600)
    engine.zone_monitor.set_zones([
        Zone("dock", "Loading dock", [(0, 0), (1, 0), (1, 1), (0, 1)], "time_limited",
             time_limit=10_000, load_score=2),
    ])
    alerts = feed_engine(engine, skeleton(trunk=70, knee_flex=100, arm=150), 8)
    ergo = [a for a in alerts if a.alert_type == "ergo_risk"]
    assert len(ergo) == 1
    a = ergo[0]
    assert a.severity in ("high", "critical")
    assert a.details["zone_id"] == "dock" and a.details["reba_score"] >= 8 and a.details["duration"] >= 3
    assert "Ergonomic risk: REBA" in a.message
    snap = engine.ergonomics_snapshot[1]
    assert snap["reliable"] and snap["level"] >= 4 and snap["angles"]["trunk"] == pytest.approx(70, abs=1)
    assert engine.last_ergo_ms >= 0


def test_upright_person_raises_no_ergo_event(engine):
    alerts = feed_engine(engine, skeleton(), 8)
    assert not [a for a in alerts if a.alert_type == "ergo_risk"]
    assert engine.ergonomics_snapshot[1]["level_name"] == "negligible"


def test_ergonomics_can_be_disabled(tmp_config):
    tmp_config.ergonomics.enabled = False
    eng = AnomalyEngine(tmp_config)
    assert eng.ergo is None and eng.ergonomics_snapshot == {}


# --- store: schema v2 ------------------------------------------------------------------


def test_v1_database_gains_the_ergo_table_without_a_backup(tmp_path):
    db = tmp_path / "events.db"
    with sqlite3.connect(db) as conn:  # a v1 database with one event
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, severity TEXT NOT NULL, "
            "camera_id TEXT NOT NULL DEFAULT 'cam-0', track_id INTEGER, zone_id TEXT, start_ts REAL NOT NULL, "
            "end_ts REAL NOT NULL, attributes TEXT NOT NULL DEFAULT '{}', clip_path TEXT, thumbnail_path TEXT, "
            "verified INTEGER, alert_id TEXT UNIQUE, message TEXT NOT NULL DEFAULT '', confidence REAL, "
            "created_at REAL NOT NULL DEFAULT 0)")
        conn.execute("INSERT INTO events (type, severity, start_ts, end_ts, alert_id) "
                     "VALUES ('fall', 'critical', 1, 1, 'ALT-1')")
        conn.execute("PRAGMA user_version = 1")
    store = EventStore(str(db))
    assert store.total() == 1
    assert not list(tmp_path.glob("*.bak-*"))
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
        assert conn.execute("SELECT COUNT(*) FROM ergo_time").fetchone()[0] == 0


def test_ergo_time_accumulates_and_groups(tmp_path):
    store = EventStore(str(tmp_path / "e.db"))
    store.add_ergo_time("cam-0", [("2026-10-01", "zone", "dock", 4, 10.0), ("2026-10-01", "zone", "dock", 2, 5.0)])
    store.add_ergo_time("cam-0", [("2026-10-01", "zone", "dock", 4, 2.5), ("2026-10-02", "zone", "", 3, 7.0)])
    store.add_ergo_time("laptop", [("2026-10-01", "zone", "dock", 4, 1.0)])
    assert store.ergo_time("zone", "2026-10-01", "2026-10-01") == {"dock": {4: 13.5, 2: 5.0}}
    assert store.ergo_time("zone", "2026-10-01", "2026-10-02", camera_id="cam-0") == {
        "dock": {4: 12.5, 2: 5.0}, "": {3: 7.0}}
    with pytest.raises(ValueError):
        store.ergo_time("evil", "2026-10-01", "2026-10-01")


# --- API --------------------------------------------------------------------------------


@pytest.fixture
def ergo_client(tmp_config, tmp_path, monkeypatch):
    store = EventStore(str(tmp_path / "api.db"))
    flushed = []
    fake = SimpleNamespace(
        config=tmp_config,
        event_store=store,
        anomaly_engine=SimpleNamespace(ergonomics_snapshot={7: {"track_id": 7, "score": 9, "level": 4}}),
        flush_ergo_time=lambda: flushed.append(True),
    )
    monkeypatch.setattr(server, "pipeline", fake)
    monkeypatch.setattr(server, "config", tmp_config)
    server._cameras.clear()
    with TestClient(server.app) as client:
        client.store, client.flushed = store, flushed
        yield client


def test_live_endpoint(ergo_client):
    body = ergo_client.get("/api/ergonomics/live").json()
    assert body["tracks"] == [{"track_id": 7, "score": 9, "level": 4}] and body["min_confidence"] == 0.6


def test_time_endpoint_flushes_and_names_levels(ergo_client):
    ergo_client.store.add_ergo_time("cam-0", [("2026-10-01", "zone", "dock", 4, 30.0),
                                              ("2026-10-01", "zone", "dock", 0, 5.0)])
    body = ergo_client.get("/api/ergonomics/time", params={"day_from": "2026-10-01"}).json()
    assert ergo_client.flushed  # in-memory time is written before reading
    assert body["rows"] == {"dock": {"unknown": 5.0, "high": 30.0}}
    assert body["levels"] == ["unknown", "negligible", "low", "medium", "high", "very_high"]
    assert ergo_client.get("/api/ergonomics/time", params={"group_by": "person"}).status_code == 422
    assert ergo_client.get("/api/ergonomics/time", params={"day_from": "yesterday"}).status_code == 422


def test_postures_endpoint_summarises_ergo_events(ergo_client):
    for i, (part, reba, dur) in enumerate([("trunk", 9, 4.0), ("trunk", 11, 6.0), ("upper_arm", 8, 3.0)]):
        ergo_client.store.emit(Event(type="ergo_risk", severity="high", start_ts=T0 + i, end_ts=T0 + i,
                                     alert_id=f"ALT-E{i}",
                                     attributes={"dominant": part, "reba_score": reba, "duration": dur}))
    body = ergo_client.get("/api/ergonomics/postures").json()["postures"]
    assert list(body) == ["trunk", "upper_arm"]
    assert body["trunk"] == {"events": 2, "peak_reba": 11, "total_duration_s": 10.0}
