"""Unified event schema, store (filters, grouping), migration from the old table, and the bus."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from events import Event, EventBus, EventStore
from events.migrate import SCHEMA_VERSION, migrate


def ev(i=1, type="zone_intrusion", severity="high", ts=1000.0, **kw) -> Event:
    return Event(type=type, severity=severity, start_ts=ts, end_ts=ts, alert_id=f"ALT-{i:06d}",
                 message=f"event {i}", **kw)


@pytest.fixture
def store(tmp_path):
    return EventStore(str(tmp_path / "db" / "events.db"))


# --- schema ------------------------------------------------------------------------


def test_event_validates_severity_type_and_times():
    with pytest.raises(ValueError, match="severity"):
        ev(severity="urgent")
    with pytest.raises(ValueError, match="type"):
        ev(type="banana")
    with pytest.raises(ValueError, match="end_ts"):
        Event(type="fall", severity="critical", start_ts=10, end_ts=5)
    assert ev(type="rule:helmet_loading").type == "rule:helmet_loading"


def test_from_alert_moves_zone_and_clip_out_of_details_and_uses_dwell_for_start():
    alert = SimpleNamespace(
        alert_id="ALT-ABC", alert_type="time_exceeded", track_id=4, timestamp=1100.0,
        confidence=0.8, severity="medium", message="Zone violation",
        details={"zone_id": "dock", "duration": 12.5, "clip_path": "data/clips/x.mp4"},
    )
    e = Event.from_alert(alert, camera_id="cam-2", clip_path="data/clips/x.mp4", thumbnail_path="t.jpg")
    assert (e.zone_id, e.camera_id, e.track_id) == ("dock", "cam-2", 4)
    assert e.start_ts == pytest.approx(1087.5) and e.end_ts == 1100.0 and e.duration_s == 12.5
    assert "zone_id" not in e.attributes and "clip_path" not in e.attributes
    legacy = e.to_alert_dict()  # what /api/alerts and the WebSocket still send
    assert legacy["alert_type"] == "time_exceeded" and legacy["timestamp"] == 1100.0
    assert legacy["details"]["zone_id"] == "dock"


# --- store -------------------------------------------------------------------------


def test_emit_assigns_ids_and_duplicates_are_ignored(store):
    first = store.emit(ev(1))
    again = store.emit(ev(1))
    assert first.id is not None and again.id == first.id
    assert store.total() == 1


def test_query_filters_and_orders_newest_first(store):
    store.emit(ev(1, ts=100, zone_id="a", track_id=1))
    store.emit(ev(2, type="fall", severity="critical", ts=200, track_id=2))
    store.emit(ev(3, type="loitering", severity="low", ts=300, zone_id="a", camera_id="cam-9"))
    assert [e.alert_id for e in store.query()] == ["ALT-000003", "ALT-000002", "ALT-000001"]
    assert [e.alert_id for e in store.query(types=["fall", "loitering"])] == ["ALT-000003", "ALT-000002"]
    assert [e.alert_id for e in store.query(severity="critical")] == ["ALT-000002"]
    assert [e.alert_id for e in store.query(zone_id="a")] == ["ALT-000003", "ALT-000001"]
    assert [e.alert_id for e in store.query(camera_id="cam-9")] == ["ALT-000003"]
    assert [e.alert_id for e in store.query(track_id=2)] == ["ALT-000002"]
    assert [e.alert_id for e in store.query(start=150, end=250)] == ["ALT-000002"]
    assert [e.alert_id for e in store.query(limit=1, offset=1)] == ["ALT-000002"]
    assert store.total(zone_id="a") == 2


def test_time_window_matches_overlapping_events(store):
    store.emit(Event(type="loitering", severity="low", start_ts=100, end_ts=200, alert_id="ALT-L"))
    assert store.total(start=150, end=160) == 1  # the window is inside the event
    assert store.total(start=201) == 0


def test_count_groups_by_allowed_keys_only(store):
    for i, (t, z) in enumerate([("fall", "a"), ("fall", None), ("loitering", "a")]):
        store.emit(ev(i, type=t, severity="low", zone_id=z))
    assert store.count("type") == {"fall": 2, "loitering": 1}
    assert store.count("zone") == {"": 1, "a": 2}
    assert sum(store.count("hour").values()) == 3
    with pytest.raises(ValueError):
        store.count("type; DROP TABLE events")


def test_filter_values_are_parameters_not_sql(store):
    store.emit(ev(1))
    assert store.query(zone_id="x' OR '1'='1") == []
    assert store.total() == 1


def test_set_verified_and_get(store):
    e = store.emit(ev(1, attributes={"reba": 9}))
    store.set_verified(e.id, True)
    got = store.get(e.id)
    assert got.verified == 1 and got.attributes == {"reba": 9}
    assert store.get_by_alert_id("ALT-000001").id == e.id
    assert store.get(9999) is None


def test_store_rejects_directory(tmp_path):
    with pytest.raises(RuntimeError, match="is a directory"):
        EventStore(str(tmp_path))


# --- migration from the original table ---------------------------------------------


def make_legacy_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT UNIQUE, "
            "alert_type TEXT, track_id INTEGER, timestamp REAL, severity TEXT, confidence REAL, "
            "message TEXT, details TEXT, clip_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        rows = [
            ("ALT-OLD1", "zone_intrusion", 3, 1000.0, "high", 0.9, "door",
             json.dumps({"zone_id": "lab", "duration": 0.0, "clip_path": "data/clips/ALT-OLD1/clip_ALT-OLD1.mp4"}),
             "data/clips/ALT-OLD1/clip_ALT-OLD1.mp4"),
            ("ALT-OLD2", "loitering", 5, 2000.0, "low", 0.7, "loiter", json.dumps({"dwell_seconds": 12.0}), None),
            ("ALT-OLD3", "fall", 6, 3000.0, "critical", 0.9, "fall", "not json", ""),
        ]
        conn.executemany(
            "INSERT INTO events (alert_id, alert_type, track_id, timestamp, severity, confidence, "
            "message, details, clip_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_migration_backs_up_copies_rows_and_keeps_old_table(tmp_path):
    db = tmp_path / "events.db"
    make_legacy_db(db)
    store = EventStore(str(db))

    backups = list(tmp_path.glob("events.db.bak-*"))
    assert len(backups) == 1  # a timestamped copy, taken before anything changed
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
        assert "alert_type" in {r[1] for r in conn.execute("PRAGMA table_info(events)")}

    old1, old2, old3 = (store.get_by_alert_id(f"ALT-OLD{i}") for i in (1, 2, 3))
    assert old1.zone_id == "lab" and old1.camera_id == "cam-0"
    assert old1.thumbnail_path == "data/clips/ALT-OLD1/snapshot_ALT-OLD1.jpg"
    assert "clip_path" not in old1.attributes and "zone_id" not in old1.attributes
    assert old2.start_ts == pytest.approx(1988.0) and old2.end_ts == 2000.0
    assert old3.attributes == {"raw_details": "not json"} and old3.clip_path is None
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events_v0").fetchone()[0] == 3
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_is_idempotent_and_skips_backup_when_current(tmp_path):
    db = tmp_path / "events.db"
    make_legacy_db(db)
    EventStore(str(db))
    EventStore(str(db))  # second start: nothing to do
    assert migrate(db) == SCHEMA_VERSION
    assert len(list(tmp_path.glob("events.db.bak-*"))) == 1
    assert EventStore(str(db)).total() == 3


def test_new_database_needs_no_backup(tmp_path):
    EventStore(str(tmp_path / "fresh.db"))
    assert not list(tmp_path.glob("*.bak-*"))


# --- bus ---------------------------------------------------------------------------


def test_bus_runs_subscribers_in_order_and_survives_a_failing_one(store):
    seen = []
    bus = EventBus()
    bus.subscribe("store", store.emit)
    bus.subscribe("broken", lambda e: 1 / 0)
    bus.subscribe("after", lambda e: seen.append(e.id))
    bus.publish(ev(1))
    assert seen and seen[0] is not None  # later subscribers see the id the store assigned
