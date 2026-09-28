"""AnomalyEngine: loitering, alert wiring, LSTM gating, state pruning."""

import pytest

from anomaly.engine import AnomalyEngine
from anomaly.zone_monitor import Zone
from conftest import make_features, make_pose


@pytest.fixture
def engine(tmp_config):
    tmp_config.loiter.time_threshold = 10.0
    tmp_config.loiter.movement_threshold = 50.0
    eng = AnomalyEngine(tmp_config)
    eng.zone_monitor.set_zones([])
    return eng


def feed(engine, positions, t0=0.0, dt=0.1, track_id=1):
    feats = {track_id: make_features(track_id=track_id, standing_height=0.0)}
    alerts, t = [], t0
    for cx in positions:
        alerts += engine.process({track_id: make_pose(track_id=track_id, cx=cx)}, feats, t)
        t += dt
    return alerts, t


def test_untrained_lstm_is_disabled(engine):
    assert engine.temporal_classifier.loaded is False


def test_loitering_after_standing_still(engine):
    alerts, _ = feed(engine, [300.0] * 120)            # 12 s in one spot (small jitter-free)
    loiter = [a for a in alerts if a.alert_type == "loitering"]
    assert len(loiter) == 1
    assert loiter[0].severity == "low"
    assert loiter[0].details["dwell_seconds"] > 10.0


def test_walking_then_brief_pause_is_not_loitering(engine):
    walk = [100.0 + 10 * i for i in range(100)]       # 10 s walking at 100 px/s
    alerts, _ = feed(engine, walk + [walk[-1]] * 30)  # then a 3 s pause
    assert not [a for a in alerts if a.alert_type == "loitering"]


def test_slow_drift_resets_loiter_clock(engine):
    drift = [300.0 + 6 * i for i in range(150)]       # 60 px/s: leaves the radius < 1 s
    alerts, _ = feed(engine, drift)
    assert not [a for a in alerts if a.alert_type == "loitering"]


def test_loitering_cooldown(engine):
    alerts, _ = feed(engine, [300.0] * 400)            # 40 s < 60 s cooldown
    assert len([a for a in alerts if a.alert_type == "loitering"]) == 1


def test_zone_violation_becomes_alert(engine):
    engine.zone_monitor.set_frame_size(640, 480)
    engine.zone_monitor.set_zones([Zone("door", "Door", [(0, 0), (1, 0), (1, 1), (0, 1)], "restricted")])
    alerts, _ = feed(engine, [300.0])
    (a,) = alerts
    assert a.alert_type == "zone_intrusion" and a.severity == "high"
    assert a.details["zone_id"] == "door"
    assert a.alert_id.startswith("ALT-")


def test_state_pruned_when_track_disappears(engine):
    feed(engine, [300.0] * 5, track_id=3)
    engine.process({}, {}, 1.0)
    assert 3 not in engine.loiter_anchor
    assert 3 not in engine.fall_detector.tracks
