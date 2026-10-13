"""Near falls: went down suddenly (descent seen) and got back up, with no confirmed fall."""

from types import SimpleNamespace

from anomaly.fall_detector import FallDetector
from test_fall_detector import LIE, STAND, Sim

# Lying straight along the picture's up-down axis, seen from a camera high in a corner: standing
# keypoints (head at standing height, legs extended) in a box wider than tall.
LYING_SEEN_FROM_ABOVE = dict(top=200.0, w=200.0, h=180.0)


def fall(sim):
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)  # fast drop to the floor
    sim.hold(LIE, 0.4)         # down, but not still long enough to confirm (1 s)


def test_going_down_and_getting_back_up_is_one_near_fall():
    sim = Sim()
    fall(sim)
    sim.move(LIE, STAND, 0.6)
    sim.hold(STAND, 2.0)
    events = sim.det.drain_near_falls()
    assert len(events) == 1 and events[0].stage == "near_fall"
    assert sim.events == []  # no confirmed fall
    assert 0.5 <= events[0].signals["down_for_s"] <= 2.0
    sim.hold(STAND, 3.0)
    assert sim.det.drain_near_falls() == []  # one per get-up


def test_staying_down_is_a_fall_not_a_near_fall():
    sim = Sim()
    fall(sim)
    sim.hold(LIE, 4.0)
    sim.move(LIE, STAND, 0.6)
    sim.hold(STAND, 2.0)
    assert len(sim.events) == 1 and sim.events[0].stage == "confirmed"
    assert sim.det.drain_near_falls() == []


def test_never_reaching_the_ground_is_not_a_near_fall():
    # A fast crouch or sitting down: everyday actions move as fast as the start of a fall.
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, dict(top=260.0, w=60.0, h=120.0), 0.15)
    sim.hold(dict(top=260.0, w=60.0, h=120.0), 0.5)
    sim.move(dict(top=260.0, w=60.0, h=120.0), STAND, 0.5)
    sim.hold(STAND, 2.0)
    assert sim.det.drain_near_falls() == []


def test_lying_straight_seen_from_above_is_not_getting_up():
    sim = Sim()
    fall(sim)
    sim.hold(LYING_SEEN_FROM_ABOVE, 3.0)
    assert sim.det.drain_near_falls() == []


def test_too_long_on_the_floor_before_getting_up_is_not_a_near_fall():
    sim = Sim(FallDetector(stillness_seconds=30.0))  # never confirms: only the window applies
    fall(sim)
    sim.hold(LIE, FallDetector.NEAR_FALL_WINDOW_S)
    sim.move(LIE, STAND, 0.6)
    sim.hold(STAND, 2.0)
    assert sim.det.drain_near_falls() == []


def near_fall_event(ts):
    return SimpleNamespace(track_id=3, timestamp=ts, confidence=0.7, velocity=2.1,
                           signals={"down_for_s": 1.2})


def test_a_second_near_fall_within_the_hour_escalates(tmp_config):
    from anomaly.engine import AnomalyEngine

    engine = AnomalyEngine(tmp_config)
    first = engine._near_fall_alert(near_fall_event(100.0))
    assert first.alert_type == "near_fall" and first.severity == "medium"
    assert first.message == "Near fall: went down suddenly and got back up"
    second = engine._near_fall_alert(near_fall_event(1900.0))
    assert second.severity == "high" and "2 near falls in the last hour" in second.message
    later = engine._near_fall_alert(near_fall_event(9000.0))
    assert later.severity == "medium"  # the earlier ones are over an hour old


def test_getting_up_under_a_new_track_id_after_a_confirmed_fall_is_not_a_near_fall():
    # Own recording (2026-10-06): down 10 s (confirmed), got up as a new track id.
    sim = Sim()
    fall(sim)
    sim.hold(LIE, 2.0)
    assert sim.state == FallDetector.CONFIRMED
    sim.det.inherit(2, 1, sim.t)
    for params in [STAND] * 60:
        from conftest import make_pose
        sim.det.check(2, make_pose(track_id=2, **params), sim.feat, sim.t)
        sim.t += 0.04
    assert sim.det.drain_near_falls() == []
