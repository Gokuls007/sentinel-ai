"""Fall state machine on synthetic pose sequences (25 fps)."""

from anomaly.fall_detector import FallDetector
from conftest import make_features, make_pose

DT = 0.04
STAND = dict(top=200.0, w=60.0, h=180.0)
LIE = dict(top=330.0, w=180.0, h=60.0, lying=True)


def lerp(a, b, t):
    out = {k: a[k] + (b[k] - a[k]) * t for k in ("top", "w", "h")}
    out["lying"] = b.get("lying", False) if t > 0.5 else a.get("lying", False)
    out["head_visible"] = b.get("head_visible", True)
    return out


class Sim:
    def __init__(self, detector=None):
        self.det = detector or FallDetector()
        self.feat = make_features()
        self.t = 0.0
        self.events = []

    def hold(self, params, seconds):
        for _ in range(int(round(seconds / DT))):
            self.step(params)

    def move(self, start, end, seconds):
        n = int(round(seconds / DT))
        for i in range(1, n + 1):
            self.step(lerp(start, end, i / n))

    def step(self, params):
        ev = self.det.check(1, make_pose(**params), self.feat, self.t)
        if ev:
            self.events.append(ev)
        self.t += DT

    @property
    def state(self):
        return self.det.state_of(1)


def test_fall_then_lying_still_raises_exactly_one_alert():
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)       # fast drop: ~1.8 body heights / s
    sim.hold(LIE, 5.0)
    assert len(sim.events) == 1
    ev = sim.events[0]
    assert ev.stage == "confirmed"
    assert ev.velocity > 1.2
    assert 0.7 <= ev.confidence <= 0.99
    assert sim.state == FallDetector.CONFIRMED


def test_alert_waits_for_stillness():
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)
    sim.hold(LIE, 0.5)              # on the ground, but not still for 1 s yet
    assert sim.events == []
    assert sim.state == FallDetector.FALLEN


def test_getting_up_quickly_does_not_alert_and_resets():
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)
    sim.hold(LIE, 0.4)
    sim.move(LIE, STAND, 0.6)
    sim.hold(STAND, 2.0)
    assert sim.events == []
    assert sim.state == FallDetector.UPRIGHT


def test_sitting_down_slowly_is_not_a_fall():
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 3.0)       # slow descent (~0.12 body heights / s)
    sim.hold(LIE, 5.0)
    assert sim.events == []


def test_fast_crouch_that_stays_upright_times_out():
    crouch = dict(top=260.0, w=70.0, h=120.0)  # lower but still taller than wide
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, crouch, 0.2)
    sim.hold(crouch, 3.0)
    assert sim.events == []
    assert sim.state == FallDetector.UPRIGHT


def test_uncalibrated_person_never_alerts():
    sim = Sim()
    sim.feat.initial_standing_height = 0.0
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)
    sim.hold(LIE, 5.0)
    assert sim.events == []


def test_fall_without_visible_face_still_detected():
    sim = Sim()
    sim.hold(STAND, 1.0)
    lie_no_face = {**LIE, "head_visible": False}
    sim.move(STAND, lie_no_face, 0.2)
    sim.hold(lie_no_face, 3.0)
    assert len(sim.events) == 1


def test_cooldown_between_falls():
    sim = Sim(FallDetector(cooldown_seconds=30.0))
    for _ in range(2):              # two falls ~8 s apart
        sim.hold(STAND, 1.0)
        sim.move(STAND, LIE, 0.2)
        sim.hold(LIE, 3.0)
        sim.move(LIE, STAND, 1.0)
        sim.hold(STAND, 2.0)
    assert len(sim.events) == 1
    sim.hold(STAND, 25.0)           # well past the cooldown
    sim.move(STAND, LIE, 0.2)
    sim.hold(LIE, 3.0)
    assert len(sim.events) == 2


def test_prune_forgets_missing_tracks():
    det = FallDetector()
    det.check(1, make_pose(), make_features(), 0.0)
    det.check(2, make_pose(track_id=2), make_features(track_id=2), 0.0)
    det.prune({2})
    assert set(det.tracks) == {2}
