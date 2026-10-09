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
        for _ in range(round(seconds / DT)):
            self.step(params)

    def move(self, start, end, seconds):
        n = round(seconds / DT)
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


# --- seated close to a webcam (own recording, 2026-10-06) ------------------------------


def webcam_closeup(hips_visible=True, hip_y=700.0, ankle_y=None):
    """Upper body filling the right half of a 1280x720 webcam frame, cut by the bottom edge."""
    import numpy as np

    from core.pose_estimator import PoseResult

    kp = np.zeros((17, 3), np.float32)
    kp[0] = (1000, 200, 0.9)
    kp[5], kp[6] = (920, 420, 0.9), (1120, 420, 0.9)
    if hips_visible:
        kp[11], kp[12] = (1180, hip_y, 0.4), (990, hip_y, 0.4)
    if ankle_y is not None:  # a guessed "ankle" at hip height
        kp[16] = (834, ankle_y, 0.35)
    return PoseResult(track_id=1, keypoints=kp, bbox=np.array([777, 95, 1279, 711], np.float32))


def test_hips_flickering_in_and_out_of_view_is_not_a_fall():
    det = FallDetector()
    det.frame_size = (1280, 720)
    feat = make_features(standing_height=897.0)  # the torso-estimated height seen on that recording
    t = 0.0
    for i in range(60):  # hips visible, then the box centre, then visible again...
        assert det.check(1, webcam_closeup(hips_visible=i % 3 != 0), feat, t) is None
        t += 0.067
    assert det.state_of(1) == FallDetector.UPRIGHT


def test_someone_filling_the_picture_is_never_judged_fallen():
    """Even when a guessed ankle at hip height makes the hips look 'on the floor'."""
    det = FallDetector()
    det.frame_size = (1280, 720)
    feat = make_features(standing_height=897.0)
    t = 0.0
    for _ in range(150):  # 10 s still, with the misleading ankle
        assert det.check(1, webcam_closeup(ankle_y=694.0), feat, t) is None
        t += 0.067
    assert det.state_of(1) == FallDetector.UPRIGHT
    assert det.drain_possible() == []


def test_a_real_fall_still_counts_without_a_frame_size():
    sim = Sim()
    sim.hold(STAND, 1.0)
    sim.move(STAND, LIE, 0.2)
    sim.hold(LIE, 5.0)
    assert len(sim.events) == 1

