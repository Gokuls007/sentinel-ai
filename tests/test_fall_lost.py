"""The 'lost on the floor' fixes: holding a lost track, hysteresis, and the recovery retries
(region-local low threshold, rotated fallback), with synthetic poses and a fake pose model."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from anomaly.fall_detector import FallDetector
from anomaly.fall_recovery import crop_region, recover_pose, unrotate_points
from core.pose_estimator import PoseResult
from test_fall_webcam import calibrate, upper_body

FPS = 30.0


def fall_frames(rng, lying=0):
    standing = [upper_body(jitter=1.0, rng=rng) for _ in range(30)]
    falling = [upper_body(shoulder_y=200 + 25 * k, angle_deg=min(85, 12 * k), jitter=1.0, rng=rng) for k in range(1, 9)]
    on_floor = [upper_body(shoulder_y=400, angle_deg=85, jitter=1.0, rng=rng) for _ in range(lying)]
    return standing, falling, on_floor


def drive(det, feat, frames, t0=0.0):
    events = []
    for i, p in enumerate(frames):
        e = det.check(1, p, feat, t0 + i / FPS)
        if e:
            events.append(e)
    return events, t0 + len(frames) / FPS


# --- fix 1: hold a lost track ---------------------------------------------------------------------

def lost_after_landing(det, lying_seen=6):
    rng = np.random.default_rng(0)
    standing, falling, on_floor = fall_frames(rng, lying=lying_seen)
    feat = calibrate(standing)
    events, t = drive(det, feat, standing + falling + on_floor)
    return events, t


def test_without_hold_a_person_lost_on_the_floor_is_never_confirmed():
    det = FallDetector()
    events, t = lost_after_landing(det)
    assert det.state_of(1) == det.FALLEN and events == []
    assert all(det.check_missing(1, t + i / FPS) is None for i in range(150))


def test_hold_confirms_a_fall_when_the_person_vanishes_on_the_floor():
    det = FallDetector(lost_hold_seconds=5.0)
    events, t = lost_after_landing(det)
    assert events == []  # not confirmed while still visible (only 0.2 s on the floor)
    held = [det.check_missing(1, t + i / FPS) for i in range(60)]
    fired = [e for e in held if e]
    assert len(fired) == 1 and fired[0].signals["held_while_lost"] is True
    assert fired[0].timestamp - t == pytest.approx(1.0, abs=0.05)  # after the 1 s confirmation
    assert det.state_of(1) == det.CONFIRMED


def test_hold_expires_and_ignores_upright_people():
    det = FallDetector(lost_hold_seconds=0.5, stillness_seconds=1.0)
    _, t = lost_after_landing(det)
    assert all(det.check_missing(1, t + i / FPS) is None for i in range(60))  # 0.5 s hold < 1 s needed
    walker = FallDetector(lost_hold_seconds=5.0)
    rng = np.random.default_rng(1)
    standing, _, _ = fall_frames(rng)
    drive(walker, calibrate(standing), standing)
    assert walker.check_missing(1, 2.0) is None and walker.state_of(1) == walker.UPRIGHT  # walked out of view


def test_vanishing_mid_descent_away_from_the_floor_is_not_held():
    det = FallDetector(lost_hold_seconds=5.0)
    rng = np.random.default_rng(2)
    standing, falling, _ = fall_frames(rng)
    feat = calibrate(standing)
    drive(det, feat, standing + falling[:1])  # only just started dropping, still upright
    if det.state_of(1) == det.FALLING:
        assert det.check_missing(1, 2.0) is None and det.state_of(1) == det.FALLING


# --- fix 4: hysteresis ---------------------------------------------------------------------------

def test_one_upright_looking_frame_no_longer_cancels_the_ground_state():
    rng = np.random.default_rng(3)
    for hold, expect_state in ((0.0, FallDetector.UPRIGHT), (0.5, FallDetector.FALLEN)):
        det = FallDetector(upright_hold_seconds=hold)
        standing, falling, on_floor = fall_frames(rng, lying=6)
        feat = calibrate(standing)
        _, t = drive(det, feat, standing + falling + on_floor)
        blip = upper_body(jitter=0.0)  # a single mis-estimated upright pose
        det.check(1, blip, feat, t)
        assert det.state_of(1) == expect_state, hold


def test_hysteresis_still_lets_people_get_up():
    det = FallDetector(upright_hold_seconds=0.5)
    rng = np.random.default_rng(4)
    standing, falling, on_floor = fall_frames(rng, lying=6)
    feat = calibrate(standing)
    _, t = drive(det, feat, standing + falling + on_floor)
    drive(det, feat, [upper_body(jitter=1.0, rng=rng) for _ in range(20)], t0=t)  # 0.67 s upright
    assert det.state_of(1) == det.UPRIGHT


# --- fixes 2 and 3: recovery retries -------------------------------------------------------------

class FakePoseModel:
    """Returns one detection per call, from a callable(image) -> (kps (17,3), xyxy) or None."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def __call__(self, image, conf=0.25, verbose=False):
        self.calls.append((image.shape, conf))
        found = self.respond(image, conf)
        if found is None:
            return [SimpleNamespace(keypoints=None, boxes=None)]
        kps, box = found
        return [SimpleNamespace(keypoints=SimpleNamespace(data=_Tensor([kps])), boxes=_Boxes([box]))]


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:  # like ultralytics Boxes: len() and .xyxy
    def __init__(self, boxes):
        self.xyxy = _Tensor(boxes)

    def __len__(self):
        return len(self.xyxy.a)


def test_unrotate_inverts_cv2_rotation():
    img = np.zeros((40, 60, 3), np.uint8)  # h=40, w=60
    pts = np.array([[5.0, 7.0], [59.0, 0.0], [0.0, 39.0]])
    for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        for x, y in pts:
            marked = img.copy()
            marked[int(y), int(x)] = 255
            ry, rx = np.argwhere(cv2.rotate(marked, rotation)[:, :, 0] == 255)[0]
            back = unrotate_points(np.array([[rx, ry]], float), rotation, 60, 40)[0]
            assert tuple(back) == (x, y)


def test_crop_region_is_widened_and_clipped():
    assert crop_region([100, 100, 200, 140], (480, 640)) == (25, 50, 275, 190)
    assert crop_region([0, 0, 5, 5], (480, 640)) is None or crop_region([0, 0, 5, 5], (480, 640))[0] == 0


def test_low_threshold_retry_maps_keypoints_back_to_the_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    kps = np.zeros((17, 3), np.float32)
    kps[0] = (40, 30, 0.6)

    model = FakePoseModel(lambda img, conf: (kps, [10, 20, 110, 60]) if conf <= 0.15 else None)
    pose, how = recover_pose(model, frame, 7, [100, 100, 200, 140], low_conf=0.15, try_rotated=False)
    assert how == "low" and pose.track_id == 7
    assert tuple(pose.keypoints[0, :2]) == (65.0, 80.0)  # crop starts at (25, 50)
    assert tuple(pose.bbox) == (35.0, 70.0, 135.0, 110.0)
    assert model.calls == [((140, 250, 3), 0.15)]


def test_rotated_retry_finds_a_lying_person_the_upright_pass_missed():
    frame = np.zeros((480, 640, 3), np.uint8)
    region = crop_region([100, 100, 200, 140], frame.shape)  # (25, 50, 275, 190): 250 x 140

    def respond(img, conf):
        if img.shape[:2] == (140, 250):
            return None  # unrotated: nothing
        kps = np.zeros((17, 3), np.float32)
        kps[0] = (10, 20, 0.7)  # in the rotated crop
        return kps, [5, 10, 50, 200]

    pose, how = recover_pose(FakePoseModel(respond), frame, 3, [100, 100, 200, 140], low_conf=0.3, try_low=True)
    assert how == "rotated"
    # CW rotation: rotated (x, y) -> crop (y, 140 - 1 - x) -> frame (+25, +50)
    assert tuple(pose.keypoints[0, :2]) == (20 + region[0], 140 - 1 - 10 + region[1])
    x1, y1, x2, y2 = pose.bbox
    assert x1 < x2 and y1 < y2


def test_engine_uses_recovered_pose_or_holds_for_fallen_tracks_only(tmp_config):
    from anomaly.engine import AnomalyEngine

    tmp_config.fall.lost_hold_seconds = 5.0
    engine = AnomalyEngine(tmp_config)
    rng = np.random.default_rng(5)
    standing, falling, on_floor = fall_frames(rng, lying=6)
    feat = calibrate(standing)
    t = 0.0
    for p in standing + falling + on_floor:
        engine.process({1: p}, {1: feat}, t)
        t += 1 / FPS
    assert engine.fall_detector.state_of(1) == FallDetector.FALLEN
    alerts = []
    for _ in range(40):  # person gone from poses, still a known track
        alerts += engine.process({}, {1: feat}, t)
        t += 1 / FPS
    falls = [a for a in alerts if a.alert_type == "fall"]
    assert len(falls) == 1 and "no longer visible" in falls[0].message


def test_recovered_pose_feeds_the_fall_check(tmp_config):
    from anomaly.engine import AnomalyEngine

    engine = AnomalyEngine(tmp_config)  # no hold: only the recovered pose can confirm
    rng = np.random.default_rng(6)
    standing, falling, on_floor = fall_frames(rng, lying=6)
    feat = calibrate(standing)
    t = 0.0
    for p in standing + falling + on_floor:
        engine.process({1: p}, {1: feat}, t)
        t += 1 / FPS
    alerts = []
    for _ in range(40):
        lying = upper_body(shoulder_y=400, angle_deg=85, jitter=1.0, rng=rng)
        alerts += engine.process({}, {1: feat}, t, recovered={1: PoseResult(1, lying.keypoints, lying.bbox)})
        t += 1 / FPS
    assert [a.alert_type for a in alerts] == ["fall"]


# --- the "last seen lying" gate and the two recovery modes ---------------------------------------

def test_hold_ignores_someone_last_seen_bent_over():
    """Head below the hips (picking something up) is not lying; vanishing then is no fall."""
    det = FallDetector(lost_hold_seconds=5.0)
    rng = np.random.default_rng(7)
    standing, falling, _ = fall_frames(rng)
    feat = calibrate(standing)
    bent = [upper_body(shoulder_y=400, angle_deg=160, jitter=1.0, rng=rng) for _ in range(6)]
    _, t = drive(det, feat, standing + falling + bent)
    assert all(det.check_missing(1, t + i / FPS) is None for i in range(90))


def landed(seed):
    det = FallDetector()
    rng = np.random.default_rng(seed)
    standing, falling, on_floor = fall_frames(rng, lying=6)
    feat = calibrate(standing)
    _, t = drive(det, feat, standing + falling + on_floor)
    return det, feat, t, rng


def test_presence_mode_confirms_a_refound_lying_person_despite_jitter():
    det, feat, t, rng = landed(8)
    events = []
    for i in range(45):  # very jittery re-found poses at the same spot
        p = upper_body(shoulder_y=400, angle_deg=85, jitter=12.0, rng=rng)
        e = det.check_recovered(1, p, feat, t + i / FPS, mode="presence")
        if e:
            events.append(e)
    assert len(events) == 1 and events[0].signals["recovered"] is True


def test_presence_mode_does_not_confirm_someone_who_moves_or_gets_up():
    det, feat, t, rng = landed(9)
    for i in range(45):  # crawling away: 20 px per frame
        p = upper_body(cx=320 + 20 * i, shoulder_y=400, angle_deg=85, jitter=1.0, rng=rng)
        assert det.check_recovered(1, p, feat, t + i / FPS, mode="presence") is None
    det, feat, t, rng = landed(10)
    for i in range(30):  # re-found standing up
        det.check_recovered(1, upper_body(jitter=1.0, rng=rng), feat, t + i / FPS, mode="presence")
    assert det.state_of(1) == FallDetector.UPRIGHT


def test_pose_mode_is_the_normal_check():
    det, feat, t, rng = landed(11)
    events = [det.check_recovered(1, upper_body(shoulder_y=400, angle_deg=85, jitter=12.0, rng=rng), feat,
                                  t + i / FPS, mode="pose") for i in range(45)]
    assert not any(events)  # the same jitter: never still enough as a pose


# --- two-level fall alerts ----------------------------------------------------------------------

def test_reaching_the_ground_raises_one_possible_fall_then_the_confirmed_fall():
    det = FallDetector()
    rng = np.random.default_rng(12)
    standing, falling, on_floor = fall_frames(rng, lying=60)
    feat = calibrate(standing)
    confirmed, _ = drive(det, feat, standing + falling + on_floor)
    possible = det.drain_possible()
    assert len(possible) == 1 and possible[0].stage == "possible"
    assert len(confirmed) == 1 and confirmed[0].stage == "confirmed"
    assert possible[0].timestamp < confirmed[0].timestamp
    assert det.drain_possible() == []  # drained


def test_possible_fall_without_confirmation_when_the_person_gets_up():
    det = FallDetector()
    rng = np.random.default_rng(13)
    standing, falling, on_floor = fall_frames(rng, lying=6)
    feat = calibrate(standing)
    confirmed, _ = drive(det, feat, standing + falling + on_floor + [upper_body(jitter=1.0, rng=rng)] * 20)
    assert confirmed == [] and len(det.drain_possible()) == 1


def test_engine_emits_possible_fall_alert_and_dispatcher_keeps_it_off_telegram(tmp_config):
    from anomaly.engine import AnomalyEngine
    from events.schema import Event
    from notifications.dispatcher import NotificationDispatcher

    engine = AnomalyEngine(tmp_config)
    rng = np.random.default_rng(14)
    standing, falling, on_floor = fall_frames(rng, lying=60)
    feat = calibrate(standing)
    alerts, t = [], 0.0
    for p in standing + falling + on_floor:
        alerts += engine.process({1: p}, {1: feat}, t)
        t += 1 / FPS
    kinds = [(a.alert_type, a.severity) for a in alerts if "fall" in a.alert_type]
    assert kinds == [("possible_fall", "medium"), ("fall", "critical")]

    sent = []
    notifier = SimpleNamespace(name="fake", send=lambda n: sent.append(n), close=lambda: None)
    dispatcher = NotificationDispatcher([notifier], debounce_s=0, min_severity="low")
    try:
        possible = Event(type="possible_fall", severity="medium", start_ts=1.0, end_ts=1.0, track_id=1)
        confirmed = Event(type="fall", severity="critical", start_ts=2.0, end_ts=2.0, track_id=1)
        assert dispatcher.should_notify(possible) is False
        assert dispatcher.should_notify(confirmed) is True
        assert dispatcher.stats["dashboard_only"] == 1
    finally:
        dispatcher.close()
