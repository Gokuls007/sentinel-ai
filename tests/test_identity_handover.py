"""A new track id for the same person keeps that person's calibration and fall state.

Own recording (2026-10-06): the person behind the desk went from track 3 to track 8 while out of
view. Track 8 had no standing height, so it could never be judged for a fall."""

from types import SimpleNamespace

import numpy as np

from anomaly.fall_detector import FallDetector, _TrackState
from conftest import make_features, make_pose
from core.pipeline import SentinelPipeline

DT = 0.04
STAND = dict(top=200.0, w=60.0, h=180.0)
LIE = dict(top=330.0, w=180.0, h=60.0, lying=True)


def run(det, tid, feat, params, t, seconds):
    events = []
    for _ in range(round(seconds / DT)):
        ev = det.check(tid, make_pose(track_id=tid, **params), feat, t)
        if ev:
            events.append(ev)
        t += DT
    return events, t


def calibrated(det, tid=1, t=0.0):
    feat = make_features(track_id=tid)
    _, t = run(det, tid, feat, STAND, t, 1.0)
    return feat, t


def test_inherited_person_found_lying_is_a_fall_not_seen():
    det = FallDetector()
    _old_feat, t = calibrated(det, tid=3)
    det.inherit(8, 3, t)  # track 3 vanished; 8 appears where it was, already on the floor
    events, t = run(det, 8, make_features(track_id=8), LIE, t, 4.0)
    assert len(events) == 1 and events[0].signals.get("fall_not_seen") is True
    possible = det.drain_possible()
    assert len(possible) == 1 and possible[0].signals.get("fall_not_seen") is True


def test_inherited_person_found_standing_is_nothing():
    det = FallDetector()
    _, t = calibrated(det, tid=3)
    det.inherit(8, 3, t)
    events, _ = run(det, 8, make_features(track_id=8), STAND, t, 5.0)
    assert events == [] and det.state_of(8) == FallDetector.UPRIGHT


def test_found_on_floor_only_right_after_the_hand_over():
    det = FallDetector()
    _, t = calibrated(det, tid=3)
    det.inherit(8, 3, t)
    _, t = run(det, 8, make_features(track_id=8), STAND, t, FallDetector.FOUND_WINDOW_S + 0.5)
    events, _ = run(det, 8, make_features(track_id=8), dict(STAND, top=230.0), t, 2.0)  # later: normal rules
    assert events == []


def test_a_confirmed_fall_carries_over_without_a_second_alert():
    det = FallDetector()
    feat, t = calibrated(det, tid=3)
    for _ in range(5):  # fast drop
        det.check(3, make_pose(track_id=3, **LIE), feat, t)
        t += DT
    first, t = run(det, 3, feat, LIE, t, 3.0)
    assert len(first) == 1 and det.state_of(3) == FallDetector.CONFIRMED
    det.inherit(8, 3, t)
    assert det.state_of(8) == FallDetector.CONFIRMED
    again, _ = run(det, 8, make_features(track_id=8), LIE, t, 5.0)
    assert again == []  # same fall, new id: not re-alerted (cooldown carried over)


def stub_pipeline():
    det = FallDetector()
    return SimpleNamespace(
        anomaly_engine=SimpleNamespace(fall_detector=det),
        _last_boxes={}, _handover_checked=set(),
        HANDOVER_S=SentinelPipeline.HANDOVER_S, HANDOVER_NEW_S=SentinelPipeline.HANDOVER_NEW_S,
    ), det


def step(p, poses, features, ts):
    SentinelPipeline._hand_over_identities(p, poses, features, ts)


def test_pipeline_hands_over_to_an_overlapping_new_id():
    p, det = stub_pipeline()
    old = make_features(track_id=3, standing_height=171.0)
    det.tracks[3] = _TrackState()
    det.tracks[3].floor_y = 380.0
    step(p, {3: make_pose(track_id=3, **STAND)}, {3: old}, 10.0)
    new = make_features(track_id=8, standing_height=0.0, first_seen=11.0)
    step(p, {8: make_pose(track_id=8, **LIE)}, {3: old, 8: new}, 11.2)  # track 3 gone, 8 where it was
    assert new.initial_standing_height == 171.0
    assert det.tracks[8].floor_y == 380.0 and det.tracks[8].inherited_at == 11.2


def test_pipeline_does_not_hand_over_far_away_or_old_or_to_calibrated_ids():
    p, _det = stub_pipeline()
    old = make_features(track_id=3, standing_height=171.0)
    step(p, {3: make_pose(track_id=3, **STAND)}, {3: old}, 10.0)
    far = make_features(track_id=8, standing_height=0.0, first_seen=11.0)
    step(p, {8: make_pose(track_id=8, cx=1000.0, **STAND)}, {3: old, 8: far}, 11.0)
    assert far.initial_standing_height == 0.0  # no overlap: a different person
    late = make_features(track_id=9, standing_height=0.0, first_seen=30.0)
    step(p, {9: make_pose(track_id=9, **STAND)}, {3: old, 9: late}, 30.0)
    assert late.initial_standing_height == 0.0  # the old track vanished too long ago
    known = make_features(track_id=10, standing_height=150.0, first_seen=11.0)
    step(p, {10: make_pose(track_id=10, **STAND)}, {3: old, 10: known}, 11.1)
    assert known.initial_standing_height == 150.0  # already calibrated: keeps its own
    assert np.isfinite(old.initial_standing_height)
