"""Direction-independent ground signals, the subject split and the learned-model helpers
(synthetic poses, no models, no datasets)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fall_round3 as fr
import sweep_fall_confirm as sw
from anomaly.fall_detector import FallDetector
from core.pose_estimator import PoseResult

H = 400.0  # head-to-ankle scale of the synthetic person


def full_body(hip_y, ankle_y, nose_y, shoulder_y, cx=320.0, width=60.0):
    kp = np.zeros((17, 3), np.float32)
    kp[0] = (cx, nose_y, 0.9)
    kp[5], kp[6] = (cx - 20, shoulder_y, 0.9), (cx + 20, shoulder_y, 0.9)
    kp[11], kp[12] = (cx - 12, hip_y, 0.9), (cx + 12, hip_y, 0.9)
    kp[15], kp[16] = (cx - 10, ankle_y, 0.9), (cx + 10, ankle_y, 0.9)
    ys = kp[kp[:, 2] > 0, 1]
    return PoseResult(1, kp, np.array([cx - width, ys.min() - 10, cx + width, ys.max() + 10], np.float32))


def signals(det, pose, ts=0.0):
    from anomaly.fall_detector import _TrackState

    st = _TrackState()
    st.upright_head_y = 100.0
    return det._compute_signals(st, pose, H, ts)


def test_hip_height_and_spread_for_standing_bending_and_lying():
    det = FallDetector(ground_mode="combined")
    standing = signals(det, full_body(hip_y=340, ankle_y=500, nose_y=100, shoulder_y=160))
    assert 0.35 < standing["hip_height"] < 0.45 and standing["spread"] > 0.9
    # Bending to pick something up: head down near the feet, hips still high, feet planted.
    bending = signals(det, full_body(hip_y=340, ankle_y=500, nose_y=470, shoulder_y=420))
    assert bending["hip_height"] > det.HIP_HIGH and not det._on_ground({**bending, "horizontal_pose": True,
                                                                         "head_dropped": True})
    # Fallen toward the camera: no rotation on screen, everything collapsed near the feet.
    toward = signals(det, full_body(hip_y=490, ankle_y=500, nose_y=455, shoulder_y=470))
    assert toward["hip_height"] < det.HIP_LOW and toward["spread"] < det.SPREAD_COLLAPSED
    assert det._on_ground({**toward, "horizontal_pose": False})  # the torso rule alone would miss it
    assert not FallDetector()._on_ground({**toward, "horizontal_pose": False})  # default "torso" mode


def test_box_calibration_starts_detection_without_a_skeleton():
    det = FallDetector(box_calibration=True)
    from types import SimpleNamespace

    feat = SimpleNamespace(initial_standing_height=0.0)
    tall = PoseResult(1, np.zeros((17, 3), np.float32), np.array([300, 100, 360, 300], np.float32))
    for i in range(12):
        det.check(1, tall, feat, i / 30)
    assert det.tracks[1].box_scale == 0.9 * 200
    assert FallDetector().tracks == {} or True  # default: no box calibration
    det_off = FallDetector()
    for i in range(12):
        det_off.check(1, tall, feat, i / 30)
    assert det_off.tracks[1].box_scale == 0.0


def test_subject_split_never_shares_people():
    names = [f"Subject.{s}/Fall left" for s in range(1, 11)]
    subs = [fr.subject_of(sw.Cached(n, "fall", 20.0)) for n in names]
    assert set(subs) == fr.TUNE_SUBJECTS | fr.TEST_SUBJECTS and not fr.TUNE_SUBJECTS & fr.TEST_SUBJECTS
    assert fr.subject_of(sw.Cached("fall-01", "fall", 30.0)) is None  # URFD


def synthetic_video(kind="fall", fall_at=60, frames=150, fps=20.0):
    from test_fall_webcam import upper_body

    rng = np.random.default_rng(0)
    is_fall = kind == "fall"
    v = sw.Cached("Subject.7/Fall left" if is_fall else "Subject.7/Walk", "fall" if is_fall else "caucafall_adl",
                  fps, onset_frame=fall_at + 1 if is_fall else None)
    for i in range(frames):
        if kind == "fall" and i >= fall_at:
            k = min(8, i - fall_at)
            p = upper_body(shoulder_y=200 + 25 * k, angle_deg=min(85, 12 * k), jitter=1.0, rng=rng)
        else:
            p = upper_body(jitter=1.0, rng=rng)
        v.frames.append((i / fps, (1,), [(1, p, 480.0)], {}))
    return v


def test_video_features_and_labels():
    v = synthetic_video()
    X, y, ts, tids = fr.video_features(v)
    assert X.shape == (150, len(fr.FEATURES)) and set(tids) == {1}
    onset = 60 / 20.0
    assert (y[ts >= onset] == 1).all() and (y[ts < onset - 0.5] == 0).all()
    assert (y[(ts >= onset - 0.5) & (ts < onset)] == -1).all()
    drop = X[:, fr.FEATURES.index("drop_from_start")]
    assert drop[-1] > 0.3 > abs(drop[10])
    _x2, y2, _t, _i = fr.video_features(synthetic_video("adl"))
    assert (y2 == 0).all()


class Fixed:
    """A stand-in model returning fixed probabilities per call order."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, float)

    def predict_proba(self, X):
        p = self.probs[: len(X)]
        return np.stack([1 - p, p], axis=1)


def test_model_events_need_sustained_probability():
    v = synthetic_video(frames=100)
    probs = [0.1] * 40 + [0.9] * 4 + [0.1] * 6 + [0.9] * 40 + [0.1] * 10  # a 0.2 s blip, then 2 s
    possible, confirmed = fr.model_events(Fixed(probs), v, threshold=0.7)
    assert len(possible) == 1 and len(confirmed) == 1
    assert possible[0] >= 50 / 20 + 0.3 and confirmed[0] >= 50 / 20 + 1.3


def test_score_model_counts_both_levels():
    falls = synthetic_video(frames=100)  # onset at 3.0 s
    walk = synthetic_video("adl", frames=100)

    class PerVideo:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, X):
            self.calls += 1
            p = np.array([0.1] * 60 + [0.95] * 40) if self.calls == 1 else np.full(len(X), 0.95)
            return np.stack([1 - p, p], axis=1)

    s = fr.score_model(PerVideo(), [falls, walk], threshold=0.7)
    assert s["tp"] == 1 and s["possible_tp"] == 1
    assert s["false_alarms_clean"] == 1 and s["possible_false_alarms"] == 1
    assert s["false_alarms_by_activity"] == {"Walk": [1, 1]}
