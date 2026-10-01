"""Fall detection in laptop-webcam conditions: ankles out of view, people close to the camera,
and keypoint jitter (synthetic poses, no models)."""

from types import SimpleNamespace

import numpy as np

from anomaly.fall_detector import FallDetector, _TrackState
from core.pose_estimator import PoseEstimator, PoseResult


def upper_body(cx=320.0, shoulder_y=200.0, torso=150.0, angle_deg=0.0, jitter=0.0, rng=None, track_id=1):
    """Nose, shoulders and hips only (no knees/ankles), torso rotated by angle_deg from vertical."""
    kp = np.zeros((17, 3), dtype=np.float32)
    a = np.radians(angle_deg)
    hip = np.array([cx, shoulder_y + torso])
    up = np.array([np.sin(a), -np.cos(a)])  # hip -> shoulder direction
    sh = hip + up * torso
    nose = sh + up * 0.45 * torso
    side = np.array([np.cos(a), np.sin(a)]) * 30
    for i, p in ((0, nose), (5, sh - side), (6, sh + side), (11, hip - side * 0.7), (12, hip + side * 0.7)):
        noise = rng.normal(0, jitter, 2) if rng is not None else 0
        kp[i] = (*(p + noise), 0.9)
    xs, ys = kp[[0, 5, 6, 11, 12], 0], kp[[0, 5, 6, 11, 12], 1]
    pad = 25
    bbox = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], dtype=np.float32)
    return PoseResult(track_id=track_id, keypoints=kp, bbox=bbox)


def calibrate(poses, t0=0.0, fps=30.0):
    est = SimpleNamespace(track_features={})
    for i, p in enumerate(poses):
        PoseEstimator._update_features(est, p.track_id, p, t0 + i / fps)
    return est.track_features[poses[0].track_id]


def test_torso_angle_and_scale_without_ankles():
    p = upper_body(torso=150, angle_deg=0)
    assert p.body_height == 0.0  # the old calibration could never start
    assert abs(p.torso_angle) < 1
    assert 450 < p.scale_estimate < 520  # nose-to-hip ~217 px scaled to head-to-ankle
    assert abs(upper_body(angle_deg=90).torso_angle - 90) < 1


def test_webcam_person_gets_calibrated_from_the_torso():
    feat = calibrate([upper_body() for _ in range(12)])
    assert feat.initial_standing_height > 0


def test_bent_over_frames_are_not_used_for_calibration():
    feat = calibrate([upper_body(angle_deg=50) for _ in range(12)])
    assert feat.initial_standing_height == 0.0


def run(detector, frames, feat, fps=30.0, t0=0.0):
    events = []
    for i, p in enumerate(frames):
        e = detector.check(p.track_id, p, feat, t0 + i / fps)
        if e:
            events.append(e)
    return events


def test_upper_body_fall_toward_the_floor_is_detected():
    rng = np.random.default_rng(0)
    standing = [upper_body(jitter=1.5, rng=rng) for _ in range(30)]
    feat = calibrate(standing)
    # Fall: within ~0.4 s the torso tips over and drops ~200 px, then lies still (with jitter).
    falling = [upper_body(shoulder_y=200 + 25 * k, angle_deg=min(85, 12 * k), jitter=1.5, rng=rng)
               for k in range(1, 9)]
    lying = [upper_body(shoulder_y=400, angle_deg=85, jitter=1.5, rng=rng) for _ in range(75)]
    events = run(FallDetector(), standing + falling + lying, feat)
    assert len(events) == 1


def test_sitting_down_close_to_the_camera_is_not_a_fall():
    rng = np.random.default_rng(1)
    standing = [upper_body(jitter=1.5, rng=rng) for _ in range(30)]
    feat = calibrate(standing)
    # Sits down fast: drops 120 px, torso stays upright, box gets wide (close to the camera).
    sitting = [upper_body(shoulder_y=200 + 15 * k, torso=150 + 10 * k, jitter=1.5, rng=rng) for k in range(1, 9)]
    seated = [upper_body(shoulder_y=320, torso=230, angle_deg=5, jitter=1.5, rng=rng) for _ in range(120)]
    for p in seated:
        p.bbox[0] -= 200  # arms and desk make the box much wider than tall
        p.bbox[2] += 200
    assert run(FallDetector(), standing + sitting + seated, feat) == []


def test_jittery_lying_person_still_counts_as_still():
    """Frame-to-frame keypoint noise used to reset the stillness timer every frame."""
    rng = np.random.default_rng(2)
    feat = calibrate([upper_body() for _ in range(12)])
    det = FallDetector()
    st_frames = [upper_body(shoulder_y=400, angle_deg=85, jitter=3.0, rng=rng) for _ in range(30)]
    signals = []
    state = _TrackState()
    for i, p in enumerate(st_frames):
        signals.append(det._compute_signals(state, p, feat.initial_standing_height, i / 30))
        det._remember(state, p, i / 30)
    assert all(s["is_still"] for s in signals[20:])
