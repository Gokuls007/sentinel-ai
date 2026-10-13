"""Standing-height calibration skips frames where the person fills the picture."""

from conftest import make_pose
from core.pose_estimator import PoseEstimator

W, H = 1280, 720


def estimator():
    est = PoseEstimator.__new__(PoseEstimator)  # no model: only the calibration bookkeeping
    est.track_features = {}
    est.frame_size = (W, H)
    return est


def feed(est, pose, n, t0=0.0):
    for i in range(n):
        est._update_features(1, pose, t0 + i * 0.1)


def test_frames_right_at_the_camera_are_not_used_to_calibrate():
    # Own recording, 2026-10-04: calibrated at 1007 px while filling the picture, then judged at
    # ~500 px across the room; the drop to the floor read as a slow movement.
    est = estimator()
    feed(est, make_pose(cx=640.0, top=150.0, w=300.0, h=570.0), 15)  # reaches the bottom edge
    assert est.track_features[1].initial_standing_height == 0.0
    feed(est, make_pose(cx=450.0, top=180.0, w=120.0, h=480.0), 10, t0=2.0)  # across the room
    h = est.track_features[1].initial_standing_height
    assert 0 < h < 600


def test_a_person_inside_the_picture_calibrates_as_before():
    est = estimator()
    feed(est, make_pose(cx=450.0, top=180.0, w=120.0, h=480.0), 10)
    assert est.track_features[1].initial_standing_height > 0


def test_close_with_both_feet_in_view_still_calibrates():
    # URFD: people are often over half the frame tall; with the feet seen, the height is measured.
    est = estimator()
    feed(est, make_pose(cx=640.0, top=0.0, w=200.0, h=690.0), 10)  # top edge, ankles at y=690
    assert est.track_features[1].initial_standing_height > 0

