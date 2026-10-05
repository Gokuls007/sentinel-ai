import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.pose_estimator import PoseResult, TrackFeatures


def make_pose(track_id=1, cx=320.0, top=200.0, w=60.0, h=180.0, head_visible=True,
              lying=False) -> PoseResult:
    """A synthetic 17-keypoint pose inside the bbox (cx, top, w, h).

    Standing: nose near the top, hips at 55 % height, ankles at the bottom.
    Lying: the body is horizontal, so every keypoint sits near the bbox's vertical centre.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    x1, x2, y2 = cx - w / 2, cx + w / 2, top + h
    if lying:
        ys = {"nose": top + 0.3 * h, "shoulder": top + 0.4 * h, "hip": top + 0.55 * h, "ankle": top + 0.7 * h}
        xs = {"nose": x1 + 0.05 * w, "shoulder": x1 + 0.25 * w, "hip": cx, "ankle": x2 - 0.05 * w}
    else:
        ys = {"nose": top + 0.05 * h, "shoulder": top + 0.25 * h, "hip": top + 0.55 * h, "ankle": y2}
        xs = {"nose": cx, "shoulder": cx, "hip": cx, "ankle": cx}
    parts = {"nose": [0], "shoulder": [5, 6], "hip": [11, 12], "ankle": [15, 16]}
    for name, idxs in parts.items():
        for i in idxs:
            kp[i] = (xs[name], ys[name], 0.9)
    if not head_visible:
        kp[0] = 0
    return PoseResult(track_id=track_id, keypoints=kp, bbox=np.array([x1, top, x2, y2], dtype=np.float32))


def make_features(track_id=1, standing_height=171.0, first_seen=0.0) -> TrackFeatures:
    feat = TrackFeatures(track_id=track_id, first_seen=first_seen)
    feat.initial_standing_height = standing_height
    return feat


@pytest.fixture
def tmp_config(tmp_path):
    """A SentinelConfig whose files all live in a temp dir (no .env, no real zones)."""
    from config.settings import SentinelConfig

    cfg = SentinelConfig()
    cfg.zone.zones_file = str(tmp_path / "zones.json")
    cfg.output.db_path = str(tmp_path / "events.db")
    cfg.output.clips_dir = str(tmp_path / "clips")
    cfg.anomaly.lstm_model_path = str(tmp_path / "missing_lstm.pt")
    cfg.objects.enabled = False  # YOLO-World is mocked where it is tested (no weights in CI)
    cfg.pose.backend = "yolo"  # RTMPose is mocked where it is tested (no rtmlib in CI)
    cfg.objects.cache_dir = str(tmp_path / "cache")
    return cfg
