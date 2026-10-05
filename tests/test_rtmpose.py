"""RTMPose-m backend (rtmlib mocked): same PoseResult format, top-down on the tracker's boxes,
and a fallback to the YOLO pose model with the reason when it can't load."""

import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest


class FakeRTMPose:
    calls: ClassVar[list] = []

    def __init__(self, onnx_model, model_input_size, backend, device):
        self.session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"] if device == "cuda" else
                                       ["CPUExecutionProvider"])

    def __call__(self, image, bboxes=()):
        FakeRTMPose.calls.append(list(bboxes))
        n = len(bboxes)
        kps = np.stack([np.tile([[b[0] + 5, b[1] + 5]], (17, 1)) for b in bboxes]) if n else np.zeros((0, 17, 2))
        scores = np.full((n, 17), 1.3)  # SimCC scores can pass 1: clipped
        return kps, scores


@pytest.fixture
def rtm(monkeypatch):
    FakeRTMPose.calls = []
    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(RTMPose=FakeRTMPose))
    from core.rtmpose_estimator import RTMPoseEstimator

    return RTMPoseEstimator(device="cpu")


def test_one_pose_per_tracked_box_in_the_usual_format(rtm):
    frame = np.zeros((480, 640, 3), np.uint8)
    out = rtm.estimate(frame, [7, 9], [np.array([10, 20, 110, 320]), np.array([300, 40, 400, 400])], 1.0)
    assert sorted(out) == [7, 9]
    p = out[7]
    assert p.keypoints.shape == (17, 3) and p.keypoints[0].tolist() == [15.0, 25.0, 1.0]  # x, y, conf (clipped)
    assert p.bbox.tolist() == [10, 20, 110, 320] and p.track_id == 7
    assert 7 in rtm.get_all_features()  # features/history updated like the YOLO estimator


def test_no_boxes_means_no_call(rtm):
    assert rtm.estimate(np.zeros((10, 10, 3), np.uint8), [], [], 1.0) == {}
    assert FakeRTMPose.calls == []  # rtmlib would treat "no boxes" as the whole image


def test_pipeline_falls_back_to_yolo_pose_with_the_reason(monkeypatch, tmp_config):
    from core import pipeline as pl

    monkeypatch.setitem(sys.modules, "rtmlib", None)  # import fails
    made = []
    monkeypatch.setattr(pl, "PoseEstimator", lambda **kw: made.append(kw) or SimpleNamespace(name="yolo"))
    tmp_config.pose.backend = "rtmpose"
    _est, status = pl.SentinelPipeline._make_pose_estimator(tmp_config)
    assert made and status.startswith("pose: yolov8n-pose (rtmpose off:")
    tmp_config.pose.backend = "yolo"
    _est, status = pl.SentinelPipeline._make_pose_estimator(tmp_config)
    assert status == "pose: yolov8n-pose"


def test_pipeline_uses_rtmpose_when_it_loads(monkeypatch, tmp_config):
    from core import pipeline as pl

    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(RTMPose=FakeRTMPose))
    tmp_config.pose.backend = "rtmpose"
    est, status = pl.SentinelPipeline._make_pose_estimator(tmp_config)
    assert est.name == "rtmpose-m" and status.startswith("pose: rtmpose-m")
