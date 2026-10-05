"""RTMPose-m pose estimation (rtmlib + onnxruntime-gpu), a drop-in for the YOLO pose model.

RTMPose is top-down: it estimates one pose per person box, using the person tracker's boxes
(YOLOv8 + ByteTrack), so every tracked person gets a pose and nothing needs matching. Output is
the same ``PoseResult`` as the YOLO pose model (17 COCO keypoints, x/y/confidence), so all the
code downstream is unchanged. Confidences are SimCC scores, clipped to 0-1.

Runtime: onnxruntime-gpu built for CUDA 12 (1.23.x), reusing PyTorch's CUDA/cuDNN libraries
(PyTorch is imported first). If rtmlib, onnxruntime or the weights are missing, construction
raises and the pipeline falls back to the YOLO pose model, saying why.
"""

from __future__ import annotations

import logging

import numpy as np

from core.pose_estimator import PoseEstimator, PoseResult

logger = logging.getLogger("sentinel.pose.rtmpose")

RTMPOSE_M = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")


class RTMPoseEstimator(PoseEstimator):
    """Same interface as ``PoseEstimator`` (estimate, features, history); RTMPose-m inside."""

    name = "rtmpose-m"

    def __init__(self, model: str = RTMPOSE_M, input_size: tuple[int, int] = (192, 256), sequence_length: int = 30,
                 device: str = "auto"):
        import torch  # first: onnxruntime then reuses PyTorch's CUDA/cuDNN libraries
        from rtmlib import RTMPose

        self.seq_len = sequence_length
        self.conf_threshold = 0.0
        use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        self.model = RTMPose(onnx_model=model, model_input_size=input_size, backend="onnxruntime",
                             device="cuda" if use_cuda else "cpu")
        session = getattr(self.model, "session", None)
        providers = session.get_providers() if session is not None else []
        self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        if use_cuda and self.device != "cuda":
            logger.warning("RTMPose is running on the CPU (onnxruntime has no CUDA provider)")
        self.track_features = {}
        self.stale_timeout = 10.0
        logger.info("RTMPose-m ready on %s", self.device)

    def estimate(self, frame: np.ndarray, track_ids: list[int], bboxes: list[np.ndarray],
                 timestamp: float) -> dict[int, PoseResult]:
        results: dict[int, PoseResult] = {}
        if track_ids:  # rtmlib treats "no boxes" as "the whole image": never call it empty
            boxes = [np.asarray(b, float).tolist() for b in bboxes]
            kps, scores = self.model(frame, bboxes=boxes)
            for tid, box, kp, sc in zip(track_ids, boxes, kps, scores, strict=True):
                keypoints = np.concatenate([np.asarray(kp, np.float32).reshape(17, 2),
                                            np.clip(np.asarray(sc, np.float32).reshape(17, 1), 0.0, 1.0)], axis=1)
                pose = PoseResult(track_id=tid, keypoints=keypoints, bbox=np.asarray(box, np.float32))
                results[tid] = pose
                self._update_features(tid, pose, timestamp)
        self._cleanup_stale(timestamp)
        return results
