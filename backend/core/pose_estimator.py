import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO

# Constants
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KEYPOINT_MIN_CONF = 0.3  # below this a keypoint is treated as not visible


@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray  # (17, 3) -> x, y, conf  (undetected keypoints are (0, 0, 0))
    bbox: np.ndarray  # [x1, y1, x2, y2]

    def _visible(self, idx: int) -> bool:
        return float(self.keypoints[idx, 2]) >= KEYPOINT_MIN_CONF

    def _mean_visible(self, a: int, b: int) -> np.ndarray | None:
        pts = [self.keypoints[i, :2] for i in (a, b) if self._visible(i)]
        return np.mean(pts, axis=0) if pts else None

    @property
    def bbox_center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def mid_hip(self) -> np.ndarray:
        """Hip centre; falls back to the bbox centre when both hips are occluded."""
        hip = self._mean_visible(LEFT_HIP, RIGHT_HIP)
        return hip if hip is not None else self.bbox_center

    @property
    def mid_shoulder(self) -> np.ndarray:
        sh = self._mean_visible(LEFT_SHOULDER, RIGHT_SHOULDER)
        return sh if sh is not None else self.bbox_center

    @property
    def head_valid(self) -> bool:
        return self._visible(NOSE)

    @property
    def head_y(self) -> float:
        """Nose y; falls back to the top of the bbox when the face isn't visible."""
        return float(self.keypoints[NOSE, 1]) if self.head_valid else float(self.bbox[1])

    @property
    def body_height(self) -> float:
        """Head-to-ankle height in px, or 0.0 if head or both ankles aren't visible."""
        ankle = self._mean_visible(LEFT_ANKLE, RIGHT_ANKLE)
        if ankle is None or not self.head_valid:
            return 0.0
        return abs(float(ankle[1]) - self.head_y)

@dataclass
class TrackFeatures:
    track_id: int
    pose_sequence: deque = field(default_factory=lambda: deque(maxlen=30))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    first_seen: float = field(default_factory=time.time)
    initial_standing_height: float = 0.0
    _height_samples: list[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    @property
    def speed(self) -> float:
        if len(self.centroid_history) < 2:
            return 0.0
        p1 = self.centroid_history[-2]
        p2 = self.centroid_history[-1]
        t1 = self.timestamps[-2]
        t2 = self.timestamps[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist = np.linalg.norm(p2 - p1)
        return dist / dt

    @property
    def displacement(self) -> float:
        if len(self.centroid_history) < 1:
            return 0.0
        return np.linalg.norm(self.centroid_history[-1] - self.centroid_history[0])

    @property
    def time_tracked(self) -> float:
        """Seconds between first and latest sighting, in the frames' own clock.

        Using time.time() here broke video-file demos, whose timestamps start at 0.
        """
        if not self.timestamps:
            return 0.0
        return float(self.timestamps[-1] - self.first_seen)


    @property
    def direction(self) -> np.ndarray | None:
        if len(self.centroid_history) < 5:
            return None
        v = self.centroid_history[-1] - self.centroid_history[-5]
        mag = np.linalg.norm(v)
        return v / mag if mag > 0 else None

    def get_pose_tensor(self) -> np.ndarray | None:
        if len(self.pose_sequence) < 10:
            return None
        # Return (seq_len, 17, 2)
        return np.array([p[:, :2] for p in self.pose_sequence])

    def get_flat_tensor(self) -> np.ndarray | None:
        tensor = self.get_pose_tensor()
        if tensor is None:
            return None
        # Flatten to (seq_len, 34)
        return tensor.reshape(len(self.pose_sequence), 34)

class PoseEstimator:
    def __init__(self, model_path: str = "yolov8n-pose.pt", sequence_length: int = 30, 
                 confidence_threshold: float = 0.3, device: str = "auto"):
        
        self.conf_threshold = confidence_threshold
        self.seq_len = sequence_length
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Initializing PoseEstimator on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.track_features: dict[int, TrackFeatures] = {}
        self.stale_timeout = 10.0

    def estimate(self, frame: np.ndarray, track_ids: list[int], bboxes: list[np.ndarray], 
                 timestamp: float) -> dict[int, PoseResult]:
        
        # Run pose model on full frame
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        if not results or results[0].keypoints is None:
            return {}

        pose_data = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
        pose_bboxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        
        results_dict = {}
        
        # Match poses to our tracked persons via IoU
        for tid, det_bbox in zip(track_ids, bboxes, strict=True):
            best_iou = 0.3
            best_idx = -1
            
            for i, p_bbox in enumerate(pose_bboxes):
                iou = self._compute_iou(det_bbox, p_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx != -1:
                pose_kpts = pose_data[best_idx]
                pose_res = PoseResult(track_id=tid, keypoints=pose_kpts, bbox=pose_bboxes[best_idx])
                results_dict[tid] = pose_res
                self._update_features(tid, pose_res, timestamp)
        
        self._cleanup_stale(timestamp)
        return results_dict

    def _compute_iou(self, box1, box2) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _update_features(self, track_id: int, pose: PoseResult, timestamp: float):
        if track_id not in self.track_features:
            self.track_features[track_id] = TrackFeatures(track_id=track_id, first_seen=timestamp)
            
        feat = self.track_features[track_id]
        feat.pose_sequence.append(pose.keypoints)
        feat.centroid_history.append(pose.mid_hip)
        feat.timestamps.append(timestamp)
        feat.last_updated = timestamp
        
        # Calibration: median head-to-ankle height over 10 upright frames with a
        # fully visible body (lying/occluded frames would give a wrong reference).
        bw, bh = pose.bbox[2] - pose.bbox[0], pose.bbox[3] - pose.bbox[1]
        upright = bh > 0 and bw / bh < 1.0
        if feat.initial_standing_height == 0.0 and upright and pose.body_height > 0:
            feat._height_samples.append(pose.body_height)
            if len(feat._height_samples) >= 10:
                feat.initial_standing_height = float(np.median(feat._height_samples))
                # print(f"DEBUG: Calibrated Track {track_id} height to {feat.initial_standing_height:.1f}px")

    def _cleanup_stale(self, current_time: float):
        to_remove = [tid for tid, feat in self.track_features.items() 
                     if current_time - feat.last_updated > self.stale_timeout]
        for tid in to_remove:
            del self.track_features[tid]
            # print(f"DEBUG: Cleaned up stale track {tid}")

    def get_features(self, track_id: int) -> TrackFeatures | None:
        return self.track_features.get(track_id)

    def get_all_features(self) -> dict[int, TrackFeatures]:
        return self.track_features
