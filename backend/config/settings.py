import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    pose_model_path: str = "yolov8n-pose.pt"
    ppe_model_path: str = "models/ppe_best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    classes: List[int] = field(default_factory=lambda: [0])  # person only

@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8

@dataclass
class PoseConfig:
    sequence_length: int = 30
    num_keypoints: int = 17
    confidence_threshold: float = 0.3

@dataclass
class FallDetectorConfig:
    velocity_threshold: float = 15.0
    aspect_ratio_threshold: float = 1.2
    head_drop_ratio: float = 0.5
    stillness_frames: int = 15
    stillness_threshold: float = 2.0

@dataclass
class ZoneConfig:
    zones_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "zones.json"
    ))
    
    def load_zones(self) -> List[Dict]:
        if not os.path.exists(self.zones_file):
            print(f"DEBUG: Zones file not found at {self.zones_file}")
            return []
        with open(self.zones_file, "r") as f:
            data = json.load(f)
            return data.get("zones", [])

@dataclass
class LoiterConfig:
    time_threshold: float = 30.0
    movement_threshold: float = 50.0

@dataclass
class AnomalyConfig:
    lstm_model_path: str = "models/action_lstm.pt"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    num_action_classes: int = 7
    action_labels: List[str] = field(default_factory=lambda: [
        "walking", "running", "standing", "sitting", "fallen", "fighting", "loitering"
    ])
    anomaly_threshold: float = 0.7

@dataclass
class OutputConfig:
    db_path: str = "data/events.db"
    clips_dir: str = "data/clips"
    clip_duration: int = 10
    webhook_url: str = ""
    email_enabled: bool = False

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    ws_path: str = "/ws/feed"

@dataclass
class SentinelConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    fall: FallDetectorConfig = field(default_factory=FallDetectorConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    loiter: LoiterConfig = field(default_factory=LoiterConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    source: str = "0"
    target_fps: int = 25
    frame_width: int = 1280
    frame_height: int = 720
