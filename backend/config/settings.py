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
    # Speeds are in "standing body heights per second", so the thresholds do not
    # depend on camera resolution, distance to the camera, or frame rate.
    descent_speed_threshold: float = 1.2   # hip moving down this fast starts a fall
    aspect_ratio_threshold: float = 1.2    # bbox width/height above this = lying down
    head_drop_ratio: float = 0.5           # head this far (x body height) below standing
    stillness_seconds: float = 1.0         # lying still this long confirms the fall
    stillness_speed_threshold: float = 0.15
    fallen_timeout_seconds: float = 5.0    # gave up waiting for stillness -> upright
    cooldown_seconds: float = 30.0         # one fall alert per person per 30 s

@dataclass
class ZoneConfig:
    zones_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "zones.json"
    ))
    # Minimum time between repeated alerts for the same person in the same zone.
    alert_cooldown: float = 30.0
    
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
    loop: bool = False  # restart video files when they end (demo / kiosk mode)
    target_fps: int = 25
    frame_width: int = 1280
    frame_height: int = 720
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ])

    @classmethod
    def from_env(cls, env_file: Optional[str] = ".env") -> "SentinelConfig":
        """Build a config from environment variables (and an optional .env file).

        Every variable documented in .env.example is honoured; anything unset keeps
        the dataclass default. CLI flags in main.py override these afterwards.
        """
        if env_file:
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=False)
            except ImportError:
                pass

        def env(name, cast=str, default=None):
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            if cast is bool:
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return cast(raw)

        cfg = cls()
        cfg.source = env("VIDEO_SOURCE", str, cfg.source)
        cfg.loop = env("LOOP_VIDEO", bool, cfg.loop)
        cfg.target_fps = env("TARGET_FPS", int, cfg.target_fps)
        cfg.frame_width = env("FRAME_WIDTH", int, cfg.frame_width)
        cfg.frame_height = env("FRAME_HEIGHT", int, cfg.frame_height)

        cfg.detector.model_path = env("DETECTION_MODEL", str, cfg.detector.model_path)
        cfg.detector.pose_model_path = env("POSE_MODEL", str, cfg.detector.pose_model_path)
        cfg.detector.confidence_threshold = env("DETECTION_CONFIDENCE", float,
                                                cfg.detector.confidence_threshold)
        cfg.detector.iou_threshold = env("IOU_THRESHOLD", float, cfg.detector.iou_threshold)
        if env("FORCE_CPU", bool, False):
            cfg.detector.device = "cpu"

        cfg.anomaly.lstm_model_path = env("LSTM_MODEL_PATH", str, cfg.anomaly.lstm_model_path)
        cfg.anomaly.anomaly_threshold = env("ANOMALY_THRESHOLD", float,
                                            cfg.anomaly.anomaly_threshold)
        cfg.zone.zones_file = env("ZONES_FILE", str, cfg.zone.zones_file)

        cfg.output.db_path = env("DB_PATH", str, cfg.output.db_path)
        cfg.output.clips_dir = env("CLIPS_DIR", str, cfg.output.clips_dir)
        cfg.output.clip_duration = env("CLIP_DURATION", int, cfg.output.clip_duration)
        cfg.output.webhook_url = env("WEBHOOK_URL", str, cfg.output.webhook_url)

        cfg.server.host = env("HOST", str, cfg.server.host)
        cfg.server.port = env("PORT", int, cfg.server.port)
        origins = env("CORS_ORIGINS", str, None)
        if origins:
            cfg.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        return cfg
