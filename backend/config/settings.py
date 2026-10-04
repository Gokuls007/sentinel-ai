import json
import os
from dataclasses import dataclass, field

from ergonomics.config import ErgonomicsConfig


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    pose_model_path: str = "yolov8n-pose.pt"
    ppe_model_path: str = "models/ppe_best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    classes: list[int] = field(default_factory=lambda: [0])  # person only

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
    # Keeping a fallen person who the detector loses (lying people are often missed).
    # Measured one by one in training/sweep_fall_confirm.py; 0 / False = off.
    lost_hold_seconds: float = 0.0         # hold a lost falling/fallen track at its last spot
    recovery_low_conf: float = 0.0         # >0: retry pose near the last box down to this confidence
    recovery_rotated: bool = False         # also retry on that region rotated 90 degrees
    recovery_window_seconds: float = 3.0   # retries only this long after the fall started
    recovery_mode: str = "pose"            # re-found person: "pose" (normal check) or "presence"
    upright_hold_seconds: float = 0.0      # "looks upright" must last this long to leave the ground
    # "combined" (default since 2026-10-02): torso rule plus direction-independent signals
    # (hips near the floor, skeleton collapse), vetoed while the hips are still high (bending
    # over). "torso": the earlier torso-angle-only rule. See docs/BENCHMARKS.md, unseen subjects.
    ground_mode: str = "combined"
    box_calibration: bool = False          # standing height from upright boxes when no skeleton

@dataclass
class ZoneConfig:
    zones_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "zones.json"
    ))
    # Minimum time between repeated alerts for the same person in the same zone.
    alert_cooldown: float = 30.0
    
    def load_zones(self) -> list[dict]:
        if not os.path.exists(self.zones_file):
            print(f"DEBUG: Zones file not found at {self.zones_file}")
            return []
        with open(self.zones_file) as f:
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
    action_labels: list[str] = field(default_factory=lambda: [
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
class NotificationConfig:
    # At most one notification per (event type, track_id) within this many seconds.
    debounce_s: float = 60.0
    # Events below this severity are not sent (low | medium | high | critical).
    min_severity: str = "medium"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: list[str] = field(default_factory=list)
    smtp_starttls: bool = True
    smtp_ssl: bool = False

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_from and self.smtp_to)


@dataclass
class LLMConfig:
    # Used only by features that answer a request (search); never in the per-frame loop.
    provider: str = "nvidia"  # nvidia | anthropic
    model: str = ""  # empty = the provider's default (llm/factory.py)
    nvidia_api_key: str = ""
    anthropic_api_key: str = ""
    base_url: str = ""  # override the OpenAI-compatible endpoint (nvidia provider)
    effort: str = "medium"  # anthropic only: low | medium | high
    timeout_s: float = 90.0


@dataclass
class SearchConfig:
    max_steps: int = 6  # model turns per question (each may call several tools)
    max_tokens_per_question: int = 60_000
    # Questions per minute, all clients together (each one costs API calls).
    rate_limit_per_min: int = 10
    # Allow search from other machines on the network (off: this computer only).
    allow_remote: bool = False


@dataclass
class RulesConfig:
    # Fall and restricted-zone alerts as built-in rules instead of the hard-coded alerts. Off
    # until the parity report (scripts/rules_parity.py) shows they match on the sample videos.
    builtins: bool = False
    # Rule compilations per minute, all clients together (each one costs an LLM call or two).
    compile_rate_limit_per_min: int = 10


DEFAULT_OBJECT_CLASSES = ["cardboard box", "chair", "ladder", "backpack", "hard hat", "safety vest", "forklift"]


@dataclass
class ObjectsConfig:
    """Open-vocabulary objects (YOLO-World) in Warehouse mode, next to the person tracker and pose."""
    enabled: bool = True
    model_path: str = "yolov8s-worldv2.pt"
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_OBJECT_CLASSES))
    confidence: float = 0.3  # open-vocabulary scores run low; below ~0.3 a soft bag can read as a box
    every_n_frames: int = 1  # run on every Nth frame (the last result is reused in between)
    # Several prompts per class (YOLO-World is sensitive to wording); the best one wins per object.
    synonyms: dict[str, list[str]] = field(default_factory=lambda: {
        "chair": ["chair", "office chair", "gaming chair", "wooden chair"],
        "cardboard box": ["cardboard box", "shipping box", "carton"],
    })
    # Per-class confidence floors (others use ``confidence``); set from the smoke test.
    floors: dict[str, float] = field(default_factory=lambda: {"chair": 0.25, "cardboard box": 0.25})
    vote_window: int = 15  # frames: the class shown is the one given most often over this window
    min_hits: int = 3      # sightings before an object is shown (drops one-frame false hits)
    imgsz: int = 640
    cache_dir: str = "data/cache"  # encoded class names (CLIP runs only when the list changes)


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
    ergonomics: ErgonomicsConfig = field(default_factory=ErgonomicsConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    
    source: str = "0"
    camera_id: str = "cam-0"  # recorded on every event; one pipeline = one camera
    # Allow starting the laptop webcam from other machines on the network (off: this computer only).
    allow_remote_camera_control: bool = False
    # What "My Camera" opens instead of a webcam index, e.g. an IP camera URL or a video file
    # (files loop). Empty = the device index chosen in the dashboard.
    laptop_camera_source: str = ""
    loop: bool = False  # restart video files when they end (demo / kiosk mode)
    target_fps: int = 25
    frame_width: int = 1280
    frame_height: int = 720
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ])

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "SentinelConfig":
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
        cfg.objects.enabled = env("OBJECTS_ENABLED", bool, cfg.objects.enabled)
        cfg.objects.model_path = env("OBJECT_MODEL", str, cfg.objects.model_path)
        cfg.objects.confidence = env("OBJECT_CONFIDENCE", float, cfg.objects.confidence)
        cfg.objects.every_n_frames = max(1, env("OBJECT_EVERY_N_FRAMES", int, cfg.objects.every_n_frames))
        raw_syn = env("OBJECT_SYNONYMS", str, "")  # JSON, e.g. {"chair": ["chair", "stool"]}
        if raw_syn:
            cfg.objects.synonyms = {str(k): [str(x) for x in v] for k, v in json.loads(raw_syn).items()}
        raw_floors = env("OBJECT_FLOORS", str, "")  # e.g. chair=0.25,cardboard box=0.25
        if raw_floors:
            cfg.objects.floors = {k.strip(): float(v) for k, v in
                                  (pair.split("=", 1) for pair in raw_floors.split(",") if "=" in pair)}
        raw_classes = env("OBJECT_CLASSES", str, "")
        if raw_classes:
            cfg.objects.classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
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
        cfg.camera_id = env("CAMERA_ID", str, cfg.camera_id)
        cfg.allow_remote_camera_control = env("ALLOW_REMOTE_CAMERA_CONTROL", bool, False)
        cfg.laptop_camera_source = env("LAPTOP_CAMERA_SOURCE", str, "")

        n = cfg.notifications
        n.debounce_s = env("NOTIFY_DEBOUNCE_S", float, n.debounce_s)
        n.min_severity = env("NOTIFY_MIN_SEVERITY", str, n.min_severity)
        n.telegram_bot_token = env("TELEGRAM_BOT_TOKEN", str, n.telegram_bot_token)
        n.telegram_chat_id = env("TELEGRAM_CHAT_ID", str, n.telegram_chat_id)
        n.smtp_host = env("SMTP_HOST", str, n.smtp_host)
        n.smtp_port = env("SMTP_PORT", int, n.smtp_port)
        n.smtp_user = env("SMTP_USER", str, n.smtp_user)
        n.smtp_password = env("SMTP_PASSWORD", str, n.smtp_password)
        n.smtp_from = env("SMTP_FROM", str, n.smtp_from)
        recipients = env("SMTP_TO", str, None)
        if recipients:
            n.smtp_to = [r.strip() for r in recipients.split(",") if r.strip()]
        n.smtp_starttls = env("SMTP_STARTTLS", bool, n.smtp_starttls)
        n.smtp_ssl = env("SMTP_SSL", bool, n.smtp_ssl)

        llm = cfg.llm
        llm.provider = env("LLM_PROVIDER", str, llm.provider).strip().lower()
        llm.model = env("LLM_MODEL", str, llm.model)
        llm.nvidia_api_key = env("NVIDIA_API_KEY", str, llm.nvidia_api_key)
        llm.anthropic_api_key = env("ANTHROPIC_API_KEY", str, llm.anthropic_api_key)
        llm.base_url = env("LLM_BASE_URL", str, llm.base_url)
        llm.effort = env("LLM_EFFORT", str, llm.effort)
        llm.timeout_s = env("LLM_TIMEOUT_S", float, llm.timeout_s)
        s = cfg.search
        s.max_steps = env("SEARCH_MAX_STEPS", int, s.max_steps)
        s.max_tokens_per_question = env("SEARCH_MAX_TOKENS", int, s.max_tokens_per_question)
        s.rate_limit_per_min = env("SEARCH_RATE_LIMIT_PER_MIN", int, s.rate_limit_per_min)
        s.allow_remote = env("ALLOW_REMOTE_SEARCH", bool, s.allow_remote)
        cfg.rules.builtins = env("RULES_BUILTINS", bool, cfg.rules.builtins)
        cfg.rules.compile_rate_limit_per_min = env("RULES_COMPILE_RATE_LIMIT_PER_MIN", int,
                                                   cfg.rules.compile_rate_limit_per_min)

        cfg.server.host = env("HOST", str, cfg.server.host)
        cfg.server.port = env("PORT", int, cfg.server.port)
        origins = env("CORS_ORIGINS", str, None)
        if origins:
            cfg.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        return cfg
