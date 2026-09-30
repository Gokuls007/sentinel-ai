import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import cv2
import numpy as np

from anomaly.engine import AnomalyEngine
from config.settings import SentinelConfig
from output.clip_recorder import ClipRecorder
from output.event_logger import EventLogger
from output.webhook import WebhookNotifier

from .detector import Detector, FrameDetections
from .pose_estimator import PoseEstimator, PoseResult
from .utils import to_serializable
from .video_source import VideoSource

logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    frame: np.ndarray
    timestamp: float
    detections: FrameDetections
    poses: dict[int, PoseResult]
    alerts: list = field(default_factory=list) # Will hold AnomalyAlert later
    annotated_frame: np.ndarray | None = None
    annotated_frame_base64: str | None = None # For pre-optimized transmission
    processing_time_ms: float = 0.0
    total_alerts: int = 0
    frame_number: int = 0

    def to_dict(self) -> dict:
        data = {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "processing_time_ms": self.processing_time_ms,
            "detections": self.detections.to_dict(),
            "alerts": [a.to_dict() if hasattr(a, 'to_dict') else str(a) for a in self.alerts],
            "stats": {
                "person_count": self.detections.person_count,
                "active_tracks": len(self.poses),
                "alert_count": self.total_alerts,
                "processing_time_ms": self.processing_time_ms
            }
        }
        # The JPEG is sent once, as the top-level "image" field of the WebSocket message.
        return to_serializable(data)

class SentinelPipeline:
    SKELETON: ClassVar[list[tuple[int, int]]] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
        (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.frame_count = 0
        self.total_alerts = 0
        self.start_time = time.time()
        
        self._on_alert: Callable | None = None
        self._on_frame: Callable | None = None
        
        # Instantiate layers
        self.video_source = VideoSource(
            source=config.source,
            target_fps=config.target_fps,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
            loop=config.loop,
        )
        
        self.detector = Detector(
            model_path=config.detector.model_path,
            confidence_threshold=config.detector.confidence_threshold,
            iou_threshold=config.detector.iou_threshold,
            device=config.detector.device,
            classes=config.detector.classes
        )
        
        self.pose_estimator = PoseEstimator(
            model_path=config.detector.pose_model_path,
            sequence_length=config.pose.sequence_length,
            confidence_threshold=config.pose.confidence_threshold,
            device=config.detector.device
        )
        
        self.anomaly_engine = AnomalyEngine(config)
        
        # Output layer
        self.event_logger = EventLogger(config.output.db_path)
        self.clip_recorder = ClipRecorder(
            clips_dir=config.output.clips_dir,
            buffer_seconds=config.output.clip_duration,
            fps=config.target_fps
        )
        self.webhook = WebhookNotifier(config.output.webhook_url) if config.output.webhook_url else None

        logger.info("Pipeline initialized successfully")

    def on_alert(self, callback: Callable):
        self._on_alert = callback
        return callback

    def on_frame(self, callback: Callable):
        self._on_frame = callback
        return callback

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        self.frame_count += 1
        processing_start = time.time()
        
        # 1. Detection & Tracking
        detections = self.detector.detect_and_track(frame)
        
        # 2. Pose Estimation
        person_detections = [d for d in detections.detections if d.class_name == "person" and d.track_id is not None]
        track_ids = [d.track_id for d in person_detections]
        bboxes = [d.bbox for d in person_detections]
        poses = self.pose_estimator.estimate(frame, track_ids, bboxes, timestamp)
        
        # 3. Anomaly Detection
        all_features = self.pose_estimator.get_all_features()
        alerts = self.anomaly_engine.process(poses, all_features, timestamp)
        
        # 4. Annotation (clips and snapshots use the annotated frame)
        annotated_frame = self._annotate_frame(frame.copy(), detections, poses, alerts)

        # 5. Persistence & Callbacks
        for alert in alerts:
            self.total_alerts += 1
            # Forensic clip (pre + post alert) is encoded in the background.
            clip_path = self.clip_recorder.save_clip(alert.alert_id, alert.timestamp,
                                                     snapshot=annotated_frame)
            alert.details = {**alert.details, "clip_path": clip_path.replace("\\", "/")}
            self.event_logger.log_event(alert, clip_path)
            if self.webhook:
                self.webhook.send(to_serializable(alert.to_dict()))
            if self._on_alert:
                self._on_alert(alert)
        self.clip_recorder.add_frame(annotated_frame, timestamp)

        processing_time_ms = (time.time() - processing_start) * 1000
        
        result = FrameResult(
            frame=frame,
            timestamp=timestamp,
            detections=detections,
            poses=poses,
            alerts=alerts,
            annotated_frame=annotated_frame,
            processing_time_ms=processing_time_ms,
            total_alerts=self.total_alerts,
            frame_number=self.frame_count
        )
        
        if self._on_frame:
            self._on_frame(result)
            
        return result

    def _annotate_frame(self, frame, detections, poses, alerts) -> np.ndarray:
        # 1. Draw Zone Overlays
        zones = self.anomaly_engine.zone_overlay_data
        for zone in zones:
            poly = np.array(zone["polygon"])
            overlay = frame.copy()
            
            # Color mapping
            if zone["type"] == "restricted":
                color = (0, 0, 255) # Red
            elif zone["type"] == "one_way":
                color = (255, 0, 255) # Purple
            else:
                color = (0, 165, 255) # Orange
                
            cv2.fillPoly(overlay, [poly.astype(int)], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [poly.astype(int)], True, color, 2)
            cv2.putText(frame, zone["name"], (poly[0][0], poly[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Draw Detections & Skeletons
        alerted_track_ids = [getattr(a, 'track_id', -1) for a in alerts]
        
        for det in detections.detections:
            tid = det.track_id
            bbox = det.bbox.astype(int)
            
            # Highlight red if alert active for this track
            is_alerted = tid in alerted_track_ids
            color = (0, 0, 255) if is_alerted else (0, 255, 0)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"ID:{tid} {det.confidence:.2f}", (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw skeleton if pose exists
            if tid in poses:
                self._draw_skeleton(frame, poses[tid])

        # 3. Draw Alert Banners (Top of frame)
        severity_colors = {
            "critical": (0, 0, 255),  # Red
            "high": (0, 69, 255),     # Orange-Red
            "medium": (0, 165, 255),  # Orange
            "low": (0, 255, 255)      # Yellow
        }
        
        for i, alert in enumerate(alerts[:3]): # Max 3 banners
            color = severity_colors.get(alert.severity, (255, 255, 255))
            # Background bar
            cv2.rectangle(frame, (10, 10 + i * 40), (450, 45 + i * 40), color, -1)
            # Text
            cv2.putText(frame, f"ALERT: {alert.message}", (15, 35 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_skeleton(self, frame: np.ndarray, pose: PoseResult):
        for start_idx, end_idx in self.SKELETON:
            kp1, kp2 = pose.keypoints[start_idx], pose.keypoints[end_idx]
            if kp1[2] > 0.3 and kp2[2] > 0.3:
                cv2.line(frame, tuple(kp1[:2].astype(int)), tuple(kp2[:2].astype(int)), (0, 255, 255), 2)
        
        for kp in pose.keypoints:
            if kp[2] > 0.3:
                cv2.circle(frame, tuple(kp[:2].astype(int)), 4, (255, 255, 0), -1)
            
        return frame

    def run(self):
        self.video_source.start()
        logger.info(f"Pipeline running on source: {self.config.source}")
        clip_fps_set = False
        try:
            while self.video_source.is_running:
                result = self.video_source.read()
                if result is None:
                    time.sleep(0.005)
                    continue

                if not clip_fps_set and self.video_source.source_fps:
                    # Clips play back at the source's real frame rate.
                    self.clip_recorder.set_fps(self.video_source.source_fps)
                    clip_fps_set = True

                frame, ts = result
                _ = self.process_frame(frame, ts)
                
                if self.frame_count % 100 == 0:
                    logger.info(f"Frame {self.frame_count} | Stats: {self.stats}")
                    
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.stop()

    def stop(self):
        self.video_source.stop()
        self.clip_recorder.flush()
        if self.webhook:
            self.webhook.close()
        uptime = time.time() - self.start_time
        logger.info(
            f"Pipeline stopped. Uptime: {uptime:.1f}s | Total Frames: {self.frame_count} "
            f"| Total Alerts: {self.total_alerts}"
        )

    @property
    def stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "frames_processed": self.frame_count,
            "total_alerts": self.total_alerts,
            "uptime_seconds": round(uptime, 1),
            "avg_fps": round(self.frame_count / uptime, 1) if uptime > 0 else 0,
            "source_stats": self.video_source.stats,
            "active_tracks": len(self.pose_estimator.get_all_features())
        }
