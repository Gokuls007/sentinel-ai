import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .fall_detector import FallDetector, FallEvent
from .zone_monitor import ZoneMonitor, ZoneViolation
from .temporal_model import TemporalClassifier
from core.pose_estimator import PoseResult, TrackFeatures
from config.settings import SentinelConfig

logger = logging.getLogger("sentinel.anomaly.engine")

@dataclass
class AnomalyAlert:
    alert_id: str
    alert_type: str  # "fall", "zone_intrusion", "loitering", "action", "ppe"
    track_id: int
    timestamp: float
    confidence: float
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "severity": self.severity,
            "message": self.message,
            "details": self.details
        }

class AnomalyEngine:
    SEVERITY_MAP = {
        "fall": "critical",
        "zone_intrusion": "high",
        "time_exceeded": "medium",
        "wrong_direction": "medium",
        "loitering": "low",
        "fighting": "critical",
        "ppe": "high"
    }

    def __init__(self, config: SentinelConfig):
        self.config = config
        self._alert_counter = 0
        
        # Instantiate detectors
        self.fall_detector = FallDetector(
            velocity_threshold=config.fall.velocity_threshold,
            aspect_ratio_threshold=config.fall.aspect_ratio_threshold,
            head_drop_ratio=config.fall.head_drop_ratio,
            stillness_frames=config.fall.stillness_frames,
            stillness_threshold=config.fall.stillness_threshold
        )
        
        self.zone_monitor = ZoneMonitor(
            zones_file=config.zone.zones_file,
            frame_width=config.frame_width,
            frame_height=config.frame_height
        )
        
        self.temporal_classifier = TemporalClassifier(
            model_path=config.anomaly.lstm_model_path,
            hidden_size=config.anomaly.lstm_hidden_size,
            num_layers=config.anomaly.lstm_num_layers,
            device=config.detector.device
        )
        
        # Loitering state
        self.last_loiter_alert: Dict[int, float] = {}
        self.loiter_cooldown = 60.0

    def process(self, poses: Dict[int, PoseResult], features: Dict[int, TrackFeatures], 
                timestamp: float) -> List[AnomalyAlert]:
        
        alerts = []
        
        for tid, pose in poses.items():
            feat = features.get(tid)
            if not feat: continue
            
            # 1. Fall Detection
            fall_reported = False
            fall_event = self.fall_detector.check(tid, pose, feat, timestamp)
            if fall_event and fall_event.stage in ["fallen", "confirmed"]:
                fall_reported = True
                alerts.append(self._create_alert(
                    alert_type="fall",
                    track_id=tid,
                    timestamp=timestamp,
                    confidence=fall_event.confidence,
                    message=f"Fall detected ({fall_event.stage})",
                    details=fall_event.signals
                ))

            # 2. Zone Monitoring
            zone_violations = self.zone_monitor.check(tid, pose.mid_hip, timestamp)
            for violation in zone_violations:
                alerts.append(self._create_alert(
                    alert_type="zone_intrusion" if violation.violation_type == "intrusion" else violation.violation_type,
                    track_id=tid,
                    timestamp=timestamp,
                    confidence=violation.confidence,
                    message=f"Zone violation: {violation.zone_name} ({violation.violation_type})",
                    details={"zone_id": violation.zone_id, "duration": violation.duration}
                ))

            # 3. Loitering
            loiter_alert = self._check_loitering(tid, feat, timestamp)
            if loiter_alert:
                alerts.append(loiter_alert)
                
            # 4. Temporal Action Classification (LSTM)
            # Get flat sequence (seq_len, 34)
            pose_seq = feat.get_flat_tensor()
            if pose_seq is not None:
                action_res = self.temporal_classifier.is_anomaly(pose_seq)
                if action_res:
                    label, conf = action_res
                    
                    # Deduplication: if FallDetector already reported a fall, ignore LSTM fall
                    if label == "fallen" and fall_reported:
                        continue
                        
                    alerts.append(self._create_alert(
                        alert_type=label if label in self.SEVERITY_MAP else "action",
                        track_id=tid,
                        timestamp=timestamp,
                        confidence=conf,
                        message=f"Behavioral anomaly: {label.upper()}",
                        details={"action": label}
                    ))
                    
        return alerts

    def _check_loitering(self, track_id: int, features: TrackFeatures, 
                         timestamp: float) -> Optional[AnomalyAlert]:
        
        if (features.time_tracked > self.config.loiter.time_threshold and 
            features.displacement < self.config.loiter.movement_threshold):
            
            if track_id in self.last_loiter_alert:
                if timestamp - self.last_loiter_alert[track_id] < self.loiter_cooldown:
                    return None
            
            self.last_loiter_alert[track_id] = timestamp
            return self._create_alert(
                alert_type="loitering",
                track_id=track_id,
                timestamp=timestamp,
                confidence=0.7,
                message="Loitering detected (stationary for long duration)",
                details={"time_tracked": features.time_tracked, "displacement": features.displacement}
            )
        return None

    def _create_alert(self, alert_type: str, track_id: int, timestamp: float, 
                      confidence: float, message: str, details: Dict) -> AnomalyAlert:
        
        self._alert_counter += 1
        # Generates ALT-XXXXXX format
        alert_id = f"ALT-{uuid.uuid4().hex[:6].upper()}"
        severity = self.SEVERITY_MAP.get(alert_type, "low")
        
        return AnomalyAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            track_id=track_id,
            timestamp=timestamp,
            confidence=confidence,
            severity=severity,
            message=message,
            details=details
        )

    @property
    def zone_overlay_data(self) -> List[Dict]:
        return self.zone_monitor.get_zones_for_overlay()
