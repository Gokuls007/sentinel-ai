import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .fall_detector import FallDetector
from .zone_monitor import ZoneMonitor
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
        f = config.fall
        self.fall_detector = FallDetector(
            descent_speed_threshold=f.descent_speed_threshold,
            aspect_ratio_threshold=f.aspect_ratio_threshold,
            head_drop_ratio=f.head_drop_ratio,
            stillness_seconds=f.stillness_seconds,
            stillness_speed_threshold=f.stillness_speed_threshold,
            fallen_timeout_seconds=f.fallen_timeout_seconds,
            cooldown_seconds=f.cooldown_seconds,
        )

        self.zone_monitor = ZoneMonitor(
            zones_file=config.zone.zones_file,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
            alert_cooldown=config.zone.alert_cooldown,
        )
        
        self.temporal_classifier = TemporalClassifier(
            model_path=config.anomaly.lstm_model_path,
            hidden_size=config.anomaly.lstm_hidden_size,
            num_layers=config.anomaly.lstm_num_layers,
            device=config.detector.device
        )
        
        # Loitering state: where each person has been hanging around, and since when.
        self.loiter_anchor: Dict[int, Tuple[np.ndarray, float]] = {}
        self.last_loiter_alert: Dict[int, float] = {}
        self.loiter_cooldown = 60.0

    def process(self, poses: Dict[int, PoseResult], features: Dict[int, TrackFeatures], 
                timestamp: float) -> List[AnomalyAlert]:
        
        alerts = []

        # Drop state for people who are no longer tracked (bounded memory).
        self.fall_detector.prune(features.keys())
        self.zone_monitor.prune(features.keys())
        for state in (self.last_loiter_alert, self.loiter_anchor):
            for tid in [t for t in state if t not in features]:
                del state[tid]

        for tid, pose in poses.items():
            feat = features.get(tid)
            if not feat: continue

            # 1. Fall Detection (one alert per fall, raised when the fall is confirmed)
            fall_reported = False
            fall_event = self.fall_detector.check(tid, pose, feat, timestamp)
            if fall_event:
                fall_reported = True
                alerts.append(self._create_alert(
                    alert_type="fall",
                    track_id=tid,
                    timestamp=timestamp,
                    confidence=fall_event.confidence,
                    message="Fall detected: person down and not moving",
                    details={**fall_event.signals, "peak_descent_speed": fall_event.velocity}
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
            loiter_alert = self._check_loitering(tid, pose.mid_hip, timestamp)
            if loiter_alert:
                alerts.append(loiter_alert)
                
            # 4. Temporal Action Classification (LSTM)
            # Get flat sequence (seq_len, 34)
            pose_seq = feat.get_flat_tensor()
            if pose_seq is not None:
                action_res = self.temporal_classifier.is_anomaly(
                    pose_seq, threshold=self.config.anomaly.anomaly_threshold)
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

    def _check_loitering(self, track_id: int, position: np.ndarray,
                         timestamp: float) -> Optional[AnomalyAlert]:
        """Alert when a person stays within movement_threshold px of one spot for
        longer than time_threshold seconds. Moving further away restarts the clock."""
        position = np.asarray(position, dtype=float)
        anchor = self.loiter_anchor.get(track_id)
        if anchor is None or np.linalg.norm(position - anchor[0]) > self.config.loiter.movement_threshold:
            self.loiter_anchor[track_id] = (position.copy(), timestamp)
            return None

        dwell = timestamp - anchor[1]
        if dwell <= self.config.loiter.time_threshold:
            return None
        last = self.last_loiter_alert.get(track_id)
        if last is not None and timestamp - last < self.loiter_cooldown:
            return None

        self.last_loiter_alert[track_id] = timestamp
        return self._create_alert(
            alert_type="loitering",
            track_id=track_id,
            timestamp=timestamp,
            confidence=0.7,
            message=f"Loitering: stayed in one spot for {dwell:.0f}s",
            details={"dwell_seconds": round(dwell, 1),
                     "radius_px": self.config.loiter.movement_threshold},
        )

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
