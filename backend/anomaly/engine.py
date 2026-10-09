import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from config.settings import SentinelConfig
from core.pose_estimator import PoseResult, TrackFeatures
from ergonomics import ErgoTracker

from .fall_detector import FallDetector
from .temporal_model import TemporalClassifier
from .zone_monitor import ZoneMonitor

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
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    SEVERITY_MAP: ClassVar[dict[str, str]] = {
        "fall": "critical",
        "possible_fall": "medium",
        "zone_intrusion": "high",
        "time_exceeded": "medium",
        "wrong_direction": "medium",
        "loitering": "low",
        "ergo_risk": "high",
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
            lost_hold_seconds=f.lost_hold_seconds,
            upright_hold_seconds=f.upright_hold_seconds,
            ground_mode=f.ground_mode,
            box_calibration=f.box_calibration,
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
        
        # Ergonomics: REBA from the pose already computed (no extra model).
        self.ergo = ErgoTracker(config.ergonomics) if config.ergonomics.enabled else None
        self.last_ergo_ms = 0.0

        # Loitering state: where each person has been hanging around, and since when.
        self.loiter_anchor: dict[int, tuple[np.ndarray, float]] = {}
        self.last_loiter_alert: dict[int, float] = {}
        self.loiter_cooldown = 60.0

    def process(self, poses: dict[int, PoseResult], features: dict[int, TrackFeatures],
                timestamp: float, recovered: dict[int, PoseResult] | None = None) -> list[AnomalyAlert]:
        """``recovered``: poses found again for fallen people the detector lost (see
        anomaly/fall_recovery.py). They feed fall detection only."""
        alerts = []
        recovered = recovered or {}

        # Fallen people with no pose this frame: a recovered pose, or held at the last spot.
        for tid in [t for t in features if t not in poses and self.fall_detector.is_down(t)]:
            if tid in recovered:
                event = self.fall_detector.check_recovered(tid, recovered[tid], features[tid], timestamp,
                                                           mode=self.config.fall.recovery_mode)
            else:
                event = self.fall_detector.check_missing(tid, timestamp)
            if event:
                alerts.append(self._fall_alert(tid, timestamp, event))

        # Drop state for people who are no longer tracked (bounded memory).
        self.fall_detector.prune(features.keys())
        self.zone_monitor.prune(features.keys())
        for state in (self.last_loiter_alert, self.loiter_anchor):
            for tid in [t for t in state if t not in features]:
                del state[tid]
        if self.ergo:
            self.ergo.prune(features.keys())
        ergo_ms = 0.0

        for tid, pose in poses.items():
            feat = features.get(tid)
            if not feat:
                continue

            # 1. Fall Detection (one alert per fall, raised when the fall is confirmed)
            fall_reported = False
            fall_event = self.fall_detector.check(tid, pose, feat, timestamp)
            if fall_event:
                fall_reported = True
                alerts.append(self._fall_alert(tid, timestamp, fall_event))

            # 2. Zone Monitoring
            zone_violations = self.zone_monitor.check(tid, pose.mid_hip, timestamp)
            for violation in zone_violations:
                alerts.append(self._create_alert(
                    alert_type=(
                        "zone_intrusion" if violation.violation_type == "intrusion" else violation.violation_type
                    ),
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

            # 4. Ergonomics (REBA): load comes from the zone the person works in, if any
            if self.ergo:
                started = time.perf_counter()
                zones = self.zone_monitor.zones_of(tid)
                load = max((z.load_score for z in zones), default=0)
                box_h = float(pose.bbox[3] - pose.bbox[1]) / max(1, self.config.frame_height)
                _, ergo_alert = self.ergo.update(
                    tid, pose.keypoints, timestamp, load=load, zone_ids=[z.id for z in zones],
                    box_height_frac=box_h)
                ergo_ms += (time.perf_counter() - started) * 1000
                if ergo_alert:
                    alerts.append(self._ergo_alert(ergo_alert))
                
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
                    
        self.last_ergo_ms = ergo_ms
        # Two-level fall alerts: the early "possible fall" (reached the ground) for the dashboard.
        for event in self.fall_detector.drain_possible():
            alerts.append(self._create_alert(
                alert_type="possible_fall",
                track_id=event.track_id,
                timestamp=event.timestamp,
                confidence=event.confidence,
                message=("Possible fall: person found on the floor, checking whether they stay down"
                         if event.signals.get("fall_not_seen")
                         else "Possible fall: person went down, checking whether they stay down"),
                details={**event.signals, "peak_descent_speed": event.velocity},
            ))
        return alerts

    @property
    def ergonomics_snapshot(self) -> dict[int, dict]:
        """Current smoothed REBA per tracked person (for the dashboard)."""
        if not self.ergo:
            return {}
        return {tid: view.as_dict() for tid, view in self.ergo.current.items()}

    def _ergo_alert(self, a) -> AnomalyAlert:
        part = a.dominant.replace("_", " ")
        alert = self._create_alert(
            alert_type="ergo_risk",
            track_id=a.track_id,
            timestamp=a.timestamp,
            confidence=a.confidence,
            message=(f"Ergonomic risk: REBA {a.peak_score} ({a.level_name.replace('_', ' ')}) "
                     f"for {a.duration_s:.0f}s, mainly {part}"),
            details={
                "reba_score": a.peak_score, "risk_level": a.level_name, "dominant": a.dominant,
                "duration": a.duration_s, "view_confidence": a.confidence, "angles": a.angles,
                **({"zone_id": a.zone_ids[0]} if a.zone_ids else {}),
            },
        )
        alert.severity = "critical" if a.level >= 5 else "high"
        return alert

    def _check_loitering(self, track_id: int, position: np.ndarray,
                         timestamp: float) -> AnomalyAlert | None:
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

    def _fall_alert(self, tid: int, timestamp: float, fall_event) -> AnomalyAlert:
        held = bool(fall_event.signals.get("held_while_lost"))
        unseen = bool(fall_event.signals.get("fall_not_seen"))
        return self._create_alert(
            alert_type="fall",
            track_id=tid,
            timestamp=timestamp,
            confidence=fall_event.confidence,
            message=("Fall detected: person went down and is no longer visible" if held
                     else "Person found on the floor and not moving (the fall itself wasn't seen)" if unseen
                     else "Fall detected: person down and not moving"),
            details={**fall_event.signals, "peak_descent_speed": fall_event.velocity},
        )

    def _create_alert(self, alert_type: str, track_id: int, timestamp: float,
                      confidence: float, message: str, details: dict) -> AnomalyAlert:
        
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
    def zone_overlay_data(self) -> list[dict]:
        return self.zone_monitor.get_zones_for_overlay()
