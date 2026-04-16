import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from core.pose_estimator import PoseResult, TrackFeatures

@dataclass
class FallEvent:
    track_id: int
    timestamp: float
    confidence: float
    stage: str # "falling", "fallen", "confirmed"
    signals: Dict[str, bool]
    head_y: float
    hip_y: float
    velocity: float
    aspect_ratio: float

class FallDetector:
    UPRIGHT = "upright"
    FALLING = "falling"
    FALLEN = "fallen"
    CONFIRMED = "confirmed"

    def __init__(self, velocity_threshold=15.0, aspect_ratio_threshold=1.2, 
                 head_drop_ratio=0.5, stillness_frames=15, stillness_threshold=2.0, 
                 cooldown_seconds=30.0):
        
        self.velocity_threshold = velocity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.head_drop_ratio = head_drop_ratio
        self.stillness_frames = stillness_frames
        self.stillness_threshold = stillness_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Per-track state
        self.states: Dict[int, str] = {}
        self.stillness_counters: Dict[int, int] = {}
        self.last_alert_time: Dict[int, float] = {}
        self.prev_hip_y: Dict[int, float] = {}
        self.fallen_start_time: Dict[int, float] = {}

    def check(self, track_id: int, pose: PoseResult, features: TrackFeatures, 
              timestamp: float) -> Optional[FallEvent]:
        
        # Initialize state if new track
        if track_id not in self.states:
            self.states[track_id] = self.UPRIGHT
            self.stillness_counters[track_id] = 0
            
        # Check cooldown
        if track_id in self.last_alert_time:
            if timestamp - self.last_alert_time[track_id] < self.cooldown_seconds:
                # If we were in CONFIRMED state and cooldown passed, reset to UPRIGHT
                if self.states[track_id] == self.CONFIRMED:
                    self.states[track_id] = self.UPRIGHT
                # Still in cooldown window
                if self.states[track_id] == self.CONFIRMED:
                    return None

        signals = self._compute_signals(track_id, pose, features)
        current_state = self.states[track_id]
        event = None
        
        # 1. UPRIGHT state
        if current_state == self.UPRIGHT:
            if signals["rapid_descent"]:
                self.states[track_id] = self.FALLING
        
        # 2. FALLING state
        elif current_state == self.FALLING:
            if signals["horizontal_pose"] and signals["head_dropped"]:
                self.states[track_id] = self.FALLEN
                self.fallen_start_time[track_id] = timestamp
                event = self._make_event(track_id, timestamp, 0.7, "fallen", signals, pose)
            elif not signals["rapid_descent"]:
                # Person recovered or sit down slowly
                self.states[track_id] = self.UPRIGHT
        
        # 3. FALLEN state
        elif current_state == self.FALLEN:
            if signals["is_still"]:
                self.stillness_counters[track_id] += 1
            else:
                self.stillness_counters[track_id] = max(0, self.stillness_counters[track_id] - 2)
            
            if self.stillness_counters[track_id] >= self.stillness_frames:
                self.states[track_id] = self.CONFIRMED
                self.last_alert_time[track_id] = timestamp
                event = self._make_event(track_id, timestamp, 0.95, "confirmed", signals, pose)
            else:
                # Check for 5s timeout if counter < 5
                time_in_fallen = timestamp - self.fallen_start_time.get(track_id, timestamp)
                if time_in_fallen > 5.0 and self.stillness_counters[track_id] < 5:
                    self.states[track_id] = self.UPRIGHT
                    self.stillness_counters[track_id] = 0
        
        # 4. CONFIRMED state
        elif current_state == self.CONFIRMED:
            # We stay in CONFIRMED until cooldown expires (handled at start of check)
            # Or if they stand up (added for safety reset)
            if not signals["horizontal_pose"] and not signals["head_dropped"]:
                self.states[track_id] = self.UPRIGHT
                self.stillness_counters[track_id] = 0

        # Update tracking variables
        self.prev_hip_y[track_id] = pose.mid_hip[1]
        
        return event

    def _compute_signals(self, track_id: int, pose: PoseResult, features: TrackFeatures) -> Dict:
        # Signal 1: Rapid Descent
        velocity = 0.0
        if track_id in self.prev_hip_y:
            velocity = pose.mid_hip[1] - self.prev_hip_y[track_id]
        rapid_descent = velocity > self.velocity_threshold
            
        # Signal 2: Horizontal Pose
        bbox_w = pose.bbox[2] - pose.bbox[0]
        bbox_h = pose.bbox[3] - pose.bbox[1]
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0
        horizontal_pose = aspect_ratio > self.aspect_ratio_threshold
        
        # Signal 3: Head Dropped
        head_dropped = False
        if features.initial_standing_height > 0:
            # Reference: first hip_y coordinate when they were standing
            # (Note: centroid_history[0] is initialized in TrackFeatures)
            reference_y = features.centroid_history[0][1]
            drop_dist = pose.head_y - reference_y
            head_dropped = (drop_dist / features.initial_standing_height) > self.head_drop_ratio

        # Signal 4: Stillness
        is_still = False
        if len(features.centroid_history) >= 3:
            # Check average movement over last 3 frames
            movements = [np.linalg.norm(features.centroid_history[i] - features.centroid_history[i-1]) 
                         for i in range(-1, -3, -1)]
            avg_move = np.mean(movements)
            is_still = avg_move < self.stillness_threshold
            
        return {
            "rapid_descent": rapid_descent,
            "horizontal_pose": horizontal_pose,
            "head_dropped": head_dropped,
            "is_still": is_still,
            "velocity_value": velocity,
            "aspect_ratio_value": aspect_ratio
        }

    def _make_event(self, track_id: int, timestamp: float, confidence: float, stage: str, 
                    signals: Dict, pose: PoseResult) -> FallEvent:
        return FallEvent(
            track_id=track_id,
            timestamp=timestamp,
            confidence=confidence,
            stage=stage,
            signals=signals,
            head_y=pose.head_y,
            hip_y=pose.mid_hip[1],
            velocity=signals.get("velocity_value", 0.0),
            aspect_ratio=signals.get("aspect_ratio_value", 0.0)
        )

    def reset_track(self, track_id: int):
        if track_id in self.states: del self.states[track_id]
        if track_id in self.stillness_counters: del self.stillness_counters[track_id]
        if track_id in self.last_alert_time: del self.last_alert_time[track_id]
        if track_id in self.prev_hip_y: del self.prev_hip_y[track_id]
        if track_id in self.fallen_start_time: del self.fallen_start_time[track_id]
