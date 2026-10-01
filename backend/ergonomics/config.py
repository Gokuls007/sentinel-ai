"""Ergonomics settings (REBA from 2D pose)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ErgonomicsConfig:
    enabled: bool = True
    keypoint_min_conf: float = 0.3      # below this a keypoint is missing, never guessed
    # View confidence from shoulder width / torso length (see angles.view_confidence):
    side_ratio: float = 0.35            # at or below: a side view, angles are reliable (confidence 1)
    front_ratio: float = 0.75           # at or above: facing the camera, angles unreliable (confidence 0)
    # Facing direction: nose x relative to the ears (head-local, so trunk/neck tilt doesn't
    # change it), or relative to the shoulder midpoint when no ear is visible. Offsets are
    # divided by torso length; below these the person faces the camera (or away): unclear.
    facing_min_offset_ear: float = 0.05
    facing_min_offset_shoulder: float = 0.12
    unclear_facing_penalty: float = 0.5  # confidence multiplier when flexion/extension sign is unknown
    missing_part_penalty: float = 0.15  # confidence lost per missing optional body part
    min_confidence: float = 0.6         # scores below this are shown greyed out and never alert
    wrist_default_score: int = 1        # wrist angle can't be measured from COCO keypoints
    smoothing_window_s: float = 1.0     # median over this window
    alert_level: int = 4                # alert at >= this risk level index (4 = high, 5 = very high)
    alert_after_s: float = 3.0          # ...sustained this long
    alert_cooldown_s: float = 60.0      # at most one ergo_risk event per track per this many seconds
    static_after_s: float = 60.0        # posture held this long adds REBA's +1 activity score
    static_max_trunk_change_deg: float = 10.0
