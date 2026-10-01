"""Joint angles for REBA from 17 COCO keypoints (2D image coordinates, y pointing down).

Conventions:
- Angles are in degrees. A keypoint below ``keypoint_min_conf`` is missing; any angle that
  needs it is ``None`` rather than a guess.
- Flexion is positive and extension negative, for the trunk, the neck and the upper arms.
  The sign comes from the facing direction (``facing``: +1 = the person faces image-right,
  -1 = image-left), estimated from the nose relative to the shoulder midpoint. When facing is
  unclear (looking straight at or away from the camera) the sign is unknown: magnitudes are
  reported as flexion and confidence is lowered, because REBA scores extension differently.
- The lower arm and knee scores use flexion = 180 - (interior joint angle), so a straight
  limb is 0 degrees of flexion.

2D limits: angles are projections onto the image plane. They are only trustworthy when the
person is seen from the side, which ``view_confidence`` measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ergonomics.config import ErgonomicsConfig

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


@dataclass
class PostureAngles:
    trunk: float | None = None          # signed: + flexion (leaning forward), - extension
    neck: float | None = None           # signed, relative to the trunk
    upper_arm_left: float | None = None   # signed: + raised forward, - behind the body
    upper_arm_right: float | None = None
    lower_arm_left: float | None = None   # flexion: 0 = straight
    lower_arm_right: float | None = None
    knee_left: float | None = None        # flexion: 0 = straight
    knee_right: float | None = None
    facing: int | None = None           # +1 image-right, -1 image-left, None unclear
    shoulder_torso_ratio: float | None = None
    confidence: float = 0.0             # 0..1, how far these 2D angles can be trusted
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 1)  # noqa: E731
        return {
            "trunk": r(self.trunk), "neck": r(self.neck),
            "upper_arm_left": r(self.upper_arm_left), "upper_arm_right": r(self.upper_arm_right),
            "lower_arm_left": r(self.lower_arm_left), "lower_arm_right": r(self.lower_arm_right),
            "knee_left": r(self.knee_left), "knee_right": r(self.knee_right),
            "facing": self.facing, "shoulder_torso_ratio": r(self.shoulder_torso_ratio),
            "confidence": round(float(self.confidence), 2),
        }


def _pt(kp: np.ndarray, i: int, min_conf: float) -> np.ndarray | None:
    return kp[i, :2].astype(float) if float(kp[i, 2]) >= min_conf else None


def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


def _mid_both(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    """Midpoint only when both sides are visible (a single side would bias the midline)."""
    return (a + b) / 2 if a is not None and b is not None else None


def signed_angle(v: np.ndarray, ref: np.ndarray, facing: int | None) -> float:
    """Angle from ``ref`` to ``v`` in degrees: positive (flexion) when ``v`` has turned toward
    the side the person faces, negative (extension) when it turned away.

    Works for any reference direction (trunk vs. up, neck vs. trunk, arm vs. trunk-down): the
    horizontal part of (unit v - unit ref) says which way v moved in the image, and comparing
    it with ``facing`` says forward or backward. With facing unknown the magnitude is returned.
    """
    nv, nr = np.linalg.norm(v), np.linalg.norm(ref)
    if nv == 0 or nr == 0:
        return 0.0
    u, r = v / nv, ref / nr
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, r))))))
    if facing is None:
        return angle
    moved_x = u[0] - r[0]
    return angle if moved_x * facing >= 0 else -angle


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at b, in degrees (180 = straight)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))


def facing_direction(nose, ear, shoulder_mid, torso_len, cfg: ErgonomicsConfig) -> int | None:
    """+1 facing image-right, -1 image-left, None unclear (facing or turned from the camera).

    The nose sits in front of the ears whatever the trunk or neck does, so the ear reference
    keeps working when someone leans or tilts their head back (extension). The shoulder
    midpoint is the fallback; it fails for extension, because leaning back moves the head
    behind the shoulders.
    """
    if nose is None or torso_len <= 0:
        return None
    if ear is not None:
        ref, threshold = ear, cfg.facing_min_offset_ear
    elif shoulder_mid is not None:
        ref, threshold = shoulder_mid, cfg.facing_min_offset_shoulder
    else:
        return None
    offset = (nose[0] - ref[0]) / torso_len
    if abs(offset) < threshold:
        return None
    return 1 if offset > 0 else -1


def view_confidence(ratio: float | None, cfg: ErgonomicsConfig) -> float:
    """Shoulder width / torso length: small in a side view (reliable), large when facing the
    camera (unreliable). Bending toward the camera shortens the apparent torso, which also
    raises the ratio, and that view is equally unreliable."""
    if ratio is None:
        return 0.0
    if ratio <= cfg.side_ratio:
        return 1.0
    if ratio >= cfg.front_ratio:
        return 0.0
    return (cfg.front_ratio - ratio) / (cfg.front_ratio - cfg.side_ratio)


def compute_angles(keypoints: np.ndarray, cfg: ErgonomicsConfig | None = None) -> PostureAngles:
    cfg = cfg or ErgonomicsConfig()
    kp = np.asarray(keypoints, dtype=float)
    p = lambda i: _pt(kp, i, cfg.keypoint_min_conf)  # noqa: E731
    out = PostureAngles()

    ls, rs, lh, rh = p(L_SHOULDER), p(R_SHOULDER), p(L_HIP), p(R_HIP)
    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)
    if shoulder_mid is None or hip_mid is None:
        out.notes.append("trunk not visible")
        return out  # without a trunk there is no REBA score

    trunk_vec = shoulder_mid - hip_mid            # hips -> shoulders
    torso_len = float(np.linalg.norm(trunk_vec))
    if torso_len < 1:
        out.notes.append("degenerate torso")
        return out

    nose = p(NOSE)
    ear = _mid(p(L_EAR), p(R_EAR))
    out.facing = facing_direction(nose, ear, shoulder_mid, torso_len, cfg)
    up = np.array([0.0, -1.0])
    out.trunk = signed_angle(trunk_vec, up, out.facing)

    # Neck: shoulder midpoint -> head (ears preferred: the nose sits in front of the neck axis).
    head = ear if ear is not None else nose
    if head is not None:
        out.neck = signed_angle(head - shoulder_mid, trunk_vec, out.facing)

    # Upper arms: relative to the trunk pointing down (arm hanging = 0, forward horizontal = 90).
    trunk_down = -trunk_vec
    for side, (sh, el, wr) in {"left": (ls, p(L_ELBOW), p(L_WRIST)), "right": (rs, p(R_ELBOW), p(R_WRIST))}.items():
        if sh is not None and el is not None:
            setattr(out, f"upper_arm_{side}", signed_angle(el - sh, trunk_down, out.facing))
            if wr is not None:
                setattr(out, f"lower_arm_{side}", 180.0 - interior_angle(sh, el, wr))

    for side, (hp, kn, an) in {"left": (lh, p(L_KNEE), p(L_ANKLE)), "right": (rh, p(R_KNEE), p(R_ANKLE))}.items():
        if hp is not None and kn is not None and an is not None:
            setattr(out, f"knee_{side}", 180.0 - interior_angle(hp, kn, an))

    # Confidence: view (side = good), facing clarity, and missing parts.
    both_shoulders = _mid_both(ls, rs)
    if both_shoulders is not None:
        out.shoulder_torso_ratio = float(np.linalg.norm(ls - rs)) / torso_len
        confidence = view_confidence(out.shoulder_torso_ratio, cfg)
    else:
        # One shoulder hidden behind the body is itself a sign of a side view.
        out.shoulder_torso_ratio = 0.0
        confidence = 1.0
    if out.facing is None:
        confidence *= cfg.unclear_facing_penalty
        out.notes.append("facing direction unclear: flexion/extension sign unknown")
    missing = sum(
        v is None for v in (
            out.neck,
            out.upper_arm_left if out.upper_arm_right is None else out.upper_arm_right,
            out.lower_arm_left if out.lower_arm_right is None else out.lower_arm_right,
            out.knee_left if out.knee_right is None else out.knee_right,
        )
    )
    confidence -= cfg.missing_part_penalty * missing
    out.confidence = max(0.0, min(1.0, confidence))
    return out
