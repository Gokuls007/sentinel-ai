"""Head pose for Exam Hall: yaw and pitch per person, behind an interface.

The default estimator works from COCO face keypoints (nose, eyes, ears) only: no extra model.
A dedicated head-pose model can replace it later by implementing ``HeadPoseEstimator``; it
gets the frame and the person's box as well, so it can crop the head itself.

Conventions (image terms, the camera faces the students):
- ``yaw``: degrees, + = the face points toward the image's right, - = toward its left. About
  +-75 when only one ear is visible (a profile view).
- ``pitch``: degrees, + = head tilted down (toward the desk or lap). Only meaningful against a
  seat's own baseline: the camera's height and angle shift it for everyone.
- ``confidence``: 0-1, from the keypoints used. Low confidence never flags (exam-hall.md 1.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
PROFILE_YAW = 75.0


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float | None
    confidence: float
    method: str


class HeadPoseEstimator(Protocol):
    """Anything that turns a person's keypoints (and optionally the frame) into a head pose."""

    name: str

    def estimate(self, keypoints: np.ndarray, frame: np.ndarray | None = None,
                 box: tuple[float, float, float, float] | None = None) -> HeadPose | None: ...


class KeypointHeadPose:
    """2D estimate from the nose's position relative to the ears (or eyes)."""

    name = "keypoints-2d"

    def __init__(self, min_conf: float = 0.4):
        self.min_conf = min_conf

    def estimate(self, keypoints, frame=None, box=None) -> HeadPose | None:
        kp = np.asarray(keypoints, float)
        if kp.shape[0] < 5 or kp[NOSE, 2] < self.min_conf:
            return None
        ok = lambda i: kp[i, 2] >= self.min_conf  # noqa: E731
        nose = kp[NOSE, :2]
        pairs = [(L_EAR, R_EAR, 1.0, "ears"), (L_EYE, R_EYE, 0.45, "eyes")]  # eye span ~0.45 ear span
        for a, b, scale, method in pairs:
            if ok(a) and ok(b):
                mid = (kp[a, :2] + kp[b, :2]) / 2
                span = abs(kp[a, 0] - kp[b, 0]) / scale  # in ear-span units
                if span < 4:
                    continue
                s = float(np.clip((nose[0] - mid[0]) / (span / 2), -1, 1))
                yaw = math.degrees(math.asin(s))
                # Nose below the ear (or eye) line, in ear spans. Compared only with the seat's
                # own baseline, so the constant offset of the camera angle cancels out.
                drop = (nose[1] - mid[1]) / span
                pitch = math.degrees(math.atan2(drop, 0.5))
                conf = float(min(kp[NOSE, 2], kp[a, 2], kp[b, 2])) * (1.0 if method == "ears" else 0.85)
                return HeadPose(yaw, pitch, conf, method)
        for ear in (L_EAR, R_EAR):
            if ok(ear):
                yaw = PROFILE_YAW if nose[0] > kp[ear, 0] else -PROFILE_YAW
                return HeadPose(yaw, None, float(min(kp[NOSE, 2], kp[ear, 2])) * 0.8, "profile")
        return None


def default_estimator() -> HeadPoseEstimator:
    return KeypointHeadPose()
