"""Per-person signals the rules need that aren't computed elsewhere: head turn, looking down
and holding an object. All 2D estimates from COCO keypoints, each with a confidence; a
low-confidence estimate never makes a condition true.
"""

from __future__ import annotations

import math

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_WRIST, R_WRIST = 9, 10
MIN_CONF = 0.5


def head_turn(kp: np.ndarray, min_conf: float = MIN_CONF) -> tuple[str, float, float] | None:
    """(direction, angle in degrees, confidence) of the head turn, or None if it can't be told.

    Direction is the person's own left or right. The camera image is not mirrored, so a nose
    moving toward the image's right means the person turned to *their* left.
    - Both ears visible: the nose's offset from the ear midpoint, as a share of half the ear
      span, read as sin(angle).
    - One ear visible: a profile view, about 75 degrees, toward the side the hidden ear is on.
    """
    kp = np.asarray(kp, float)
    if kp.shape[0] < 5 or kp[NOSE, 2] < min_conf:
        return None
    le, re_ = kp[L_EAR, 2] >= min_conf, kp[R_EAR, 2] >= min_conf
    if le and re_:
        mid = (kp[L_EAR, 0] + kp[R_EAR, 0]) / 2
        half = abs(kp[L_EAR, 0] - kp[R_EAR, 0]) / 2
        if half < 2:
            return None
        s = float(np.clip((kp[NOSE, 0] - mid) / half, -1, 1))
        angle = math.degrees(math.asin(abs(s)))
        conf = float(min(kp[NOSE, 2], kp[L_EAR, 2], kp[R_EAR, 2]))
        return ("left" if s > 0 else "right"), angle, conf
    if le or re_:
        ear = L_EAR if le else R_EAR
        # The nose sits on the side the face points to.
        direction = "left" if kp[NOSE, 0] > kp[ear, 0] else "right"
        return direction, 75.0, float(min(kp[NOSE, 2], kp[ear, 2])) * 0.8
    return None


def looking_down(kp: np.ndarray, min_conf: float = MIN_CONF, pitch_limit: float = 0.35) -> bool | None:
    """True when the nose is well below the ear line (``pitch_limit`` ear spans), None if the
    ears or nose aren't visible clearly."""
    kp = np.asarray(kp, float)
    if kp.shape[0] < 5 or min(kp[NOSE, 2], kp[L_EAR, 2], kp[R_EAR, 2]) < min_conf:
        return None
    span = abs(kp[L_EAR, 0] - kp[R_EAR, 0])
    if span < 4:
        return None
    ear_y = (kp[L_EAR, 1] + kp[R_EAR, 1]) / 2
    return bool((kp[NOSE, 1] - ear_y) / span > pitch_limit)


def holding(kp: np.ndarray, body_height: float, objects: list[tuple[str, tuple]],
            reach: float = 0.25, min_conf: float = 0.3) -> set[str]:
    """Object classes at one of the person's hands: the wrist is inside the object's box grown
    by ``reach`` body heights. ``objects`` is [(class name, (x1, y1, x2, y2))]."""
    kp = np.asarray(kp, float)
    wrists = [kp[i, :2] for i in (L_WRIST, R_WRIST) if kp.shape[0] > i and kp[i, 2] >= min_conf]
    if not wrists or not objects:
        return set()
    pad = reach * max(body_height, 1.0)
    found = set()
    for name, (x1, y1, x2, y2) in objects:
        for w in wrists:
            if x1 - pad <= w[0] <= x2 + pad and y1 - pad <= w[1] <= y2 + pad:
                found.add(name)
    return found
