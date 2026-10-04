"""Which body parts the camera can see, per person and for the scene.

Fall detection and ergonomics need hips and legs. A laptop webcam on a desk, or a camera
mounted too close, sees only the upper body; then everything quietly scores nothing. The view
check says so instead: a banner when, for a few seconds, nobody in view has hips and knees.
"""

from __future__ import annotations

from collections import deque

import numpy as np

PARTS = {
    "head": (0, 1, 2),
    "shoulders": (5, 6),
    "wrists": (9, 10),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}
UPPER_BODY_MESSAGE = ("Only upper body visible. Fall detection and ergonomics need a full-body view: "
                      "place the camera 2–4 m away, ideally side-on.")


def visible_parts(keypoints, min_conf: float = 0.3) -> dict[str, bool]:
    """Part -> visible (at least one of its keypoints at ``min_conf`` or more)."""
    kp = np.asarray(keypoints, float)
    return {part: bool(any(kp[i, 2] >= min_conf for i in idx if i < kp.shape[0])) for part, idx in PARTS.items()}


def full_body(parts: dict[str, bool]) -> bool:
    return parts["hips"] and parts["knees"]


class ViewCheck:
    """Scene-level: "upper body only" once, for ``window_s``, people were in view in most frames
    and none of them had hips and knees visible."""

    def __init__(self, window_s: float = 5.0, present_share: float = 0.8, min_conf: float = 0.3):
        self.window_s, self.present_share, self.min_conf = window_s, present_share, min_conf
        self._frames: deque[tuple[float, bool, bool]] = deque()  # (ts, anyone, anyone full-body)
        self.upper_body_only = False

    def update(self, keypoints_by_track: dict, ts: float) -> bool:
        parts = [visible_parts(kp, self.min_conf) for kp in keypoints_by_track.values()]
        self._frames.append((ts, bool(parts), any(full_body(p) for p in parts)))
        while self._frames and ts - self._frames[0][0] > self.window_s:
            self._frames.popleft()
        span = ts - self._frames[0][0] if self._frames else 0.0
        with_people = [f for f in self._frames if f[1]]
        self.upper_body_only = (
            span >= self.window_s * 0.9
            and len(with_people) >= self.present_share * len(self._frames)
            and not any(f[2] for f in with_people)
        )
        return self.upper_body_only

    def snapshot(self) -> dict:
        return {"upper_body_only": self.upper_body_only,
                "message": UPPER_BODY_MESSAGE if self.upper_body_only else None}


def ergo_reason(keypoints, confidence: float, min_confidence: float, box_height_frac: float | None,
                min_conf: float = 0.3, min_height_frac: float = 0.2) -> str:
    """Why a person's REBA isn't trusted, in plain words (instead of "low confidence")."""
    parts = visible_parts(keypoints, min_conf)
    if not parts["shoulders"]:
        return "shoulders not visible"
    if not parts["hips"]:
        return "hips not visible"
    if not parts["knees"]:
        return "legs not visible"
    if box_height_frac is not None and box_height_frac < min_height_frac:
        return "too small or far away"
    if confidence < min_confidence:
        return "facing the camera (REBA needs a side view)"
    return "keypoints unclear"
