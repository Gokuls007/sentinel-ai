"""Exam setup check, live: can the camera see the students well enough?

Over a short window (``window_s``, so one bad frame doesn't flicker the checklist):
- **view**: heads and shoulders visible for most people;
- **size**: median shoulder width at least ``min_shoulder_px`` (smaller, and keypoints get
  unreliable: "Students are too small in the image...");
- **lighting**: most people's head and shoulder keypoints confident;
- **people**: how many are in view.
"""

from __future__ import annotations

from collections import deque

import numpy as np

HEAD_SHOULDERS = (0, 1, 2, 5, 6)
L_SH, R_SH = 5, 6

TOO_SMALL = "Students are too small in the image; move the camera closer or use fewer seats in view."
TOO_DARK = "Keypoints are uncertain for most people; add light or reduce glare."
NO_VIEW = "Heads and shoulders aren't visible for most people; raise or angle the camera."


class SetupCheck:
    def __init__(self, window_s: float = 3.0, min_shoulder_px: float = 40.0, min_conf: float = 0.5,
                 visible_conf: float = 0.3, share: float = 0.6):
        self.window_s = window_s
        self.min_shoulder_px = min_shoulder_px
        self.min_conf = min_conf
        self.visible_conf = visible_conf
        self.share = share
        self._frames: deque = deque()  # (ts, people, visible, confident, shoulder widths)

    def update(self, keypoints: list[np.ndarray], ts: float) -> None:
        visible = confident = 0
        widths = []
        for kp in keypoints:
            kp = np.asarray(kp, float)
            confs = kp[list(HEAD_SHOULDERS), 2]
            if kp[0, 2] >= self.visible_conf and kp[L_SH, 2] >= self.visible_conf and kp[R_SH, 2] >= self.visible_conf:
                visible += 1
                widths.append(abs(kp[L_SH, 0] - kp[R_SH, 0]))
            if float(np.mean(confs)) >= self.min_conf:
                confident += 1
        self._frames.append((ts, len(keypoints), visible, confident, widths))
        while self._frames and ts - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    def snapshot(self) -> dict:
        if not self._frames:
            return {"people": 0, "checks": [], "ok": False}
        people = round(float(np.median([f[1] for f in self._frames])))
        total = sum(f[1] for f in self._frames)
        widths = [w for f in self._frames for w in f[4]]
        checks = []
        if total:
            view_ok = sum(f[2] for f in self._frames) / total >= self.share
            light_ok = sum(f[3] for f in self._frames) / total >= self.share
            size = float(np.median(widths)) if widths else 0.0
            size_ok = bool(widths) and size >= self.min_shoulder_px
            checks = [
                {"id": "view", "ok": view_ok, "label": "Heads and shoulders visible",
                 "message": None if view_ok else NO_VIEW},
                {"id": "size", "ok": size_ok, "label": "Students large enough in the image",
                 "message": None if size_ok else TOO_SMALL, "value": round(size, 1)},
                {"id": "lighting", "ok": light_ok, "label": "Lighting", "message": None if light_ok else TOO_DARK},
            ]
        return {"people": people, "checks": checks, "ok": bool(checks) and all(c["ok"] for c in checks)}
