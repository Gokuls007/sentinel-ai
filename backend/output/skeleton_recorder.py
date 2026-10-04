"""Skeleton-only incident record: keypoints instead of video.

In skeleton-only mode no MP4 clip, JPEG snapshot or recording is ever written. Each event
instead gets ``skeleton_<alert_id>.json`` in its folder under ``clips_dir``: every person's 17
COCO keypoints, normalised to the frame (0-1), from ``pre_s`` before the alert to ``post_s``
after, played back as a stick figure on the event page. No pixels are stored.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque

logger = logging.getLogger("sentinel.output.skeleton")


class SkeletonRecorder:
    def __init__(self, clips_dir: str, pre_s: float = 10.0, post_s: float = 5.0):
        self.clips_dir = clips_dir
        self.pre_s = pre_s
        self.post_s = post_s
        self._frames: deque = deque()  # (ts, people)
        self._pending: list[dict] = []  # saves waiting for their post-alert seconds

    @staticmethod
    def frame_people(poses: dict, width: int, height: int) -> list[dict]:
        """``{track_id: pose}`` -> compact, normalised people for one frame."""
        w, h = max(width, 1), max(height, 1)
        people = []
        for tid, p in poses.items():
            kp = [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(c), 2)] for x, y, c in p.keypoints]
            x1, y1, x2, y2 = (float(v) for v in p.bbox)
            people.append({"track_id": int(tid), "keypoints": kp,
                           "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]})
        return people

    def path_for(self, alert_id: str) -> str:
        return os.path.join(self.clips_dir, alert_id, f"skeleton_{alert_id}.json")

    def save(self, alert_id: str, ts: float, track_id: int | None = None, meta: dict | None = None) -> str:
        """Schedule a save; it's written once ``post_s`` of frames after ``ts`` have arrived (or on
        ``flush``). Returns the path it will have."""
        path = self.path_for(alert_id)
        self._pending.append({"alert_id": alert_id, "ts": ts, "track_id": track_id, "meta": meta or {}, "path": path})
        return path

    def add_frame(self, ts: float, people: list[dict]) -> None:
        self._frames.append((ts, people))
        horizon = self.pre_s + self.post_s + 1.0
        while self._frames and ts - self._frames[0][0] > horizon:
            self._frames.popleft()
        due = [p for p in self._pending if ts >= p["ts"] + self.post_s]
        for p in due:
            self._pending.remove(p)
            self._write(p)

    def flush(self) -> None:
        """Write pending saves with what there is (the source stopped)."""
        for p in self._pending:
            self._write(p)
        self._pending.clear()

    def _write(self, p: dict) -> None:
        start, end = p["ts"] - self.pre_s, p["ts"] + self.post_s
        frames = [{"t": round(t - p["ts"], 3), "people": people} for t, people in self._frames if start <= t <= end]
        doc = {"alert_id": p["alert_id"], "alert_ts": p["ts"], "track_id": p["track_id"], "pre_s": self.pre_s,
               "post_s": self.post_s, "keypoints": "COCO-17, x/y normalised to the frame, c = confidence",
               **p["meta"], "frames": frames}
        try:
            os.makedirs(os.path.dirname(p["path"]), exist_ok=True)
            tmp = p["path"] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, separators=(",", ":"))
            os.replace(tmp, p["path"])
        except OSError as e:
            logger.warning("could not save skeleton for %s: %s", p["alert_id"], e)
