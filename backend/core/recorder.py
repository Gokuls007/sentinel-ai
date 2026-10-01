"""Record a camera's raw frames (no overlays) to an MP4 file.

Used for test footage: fall and squat clips for training/eval_false_alarms.py, and the
ergonomics clip for scripts/label_reba.py (labelling must see the raw frame, not Sentinel's
skeleton). Frames are written on a fixed-fps timeline from their timestamps, duplicating or
dropping frames as needed, so playback runs at real speed even when a webcam delivers
frames irregularly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("sentinel.recorder")


class Recorder:
    def __init__(self, directory: str, prefix: str = "recording", fps: float = 15.0,
                 max_seconds: float = 600.0):
        self.directory = directory
        self.prefix = prefix
        self.fps = fps
        self.max_seconds = max_seconds
        self._lock = threading.Lock()
        self._writer: cv2.VideoWriter | None = None
        self._pending = False  # start() was called; the writer opens on the first frame
        self.path: str | None = None
        self._t0: float | None = None
        self._written = 0
        self.started_at: float | None = None
        self.last_path: str | None = None
        self.last_seconds: float = 0.0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def seconds(self) -> float:
        return self._written / self.fps if self._writer is not None else 0.0

    def start(self) -> str:
        with self._lock:
            if self._writer is not None:
                return self.path
            os.makedirs(self.directory, exist_ok=True)
            name = f"{self.prefix}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            self.path = os.path.join(self.directory, name)
            self._t0 = None
            self._written = 0
            self.started_at = time.time()
            self._writer = None
            self._pending = True
            logger.info("recording to %s", self.path)
            return self.path

    def write(self, frame: np.ndarray, timestamp: float) -> None:
        """Add a frame; called from the capture thread. Cheap when not recording."""
        if not self._pending and self._writer is None:
            return
        with self._lock:
            if self._writer is None:
                if not self._pending:
                    return
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self._pending = False
                if not self._writer.isOpened():
                    logger.error("could not open %s for writing", self.path)
                    self._writer = None
                    return
                self._t0 = timestamp
            due = int((timestamp - self._t0) * self.fps) + 1  # frames that should exist by now
            while self._written < due:
                self._writer.write(frame)
                self._written += 1
            if self._written / self.fps >= self.max_seconds:
                self._close()

    def stop(self) -> dict:
        with self._lock:
            self._pending = False
            self._close()
            return {"path": self.last_path, "seconds": round(self.last_seconds, 1)}

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self.last_path, self.last_seconds = self.path, self._written / self.fps
            logger.info("saved %s (%.1f s)", self.path, self.last_seconds)
        self._writer = None

    def status(self) -> dict:
        active = self.recording or self._pending
        return {"recording": active, "path": self.path if active else None, "seconds": round(self.seconds, 1),
                "last_path": self.last_path, "last_seconds": round(self.last_seconds, 1),
                "max_seconds": self.max_seconds}
