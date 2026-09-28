"""Forensic clips: a rolling pre-alert buffer plus post-alert frames, saved per alert.

* Frames are buffered JPEG-compressed and downscaled (a 10 s buffer is a few MB, not
  hundreds), and they are the *annotated* frames, so clips show boxes, skeletons and
  the alert banner.
* When an alert fires, the clip keeps recording for ``post_seconds`` and is then
  encoded on a background thread, so the pipeline never blocks on disk/video I/O.
* Encoding uses ffmpeg (H.264, playable in browsers) when available via imageio-ffmpeg,
  falling back to OpenCV's mp4v writer.
* A JPEG snapshot of the alert frame is written immediately.
* Durations and playback speed follow the frame timestamps, so a clip is real-time
  even when processing runs slower than the camera (frames get dropped upstream).
"""

import logging
import os
import queue
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sentinel.output.recorder")

try:  # bundled ffmpeg binary, no system install needed
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover - optional dependency
    FFMPEG = None


@dataclass
class _Recording:
    alert_id: str
    path: str
    frames: List[Tuple[bytes, float]]  # (JPEG, timestamp)
    remaining: int  # frame-count cap on post-alert recording
    end_time: float = float("inf")  # stop post-alert recording at this timestamp
    done: threading.Event = field(default_factory=threading.Event)


class ClipRecorder:
    def __init__(self, clips_dir="data/clips", buffer_seconds=10, fps=25, post_seconds=5,
                 max_width=960, jpeg_quality=80):
        self.clips_dir = clips_dir
        self.buffer_seconds = buffer_seconds
        self.post_seconds = post_seconds
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.frame_buffer: deque = deque(maxlen=1)  # (JPEG, timestamp); sized by set_fps
        self.set_fps(fps)
        self._active: List[_Recording] = []
        self._lock = threading.Lock()
        self._jobs: "queue.Queue[Optional[_Recording]]" = queue.Queue()
        self._worker = threading.Thread(target=self._write_loop, daemon=True, name="clip-writer")
        self._worker.start()
        os.makedirs(clips_dir, exist_ok=True)

    def set_fps(self, fps: float):
        """Match the source frame rate so clip durations (and playback speed) are right."""
        self.fps = max(1, int(round(fps)))
        self.pre_frames = max(1, int(self.buffer_seconds * self.fps))
        self.post_frames = max(0, int(self.post_seconds * self.fps))
        self.frame_buffer = deque(self.frame_buffer, maxlen=self.pre_frames)

    # -- frames -------------------------------------------------------------------------

    def _encode(self, frame: np.ndarray) -> Optional[bytes]:
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            frame = cv2.resize(frame, (self.max_width, int(h * scale) // 2 * 2))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else None

    def add_frame(self, frame: np.ndarray, timestamp: float = 0.0):
        data = self._encode(frame)
        if data is None:
            return
        item = (data, float(timestamp))
        self.frame_buffer.append(item)
        # Keep only the last buffer_seconds (the maxlen is just a frame-count cap).
        while len(self.frame_buffer) > 1 and timestamp - self.frame_buffer[0][1] > self.buffer_seconds:
            self.frame_buffer.popleft()
        with self._lock:
            finished = []
            for rec in self._active:
                rec.frames.append(item)
                rec.remaining -= 1
                if rec.remaining <= 0 or timestamp >= rec.end_time:
                    finished.append(rec)
            for rec in finished:
                self._active.remove(rec)
                self._jobs.put(rec)

    # -- alerts -------------------------------------------------------------------------

    def save_clip(self, alert_id: str, alert_timestamp: float = 0.0,
                  snapshot: Optional[np.ndarray] = None) -> str:
        """Start a clip for ``alert_id``; returns the path the MP4 will be written to."""
        safe_id = "".join(c for c in alert_id if c.isalnum() or c in "-_")
        incident_dir = os.path.join(self.clips_dir, safe_id)
        os.makedirs(incident_dir, exist_ok=True)
        clip_path = os.path.join(incident_dir, f"clip_{safe_id}.mp4")
        if snapshot is not None:
            cv2.imwrite(os.path.join(incident_dir, f"snapshot_{safe_id}.jpg"), snapshot)

        rec = _Recording(alert_id=safe_id, path=clip_path, frames=list(self.frame_buffer),
                         remaining=self.post_frames,
                         end_time=alert_timestamp + self.post_seconds if alert_timestamp else float("inf"))
        if not rec.frames and rec.remaining == 0:
            logger.warning("Attempted to save clip with empty buffer")
            return ""
        if rec.remaining == 0:
            self._jobs.put(rec)
        else:
            with self._lock:
                self._active.append(rec)
        return clip_path

    def flush(self, timeout: float = 30.0):
        """Finish all clips now (e.g. at shutdown or the end of a video)."""
        with self._lock:
            pending, self._active = self._active, []
        for rec in pending:
            self._jobs.put(rec)
        done = threading.Event()
        self._jobs.put(_Recording("__flush__", "", [], 0, done=done))
        done.wait(timeout)

    # -- writer thread ------------------------------------------------------------------

    def _write_loop(self):
        while True:
            rec = self._jobs.get()
            if rec is None:
                return
            if rec.alert_id == "__flush__":
                rec.done.set()
                continue
            try:
                self._write(rec)
            except Exception as e:  # never let one bad clip kill the writer
                logger.error("Failed to save clip %s: %s", rec.path, e)
            finally:
                rec.frames.clear()
                rec.done.set()

    def _write(self, rec: _Recording):
        fps = self._playback_fps(rec)
        frames = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b, _ in rec.frames]
        frames = [f for f in frames if f is not None]
        if not frames:
            return
        h, w = frames[0].shape[:2]
        h, w = h // 2 * 2, w // 2 * 2  # H.264 (yuv420p) needs even dimensions
        frames = [f[:h, :w] for f in frames]
        if FFMPEG and self._write_ffmpeg(rec.path, frames, w, h, fps):
            logger.info("Saved forensic clip (H.264): %s", rec.path)
            return
        out = cv2.VideoWriter(rec.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not out.isOpened():
            logger.error("Could not open video writer for %s", rec.path)
            return
        for f in frames:
            out.write(f)
        out.release()
        logger.info("Saved forensic clip (mp4v): %s", rec.path)

    def _playback_fps(self, rec: _Recording) -> float:
        """Frames per second actually recorded (from timestamps), else the nominal fps."""
        if len(rec.frames) >= 2:
            span = rec.frames[-1][1] - rec.frames[0][1]
            if span > 0:
                return float(min(60.0, max(1.0, (len(rec.frames) - 1) / span)))
        return float(self.fps)

    def _write_ffmpeg(self, path: str, frames, w: int, h: int, fps: float) -> bool:
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait(timeout=120)
            return proc.returncode == 0 and os.path.exists(path)
        except Exception as e:
            logger.warning("ffmpeg encode failed (%s); falling back to OpenCV", e)
            return False
