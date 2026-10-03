"""Threaded frame capture from a webcam, network stream or video file.

* Live sources (webcam index, rtsp/http(s) URL): frames are timestamped with the wall
  clock and the queue holds only the newest couple of frames, so the dashboard shows
  what is happening *now* even if inference is slower than the camera.
* Video files: frames are timestamped on the video's own timeline (frame / fps), so
  speeds used by the fall detector stay correct even when processing is slower than
  real time. Playback is paced at the file's native FPS; with ``loop=True`` the file
  restarts when it ends (demo / kiosk mode) and timestamps keep increasing.
"""

import logging
import queue
import sys
import threading
import time

import cv2
from numpy import ndarray

logger = logging.getLogger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


class VideoSource:
    def __init__(self, source: str = "0", queue_size: int = 2, target_fps: int = 25,
                 frame_width: int = 1280, frame_height: int = 720, reconnect_delay: float = 5.0,
                 loop: bool = False):
        try:
            self.source = int(source)  # "0" -> webcam 0
        except (TypeError, ValueError):
            self.source = source

        self.is_file = isinstance(self.source, str) and not self.source.lower().startswith(STREAM_PREFIXES)
        self.is_live = not self.is_file
        self.loop = loop and self.is_file
        self.target_fps = target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reconnect_delay = reconnect_delay

        self.frame_queue: queue.Queue[tuple[ndarray, float]] = queue.Queue(maxsize=max(1, queue_size))
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.recorder = None  # optional core.recorder.Recorder fed every captured frame
        self.paused = False  # set by the pipeline when demo footage is switched off

        # Stats
        self.frames_read = 0
        self.frames_dropped = 0
        self.loops_completed = 0
        self.actual_fps = 0.0
        self.source_fps = 0.0
        self.hardware_error = False
        self.error_message = ""
        self.finished = False
        self._fps_counter = 0
        self._fps_start_time = time.time()

    # -- lifecycle ----------------------------------------------------------------------

    def start(self) -> "VideoSource":
        self.stop_event.clear()
        self.finished = False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True,
                                               name="video-capture")
        self.capture_thread.start()
        logger.info("VideoSource started for source: %s", self.source)
        return self

    def stop(self):
        """Stop capturing and release the device (a webcam's light goes off). The capture thread
        releases its own capture when it exits; releasing it from here while that thread is
        still inside ``read()`` can leave the device held on Windows, so only release here if
        the thread has already finished."""
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.cap and not (self.capture_thread and self.capture_thread.is_alive()):
            self.cap.release()
        logger.info("VideoSource stopped")

    def read(self) -> tuple[ndarray, float] | None:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    # -- capture ------------------------------------------------------------------------

    def _connect(self) -> bool:
        if self.cap:
            self.cap.release()
        if isinstance(self.source, int) and sys.platform == "win32":
            # DirectShow opens laptop webcams in about a second; the default (MSMF) backend
            # can take 10+ s or fail. Fall back to the default if DirectShow can't open it.
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.hardware_error = True
            self.error_message = f"Failed to open source: {self.source}"
            logger.error(self.error_message)
            return False

        self.hardware_error = False
        self.error_message = ""
        if isinstance(self.source, int):  # ask webcams for the configured resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps and 1 <= fps <= 240 else float(self.target_fps)
        logger.info("Connected to source. Resolution: %.0fx%.0f, FPS: %.1f",
                    self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    self.source_fps)
        return True

    def _put(self, item: tuple[ndarray, float]):
        """Keep only the newest frames: drop the oldest when the consumer is behind."""
        while True:
            try:
                self.frame_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

    def _capture_loop(self):
        if not self._connect() and not self.is_live:
            self.finished = True
            return  # a missing file can't be retried

        # Files: timestamps follow the video timeline, anchored to the start time.
        timeline_origin = time.time()
        timeline_offset = 0.0  # seconds of video already played in previous loops
        frame_index = 0

        while not self.stop_event.is_set():
            if self.paused:  # don't decode frames nobody will process
                self.stop_event.wait(0.1)
                continue
            if self.cap is None or not self.cap.isOpened():
                logger.info("Reconnecting to source in %.0fs...", self.reconnect_delay)
                if self.stop_event.wait(self.reconnect_delay):
                    break
                self._connect()
                continue

            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    if self.loop and frame_index > 0:
                        self.loops_completed += 1
                        timeline_offset += frame_index / self.source_fps
                        frame_index = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.info("End of video file reached")
                    break
                logger.warning("Failed to read frame, attempting reconnect...")
                self.cap.release()
                continue

            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            if self.is_file:
                timestamp = timeline_origin + timeline_offset + frame_index / self.source_fps
                frame_index += 1
            else:
                timestamp = time.time()

            self._put((frame, timestamp))
            recorder = self.recorder
            if recorder is not None:
                recorder.write(frame, timestamp)
            self.frames_read += 1
            self._fps_counter += 1

            now = time.time()
            if now - self._fps_start_time >= 1.0:
                self.actual_fps = self._fps_counter / (now - self._fps_start_time)
                self._fps_counter = 0
                self._fps_start_time = now

            # Pace files at their native speed; webcams are paced by the device itself.
            if self.is_file:
                sleep_time = 1.0 / self.source_fps - (time.time() - loop_start)
                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

        self.finished = True
        if self.cap:
            self.cap.release()

    # -- status -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while frames are still being captured or are waiting in the queue."""
        if self.stop_event.is_set():
            return False
        capturing = self.capture_thread is not None and self.capture_thread.is_alive()
        return capturing or not self.frame_queue.empty()

    @property
    def stats(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "actual_fps": round(self.actual_fps, 2),
            "source_fps": round(self.source_fps, 2),
            "queue_size": self.frame_queue.qsize(),
            "is_file": self.is_file,
            "loop": self.loop,
            "loops_completed": self.loops_completed,
            "finished": self.finished,
            "hardware_error": self.hardware_error,
            "error_message": self.error_message,
        }
