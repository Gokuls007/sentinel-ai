import time
import threading
import queue
import cv2
import logging
from typing import Optional, Tuple, Dict
from numpy import ndarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, source: str = "0", queue_size: int = 128, target_fps: int = 25, 
                 frame_width: int = 1280, frame_height: int = 720, reconnect_delay: float = 5.0):
        # Convert "0" to integer for webcam
        try:
            self.source = int(source)
        except ValueError:
            self.source = source
            
        self.queue_size = queue_size
        self.target_fps = target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reconnect_delay = reconnect_delay
        
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Stats
        self.frames_read = 0
        self.frames_dropped = 0
        self.actual_fps = 0.0
        self.hardware_error = False
        self.error_message = ""
        self._fps_counter = 0
        self._fps_start_time = time.time()

    def start(self) -> 'VideoSource':
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"VideoSource started for source: {self.source}")
        return self

    def stop(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("VideoSource stopped")

    def read(self) -> Optional[Tuple[ndarray, float]]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _connect(self) -> bool:
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.hardware_error = True
            self.error_message = f"Failed to open source: {self.source}"
            logger.error(self.error_message)
            return False
        
        self.hardware_error = False
        self.error_message = ""
            
        # Set resolution for webcams
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Connected to source. Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        return True

    def _capture_loop(self):
        if not self._connect():
            # If initial connect fails and it's RTSP, we'll try in the loop
            if not isinstance(self.source, str) or not (self.source.startswith("rtsp://") or self.source.startswith("http://") or self.source.startswith("https://")):
                return

        frame_interval = 1.0 / self.target_fps
        
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                logger.info(f"Reconnecting to source in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                if not self._connect():
                    continue

            start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                if isinstance(self.source, str) and not (self.source.startswith("rtsp://") or self.source.startswith("http://")):
                    logger.info("End of video file reached")
                    break
                else:
                    logger.warning("Failed to read frame, attempting reconnect...")
                    self.cap.release()
                    continue

            # Resize if necessary
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            timestamp = time.time()
            
            # Thread-safe queue management
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            
            self.frame_queue.put((frame, timestamp))
            self.frames_read += 1
            self._fps_counter += 1
            
            # FPS Tracking
            current_time = time.time()
            if current_time - self._fps_start_time >= 1.0:
                self.actual_fps = self._fps_counter / (current_time - self._fps_start_time)
                self._fps_counter = 0
                self._fps_start_time = current_time

            # Throttling
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.cap:
            self.cap.release()

    @property
    def is_running(self) -> bool:
        return self.capture_thread is not None and self.capture_thread.is_alive() and not self.stop_event.is_set()

    @property
    def stats(self) -> Dict:
        return {
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "actual_fps": round(self.actual_fps, 2),
            "queue_size": self.frame_queue.qsize(),
            "hardware_error": self.hardware_error,
            "error_message": self.error_message
        }
