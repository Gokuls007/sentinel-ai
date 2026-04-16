import cv2
import os
import logging
import numpy as np
from collections import deque
from datetime import datetime

logger = logging.getLogger("sentinel.output.recorder")

class ClipRecorder:
    def __init__(self, clips_dir="data/clips", buffer_seconds=10, fps=25):
        self.clips_dir = clips_dir
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)
            logger.info(f"Created clips directory: {clips_dir}")

    def add_frame(self, frame: np.ndarray, timestamp: float):
        # Store frame and its timestamp
        self.frame_buffer.append((frame.copy(), timestamp))

    def save_clip(self, alert_id: str, alert_timestamp: float) -> str:
        if not self.frame_buffer:
            logger.warning("Attempted to save clip with empty buffer")
            return ""

        # Create alert-specific subfolder
        incident_dir = os.path.join(self.clips_dir, alert_id)
        if not os.path.exists(incident_dir):
            os.makedirs(incident_dir)

        # filename format: data/clips/ALT-XXXXXX/clip.mp4
        clip_name = f"clip_{alert_id}.mp4"
        clip_path = os.path.join(incident_dir, clip_name)
        
        h, w = self.frame_buffer[0][0].shape[:2]
        
        # Use 'mp4v' or 'XVID' codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, self.fps, (w, h))

        try:
            for frame, ts in self.frame_buffer:
                out.write(frame)
            out.release()
            logger.info(f"Saved forensic clip: {clip_path}")
            return clip_path
        except Exception as e:
            logger.error(f"Failed to save clip: {e}")
            if out: out.release()
            return ""
