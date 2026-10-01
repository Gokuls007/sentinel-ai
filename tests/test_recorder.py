"""Raw-frame recorder: real-speed timeline, auto-stop, and the record API."""

import cv2
import numpy as np

from core.recorder import Recorder


def frame(v=0):
    return np.full((72, 128, 3), v, np.uint8)


def frames_in(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def test_irregular_frames_play_back_at_real_speed(tmp_path):
    rec = Recorder(str(tmp_path), fps=10)
    path = rec.start()
    assert rec.status()["recording"]
    # 2.8 s of capture with uneven gaps (a webcam at ~7 fps): a 10 fps file -> ~28 frames.
    t = 100.0
    for gap in [0.1, 0.2, 0.15, 0.05, 0.2] * 4:
        rec.write(frame(), t)
        t += gap
    rec.write(frame(), t)
    info = rec.stop()
    n, fps = frames_in(path)
    assert fps == 10 and 28 <= n <= 30
    assert info["path"] == path and abs(info["seconds"] - n / 10) < 1e-6
    assert not rec.status()["recording"]


def test_not_recording_ignores_frames_and_max_length_auto_stops(tmp_path):
    rec = Recorder(str(tmp_path), fps=10, max_seconds=1.0)
    rec.write(frame(), 0.0)  # before start: ignored
    assert list(tmp_path.iterdir()) == []
    path = rec.start()
    for i in range(30):
        rec.write(frame(), i * 0.1)
    assert not rec.recording and rec.last_path == path
    assert frames_in(path)[0] == 10

