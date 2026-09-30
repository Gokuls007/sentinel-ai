"""VideoSource (file timeline, EOF, looping) and ClipRecorder."""

import itertools
import time

import cv2
import numpy as np
import pytest

from core.video_source import VideoSource
from output.clip_recorder import ClipRecorder


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "tiny.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, np.uint8)
        out.write(frame)
    out.release()
    return path


def drain(src, timeout=10.0):
    frames, deadline = [], time.time() + timeout
    while src.is_running and time.time() < deadline:
        item = src.read()
        if item is None:
            time.sleep(0.002)
        else:
            frames.append(item)
    return frames


def test_file_source_delivers_every_frame_with_timeline_timestamps(video):
    src = VideoSource(video, loop=False).start()
    frames = drain(src)
    src.stop()
    assert len(frames) == 20
    ts = [t for _, t in frames]
    assert np.allclose(np.diff(ts), 1 / 20, atol=1e-6)
    assert src.stats["finished"] and src.stats["is_file"]


def test_file_source_loops(video):
    src = VideoSource(video, loop=True).start()
    frames, deadline = [], time.time() + 10
    while len(frames) < 45 and time.time() < deadline:
        item = src.read()
        if item is None:
            time.sleep(0.002)
        else:
            frames.append(item)
    src.stop()
    assert len(frames) >= 45
    assert src.loops_completed >= 2
    ts = [t for _, t in frames]
    assert all(b > a for a, b in itertools.pairwise(ts))  # timeline keeps increasing across loops


def test_missing_file_reports_error(tmp_path):
    src = VideoSource(str(tmp_path / "nope.mp4")).start()
    drain(src, timeout=3)
    src.stop()
    assert src.hardware_error and "Failed to open" in src.error_message


def test_clip_has_pre_and_post_alert_footage(tmp_path):
    rec = ClipRecorder(clips_dir=str(tmp_path), buffer_seconds=2, fps=10, post_seconds=1)
    frame = np.zeros((120, 160, 3), np.uint8)
    t = 100.0
    for _ in range(40):                       # 4 s of history; only the last 2 s are kept
        rec.add_frame(frame, t)
        t += 0.1
    path = rec.save_clip("ALT-T1", t, snapshot=frame)
    for _ in range(15):                       # 1.5 s after the alert; clip stops at 1 s
        rec.add_frame(frame, t)
        t += 0.1
    rec.flush()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert 28 <= n <= 33                      # ~2 s pre + ~1 s post at 10 fps
    assert fps == pytest.approx(10, abs=0.5)
    assert (tmp_path / "ALT-T1" / "snapshot_ALT-T1.jpg").is_file()


def test_clip_playback_fps_follows_timestamps(tmp_path):
    rec = ClipRecorder(clips_dir=str(tmp_path), buffer_seconds=10, fps=25, post_seconds=0)
    frame = np.zeros((64, 64, 3), np.uint8)
    for i in range(50):                       # processed at 5 fps although nominal is 25
        rec.add_frame(frame, i * 0.2)
    path = rec.save_clip("ALT-T2", 10.0)
    rec.flush()
    cap = cv2.VideoCapture(path)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(5, abs=0.5)
    cap.release()
