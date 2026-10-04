"""Skeleton-only mode: events keep keypoints (10 s before, 5 s after), never video or images."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from output.skeleton_recorder import SkeletonRecorder


def pose(x):
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0], kp[:, 1], kp[:, 2] = x, 360.0, 0.9
    return SimpleNamespace(keypoints=kp, bbox=(x - 50, 100, x + 50, 600))


def test_keypoints_around_the_alert_are_saved_once_the_post_seconds_arrive(tmp_path):
    rec = SkeletonRecorder(str(tmp_path), pre_s=10, post_s=5)
    path = None
    for i in range(300):  # 30 s at 10 fps; the alert at 20 s
        ts = i / 10
        rec.add_frame(ts, SkeletonRecorder.frame_people({3: pose(100 + i)}, 1280, 720))
        if i == 200:
            path = rec.save("ALT-1", ts, track_id=3, meta={"type": "fall"})
        if i == 240:
            assert not (tmp_path / "ALT-1" / "skeleton_ALT-1.json").exists()  # still collecting
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert doc["alert_id"] == "ALT-1" and doc["type"] == "fall" and doc["track_id"] == 3
    times = [f["t"] for f in doc["frames"]]
    assert times[0] == pytest.approx(-10) and times[-1] == pytest.approx(5)
    person = doc["frames"][0]["people"][0]
    assert person["track_id"] == 3 and len(person["keypoints"]) == 17
    assert all(0 <= v <= 1 for kp in person["keypoints"] for v in kp[:2])  # normalised, no pixels
    assert sorted(p.name for p in (tmp_path / "ALT-1").iterdir()) == ["skeleton_ALT-1.json"]  # nothing else


def test_flush_writes_pending_skeletons_when_the_source_stops(tmp_path):
    rec = SkeletonRecorder(str(tmp_path))
    rec.add_frame(1.0, SkeletonRecorder.frame_people({1: pose(100)}, 640, 480))
    path = rec.save("ALT-2", 1.0)
    rec.flush()
    assert len(json.loads(Path(path).read_text(encoding="utf-8"))["frames"]) == 1


def test_notifications_are_text_only_without_a_thumbnail():
    from events.schema import Event
    from notifications.base import summarize

    n = summarize(Event("fall", "critical", 1.0, 1.0, alert_id="ALT-3"))
    assert n.image_path is None


# --- API ----------------------------------------------------------------------------------------


@pytest.fixture
def api(tmp_config, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True
    fake = SimpleNamespace(config=tmp_config, skeleton_only=False)
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "pipeline", fake)
    monkeypatch.setattr(server, "_retention_worker", None)
    monkeypatch.setattr(server, "_recordings_dir", lambda: tmp_path / "recordings")
    server._cameras.clear()
    with TestClient(server.app) as c:
        c.fake = fake
        yield c
    server._cameras.clear()


def test_skeleton_endpoint_serves_only_files_under_clips_dir(api, tmp_config, tmp_path):
    folder = tmp_path / "clips" / "ALT-9"
    folder.mkdir(parents=True)
    (folder / "skeleton_ALT-9.json").write_text('{"frames": []}')
    assert api.get("/api/skeletons/ALT-9").json() == {"frames": []}
    assert api.get("/api/skeletons/ALT-404").status_code == 404
    assert api.get("/api/skeletons/..%2F..%2Fsecret").status_code == 404


def test_turning_skeleton_only_on_reaches_running_pipelines_and_blocks_recording(api):
    from api import server

    laptop = SimpleNamespace(skeleton_only=False, video_source=SimpleNamespace(recorder=None))
    server._cameras[server.LAPTOP_CAMERA] = laptop
    api.put("/api/privacy", json={"skeleton_only": True})
    assert api.fake.skeleton_only and laptop.skeleton_only
    r = api.post("/api/cameras/laptop/record/start", json={})
    assert r.status_code == 409 and "Skeleton-only" in r.json()["detail"]
    api.put("/api/privacy", json={"skeleton_only": False})
    assert not laptop.skeleton_only
    server._cameras.clear()  # the fake has no stop(); the server stops cameras on shutdown
