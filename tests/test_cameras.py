"""Laptop-webcam camera API with a fake pipeline (no models, no real camera)."""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import server
from events import EventBus


class FakeSource:
    def __init__(self):
        self.stopped = threading.Event()
        self.stats = {"error_message": "", "hardware_error": False}

    def stop(self):
        self.stopped.set()


class FakePipeline:
    """Stands in for SentinelPipeline: 'captures' until its source is stopped."""

    instances: list = []  # noqa: RUF012

    def __init__(self, cfg):
        self.config = cfg
        self.event_bus = EventBus()
        self.video_source = FakeSource()
        self.frame_count = 0
        self.stopped = False
        self._on_frame = None
        self.anomaly_engine = SimpleNamespace(zone_overlay_data=[], fall_detector=SimpleNamespace(state_of=str))
        self.pose_estimator = SimpleNamespace(get_all_features=dict)
        FakePipeline.instances.append(self)

    @property
    def stats(self):
        return {"avg_fps": 12.5, "frames_processed": self.frame_count}

    def on_frame(self, callback):
        self._on_frame = callback
        return callback

    def run(self):
        while not self.video_source.stopped.wait(0.02):
            self.frame_count += 1
            if self._on_frame:
                self._on_frame(SimpleNamespace(frame_number=self.frame_count, annotated_frame=None,
                                               to_dict=lambda: {"stats": {}, "n": self.frame_count}))
        self.stop()

    def stop(self):
        self.stopped = True


@pytest.fixture
def cam_client(tmp_config, monkeypatch):
    FakePipeline.instances.clear()
    primary = SimpleNamespace(config=tmp_config, stats={"avg_fps": 9.9})
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "SentinelPipeline", FakePipeline)
    tmp_config.allow_remote_camera_control = True  # TestClient's host is "testclient"
    for d in (server._cameras, server._camera_state, server._camera_frames, server._camera_threads):
        d.clear()
    with TestClient(server.app) as c:
        yield c
    server.stop_cameras()
    for thread in list(server._camera_threads.values()):
        thread.join(timeout=5)


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def laptop(client):
    return next(c for c in client.get("/api/cameras").json() if c["id"] == "laptop")


def test_start_run_and_stop_the_laptop_camera(cam_client, tmp_config):
    assert laptop(cam_client)["status"] == "stopped"
    r = cam_client.post("/api/cameras/laptop/start", json={"index": 1})
    assert r.status_code == 202 and r.json()["status"] == "starting"
    assert wait_for(lambda: laptop(cam_client)["status"] == "running")

    fake = FakePipeline.instances[0]
    assert fake.config.source == "1" and fake.config.camera_id == "laptop" and fake.config.loop is False
    with open(fake.config.zone.zones_file, encoding="utf-8") as f:
        assert json.load(f) == {"zones": []}  # no borrowed zones
    assert tmp_config.camera_id == "cam-0"  # the primary config is untouched
    assert cam_client.get("/api/stats", params={"camera": "laptop"}).json()["avg_fps"] == 12.5

    # starting again while running is a no-op
    assert cam_client.post("/api/cameras/laptop/start", json={}).json()["status"] == "running"
    assert len(FakePipeline.instances) == 1

    stopped = cam_client.post("/api/cameras/laptop/stop", json={}).json()
    assert stopped["status"] == "stopped"
    assert fake.video_source.stopped.is_set() and fake.stopped  # the device is released
    assert cam_client.get("/api/stats", params={"camera": "laptop"}).status_code == 404


def test_websocket_streams_the_requested_cameras_frames(cam_client):
    cam_client.post("/api/cameras/laptop/start", json={"index": 0})
    assert wait_for(lambda: "laptop" in server._camera_frames)
    with cam_client.websocket_connect("/ws/feed?camera=laptop") as ws:
        assert ws.receive_json()["type"] == "history"
        frame = ws.receive_json()
        assert frame["type"] == "frame" and frame["data"]["n"] >= 1


def test_camera_control_requires_this_computer_and_json(cam_client, tmp_config):
    tmp_config.allow_remote_camera_control = False
    assert cam_client.post("/api/cameras/laptop/start", json={}).status_code == 403
    tmp_config.allow_remote_camera_control = True
    # A cross-site page can POST text/plain without a CORS preflight; even with a JSON-looking
    # body it must be refused (FastAPI rejects the body with 422, or our check with 415).
    for body in ("index=0", '{"index": 0}'):
        r = cam_client.post("/api/cameras/laptop/start", content=body, headers={"content-type": "text/plain"})
        assert r.status_code in (415, 422), r.status_code
    r = cam_client.post("/api/cameras/laptop/stop", content="{}", headers={"content-type": "text/plain"})
    assert r.status_code == 415
    assert cam_client.post("/api/cameras/laptop/start", json={"index": 42}).status_code == 422
    assert FakePipeline.instances == []


def test_stop_during_startup_releases_everything(cam_client, monkeypatch):
    gate = threading.Event()

    class SlowPipeline(FakePipeline):
        def __init__(self, cfg):
            gate.wait(5)  # "loading models"
            super().__init__(cfg)

    monkeypatch.setattr(server, "SentinelPipeline", SlowPipeline)
    cam_client.post("/api/cameras/laptop/start", json={})
    stopping = threading.Thread(target=lambda: cam_client.post("/api/cameras/laptop/stop", json={}))
    stopping.start()
    time.sleep(0.1)
    gate.set()
    stopping.join(5)
    assert wait_for(lambda: FakePipeline.instances and FakePipeline.instances[0].stopped)
    assert laptop(cam_client)["status"] == "stopped"
    assert "laptop" not in server._cameras


def test_unknown_camera_is_404(cam_client):
    assert cam_client.get("/api/tracks", params={"camera": "garage"}).status_code == 404
    assert cam_client.get("/api/stats", params={"camera": "cam-0"}).json()["avg_fps"] == 9.9  # primary
