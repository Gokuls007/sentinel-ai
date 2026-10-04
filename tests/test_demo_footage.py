"""'Test with demo footage': clips come only from a server-built list; playing one starts the
"test" camera in warehouse mode with notifications off."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def demo_api(tmp_config, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import server

    rec = tmp_path / "recordings"
    rec.mkdir()
    (rec / "lifting_side.mp4").write_bytes(b"x")
    tmp_config.output.db_path = str(tmp_path / "events.db")
    tmp_config.notifications.telegram_bot_token = "secret"
    tmp_config.allow_remote_camera_control = True
    monkeypatch.setattr(server, "pipeline", SimpleNamespace(config=tmp_config))
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "_recordings_dir", lambda: rec)
    started = []
    monkeypatch.setattr(server, "_run_camera", lambda cam, cfg, mode=None: started.append((cam, cfg, mode)))
    server._cameras.clear()
    server._camera_state.pop(server.TEST_CAMERA, None)
    with TestClient(server.app) as c:
        c.started = started
        yield c


def test_clips_include_my_recordings_and_hide_paths(demo_api):
    clips = demo_api.get("/api/demo/clips").json()["clips"]
    mine = [c for c in clips if c["source"] == "recording"]
    assert mine == [{"id": "rec-lifting_side", "label": "lifting_side.mp4 (your recording)", "source": "recording"}]
    assert all("path" not in c for c in clips)
    assert all("subject 6" not in c["label"] for c in clips)  # held-out subjects never offered


def test_play_starts_a_warehouse_test_camera_without_notifications(demo_api):
    import time

    r = demo_api.post("/api/demo/play", json={"clip": "rec-lifting_side"})
    assert r.status_code == 202 and r.json()["label"].startswith("lifting_side.mp4")
    for _ in range(50):
        if demo_api.started:
            break
        time.sleep(0.02)
    cam, cfg, mode = demo_api.started[0]
    assert cam == "test" and mode == "warehouse" and cfg.loop and cfg.source.endswith("lifting_side.mp4")
    assert cfg.notifications.telegram_bot_token == "" and cfg.output.webhook_url == ""
    assert demo_api.post("/api/demo/play", json={"clip": "../../etc/passwd"}).status_code == 404
    assert demo_api.post("/api/demo/stop", json={}).json()["id"] == "test"


def test_event_bus_unsubscribe():
    from events.bus import EventBus

    bus, seen = EventBus(), []
    bus.subscribe("store", lambda e: seen.append(("store", e)))
    bus.subscribe("websocket", lambda e: seen.append(("ws", e)))
    bus.unsubscribe("store")
    bus.publish(SimpleNamespace(alert_id="a1"))
    assert [n for n, _e in seen] == ["ws"]
