"""REST + WebSocket API without a running pipeline (no models needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from api import server


@pytest.fixture
def client(tmp_config, monkeypatch):
    monkeypatch.setattr(server, "pipeline", None)
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "_latest_message", None)
    monkeypatch.setattr(server, "_latest_frame_number", -1)
    server.alert_history.clear()
    with TestClient(server.app) as c:
        yield c
    server.alert_history.clear()


def alert(i, severity="high", alert_type="zone_intrusion"):
    return {"alert_id": f"ALT-{i:06d}", "alert_type": alert_type, "track_id": i, "timestamp": float(i),
            "confidence": 0.9, "severity": severity, "message": f"alert {i}", "details": {}}


def test_health_reports_degraded_without_pipeline(client):
    body = client.get("/api/health").json()
    assert body["status"] == "degraded" and body["pipeline"] is False


def test_endpoints_needing_pipeline_return_503(client):
    for path in ("/api/stats", "/api/zones", "/api/tracks"):
        assert client.get(path).status_code == 503


def test_alerts_from_memory_newest_first_and_filtered(client):
    server.alert_history.extend([alert(1), alert(2, "low", "loitering"), alert(3)])
    ids = [a["alert_id"] for a in client.get("/api/alerts").json()]
    assert ids == ["ALT-000003", "ALT-000002", "ALT-000001"]
    low = client.get("/api/alerts", params={"severity": "low"}).json()
    assert [a["alert_type"] for a in low] == ["loitering"]
    assert len(client.get("/api/alerts", params={"limit": 1}).json()) == 1
    assert client.get("/api/alerts", params={"limit": 0}).status_code == 422


def test_clip_and_snapshot_serving(client, tmp_config):
    from pathlib import Path

    d = Path(tmp_config.output.clips_dir) / "ALT-ABC123"
    d.mkdir(parents=True)
    (d / "clip_ALT-ABC123.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (d / "snapshot_ALT-ABC123.jpg").write_bytes(b"\xff\xd8\xff")
    r = client.get("/api/clips/ALT-ABC123")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert client.get("/api/snapshots/ALT-ABC123").status_code == 200
    assert client.get("/api/clips/ALT-NOPE").status_code == 404


@pytest.mark.parametrize("bad", ["..", "../../etc/passwd", "a.b", "ALT 1", "..\\x"])
def test_incident_ids_outside_clips_dir_rejected(bad):
    assert server._incident_file(bad, "clip", "mp4") is None


@pytest.mark.parametrize("url", ["/api/clips/..%2F..%2Fetc", "/api/clips/a.b", "/api/nope"])
def test_bad_api_urls_are_404(client, url):
    assert client.get(url).status_code == 404


def test_websocket_sends_history_then_frames_and_alerts(client):
    server.alert_history.extend([alert(1), alert(2)])
    with client.websocket_connect("/ws/feed") as ws:
        first = ws.receive_json()
        assert first["type"] == "history"
        assert [a["alert_id"] for a in first["alerts"]] == ["ALT-000001", "ALT-000002"]

        server._latest_message = json.dumps({"type": "frame", "image": None, "data": {"frame_number": 5}})
        server._latest_frame_number = 5
        msg = ws.receive_json()
        assert msg["type"] == "frame" and msg["data"]["frame_number"] == 5

        server._broadcast_alert(json.dumps({"type": "alert", "alert": alert(9)}))
        msg = ws.receive_json()
        while msg["type"] != "alert":  # frames are only re-sent when new, but be tolerant
            msg = ws.receive_json()
        assert msg["alert"]["alert_id"] == "ALT-000009"
    assert not server._clients  # disconnected client was removed


def test_tracks_stats_zones_with_numpy_values(client, monkeypatch, tmp_path):
    """Pose features are numpy types; the JSON endpoints must still serialise them."""
    from types import SimpleNamespace

    import numpy as np

    from anomaly.fall_detector import FallDetector
    from anomaly.zone_monitor import ZoneMonitor
    from conftest import make_features

    feat = make_features(track_id=4)
    feat.centroid_history.extend([np.array([1.0, 2.0], np.float32), np.array([4.0, 6.0], np.float32)])
    feat.timestamps.extend([np.float32(1.0), np.float32(2.0)])
    zones = ZoneMonitor(zones_file=str(tmp_path / "z.json"))
    fake = SimpleNamespace(
        pose_estimator=SimpleNamespace(get_all_features=lambda: {4: feat}),
        anomaly_engine=SimpleNamespace(fall_detector=FallDetector(), zone_overlay_data=zones.get_zones_for_overlay()),
        stats={"avg_fps": np.float32(17.5), "active_tracks": np.int64(1)},
    )
    monkeypatch.setattr(server, "pipeline", fake)
    (track,) = client.get("/api/tracks").json()
    assert track["track_id"] == 4 and track["speed"] == 5.0 and track["fall_state"] == "upright"
    assert client.get("/api/stats").json()["avg_fps"] == 17.5
    assert len(client.get("/api/zones").json()) == 3
