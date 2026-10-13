"""Room map: floor, leg-hiding furniture and standing height learned from people."""

import numpy as np

from anomaly.fall_detector import FallDetector
from core.pose_estimator import PoseResult
from scene.room_model import RoomModel

W, H = 1280, 720


def person(cx, top, h, feet=True, tid=1):
    """Upright person, box from ``top`` to ``top + h``; with ``feet`` the whole leg is seen."""
    kp = np.zeros((17, 3), np.float32)
    y = lambda f: top + f * h  # noqa: E731
    for i, (dx, f) in {0: (0, 0.06), 5: (-0.08, 0.2), 6: (0.08, 0.2), 11: (-0.05, 0.52), 12: (0.05, 0.52),
                       13: (-0.05, 0.75), 14: (0.05, 0.75), 15: (-0.05, 0.98), 16: (0.05, 0.98)}.items():
        kp[i] = (cx + dx * h, y(f), 0.9)
    if not feet:
        kp[13:17, 2] = 0.0
    box = np.array([cx - 0.15 * h, top, cx + 0.15 * h, top + h], np.float32)
    return PoseResult(track_id=tid, keypoints=kp, bbox=box)


def room_seen(frames=200):
    room = RoomModel(W, H)
    rng = np.random.default_rng(0)
    for i in range(frames):
        foot = rng.uniform(450, 690)
        seen = 2.0 * foot - 700  # top of the head to the feet; further away (higher foot) = smaller
        h = seen / 0.98  # the ankles sit at 98% of the box
        poses = {1: person(rng.uniform(200, 1000), foot - seen, h)}
        # behind a couch whose top edge is at y=500, x 100-500: box cut there, no feet
        poses[2] = person(rng.uniform(200, 400), 250, 250, feet=False, tid=2)
        room.update(poses, i * 0.1)
    return room


def test_floor_furniture_and_standing_height_are_learned():
    room = room_seen()
    s = room.status()
    assert s["floor_cells"] > 5 and s["furniture_cells"] > 0 and s["height_ready"]
    assert abs(room.standing_height(600.0) - 500.0) < 30  # h = 2 * 600 - 700


def test_legs_hidden_only_where_the_furniture_is():
    room = room_seen()
    assert room.legs_hidden([250, 250, 330, 500])        # box ends on the couch edge
    assert not room.legs_hidden([800, 300, 900, 650])     # ends on the floor elsewhere


def test_one_distance_is_not_enough_for_a_height_ruler():
    # Kitchen recording, 2026-10-12: only one person's feet seen, all at one spot.
    room = RoomModel(W, H)
    for i in range(200):
        room.update({1: person(640.0, 250.0, 440.0)}, i * 0.1)
    assert not room.status()["height_ready"] and room.standing_height(600) is None


def test_saved_and_reloaded_unless_the_camera_sees_another_view(tmp_path):
    path = str(tmp_path / "room_laptop.json")
    frame = np.full((H, W, 3), 90, np.uint8)
    room = RoomModel(W, H, path=path)
    for i in range(200):
        room.update({2: person(300.0, 250.0, 250.0, feet=False, tid=2)}, i * 0.1, frame)
    room.save()
    again = RoomModel(W, H, path=path)
    again.update({}, 0.0, frame)
    assert again.status()["furniture_cells"] == room.status()["furniture_cells"] > 0
    moved = RoomModel(W, H, path=path)
    moved.update({}, 0.0, np.full((H, W, 3), 200, np.uint8))  # the laptop was moved
    assert moved.status()["furniture_cells"] == 0


def test_fall_rule_with_legs_hidden_asks_for_a_lying_torso_and_the_head_down():
    fd = FallDetector(ground_mode="combined")
    bent = {"horizontal_pose": False, "head_dropped": False, "head_known": True, "hip_height": 0.05,
            "spread": 0.4, "torso_angle": 40.0, "feet_seen": True, "legs_hidden": True}
    assert not fd._on_ground(bent)  # bent over a sink behind a couch
    assert not fd._on_ground({**bent, "horizontal_pose": True, "torso_angle": 70.0})  # head still up
    assert fd._on_ground({**bent, "horizontal_pose": True, "torso_angle": 70.0, "head_dropped": True})


def test_room_api_shows_hides_and_relearns(tmp_config, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True  # TestClient's host is "testclient"
    room = room_seen()
    fake = SimpleNamespace(room=room, show_room_map=False, config=tmp_config)
    monkeypatch.setattr(server, "pipeline", fake)
    monkeypatch.setattr(server, "config", tmp_config)
    with TestClient(server.app) as c:
        body = c.get("/api/room").json()
        assert body["furniture_cells"] > 0 and body["show"] is False
        assert c.put("/api/room", json={"show": True}).json()["show"] is True and fake.show_room_map
        assert c.post("/api/room/reset", json={}).json()["furniture_cells"] == 0
