"""Zone polygons whose edges cross are rejected, with a suggested fix."""

import json

import pytest

from anomaly.geometry import convex_hull, fix_polygon, self_intersections
from anomaly.zone_monitor import ZoneMonitor

BOWTIE = [(0.1, 0.1), (0.5, 0.5), (0.5, 0.1), (0.1, 0.5)]
SQUARE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
# The user's real "Zone 1" from the laptop camera (12 points, 4 crossings).
ZONE_1 = [(0.0877, 0.381), (0.2401, 0.3981), (0.1012, 0.6828), (0.2208, 0.7223), (0.2285, 0.5611),
          (0.0549, 0.5885), (0.0491, 0.6605), (0.0606, 0.7257), (0.0761, 0.7189), (0.0915, 0.6948),
          (0.2343, 0.5165), (0.2401, 0.4787)]


def test_crossing_edges_are_found():
    assert self_intersections(BOWTIE) == [(0, 2)]
    assert self_intersections(SQUARE) == []
    assert len(self_intersections(ZONE_1)) == 4
    concave = [(0.1, 0.1), (0.5, 0.1), (0.3, 0.3), (0.5, 0.5), (0.1, 0.5)]  # an arrow shape: fine
    assert self_intersections(concave) == []


def test_fix_keeps_the_points_when_it_can():
    fixed, method = fix_polygon(BOWTIE)
    assert method == "reordered" and sorted(fixed) == sorted(BOWTIE) and self_intersections(fixed) == []
    fixed, method = fix_polygon(ZONE_1)
    assert method == "reordered" and len(fixed) == 12 and self_intersections(fixed) == []
    assert fix_polygon(SQUARE) == (SQUARE, "unchanged")


def test_hull_when_reordering_cannot_work():
    assert sorted(convex_hull([*SQUARE, (0.3, 0.3)])) == sorted(SQUARE)


def test_existing_crossing_zone_is_flagged_with_a_fix(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [{"id": "z1", "name": "Zone 1", "zone_type": "restricted",
                                           "polygon": [list(p) for p in ZONE_1]}]}))
    (zone,) = ZoneMonitor(zones_file=str(path)).get_zones_for_overlay()
    assert zone["self_intersecting"] and zone["fix"]["method"] == "reordered"
    assert self_intersections([tuple(p) for p in zone["fix"]["polygon_normalized"]]) == []


@pytest.fixture
def zones_api(tmp_config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import server

    tmp_config.output.db_path = str(tmp_path / "events.db")
    monitor = ZoneMonitor(zones_file=str(tmp_path / "zones_laptop.json"))
    p = SimpleNamespace(config=tmp_config, anomaly_engine=SimpleNamespace(zone_monitor=monitor,
                                                                          zone_overlay_data=[]))
    monkeypatch.setattr(server, "pipeline", p)
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.allow_remote_camera_control = True
    server._cameras.clear()
    with TestClient(server.app) as c:
        yield c


def zone(points):
    return {"zones": [{"id": "z1", "name": "Zone 1", "zone_type": "restricted", "polygon": [list(p) for p in points]}]}


def test_saving_a_crossing_zone_is_rejected_with_the_fix(zones_api):
    r = zones_api.put("/api/zones?camera=cam-0", json=zone(BOWTIE))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "cross" in detail["message"] and detail["fix"]["method"] == "reordered"
    ok = zones_api.put("/api/zones?camera=cam-0", json=zone(detail["fix"]["polygon"]))
    assert ok.status_code == 200 and ok.json()[0]["self_intersecting"] is False
