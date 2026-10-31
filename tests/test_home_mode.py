"""Home care mode: the body-risk engine with bed/chair/couch found automatically and without the
warehouse-only rules (lifting, standing on chairs, hazards, loitering, REBA)."""

from types import SimpleNamespace

import pytest


def test_home_objects_are_furniture_and_loose_floor_things():
    from config.settings import DEFAULT_HOME_OBJECTS, OBJECT_MODES, default_mode_classes

    assert "home" in OBJECT_MODES
    # Furniture people sit or lie on, and loose things that become trip hazards on the floor.
    assert default_mode_classes()["home"] == DEFAULT_HOME_OBJECTS == [
        "bed", "chair", "couch", "backpack", "handbag", "suitcase", "pillow", "cardboard box", "shoe", "clothes"]


def test_home_runs_the_body_engine_but_skips_workplace_alerts():
    from core.pipeline import BODY_MODES, HOME_SKIPPED

    assert "home" in BODY_MODES and "warehouse" in BODY_MODES
    assert set(HOME_SKIPPED) == {"loitering", "ergo_risk"}


@pytest.fixture
def api(tmp_config, monkeypatch):
    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "pipeline", SimpleNamespace(config=tmp_config, mode_classes={}))
    monkeypatch.setattr(server, "_retention_worker", None)
    server._cameras.clear()
    with TestClient(server.app) as c:
        yield c


def test_the_api_accepts_home_mode_and_its_object_list(api):
    assert api.put("/api/app", json={"mode": "home"}).json()["mode"] == "home"
    assert api.get("/api/app").json()["mode"] == "home"
    assert api.get("/api/objects", params={"mode": "home"}).json()["classes"][:3] == ["bed", "chair", "couch"]
    body = {"mode": "home", "classes": ["bed", "chair", "couch", "toilet"]}
    assert api.put("/api/objects", json=body).status_code == 200
