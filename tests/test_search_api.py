"""POST /api/search (streamed) and GET /api/search/status with a scripted LLM."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "search"))

from fakes import ScriptedLLM, answer, calls
from seed import EVENT_IDS, build_store

from api import server
from llm import LLMError

FALL = EVENT_IDS["wed_fall"]


@pytest.fixture
def api(tmp_config, tmp_path, monkeypatch):
    store = build_store(str(tmp_path / "search.db"))
    engine = SimpleNamespace(zone_overlay_data=[{"id": "loading_dock", "name": "Loading Dock"}])
    monkeypatch.setattr(server, "pipeline", SimpleNamespace(config=tmp_config, event_store=store,
                                                            anomaly_engine=engine))
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.search.allow_remote = True  # TestClient's host is "testclient"
    server._search_times.clear()
    state = {"llm": None}

    def fake_build(cfg):
        if state["llm"] is None:
            raise LLMError("LLM_PROVIDER=nvidia needs NVIDIA_API_KEY in .env")
        return state["llm"]

    monkeypatch.setattr("llm.build_llm", fake_build)
    with TestClient(server.app) as c:
        c.state = state
        c.cfg = tmp_config
        yield c


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_search_streams_steps_then_answer_with_cited_events(api):
    api.state["llm"] = ScriptedLLM(
        calls(("find_events", {"types": ["fall"], "zone": "dock", "limit": 1})),
        answer(f"The latest fall was on the Loading Dock [#{FALL}]; also [#77]."),
    )
    r = api.post("/api/search", json={"question": "latest fall at the dock?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    steps = events(r)
    assert [s["type"] for s in steps] == ["tool_call", "tool_result", "done"]
    assert steps[0]["arguments"]["zone"] == "dock"
    done = steps[-1]
    assert done["stop"] == "answered" and done["removed_citations"] == [77]
    assert [c["id"] for c in done["citations"]] == [FALL]
    assert done["citations"][0]["type"] == "fall" and "clip_url" in done["citations"][0]
    assert "[#77]" not in done["answer"] and done["provider"] == "scripted"


def test_llm_failure_mid_run_is_reported_in_the_stream(api):
    api.state["llm"] = ScriptedLLM(LLMError("nvidia returned HTTP 503"))
    done = events(api.post("/api/search", json={"question": "q"}))[-1]
    assert done["type"] == "done" and done["stop"] == "llm_error" and "503" in done["error"]


def test_unconfigured_provider_is_503_and_status_explains(api):
    r = api.post("/api/search", json={"question": "q"})
    assert r.status_code == 503 and "NVIDIA_API_KEY" in r.json()["detail"]
    s = api.get("/api/search/status").json()
    assert s["enabled"] is False and s["provider"] == "nvidia" and "NVIDIA_API_KEY" in s["reason"]
    assert s["model"] == "nvidia/nemotron-3-super-120b-a12b"


def test_status_when_configured_never_includes_keys(api):
    api.state["llm"] = ScriptedLLM()
    api.cfg.llm.nvidia_api_key = "nvapi-secret"
    s = api.get("/api/search/status")
    assert s.json() == {"enabled": True, "provider": "scripted", "model": "scripted-1", "reason": None}
    assert "nvapi-secret" not in s.text


def test_search_is_local_only_by_default_and_needs_json(api):
    api.state["llm"] = ScriptedLLM(answer("ok"), answer("ok"))
    api.cfg.search.allow_remote = False
    assert api.post("/api/search", json={"question": "q"}).status_code == 403
    api.cfg.search.allow_remote = True
    r = api.post("/api/search", content="question=q", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code in (415, 422)


def test_question_validation_and_rate_limit(api):
    api.state["llm"] = ScriptedLLM(*[answer("ok") for _ in range(5)])
    assert api.post("/api/search", json={"question": ""}).status_code == 422
    assert api.post("/api/search", json={"question": "x" * 501}).status_code == 422
    api.cfg.search.rate_limit_per_min = 2
    codes = [api.post("/api/search", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_zone_names_come_from_running_cameras(api):
    assert server._zone_names() == {"loading_dock": "Loading Dock"}
