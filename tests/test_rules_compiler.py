"""The rule compiler (scripted LLM: no network), the rules API and the pipeline hookup."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from llm.base import LLMError
from rules.compiler import RuleCompiler, tool_specs
from rules.dsl import Rule
from rules.engine import RuleEngine

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "search"))

from fakes import ScriptedLLM, answer, calls

ZONES = {"dock_1": ("Loading Dock", "time_limited"), "chem": ("Chemical Storage", "restricted")}
DWELL = {"name": "Loading dock dwell", "conditions": [{"type": "in_zone", "zone": "dock_1"}], "duration_s": 30}


def compile_with(*responses, text="alert if someone stays in the loading dock for more than 30 seconds"):
    llm = ScriptedLLM(*responses)
    return RuleCompiler(llm).compile(text, ZONES, now="Monday 2026-10-05 12:00"), llm


def test_valid_rule_with_preview_and_no_extra_calls():
    res, llm = compile_with(calls(("submit_rule", DWELL)))
    assert res.status == "rule" and res.attempts == 1 and len(llm.requests) == 1
    assert res.body.duration_s == 30 and res.preview == "Person · in Loading Dock · for 30 s → medium alert, clip"
    system = llm.requests[0]["system"]
    assert "dock_1: Loading Dock (time_limited)" in system and "Monday 2026-10-05" in system
    assert llm.requests[0]["messages"][0].content.startswith("alert if someone stays")


def test_tool_schema_is_plain_json_schema():
    submit, refuse = tool_specs()
    s = json.dumps(submit.parameters)
    assert "$defs" in submit.parameters and '"title"' not in s and "discriminator" not in s
    assert refuse.parameters["required"] == ["reason"]


def test_one_repair_turn_fixes_an_invalid_rule():
    bad = {**DWELL, "conditions": [{"type": "in_zone", "zone": "loading_zone"}], "duration_s": -5}
    res, llm = compile_with(calls(("submit_rule", bad)), calls(("submit_rule", DWELL)))
    assert res.status == "rule" and res.attempts == 2
    feedback = llm.requests[1]["messages"][-1]
    assert feedback.role == "tool" and "duration_s" in feedback.content and "loading_zone" in feedback.content


def test_still_invalid_after_the_repair_is_an_error_not_a_guess():
    bad = {**DWELL, "conditions": [{"type": "in_zone", "zone": "dock_9"}]}
    res, llm = compile_with(calls(("submit_rule", bad)), calls(("submit_rule", bad)))
    assert res.status == "error" and "dock_9" in res.error and res.body is None and len(llm.requests) == 2


def test_refusal_names_what_is_missing():
    res, _ = compile_with(calls(("refuse", {"reason": "Helmets can't be detected yet.", "unsupported": ["helmet"]})),
                          text="alert if someone in the dock has no helmet")
    assert res.status == "refusal" and "Helmets" in res.refusal and res.unsupported == ["helmet"]


def test_a_condition_from_a_later_phase_becomes_a_clear_refusal():
    near = {**DWELL, "conditions": [{"type": "near_object", "object": "forklift"}]}
    res, _ = compile_with(calls(("submit_rule", near)), text="alert if someone is near a forklift")
    assert res.status == "refusal" and "3b" in res.refusal and res.unsupported == ["near_object"]


def test_no_tool_call_gets_one_nudge_and_wrapped_arguments_are_accepted():
    res, llm = compile_with(answer("Sure! Here's your rule."), calls(("submit_rule", {"rule": DWELL})))
    assert res.status == "rule" and llm.requests[1]["messages"][-1].content == "Call submit_rule or refuse."


def test_provider_errors_and_empty_text():
    res, _ = compile_with(LLMError("quota exceeded"))
    assert res.status == "error" and "quota" in res.error
    res, llm = compile_with(text="   ")
    assert res.status == "error" and llm.requests == []


# --- API -----------------------------------------------------------------------------------------

@pytest.fixture
def rules_api(tmp_config, monkeypatch):
    from fastapi.testclient import TestClient

    from api import server
    from rules.store import RuleStore

    store = RuleStore(tmp_config.output.db_path)
    from events import EventStore

    events = EventStore(tmp_config.output.db_path)
    zones = [{"id": "dock_1", "name": "Loading Dock", "type": "time_limited"},
             {"id": "chem", "name": "Chemical Storage", "type": "restricted"}]
    reloads = []
    zone_objs = [SimpleNamespace(id="dock_1", time_limit=10.0), SimpleNamespace(id="chem", time_limit=None)]
    primary = SimpleNamespace(
        config=tmp_config, mode="warehouse", paused=False, event_store=events, rule_store=store,
        rules=RuleEngine(), reload_rules=lambda: reloads.append(1),
        anomaly_engine=SimpleNamespace(zone_overlay_data=zones, zone_monitor=SimpleNamespace(zones=zone_objs)))
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.allow_remote_camera_control = True
    tmp_config.search.allow_remote = True
    server._cameras.clear()
    server._compile_times.clear()
    scripted = []
    monkeypatch.setattr("llm.build_llm", lambda cfg: scripted.pop(0) if scripted else ScriptedLLM())
    with TestClient(server.app) as c:
        c.store, c.events, c.reloads, c.scripted = store, events, reloads, scripted
        yield c


def test_compile_then_confirm_then_manage(rules_api):
    api = rules_api
    api.scripted.append(ScriptedLLM(calls(("submit_rule", DWELL))))
    draft = api.post("/api/rules/compile", json={"text": "someone in the loading dock over 30 seconds"}).json()
    assert draft["status"] == "rule" and draft["camera_ids"] == ["cam-0"] and api.store.list() == []
    saved = api.post("/api/rules", json={"text": "someone in the loading dock over 30 seconds",
                                         "rule": {**draft["rule"], "severity": "high"}}).json()
    assert saved["id"] == "loading-dock-dwell" and saved["severity"] == "high" and api.reloads == [1]
    assert saved["preview"].endswith("high alert, clip") and saved["source_text"].startswith("someone")
    # Fired counts come from the event store.
    from events import Event

    api.events.emit(Event(type="rule:loading-dock-dwell", severity="high", start_ts=1, end_ts=31))
    listed = api.get("/api/rules").json()["rules"]
    assert listed[0]["fired"] == 1 and listed[0]["last_fired"] == 31
    off = api.patch("/api/rules/loading-dock-dwell", json={"enabled": False, "duration_s": 45}).json()
    assert off["enabled"] is False and off["duration_s"] == 45 and off["version"] == 2
    assert api.patch("/api/rules/loading-dock-dwell", json={"severity": "extreme"}).status_code == 422
    assert api.request("DELETE", "/api/rules/loading-dock-dwell", json={}).json() == {"deleted": "loading-dock-dwell"}
    assert api.request("DELETE", "/api/rules/loading-dock-dwell", json={}).status_code == 404


def test_confirm_rejects_invented_zones_and_bad_rules(rules_api):
    bad = {**DWELL, "conditions": [{"type": "in_zone", "zone": "nowhere"}]}
    r = rules_api.post("/api/rules", json={"rule": bad})
    assert r.status_code == 422 and "nowhere" in r.json()["detail"]
    r = rules_api.post("/api/rules", json={"rule": {**DWELL, "conditions": [{"type": "telepathy"}]}})
    assert r.status_code == 422


def test_compile_unavailable_and_rate_limited(rules_api, monkeypatch):
    def no_llm(cfg):
        raise LLMError("LLM_PROVIDER=nvidia needs NVIDIA_API_KEY in .env")

    monkeypatch.setattr("llm.build_llm", no_llm)
    status = rules_api.get("/api/rules/status").json()
    assert status["compile"]["enabled"] is False and "NVIDIA_API_KEY" in status["compile"]["reason"]
    assert status["builtins"] is False and {z["id"] for z in status["zones"]} == {"dock_1", "chem"}
    assert rules_api.post("/api/rules/compile", json={"text": "x"}).status_code == 503
    monkeypatch.setattr("llm.build_llm", lambda cfg: ScriptedLLM(calls(("submit_rule", DWELL))))
    from api import server

    server.pipeline.config.rules.compile_rate_limit_per_min = 2
    codes = [rules_api.post("/api/rules/compile", json={"text": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_presets(rules_api):
    presets = {p["name"]: p for p in rules_api.get("/api/presets").json()["presets"]}
    assert presets["warehouse"]["available"] and not presets["exam"]["available"]
    applied = rules_api.post("/api/presets/warehouse/apply", json={}).json()
    ids = [r["id"] for r in applied["rules"]]
    assert "wh-dwell-dock_1" in ids and "wh-posture-risk" in ids
    assert rules_api.post("/api/presets/warehouse/apply", json={}).status_code == 409
    assert rules_api.post("/api/presets/warehouse/apply", json={"replace": True}).status_code == 200
    assert rules_api.post("/api/presets/exam/apply", json={}).status_code == 400
    assert rules_api.post("/api/presets/nope/apply", json={}).status_code == 404


# --- the pipeline hookup -------------------------------------------------------------------------

@pytest.fixture
def bare_pipeline(tmp_config):
    """A pipeline object with the real analytics and rule plumbing, but no models loaded."""
    from anomaly.engine import AnomalyEngine
    from core.pipeline import SentinelPipeline
    from rules.store import RuleStore

    p = SentinelPipeline.__new__(SentinelPipeline)
    p.config = tmp_config
    p.anomaly_engine = AnomalyEngine(tmp_config)
    p.rules = RuleEngine()
    p.rule_store = RuleStore(tmp_config.output.db_path)
    p.detector = SimpleNamespace(classes=[0])
    return p


def test_rules_turn_into_alerts_with_their_actions(bare_pipeline):
    from conftest import make_pose

    p = bare_pipeline
    p.rule_store.save(Rule(id="phone", name="Phone in hand", conditions=[{"type": "holding_object",
                                                                          "object": "cell phone"}],
                           severity="high", actions=["alert"]))
    p.reload_rules()
    assert p.detector.classes == [0, 67]  # the phone class is detected only because a rule needs it
    pose = make_pose(track_id=7)
    pose.keypoints[9] = (330, 300, 0.9)  # a hand at waist height (make_pose has no wrists)
    wrist = pose.keypoints[9, :2]
    phone = SimpleNamespace(class_name="cell phone", bbox=(wrist[0] - 5, wrist[1] - 5, wrist[0] + 5, wrist[1] + 5))
    dets = SimpleNamespace(detections=[phone])
    (alert,) = p._evaluate_rules({7: pose}, {}, dets, 100.0)
    assert alert.alert_type == "rule:phone" and alert.severity == "high" and alert.track_id == 7
    assert alert.details["record_clip"] is False and alert.details["notify"] is False
    assert p._evaluate_rules({7: pose}, {}, dets, 100.1) == []  # same episode
    p.rule_store.delete("phone")
    p.reload_rules()
    assert p.detector.classes == [0]


def test_builtins_follow_the_flag(bare_pipeline):
    p = bare_pipeline
    p.reload_rules()
    assert p.rules.rules == []
    p.config.rules.builtins = True
    p.reload_rules()
    ids = [r.id for r in p.rules.rules]
    restricted = [z.id for z in p.anomaly_engine.zone_monitor.zones if z.zone_type == "restricted"]
    assert ids == ["builtin-fall", *[f"builtin-zone-{z}" for z in restricted]]


def test_rule_without_notify_is_not_sent():
    from events import Event
    from notifications.dispatcher import NotificationDispatcher

    d = NotificationDispatcher([], min_severity="low")
    quiet = Event(type="rule:x", severity="critical", start_ts=1, end_ts=1, attributes={"notify": False})
    loud = Event(type="rule:y", severity="critical", start_ts=1, end_ts=1, attributes={"notify": True})
    assert d.should_notify(quiet) is False and d.should_notify(loud) is True
