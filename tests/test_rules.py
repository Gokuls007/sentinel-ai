"""Plain-English rules (Phase 3a): the rule format, each condition, the engine's timers and
cooldowns, pose signals, the plain-words preview, storage and presets."""

from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from rules.describe import describe, warnings
from rules.dsl import Rule, RuleBody, check_references, format_errors, slugify
from rules.engine import PersonState, RuleEngine, SceneState
from rules.presets import builtin_event_type, builtin_rules, preset_rules
from rules.signals import head_turn, holding, looking_down
from rules.store import RuleStore

ZONES = {"dock_1": "Loading Dock", "chem": "Chemical Storage"}


def rule(conditions, **kw):
    return Rule(**{"id": "r1", "name": "Test", "conditions": conditions, **kw})


def body(conditions, **kw):
    return RuleBody(**{"name": "Test", "conditions": conditions, **kw})


# --- the rule format ----------------------------------------------------------------------------

@pytest.mark.parametrize("cond", [
    {"type": "in_zone", "zone": "dock_1"},
    {"type": "not_in_zone", "zone": "dock_1"},
    {"type": "fallen"},
    {"type": "stationary_for", "seconds": 30},
    {"type": "posture_risk_at_least", "level": 4},
    {"type": "count_greater_than", "n": 3},
    {"type": "count_in_zone_greater_than", "zone": "dock_1", "n": 2},
    {"type": "time_window", "start": "22:00", "end": "06:00", "days": ["mon", "fri"]},
    {"type": "holding_object", "object": "cell phone"},
    {"type": "head_turned", "direction": "left", "min_angle": 40},
    {"type": "looking_down"},
])
def test_every_condition_validates(cond):
    assert body([cond]).conditions[0].type == cond["type"]


@pytest.mark.parametrize(("cond", "msg"), [
    ({"type": "helmet_missing"}, "does not match any of the expected tags"),
    ({"type": "near_object", "object": "forklift"}, "does not match any of the expected tags"),
    ({"type": "in_zone"}, "Field required"),
    ({"type": "fallen", "zone": "x"}, "Extra inputs are not permitted"),
    ({"type": "stationary_for", "seconds": 0}, "greater than 0"),
    ({"type": "posture_risk_at_least", "level": 7}, "less than or equal to 5"),
    ({"type": "count_greater_than", "n": -1}, "greater than or equal to 0"),
    ({"type": "time_window", "start": "25:00", "end": "06:00"}, "HH:MM"),
    ({"type": "time_window", "start": "09:00", "end": "09:00"}, "empty"),
    ({"type": "time_window", "start": "09:00", "end": "17:00", "days": ["funday"]}, "Input should be"),
    ({"type": "holding_object", "object": "forklift"}, "Input should be"),
    ({"type": "head_turned", "min_angle": 5}, "greater than or equal to 15"),
])
def test_bad_conditions_are_rejected_with_clear_messages(cond, msg):
    with pytest.raises(ValidationError) as e:
        body([cond])
    assert msg in "; ".join(format_errors(e.value))


def test_rule_level_checks():
    with pytest.raises(ValidationError, match="twice"):
        body([{"type": "fallen"}, {"type": "fallen"}])
    with pytest.raises(ValidationError, match="never hold"):
        body([{"type": "in_zone", "zone": "a"}, {"type": "not_in_zone", "zone": "a"}])
    with pytest.raises(ValidationError, match="one time_window"):
        body([{"type": "time_window", "start": "01:00", "end": "02:00"},
              {"type": "time_window", "start": "03:00", "end": "04:00"}])
    with pytest.raises(ValidationError, match="'alert' action"):
        body([{"type": "fallen"}], actions=["notify"])
    with pytest.raises(ValidationError):
        body([])
    with pytest.raises(ValidationError):
        Rule(id="Bad Id!", name="x", conditions=[{"type": "fallen"}])
    b = body([{"type": "count_greater_than", "n": 1}, {"type": "time_window", "start": "22:00", "end": "06:00"}])
    assert not b.per_person and body([{"type": "fallen"}]).per_person


def test_zone_references_must_exist():
    b = body([{"type": "in_zone", "zone": "dock_9"}])
    (problem,) = check_references(b, ZONES)
    assert "dock_9" in problem and "dock_1 (Loading Dock)" in problem
    assert check_references(body([{"type": "in_zone", "zone": "dock_1"}]), ZONES) == []


def test_slugify_avoids_taken_ids():
    assert slugify("Loading Dock dwell!", set()) == "loading-dock-dwell"
    assert slugify("Loading Dock dwell", {"loading-dock-dwell"}) == "loading-dock-dwell-2"
    assert slugify("!!!", set()) == "rule"


# --- the engine --------------------------------------------------------------------------------

def person(tid=1, x=100.0, zones=(), **kw):
    return PersonState(track_id=tid, point=(x, 300.0), body_height=170.0, zones=set(zones), **kw)


def scene(ts, *persons, camera="cam-0", local=None):
    return SceneState(ts=ts, camera_id=camera, persons={p.track_id: p for p in persons},
                      local_time=local or datetime(2026, 10, 5, 12, 0))


def run(engine, frames, t0=0.0, fps=10):
    """frames: list of (seconds, [persons]) -> list of (ts, firing)."""
    out, t = [], t0
    for seconds, persons in frames:
        for _ in range(round(seconds * fps)):
            t += 1 / fps
            out += [(t, f) for f in engine.evaluate(scene(t, *persons))]
    return out


def test_dwell_rule_fires_after_its_duration_once_per_stay():
    e = RuleEngine([rule([{"type": "in_zone", "zone": "dock_1"}], duration_s=30, cooldown_s=0)])
    fired = run(e, [(29.5, [person(zones=["dock_1"])])])
    assert fired == []
    fired = run(e, [(1, [person(zones=["dock_1"])]), (60, [person(zones=["dock_1"])])], t0=29.5)
    assert len(fired) == 1 and fired[0][1].held_s == pytest.approx(30, abs=0.15)
    assert fired[0][1].zone_id == "dock_1" and fired[0][1].track_id == 1
    # Leaves, comes back: a new stay, a new alert after another 30 s.
    fired = run(e, [(3, [person()]), (31, [person(zones=["dock_1"])])], t0=100)
    assert len(fired) == 1


def test_brief_dropouts_do_not_restart_the_timer():
    e = RuleEngine([rule([{"type": "in_zone", "zone": "dock_1"}], duration_s=10)], grace_s=1.0)
    inside, out = [person(zones=["dock_1"])], [person()]
    fired = run(e, [(6, inside), (0.5, out), (4, inside)])
    assert len(fired) == 1  # 10.5 s since entering; the 0.5 s blip was ignored
    e2 = RuleEngine([rule([{"type": "in_zone", "zone": "dock_1"}], duration_s=10)], grace_s=1.0)
    assert run(e2, [(6, inside), (2, out), (6, inside)]) == []  # a real exit restarts it


def test_cooldown_spaces_out_alerts_across_episodes():
    e = RuleEngine([rule([{"type": "fallen"}], cooldown_s=60)])
    down, up = [person(fallen=True)], [person()]
    fired = run(e, [(1, down), (2, up), (1, down), (2, up)])
    assert len(fired) == 1  # the second fall came within the cooldown
    fired = run(e, [(1, down)], t0=100)
    assert len(fired) == 1


def test_each_person_has_their_own_timer():
    e = RuleEngine([rule([{"type": "in_zone", "zone": "dock_1"}], duration_s=5)])
    a, b = person(1, zones=["dock_1"]), person(2, x=400, zones=["dock_1"])
    fired = run(e, [(3, [a]), (3, [a, b]), (3, [a, b])])
    assert [f.track_id for _t, f in fired] == [1, 2]


def test_not_in_zone_and_people_leaving():
    e = RuleEngine([rule([{"type": "not_in_zone", "zone": "dock_1"}])])
    assert len(run(e, [(1, [person()])])) == 1
    assert run(e, [(1, [person(zones=["dock_1"])])], t0=5) == []
    run(e, [(3, [])], t0=10)
    assert not e._timers  # nobody there: timers dropped


def test_stationary_for_resets_when_the_person_moves():
    e = RuleEngine([rule([{"type": "stationary_for", "seconds": 10, "radius_body_heights": 0.3}])])
    assert run(e, [(8, [person(x=100)]), (8, [person(x=200)])]) == []  # moved 0.6 body heights
    fired = run(e, [(11, [person(x=210)])], t0=16)  # small drift within the radius
    assert len(fired) == 1


def test_posture_holding_head_and_looking_down_conditions():
    cases = [
        ({"type": "posture_risk_at_least", "level": 4}, {"reba_level": 4}, {"reba_level": 3}),
        ({"type": "posture_risk_at_least", "level": 4}, {"reba_level": 5}, {"reba_level": None}),
        ({"type": "holding_object", "object": "cell phone"}, {"holding": {"cell phone"}}, {"holding": {"book"}}),
        ({"type": "head_turned", "direction": "left", "min_angle": 30}, {"head": ("left", 45, 0.9)},
         {"head": ("right", 45, 0.9)}),
        ({"type": "head_turned", "direction": "either", "min_angle": 30}, {"head": ("right", 35, 0.9)},
         {"head": ("right", 20, 0.9)}),
        ({"type": "head_turned"}, {"head": ("left", 60, 0.8)}, {"head": ("left", 60, 0.3)}),  # low confidence
        ({"type": "looking_down"}, {"looking_down": True}, {"looking_down": None}),
    ]
    for cond, yes, no in cases:
        e = RuleEngine([rule([cond])])
        assert len(run(e, [(0.2, [person(**yes)])])) == 1, cond
        e = RuleEngine([rule([cond])])
        assert run(e, [(0.2, [person(**no)])]) == [], cond


def test_scene_rules_fire_once_for_the_scene():
    e = RuleEngine([rule([{"type": "count_in_zone_greater_than", "zone": "dock_1", "n": 1}], duration_s=2)])
    two = [person(1, zones=["dock_1"]), person(2, x=300, zones=["dock_1"])]
    assert run(e, [(1, two), (0.5, two[:1])]) == []
    fired = run(e, [(3, two)], t0=10)
    assert len(fired) == 1 and fired[0][1].track_id is None and fired[0][1].zone_id == "dock_1"
    e = RuleEngine([rule([{"type": "count_greater_than", "n": 2}])])
    assert run(e, [(1, two)]) == [] and len(run(e, [(1, [*two, person(3)])], t0=5)) == 1


@pytest.mark.parametrize(("start", "end", "days", "when", "inside"), [
    ("09:00", "17:00", [], datetime(2026, 10, 5, 12, 0), True),
    ("09:00", "17:00", [], datetime(2026, 10, 5, 17, 0), False),
    ("22:00", "06:00", [], datetime(2026, 10, 5, 23, 30), True),
    ("22:00", "06:00", [], datetime(2026, 10, 5, 3, 0), True),
    ("22:00", "06:00", [], datetime(2026, 10, 5, 12, 0), False),
    ("22:00", "06:00", ["fri"], datetime(2026, 10, 10, 2, 0), True),   # Sat 02:00 belongs to Friday night
    ("22:00", "06:00", ["fri"], datetime(2026, 10, 9, 2, 0), False),   # Fri 02:00 is Thursday night
    ("09:00", "17:00", ["sat", "sun"], datetime(2026, 10, 5, 12, 0), False),  # a Monday
])
def test_time_window(start, end, days, when, inside):
    e = RuleEngine([rule([{"type": "time_window", "start": start, "end": end, "days": days},
                          {"type": "count_greater_than", "n": 0}])])
    assert bool(e.evaluate(scene(1.0, person(), local=when))) is inside


def test_rules_only_run_on_their_cameras_and_reload_drops_timers():
    r = rule([{"type": "in_zone", "zone": "dock_1"}], duration_s=5, camera_ids=["cam-1"])
    e = RuleEngine([r])
    assert e.evaluate(scene(1, person(zones=["dock_1"]), camera="cam-0")) == []
    for t in range(2, 5):
        e.evaluate(scene(t, person(zones=["dock_1"]), camera="cam-1"))
    assert e._timers
    e.set_rules([r.model_copy(update={"duration_s": 10, "version": 2})])
    assert not e._timers  # an edited rule starts fresh
    e.set_rules([r.model_copy(update={"enabled": False})])
    assert e.rules == []


# --- pose signals --------------------------------------------------------------------------------

def face(nose_x=320.0, nose_y=200.0, ears=((290, 200), (350, 200)), conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[0] = (nose_x, nose_y, conf)
    k[1], k[2] = (305, 190, conf), (335, 190, conf)
    for i, e in zip((3, 4), ears, strict=True):
        if e is not None:
            k[i] = (*e, conf)
    return k


def test_head_turn_direction_angle_and_profile():
    assert head_turn(face())[1] == pytest.approx(0)
    d, angle, conf = head_turn(face(nose_x=341.2))  # sin = 21.2 / 30 -> 45 degrees
    assert d == "left" and angle == pytest.approx(45, abs=0.5) and conf == pytest.approx(0.9)
    assert head_turn(face(nose_x=290))[0] == "right"
    d, angle, _ = head_turn(face(nose_x=360, ears=((330, 200), None)))  # one ear hidden
    assert angle == 75 and d == "left"
    assert head_turn(face(conf=0.2)) is None


def test_looking_down_and_holding():
    assert looking_down(face()) is False
    assert looking_down(face(nose_y=225)) is True  # 25 px below the ear line, ear span 60
    assert looking_down(face(ears=((290, 200), None))) is None
    k = np.zeros((17, 3), np.float32)
    k[9] = (100, 300, 0.9)
    k[10] = (400, 300, 0.2)  # the right wrist isn't visible
    objects = [("cell phone", (110, 290, 130, 330)), ("book", (390, 280, 420, 320))]
    assert holding(k, 170, objects) == {"cell phone"}
    assert holding(k, 170, [("cell phone", (300, 290, 330, 330))]) == set()


# --- preview, storage, presets ------------------------------------------------------------------

def test_preview_reads_like_the_rule():
    b = body([{"type": "in_zone", "zone": "dock_1"}], duration_s=30, actions=["alert", "record_clip"])
    assert describe(b, ZONES) == "Person · in Loading Dock · for 30 s → medium alert, clip"
    b = body([{"type": "count_in_zone_greater_than", "zone": "dock_1", "n": 3},
              {"type": "time_window", "start": "22:00", "end": "06:00", "days": ["sat", "sun"]}],
             severity="high", actions=["alert", "notify"], duration_s=120)
    assert describe(b, ZONES) == ("Scene · more than 3 people in Loading Dock · between 22:00 and 06:00 on Sat, Sun"
                                  " · for 2 min → high alert, notify")


def test_preview_warnings():
    b = body([{"type": "in_zone", "zone": "dock_1"}])
    assert any("Loading Dock" in w for w in warnings(b, ZONES, "if someone is in the loading zone"))
    assert warnings(b, ZONES, "anyone in the loading dock for 30 s") == []
    turned = body([{"type": "head_turned"}], cooldown_s=5)
    w = " ".join(warnings(turned, ZONES, "head turned"))
    assert "duration" in w and "2D estimate" in w and "Cooldown" in w


def test_store_round_trip_and_preset_replacement(tmp_path):
    store = RuleStore(str(tmp_path / "events.db"))
    mine = rule([{"type": "fallen"}])
    store.save(mine)
    store.save(mine.model_copy(update={"name": "Renamed"}))
    assert [r.name for r in store.list()] == ["Renamed"] and store.get("r1").name == "Renamed"
    preset = preset_rules("warehouse", [])
    store.replace_preset("warehouse", preset)
    store.replace_preset("warehouse", preset[:1])
    assert {r.id for r in store.list()} == {"r1", preset[0].id}  # your own rule is never touched
    assert store.delete("r1") and not store.delete("r1")


def test_builtins_and_presets_from_zones():
    zones = [{"id": "chem", "name": "Chemical Storage", "zone_type": "restricted"},
             {"id": "Dock 1", "name": "Loading Dock", "zone_type": "time_limited", "time_limit": 10},
             {"id": "lane", "name": "Forklift Lane", "zone_type": "one_way"}]
    built = builtin_rules(zones, fall_cooldown_s=45)
    assert [r.id for r in built] == ["builtin-fall", "builtin-zone-chem"]
    assert builtin_event_type(built[0]) == "fall" and builtin_event_type(built[1]) == "zone_intrusion"
    assert built[0].cooldown_s == 45 and all(r.builtin for r in built)
    wh = preset_rules("warehouse", zones)
    dwell = [r for r in wh if r.id.startswith("wh-dwell")]
    assert len(dwell) == 1 and dwell[0].id == "wh-dwell-dock-1" and dwell[0].duration_s == 10
    assert builtin_event_type(wh[0]) is None
    with pytest.raises(ValueError, match="3d"):
        preset_rules("exam", zones)
