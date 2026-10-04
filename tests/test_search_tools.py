"""Search tools over the seeded event log (no LLM)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "search"))

from seed import EVENT_IDS, NOW, SEED, ZONES, build_store, now_ts

from search.tools import SearchTools, ToolError, parse_local_time, tool_specs


@pytest.fixture(scope="module")
def seeded_store(tmp_path_factory):
    return build_store(str(tmp_path_factory.mktemp("search") / "events.db"))


@pytest.fixture
def tools(seeded_store):
    return SearchTools(seeded_store, ZONES, now=now_ts)


def test_seed_ids_follow_order_and_cover_every_type(tools):
    assert tools.store.total() == len(SEED) == 45
    assert tools.store.get(EVENT_IDS["wed_fall"]).type == "fall"
    assert {e.type for e in SEED} == {"fall", "zone_intrusion", "time_exceeded", "wrong_direction", "loitering",
                                      "ergo_risk", "near_miss"}


def test_current_time_is_injected_with_calendar_ranges():
    t = SearchTools(store=None, now=now_ts).current_time()
    assert (t["now"], t["weekday"]) == ("2026-09-30T15:00:00", "Wednesday")
    r = t["ranges"]
    assert r["today"] == {"start": "2026-09-30", "end": "2026-09-30"}
    assert r["yesterday"] == {"start": "2026-09-29", "end": "2026-09-29"}
    assert r["this_week"] == {"start": "2026-09-28", "end": "2026-10-04"}
    assert r["last_week"] == {"start": "2026-09-21", "end": "2026-09-27"}
    assert r["last_weekend"] == {"start": "2026-09-26", "end": "2026-09-27"}
    assert r["last_month"] == {"start": "2026-08-01", "end": "2026-08-31"}


def test_ranges_on_a_monday_and_in_january():
    from datetime import datetime

    from search.tools import date_ranges

    r = date_ranges(datetime(2027, 1, 4, 9, 0))  # a Monday
    assert r["this_week"]["start"] == "2027-01-04" and r["last_week"] == {"start": "2026-12-28", "end": "2027-01-03"}
    assert r["last_month"] == {"start": "2026-12-01", "end": "2026-12-31"}


@pytest.mark.parametrize(("args", "expected"), [
    ({"types": ["fall"], "start": "2026-09-30", "end": "2026-09-30"}, 1),
    ({"start": "2026-09-29", "end": "2026-09-29"}, 7),
    ({"types": ["zone_intrusion"], "zone": "chemical storage", "start": "2026-09-28"}, 3),
    ({"types": ["fall"], "start": "2026-09-21", "end": "2026-09-27"}, 2),
    ({"types": ["wrong_direction"], "zone": "forklift"}, 7),
    ({"min_severity": "critical", "start": "2026-09-28"}, 6),
    ({"camera": "cam-1", "start": "2026-09-29", "end": "2026-09-29"}, 2),
    ({"types": ["ergo_risk"], "track_id": 2}, 3),
    ({"types": ["loitering"]}, 4),
])
def test_counts_match_the_hand_written_answers(tools, args, expected):
    """The eval's expected numbers, checked against the store through the tools."""
    assert tools.count_events(**args)["total"] == expected


def test_breakdown_by_zone_uses_display_names(tools):
    counts = tools.count_events(group_by="zone", start="2026-09-28", end="2026-10-04")["counts"]
    assert counts == {"(no zone)": 1, "Yard Gate [gate]": 3, "Loading Dock [loading_dock]": 5,
                      "Forklift Lane [forklift_lane]": 6, "Chemical Storage [storage]": 4,
                      "Workbench [workbench]": 3}


def test_weekday_and_hour_groups_but_never_per_track(tools):
    assert tools.count_events(group_by="weekday", types=["fall"])["counts"] == {
        "Monday": 1, "Tuesday": 2, "Wednesday": 1, "Friday": 2}
    hours = tools.count_events(group_by="hour", types=["zone_intrusion"])["counts"]
    assert max(hours, key=hours.get) == "18" and hours["18"] == 2
    with pytest.raises(ToolError):
        tools.count_events(group_by="track", types=["ergo_risk"])


def test_find_orders_and_records_seen_ids(tools):
    r = tools.find_events(types=["time_exceeded"], order="longest", limit=1)
    assert r["total_matching"] == 7 and r["returned"] == 1
    top = r["events"][0]
    assert top["id"] == EVENT_IDS["sep21_dock_longest"] and top["duration_s"] == 1200
    assert top["zone"] == "Loading Dock [loading_dock]" and top["start"] == "2026-09-21T06:40:00"
    assert tools.seen_event_ids == {top["id"]}
    newest = tools.find_events(types=["fall"], limit=1)["events"][0]
    assert newest["id"] == EVENT_IDS["wed_fall"] and newest["verified"] is True


def test_find_limit_is_capped_and_window_is_inclusive_of_bare_end_date(tools):
    assert tools.find_events(limit=500)["returned"] == 45  # cap is 50
    r = tools.find_events(types=["zone_intrusion"], start="2026-09-29T18:00:00", end="2026-09-29")
    assert {e["id"] for e in r["events"]} == {EVENT_IDS["tue_storage_evening"], EVENT_IDS["tue_gate_night"]}


def test_get_event(tools):
    e = tools.get_event(id=21)
    assert e["found"] and e["type"] == "zone_intrusion" and e["end"] == "2026-09-25T18:30:00"
    assert tools.get_event(id=999) == {"found": False, "id": 999}
    assert 21 in tools.seen_event_ids and 999 not in tools.seen_event_ids


def test_list_values(tools):
    zones = {z["id"]: z for z in tools.list_values(field="zone")["zones"]}
    assert zones["storage"]["name"] == "Chemical Storage" and zones["storage"]["events"] == 8
    assert {c["id"] for c in tools.list_values(field="camera")["cameras"]} == {"cam-0", "cam-1", "laptop"}


@pytest.mark.parametrize(("call", "message"), [
    (lambda t: t.count_events(zone="mezzanine"), "no zone matches"),
    (lambda t: t.count_events(zone="a"), "ambiguous"),
    (lambda t: t.count_events(types=["explosion"]), "unknown event type"),
    (lambda t: t.count_events(group_by="colour"), "group_by must be"),
    (lambda t: t.count_events(start="yesterday"), "can't read time"),
    (lambda t: t.count_events(start="2026-09-30", end="2026-09-29"), "end is before start"),
    (lambda t: t.find_events(order="random"), "order must be"),
    (lambda t: t.find_events(sql="DROP TABLE events"), "unknown argument"),
    (lambda t: t.run("delete_events", {}), "unknown tool"),
    (lambda t: t.get_event(id="x"), "id must be an integer"),
])
def test_bad_arguments_raise_tool_errors_for_the_model(tools, call, message):
    with pytest.raises(ToolError, match=message):
        call(tools)
    assert tools.store.total() == 45  # nothing changed


def test_parse_local_time_bare_dates():
    start = parse_local_time("2026-09-30", end_of_day=False)
    end = parse_local_time("2026-09-30", end_of_day=True)
    assert start == NOW.replace(hour=0).timestamp()
    assert 86_399 < end - start < 86_400


def test_tool_specs_are_valid_json_schema_objects():
    names = {s.name for s in tool_specs()}
    assert names == {"current_time", "list_values", "count_events", "find_events", "get_event"}
    for s in tool_specs():
        assert s.parameters["type"] == "object" and isinstance(s.parameters["properties"], dict)


def test_long_results_are_truncated():
    text = SearchTools.to_text({"x": "y" * 20_000})
    assert len(text) < 12_200 and text.endswith('lower the limit)"')
