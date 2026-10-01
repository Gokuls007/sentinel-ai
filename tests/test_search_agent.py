"""The search agent loop with a scripted LLM: tool dispatch, budgets, citations, errors."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "search"))

from fakes import ScriptedLLM, answer, calls
from seed import EVENT_IDS, ZONES, build_store, now_ts

from llm.base import LLMError, LLMResponse, Message, ToolCall, Usage
from search.agent import SearchAgent, check_citations, summarize
from search.tools import SearchTools

FALL = EVENT_IDS["wed_fall"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(str(tmp_path_factory.mktemp("agent") / "events.db"))


def agent_with(store, *responses, **kw):
    llm = ScriptedLLM(*responses)
    return SearchAgent(llm, SearchTools(store, ZONES, now=now_ts), **kw), llm


def test_tool_round_trip_and_cited_answer(store):
    agent, llm = agent_with(
        store,
        calls(("find_events", {"types": ["fall"], "start": "2026-09-30", "end": "2026-09-30"})),
        answer(f"One fall today at 08:02 on the Loading Dock [#{FALL}]."),
    )
    steps = list(agent.run("any falls today?"))
    kinds = [s["type"] for s in steps]
    assert kinds == ["tool_call", "tool_result", "done"]
    assert steps[1]["summary"] == "1 of 1 matching events"
    r = steps[-1]["result"]
    assert r.stop == "answered" and r.citations == [FALL] and r.removed_citations == []
    assert r.steps == 2 and r.usage.total == 240 and r.provider == "scripted"
    # The tool result went back to the model as a tool message tied to the call id.
    second = llm.requests[1]["messages"]
    assert second[-1].role == "tool" and second[-1].tool_call_id == "c0" and f'"id":{FALL}' in second[-1].content
    # The system prompt carries the injected "now".
    assert "2026-09-30 15:00 (Wednesday)" in llm.requests[0]["system"]


def test_citations_not_returned_by_tools_are_removed(store):
    agent, _ = agent_with(
        store,
        calls(("get_event", {"id": FALL})),
        answer(f"It was [#{FALL}], like [#4] and [#999]."),
    )
    r = agent.ask("tell me about the fall")
    assert r.citations == [FALL] and r.removed_citations == [4, 999]
    assert r.answer == f"It was [#{FALL}], like and."


def test_tool_errors_go_back_to_the_model_which_can_recover(store):
    agent, llm = agent_with(
        store,
        calls(("count_events", {"zone": "mezzanine"})),
        answer("There is no mezzanine zone; known zones are the Loading Dock, ..."),
    )
    steps = list(agent.run("falls on the mezzanine?"))
    assert steps[1]["summary"].startswith("error: no zone matches 'mezzanine'")
    assert '"error"' in llm.requests[1]["messages"][-1].content
    r = steps[-1]["result"]
    assert r.stop == "answered" and r.tool_calls[0]["error"].startswith("no zone matches")


def test_bad_json_arguments_are_reported_not_run(store):
    bad = LLMResponse(Message(role="assistant", tool_calls=[
        ToolCall(id="x", name="count_events", arguments={}, parse_error="arguments are not valid JSON")]),
        stop="tool_calls", usage=Usage(1, 1))
    agent, llm = agent_with(store, bad, answer("Sorry."))
    agent.ask("q")
    assert "not valid JSON" in llm.requests[1]["messages"][-1].content


def test_step_limit_forces_a_final_answer_without_more_tools(store):
    loop = [calls(("current_time", {})) for _ in range(3)]
    agent, llm = agent_with(store, *loop, answer("Partial: I checked the time only."), max_steps=3)
    r = agent.ask("q")
    assert r.stop == "answered" and r.steps == 4
    assert "Step limit reached" in llm.requests[-1]["messages"][-1].content


def test_step_limit_when_the_model_keeps_calling_tools(store):
    loop = [calls(("current_time", {})) for _ in range(3)]
    agent, _ = agent_with(store, *loop, max_steps=2)
    r = agent.ask("q")
    assert r.stop == "max_steps" and r.answer


def test_token_budget_stops_the_run(store):
    agent, _ = agent_with(store, *[calls(("current_time", {})) for _ in range(5)], max_tokens=250)
    r = agent.ask("q")
    assert r.stop == "token_budget" and "budget 250" in r.error


def test_llm_error_ends_cleanly(store):
    agent, _ = agent_with(store, LLMError("nvidia returned HTTP 500"))
    r = agent.ask("q")
    assert r.stop == "llm_error" and "HTTP 500" in r.error and r.answer == ""


def test_refusal_and_empty_question(store):
    agent, _ = agent_with(store, answer("", stop="refusal"))
    assert agent.ask("q").stop == "refusal"
    agent, llm = agent_with(store)
    assert agent.ask("   ").stop == "llm_error" and llm.requests == []


def test_question_length_is_capped(store):
    agent, llm = agent_with(store, answer("ok"))
    agent.ask("x" * 5000)
    assert len(llm.requests[0]["messages"][0].content) == 500


def test_bare_ids_of_returned_events_become_citations():
    text, kept, removed = check_citations("False alarms: **#4** and #31; also #77 and [#5].", {4, 31})
    assert text == "False alarms: [#4] and [#31]; also #77 and." and kept == [4, 31] and removed == [5]
    assert check_citations("Track #3 at 10:00", {9})[1] == []  # a bare number that wasn't returned stays text


def test_check_citations_dedupes_and_tidies_spacing():
    text, kept, removed = check_citations("A [#1] and [#1], B [#2] .", {1})
    assert kept == [1] and removed == [2] and text == "A [#1] and [#1], B."


@pytest.mark.parametrize(("output", "line"), [
    ({"error": "bad"}, "error: bad"),
    ({"total_matching": 9, "returned": 3}, "3 of 9 matching events"),
    ({"total": 5, "counts": {"a": 2, "b": 3}}, "5 events in 2 groups"),
    ({"total": 5}, "5 events"),
    ({"now": "2026-09-30T15:00:00", "weekday": "Wednesday"}, "Wednesday 2026-09-30 15:00:00"),
    ({"found": False, "id": 7}, "event #7 not found"),
    ({"zones": [1, 2]}, "2 zones"),
])
def test_step_summaries(output, line):
    assert summarize(output) == line
