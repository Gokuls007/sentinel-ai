"""Grading, statistics and the question file of scripts/eval_search.py (no LLM)."""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "search"))
spec = importlib.util.spec_from_file_location("eval_search", os.path.join(ROOT, "scripts", "eval_search.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

from seed import EVENT_IDS

IDS = {"a": 1, "b": 2, "c": 3}


@pytest.mark.parametrize(("text", "n", "ok"), [
    ("There were 3 falls.", 3, True),
    ("Three falls.", 3, True),
    ("At 13:20 and 03:00.", 3, False),  # times don't count
    ("Event #3 only.", 3, False),
    ("30 events", 3, False),
    ("1,300 events", 3, False),
    ("No falls (0).", 0, True),
    ("Total: 22 events", 22, True),
])
def test_mentions_number(text, n, ok):
    assert ev.mentions_number(text, n) is ok


def test_grade_count_events_none_and_labels():
    assert ev.grade({"count": 2}, "Two falls.", [], IDS) == []
    assert ev.grade({"count": 2}, "Three falls.", [], IDS) == ["count 2 not in answer"]
    assert ev.grade({"events": ["a", "b"]}, "x", [2, 1], IDS) == []
    assert ev.grade({"events": ["a"]}, "x", [1, 3], IDS) == ["events: cited [1, 3], expected [1]"]
    assert ev.grade({"none": True}, "No falls were recorded.", [], IDS) == []
    assert ev.grade({"none": True}, "Here it is.", [1], IDS) == [
        "none: cited [1]", "none: answer doesn't say there were none"]
    assert ev.grade({"labels_all": ["Loading Dock", "08:02"]}, "loading dock at 08:02", [], IDS) == []
    assert ev.grade({"labels_any": ["6 pm", "18:00"]}, "Mostly around 18:00.", [], IDS) == []
    assert ev.grade({"labels_any": ["6 pm"]}, "noon", [], IDS)


def test_unicode_spaces_and_hyphens_are_normalized():
    answer = "20 minutes in the Loading Dock zone, 6‑7 pm"
    assert ev.grade({"labels_all": ["Loading Dock"], "labels_any": ["6-7 pm"], "count": 20}, answer, [], IDS) == []


def test_grade_breakdown_needs_label_and_number_together():
    expect = {"breakdown": {"Loading Dock": 5, "Yard Gate": 3}}
    good = "- Loading Dock: 5 events\n- Yard Gate: 3 events\n- Workbench: 3"
    assert ev.grade(expect, good, [], IDS) == []
    swapped = "- Loading Dock: 3 events\n- Yard Gate: 5 events"
    assert len(ev.grade(expect, swapped, [], IDS)) == 2
    prose = "The Loading Dock had 5 events. The Yard Gate had 3."
    assert ev.grade(expect, prose, [], IDS) == []


def test_wilson_and_percentile():
    lo, hi = ev.wilson(27, 30)
    assert 0.73 < lo < 0.75 and 0.96 < hi < 0.97
    assert ev.wilson(0, 0) == (0.0, 0.0)
    assert ev.percentile([1, 2, 3, 4], 0.5) == 2.5 and ev.percentile([], 0.5) is None


def test_summary_and_markdown():
    rows = [
        {"id": "q1", "source": "claude", "category": "count+zone", "passed": True, "latency_s": 2.0, "tokens": 1000,
         "steps": 2, "stop": "answered", "removed_citations": [], "failures": []},
        {"id": "u1", "source": "user", "category": "list", "passed": False, "latency_s": 4.0, "tokens": 3000,
         "steps": 3, "stop": "max_steps", "removed_citations": [9], "failures": ["events: cited [], expected [1]"]},
    ]
    s = ev.summarize(rows)
    assert s["accuracy"] == 0.5 and s["failed_runs"] == 1 and s["removed_citations"] == 1
    assert s["by_category"] == {"count": {"passed": 1, "total": 1}, "list": {"passed": 0, "total": 1}}
    assert s["user_questions"] == 1 and s["latency_p50_s"] == 3.0
    md = ev.markdown(s, "nvidia", "nvidia/nemotron-3-super-120b-a12b", rows)
    assert md.startswith(ev.SECTION_START) and md.endswith(ev.SECTION_END)
    assert "1 / 2 (50.0%)" in md and "`u1` (events: cited [], expected [1])" in md


def test_repeated_runs_report_per_run_accuracy_and_flaky_questions():
    def row(qid, run, passed):
        return {"id": qid, "run": run, "source": "claude", "category": "count", "passed": passed, "latency_s": 1.0,
                "tokens": 10, "steps": 2, "stop": "answered", "removed_citations": [],
                "failures": [] if passed else ["count 3 not in answer"]}

    rows = [row("q1", 1, True), row("q2", 1, False), row("q3", 1, False),
            row("q1", 2, True), row("q2", 2, True), row("q3", 2, False)]
    s = ev.summarize(rows)
    assert s["runs"] == 2 and s["per_run_accuracy"] == [pytest.approx(1 / 3), pytest.approx(2 / 3)]
    assert s["flaky"] == ["q2"] and s["always_failed"] == ["q3"] and s["accuracy"] == 0.5
    md = ev.markdown(s, "nvidia", "m", rows)
    assert "2 runs of every question" in md and "33.3%, 66.7%" in md and "`q3` run 2 (count 3" in md


def test_question_file_is_well_formed():
    with open(ev.QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) and len(questions) >= 22
    kinds = {"count", "breakdown", "events", "none", "labels_all", "labels_any"}
    for q in questions:
        assert q["question"].strip() and q["category"] and q["expect"]
        assert set(q["expect"]) <= kinds, q["id"]
        for key in q["expect"].get("events", []):
            assert key in EVENT_IDS, f"{q['id']}: unknown seed key {key}"


def test_write_section_appends_then_replaces(tmp_path):
    path = tmp_path / "B.md"
    path.write_text("# B\n", encoding="utf-8")
    ev.write_section(str(path), f"{ev.SECTION_START}\none\n{ev.SECTION_END}")
    ev.write_section(str(path), f"{ev.SECTION_START}\ntwo\n{ev.SECTION_END}")
    text = path.read_text(encoding="utf-8")
    assert "## Search accuracy" in text and "two" in text and "one" not in text
