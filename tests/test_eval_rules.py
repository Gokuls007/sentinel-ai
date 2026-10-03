"""The rule-compiler eval: every case is well-formed, and grading is right (no LLM here)."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_rules import CASES, grade, markdown, summarize
from rules.dsl import RuleBody, check_references


def load():
    with open(CASES, encoding="utf-8") as f:
        return json.load(f)


def test_every_case_is_valid_and_uses_real_zones():
    doc = load()
    names = {k: v[0] for k, v in doc["zones"].items()}
    ids = [c["id"] for c in doc["cases"]]
    assert len(ids) == len(set(ids)) >= 46
    assert sum(bool(c["expect"].get("refuse")) for c in doc["cases"]) >= 10
    for case in doc["cases"]:
        for exp in [case["expect"], *case.get("accept", [])]:
            if exp.get("refuse"):
                continue
            body = RuleBody(name="x", conditions=exp["conditions"], duration_s=exp.get("duration_s", 0))
            assert check_references(body, names) == [], case["id"]


def rule_result(conditions, duration=0, **kw):
    return {"status": "rule", "attempts": 1, "rule": {"conditions": conditions, "duration_s": duration,
                                                      "severity": "medium", "actions": ["alert"], **kw}}


CASE = {"id": "t", "domain": "warehouse", "text": "x",
        "expect": {"conditions": [{"type": "in_zone", "zone": "dock_1"}, {"type": "stationary_for", "seconds": 15}],
                   "duration_s": 0, "severity": "high", "notify": True},
        "accept": [{"conditions": [{"type": "in_zone", "zone": "dock_1"}], "duration_s": 15}],
        "loose": ["radius_body_heights"]}


def test_exact_semantic_and_loose_parameters():
    order_swapped = [{"type": "stationary_for", "seconds": 15, "radius_body_heights": 0.5},
                     {"type": "in_zone", "zone": "dock_1"}]
    g = grade(CASE, rule_result(order_swapped, severity="high", actions=["alert", "notify"]))
    assert g["passed"] and g["exact"] and g["severity_ok"] and g["notify_ok"]
    g = grade(CASE, rule_result([{"type": "in_zone", "zone": "dock_1"}], 15))
    assert g["passed"] and not g["exact"] and g["severity_ok"] is False and g["notify_ok"] is False
    g = grade(CASE, rule_result([{"type": "in_zone", "zone": "dock_1"}], 20))
    assert not g["passed"] and "in_zone(zone=dock_1) for 20 s" in g["failures"][0]


def test_refusal_cases():
    case = {"id": "r", "domain": "warehouse", "text": "x", "expect": {"refuse": True}}
    assert grade(case, {"status": "refusal"})["passed"]
    g = grade(case, rule_result([{"type": "fallen"}]))
    assert not g["passed"] and "should have been refused" in g["failures"][0]
    assert not grade(CASE, {"status": "refusal", "refusal": "no"})["passed"]


def test_summary_counts_the_dangerous_failure_separately():
    rows = []
    for i, (kind, status, passed) in enumerate([("rule", "rule", True), ("rule", "refusal", False),
                                                 ("refusal", "rule", False), ("refusal", "refusal", True)]):
        rows.append({"id": f"c{i}", "run": 1, "source": "claude", "domain": "warehouse", "kind": kind,
                     "status": status, "passed": passed, "exact": passed, "attempts": 1, "latency_s": 1.0,
                     "tokens": 100, "severity_ok": None, "notify_ok": None, "failures": [] if passed else ["x"]})
    s = summarize(rows)
    assert s["passed"] == 2 and s["false_rules"] == 1 and s["wrong_refusals"] == 1
    md = markdown(s, "nvidia", "m", rows, 1)
    assert "**1**" in md and "upper bound" in md


def test_parity_matching_by_type_track_and_time():
    from rules_parity import match

    a = [("fall", 1, 10.0), ("zone_intrusion", 2, 5.0), ("zone_intrusion", 2, 40.0)]
    b = [("zone_intrusion", 2, 5.3), ("fall", 1, 10.4), ("fall", 3, 12.0)]
    matched, only_a, only_b = match(a, b, 0.5)
    assert matched == 2 and only_a == [("zone_intrusion", 2, 40.0)] and only_b == [("fall", 3, 12.0)]
    assert match([("fall", 1, 10.0)], [("fall", 1, 10.6)], 0.5)[0] == 0
