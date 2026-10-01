"""Agreement metrics of scripts/eval_ergo.py (no video, no models)."""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("eval_ergo", os.path.join(ROOT, "scripts", "eval_ergo.py"))
eval_ergo = importlib.util.module_from_spec(spec)
sys.modules["eval_ergo"] = eval_ergo
spec.loader.exec_module(eval_ergo)


def test_perfect_agreement():
    m = eval_ergo.agreement([(1, 1), (3, 3), (5, 5), (2, 2)])
    assert m["exact"] == 1.0 and m["within_one"] == 1.0 and m["kappa"] == pytest.approx(1.0)


def test_off_by_one_counts_within_one_but_not_exact():
    m = eval_ergo.agreement([(2, 3), (4, 4), (3, 2), (1, 1)])
    assert m["exact"] == 0.5 and m["within_one"] == 1.0
    assert 0 < m["kappa"] < 1


def test_weighted_kappa_penalises_big_disagreements_more():
    small = eval_ergo.weighted_kappa([(1, 2), (2, 1), (3, 3), (4, 4), (5, 5)])
    big = eval_ergo.weighted_kappa([(1, 5), (5, 1), (3, 3), (4, 4), (2, 2)])
    assert small > big


def test_known_kappa_value():
    # Hand-checked: 2x2 effective table on levels 1 and 2 only.
    pairs = [(1, 1)] * 20 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 15
    assert eval_ergo.weighted_kappa(pairs) == pytest.approx(0.4, abs=1e-9)


def test_report_separates_reliable_frames_and_excludes_unsure():
    labels = {
        "0": {"level": 1}, "30": {"level": 4}, "60": {"level": 4}, "90": {"level": None}, "120": {"level": 3},
    }
    system = {
        0: {"level": 1, "reliable": True}, 30: {"level": 4, "reliable": True},
        60: {"level": 2, "reliable": False}, 120: {"level": None, "reliable": False},
    }
    section, s = eval_ergo.report("clip.mp4", labels, system, "cpu")
    assert s["labelled"] == 4 and s["unsure"] == 1
    assert s["reliable"]["n"] == 2 and s["reliable"]["exact"] == 1.0
    assert s["all_scored"]["n"] == 3 and s["all_scored"]["exact"] == pytest.approx(2 / 3)
    assert s["coverage"] == 0.5
    assert "2D approximation of REBA" in section and "1 marked unsure" in section
