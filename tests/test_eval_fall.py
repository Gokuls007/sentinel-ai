"""Scoring and label parsing of training/eval_fall.py (no dataset, no models)."""

import importlib.util
import os
import sys
import zipfile

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("eval_fall", os.path.join(ROOT, "training", "eval_fall.py"))
eval_fall = importlib.util.module_from_spec(spec)
sys.modules["eval_fall"] = eval_fall  # dataclasses look their module up here
spec.loader.exec_module(eval_fall)
SequenceResult, score, summarize = eval_fall.SequenceResult, eval_fall.score, eval_fall.summarize


def seq(name, alerts, onset_frame=None, frames=300):
    r = SequenceResult(name=name, is_fall=onset_frame is not None, frames=frames, alert_times_s=alerts)
    return score(r, onset_frame, tolerance_s=2.0)


def test_fall_detected_after_onset_is_a_hit_with_latency():
    r = seq("fall-01", [4.5], onset_frame=91)  # onset at 3.0 s
    assert (r.tp, r.fp, r.fn) == (1, 0, 0) and r.latency_s == pytest.approx(1.5)


def test_alert_slightly_before_labelled_onset_still_counts():
    r = seq("fall-02", [2.0], onset_frame=91)  # 1 s early, inside the 2 s tolerance
    assert r.tp == 1 and r.latency_s == pytest.approx(-1.0)


def test_early_alert_is_false_positive_and_missed_fall_is_false_negative():
    r = seq("fall-03", [0.5], onset_frame=151)  # 4.5 s before onset at 5.0 s
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_repeated_alerts_in_one_fall_count_extra_as_false_positives():
    r = seq("fall-04", [3.5, 9.0], onset_frame=91)
    assert (r.tp, r.fp) == (1, 1)


def test_every_adl_alert_is_a_false_alarm():
    r = seq("adl-01", [5.0, 20.0])
    assert (r.tp, r.fp, r.fn) == (0, 2, 0)


def test_summary_metrics_and_false_alarms_per_hour_uses_adl_duration():
    results = [
        seq("fall-01", [4.0], onset_frame=91),
        seq("fall-02", [], onset_frame=91),
        seq("adl-01", [1.0], frames=54_000),  # 30 min at 30 fps
        seq("adl-02", [], frames=54_000),
    ]
    s = summarize(results)
    assert s["tp"] == 1 and s["fn"] == 1 and s["fp_in_adl_sequences"] == 1
    assert s["recall"] == pytest.approx(0.5) and s["precision"] == pytest.approx(0.5)
    assert s["adl_hours"] == pytest.approx(1.0)
    assert s["false_alarms_per_hour"] == pytest.approx(1.0)
    md = eval_fall.markdown(s, "cpu", 2.0)
    assert "1.000 h (60.0 min, 108000 frames)" in md  # the basis of the rate is always stated


def test_onsets_are_the_first_falling_frame_per_sequence(tmp_path):
    labels = tmp_path / "urfall-cam0-falls.csv"
    labels.write_text(
        "fall-01,1,-1,0.1\nfall-01,40,0,0.2\nfall-01,41,0,0.3\nfall-01,60,1,0.4\n"
        "fall-02,10,-1,0\nfall-02,25,0,0\nnot-a-row\n",
        encoding="utf-8",
    )
    assert eval_fall.fall_onsets(str(labels)) == {"fall-01": 40, "fall-02": 25}


def test_frames_from_zip_are_read_in_numeric_order(tmp_path):
    path = tmp_path / "fall-01-cam0-rgb.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in (10, 2, 1):  # stored out of order, and "10" sorts before "2" as text
            _, png = cv2.imencode(".png", np.full((4, 4, 3), i, np.uint8))
            zf.writestr(f"fall-01-cam0-rgb-{i:03d}.png" if i < 10 else f"fall-01-cam0-rgb-{i}.png", png.tobytes())
    values = [int(img[0, 0, 0]) for _, img in eval_fall.frames_from_zip(str(path))]
    assert values == [1, 2, 10]
