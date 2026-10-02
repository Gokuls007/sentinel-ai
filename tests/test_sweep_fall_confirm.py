"""Replay and scoring of training/sweep_fall_confirm.py on synthetic cached poses (no models)."""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "sweep_fall_confirm", os.path.join(ROOT, "training", "sweep_fall_confirm.py"))
sw = importlib.util.module_from_spec(spec)
sys.modules["sweep_fall_confirm"] = sw
spec.loader.exec_module(sw)

from config.settings import FallDetectorConfig
from test_fall_webcam import upper_body

FPS = 30.0
STANDING_H = 480.0


def cached(name, kind, poses, onset_frame=None, missing_after=None):
    """poses: PoseResult per frame (None = person not seen that frame)."""
    v = sw.Cached(name=name, kind=kind, fps=FPS, onset_frame=onset_frame)
    for i, p in enumerate(poses):
        rows = [] if p is None or (missing_after is not None and i >= missing_after) else [(1, p, STANDING_H)]
        v.frames.append((i / FPS, (1,), rows))
    return v


def fall_clip(lying_frames, name="fall-01", missing_after=None, jitter=1.0):
    rng = np.random.default_rng(0)
    standing = [upper_body(jitter=jitter, rng=rng) for _ in range(30)]
    falling = [upper_body(shoulder_y=200 + 25 * k, angle_deg=min(85, 12 * k), jitter=jitter, rng=rng)
               for k in range(1, 9)]
    lying = [upper_body(shoulder_y=400, angle_deg=85, jitter=jitter, rng=rng) for _ in range(lying_frames)]
    return cached(name, "fall", standing + falling + lying, onset_frame=31, missing_after=missing_after)


def standing_clip(frames=900, kind="sample"):
    rng = np.random.default_rng(1)
    return cached("walk", kind, [upper_body(jitter=1.0, rng=rng) for _ in range(frames)])


def test_long_enough_lying_is_confirmed_and_short_settings_fire_earlier():
    video = fall_clip(lying_frames=150)  # 5 s on the ground
    cfg = FallDetectorConfig()
    times = {}
    for s in (0.5, 1.0, 2.0):
        alerts = sw.replay(video, sw.make_detector(cfg, s))["alerts"]
        assert len(alerts) == 1, s
        times[s] = alerts[0]
    assert times[0.5] < times[1.0] < times[2.0]


def test_clip_ending_before_confirmation_is_not_confirmable():
    rows = sw.score_setting([fall_clip(lying_frames=45), standing_clip()], FallDetectorConfig(), 3.0)
    assert rows["tp"] == 0 and rows["on_ground"] == 1 and rows["confirmable"] == 0
    assert rows["confirmable_but_missed"] == {}


def test_five_second_setting_can_fire_despite_the_default_timeout():
    """The 5 s 'gave up' timeout would reset FALLEN before a 5 s confirmation; the sweep raises it."""
    det = sw.make_detector(FallDetectorConfig(), 5.0)
    assert det.fallen_timeout_seconds >= 7.0
    assert len(sw.replay(fall_clip(lying_frames=240), det)["alerts"]) == 1


def test_person_lost_from_view_is_explained():
    video = fall_clip(lying_frames=120, missing_after=30 + 8 + 10)  # vanishes ~0.3 s after landing
    r = sw.score_setting([video], FallDetectorConfig(), 1.0)
    assert r["tp"] == 0 and r["confirmable"] == 1 and r["confirmable_but_missed"] == {"lost from view": 1}


def test_false_alarm_rate_uses_no_fall_hours_only():
    r = sw.score_setting([fall_clip(lying_frames=150), standing_clip(900), standing_clip(900, kind="adl")],
                         FallDetectorConfig(), 1.0)
    assert r["tp"] == 1 and r["false_alarms_clean"] == 0 and r["false_alarms_per_hour"] == 0
    assert r["clean_hours"] == pytest.approx(60 / 3600)
    assert set(r["hours_by_kind"]) == {"sample", "adl"}


def test_markdown_table_marks_the_default():
    videos = [fall_clip(lying_frames=150), standing_clip()]
    rows = [sw.score_setting(videos, FallDetectorConfig(), s) for s in (0.5, 1.0)]
    md = sw.markdown(rows, 1.0, "cpu")
    assert md.startswith(sw.SECTION_START) and md.endswith(sw.SECTION_END)
    assert "| 1 s (default) |" in md and "| 0.5 s |" in md and "On the ground" in md


def test_possible_level_counts_reaching_the_ground():
    videos = [fall_clip(lying_frames=150), standing_clip()]
    r = sw.score_setting(videos, FallDetectorConfig(), 1.0)
    assert r["possible_tp"] == 1 and r["possible_recall"] == 1.0 and r["possible_false_alarms"] == 0
    short = sw.score_setting([fall_clip(lying_frames=10)], FallDetectorConfig(), 1.0)
    assert short["possible_tp"] == 1 and short["tp"] == 0  # warned, not (yet) confirmed


def test_caucafall_labels_and_discovery(tmp_path, monkeypatch):
    base = tmp_path / "caucafall" / "CAUCAFall"
    for subject, activity, labels in (("Subject.1", "Fall forward", [0, 0, 1, 1, 0]),
                                      ("Subject.1", "Kneel", [0, 0, 0])):
        d = base / subject / activity
        d.mkdir(parents=True)
        (d / "clip.avi").write_bytes(b"")
        (d / "labels.csv").write_text("frame,label\n" + "\n".join(f"{i},{v}" for i, v in enumerate(labels, 1)) + "\n",
                                      encoding="utf-8")
    monkeypatch.setattr(sw, "CAUCAFALL_DIR", str(tmp_path / "caucafall"))
    found = sw.caucafall_videos()
    assert [rel for _v, _l, rel in found] == ["Subject.1/Fall forward", "Subject.1/Kneel"]
    assert sw.caucafall_onset(found[0][1]) == 3 and sw.caucafall_onset(found[1][1]) is None


def test_heldout_table_reports_activity_breakdown():
    adl = standing_clip(kind="caucafall_adl")
    adl.name = "Subject.1/Kneel"
    videos = [fall_clip(lying_frames=150), adl]
    rows = [(label, sw.score_setting(videos, FallDetectorConfig(), 1.0, fixes=fx))
            for label, fx in (("Baseline", sw.BASELINE), sw.CANDIDATE)]
    lines = "\n".join(sw.markdown_heldout(rows, 1.0))
    assert "Held-out test: CAUCAFall" in lines and "| Kneel |" in lines and "not used" in lines
    assert sw.CANDIDATE[1] == sw.Fixes(0.0, False, True, 0.5, "presence")  # frozen
