"""Hard-negative attribution, recordings discovery and the report of training/eval_false_alarms.py."""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "eval_false_alarms", os.path.join(ROOT, "training", "eval_false_alarms.py"))
efa = importlib.util.module_from_spec(spec)
sys.modules["eval_false_alarms"] = efa  # dataclasses look their module up here
spec.loader.exec_module(efa)


def result(seconds, alerts, activities=()):
    return efa.VideoResult(label="v", path="v.mp4", source="recording", activities=list(activities),
                           seconds=seconds, frames=int(seconds * 25), alert_times_s=list(alerts))


def test_segments_count_only_labelled_squat_and_kneel_time():
    video = efa.Video(path="v.mp4", label="v", activities=["lifting", "squat"], segments=[
        {"start_s": 10, "end_s": 20, "activity": "squat"},
        {"start_s": 30, "end_s": 35, "activity": "kneel"},
        {"start_s": 40, "end_s": 50, "activity": "lifting"},  # not a hard negative
    ])
    r = efa.attribute(result(60, [12.0, 33.0, 45.0, 55.0]), video)
    assert r.hard_negative_basis == "segments"
    assert r.hard_negative_seconds == pytest.approx(15)
    assert r.hard_negative_alerts == 2  # 45 s is lifting, 55 s is unlabelled


def test_segment_past_the_end_of_the_video_is_clipped():
    video = efa.Video(path="v.mp4", label="v", segments=[{"start_s": 50, "end_s": 90, "activity": "kneel"}])
    assert efa.attribute(result(60, []), video).hard_negative_seconds == pytest.approx(10)


def test_without_segments_a_squat_video_counts_as_a_whole():
    video = efa.Video(path="v.mp4", label="v", activities=["squat", "walking"])
    r = efa.attribute(result(120, [5.0, 80.0]), video)
    assert (r.hard_negative_basis, r.hard_negative_seconds, r.hard_negative_alerts) == ("whole video", 120, 2)


def test_walking_video_is_not_a_hard_negative():
    r = efa.attribute(result(120, [5.0]), efa.Video(path="v.mp4", label="v", activities=["walking"]))
    assert (r.hard_negative_basis, r.hard_negative_seconds, r.hard_negative_alerts) == ("", 0, 0)


def test_summary_rates_and_missing_videos():
    a = efa.attribute(result(1800, [1.0, 2.0]), efa.Video(path="a", label="a", activities=["walking"]))
    b = efa.attribute(result(1800, [3.0]), efa.Video(path="b", label="b", activities=["kneel"]))
    missing = efa.VideoResult(label="m", path="m.mp4", source="sample", activities=[], missing=True)
    s = efa.summarize([a, b, missing])
    assert s["hours"] == pytest.approx(1.0) and s["alerts"] == 3 and s["alerts_per_hour"] == pytest.approx(3.0)
    assert s["hard_negative_hours"] == pytest.approx(0.5) and s["hard_negative_alerts_per_hour"] == pytest.approx(2)
    assert s["missing"] == ["m.mp4"]


def test_recordings_are_found_with_their_sidecar(tmp_path):
    (tmp_path / "ergo.mp4").write_bytes(b"")
    (tmp_path / "ergo.json").write_text(json.dumps(
        {"label": "Ergonomics clip", "activities": ["squat"], "segments": []}), encoding="utf-8")
    (tmp_path / "webcam.avi").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("not a video", encoding="utf-8")
    videos = {os.path.basename(v.path): v for v in efa.load_recordings(str(tmp_path))}
    assert set(videos) == {"ergo.mp4", "webcam.avi"}
    assert videos["ergo.mp4"].label == "Ergonomics clip" and videos["ergo.mp4"].hard_negative_activities == ["squat"]
    assert videos["webcam.avi"].activities == []


def test_committed_manifest_lists_existing_sample_clips():
    videos = efa.load_manifest()
    assert {os.path.basename(v.path) for v in videos} == {"corridor_sample.mp4", "walking_sample.mp4"}


def test_report_calls_out_missing_hard_negatives_and_little_footage():
    r = efa.attribute(result(300, []), efa.Video(path="a", label="Corridor", activities=["walking"]))
    md = efa.markdown([r], efa.summarize([r]), "cpu")
    assert md.startswith(efa.SECTION_START) and md.endswith(efa.SECTION_END)
    assert "No squat or kneel footage evaluated yet" in md and "5.0 minutes" in md


def test_write_section_replaces_in_place(tmp_path):
    path = tmp_path / "B.md"
    path.write_text(f"# B\n\n{efa.SECTION_START}\nold\n{efa.SECTION_END}\n\n## After\n", encoding="utf-8")
    efa.write_section(str(path), f"{efa.SECTION_START}\nnew\n{efa.SECTION_END}")
    text = path.read_text(encoding="utf-8")
    assert "new" in text and "old" not in text and text.endswith("## After\n")
