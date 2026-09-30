"""Tracker configuration: two-stage ByteTrack thresholds follow the detection threshold."""

from pathlib import Path

import yaml

from core.detector import TRACKER_CONFIG, TRACKER_MIN_CONF, Detector


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_default_config_keeps_weak_boxes_for_existing_tracks_only():
    cfg = load(TRACKER_CONFIG)
    assert cfg["tracker_type"] == "bytetrack"
    assert cfg["track_low_thresh"] == TRACKER_MIN_CONF < cfg["track_high_thresh"] == 0.5
    assert cfg["new_track_thresh"] > cfg["track_high_thresh"]  # a new person needs more confidence
    assert cfg["track_buffer"] >= 60


def test_config_follows_a_custom_detection_threshold():
    assert Detector._tracker_config(0.5) == TRACKER_CONFIG
    cfg = load(Detector._tracker_config(0.35))
    assert cfg["track_high_thresh"] == 0.35
    assert cfg["new_track_thresh"] == 0.45
    assert cfg["track_low_thresh"] == TRACKER_MIN_CONF
