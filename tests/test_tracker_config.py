"""Tracker configuration: two-stage ByteTrack thresholds follow the detection threshold; on synthetic
boxes at the production confidence, a fall keeps its track id and two people crossing keep theirs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


SHAPE = (720, 1280)


def tracker(conf: float = 0.25, **override):
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = load(Detector._tracker_config(conf))
    cfg.update(override)
    return BYTETracker(SimpleNamespace(**{k: v for k, v in cfg.items() if k != "tracker_type"}))


def step(t, boxes):
    """boxes: [(x1, y1, x2, y2, score)] -> {track id: box}."""
    from ultralytics.engine.results import Boxes

    data = np.array([[*b[:4], b[4], 0] for b in boxes], np.float32).reshape(-1, 6)
    return {int(r[4]): r[:4] for r in t.update(Boxes(data, SHAPE))}


def fall(t):
    """Standing, then a fast fall: the box drops, the score sags under the tracker's first stage
    for 6 frames while the box turns wide, then a confident wide box on the floor."""
    ids = []
    for _ in range(20):
        ids.append(step(t, [(600, 200, 700, 620, 0.85)]))
    for i in range(6):  # falling: lower, wider, weak
        ids.append(step(t, [(590 - 20 * i, 260 + 40 * i, 710 + 20 * i, 640, 0.15)]))
    for _ in range(15):
        ids.append(step(t, [(470, 480, 830, 640, 0.6)]))  # on the floor
    return ids


def test_a_fall_keeps_its_track_id():
    ids = fall(tracker())
    first = next(iter(ids[0]))
    assert all(list(frame) == [first] for frame in ids[-10:])


def test_the_previous_settings_lost_the_falling_person():
    """The settings before 2026-10-05 (IoU x score >= 0.2) start a new track on the floor: the
    case this guards against."""
    ids = fall(tracker(match_thresh=0.8, fuse_score=True))
    assert next(iter(ids[0])) not in ids[-1]


@pytest.mark.parametrize("gap", [0, 40], ids=["crossing", "passing close"])
def test_two_people_crossing_keep_their_own_ids(gap):
    """Walking toward each other and past: the looser overlap match mustn't swap them."""
    t = tracker()
    tracks = {}
    for i in range(60):
        a = 200 + 16 * i                      # left to right
        b = 1100 - 16 * i                     # right to left
        out = step(t, [(a, 200, a + 110, 600, 0.85), (b, 200 + gap, b + 110, 600 + gap, 0.85)])
        for tid, box in out.items():
            tracks.setdefault(tid, []).append(float(box[0]))
    long = {tid: xs for tid, xs in tracks.items() if len(xs) >= 50}
    assert len(long) == 2
    directions = sorted(np.sign(xs[-1] - xs[0]) for xs in long.values())
    assert directions == [-1, 1]  # one track kept moving right, the other left: no swap
    for xs in long.values():
        steps = np.diff(xs)
        assert np.all(np.abs(steps) < 40)  # no jump onto the other person's box
