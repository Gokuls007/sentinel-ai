"""Thin strips at the side of the picture are not people to judge (live webcam, 2026-10-12)."""

from types import SimpleNamespace

import numpy as np

from core.pipeline import SentinelPipeline

W = 1280


def det(box, cls="person"):
    return SimpleNamespace(class_name=cls, bbox=np.array(box, float), track_id=1)


def test_the_couch_strips_seen_live_are_slivers():
    assert SentinelPipeline.is_edge_sliver((0, 418, 42, 604), W)   # 42 px at the left edge
    assert SentinelPipeline.is_edge_sliver((0, 424, 76, 710), W)   # 76 px, 4x taller than wide
    assert SentinelPipeline.is_edge_sliver((1240, 100, 1280, 500), W)  # right edge


def test_real_people_are_kept():
    assert not SentinelPipeline.is_edge_sliver((269, 25, 806, 712), W)  # seated at the desk
    assert not SentinelPipeline.is_edge_sliver((0, 200, 180, 700), W)    # entering from the side
    assert not SentinelPipeline.is_edge_sliver((600, 300, 640, 500), W)  # thin but mid-picture


def test_drop_keeps_objects_and_recounts_people():
    p = SimpleNamespace(is_edge_sliver=SentinelPipeline.is_edge_sliver)
    d = SimpleNamespace(detections=[det((269, 25, 806, 712)), det((0, 418, 42, 604)),
                                    det((0, 400, 40, 600), cls="chair")], person_count=2)
    SentinelPipeline._drop_edge_slivers(p, d, W)
    assert [x.class_name for x in d.detections] == ["person", "chair"] and d.person_count == 1
