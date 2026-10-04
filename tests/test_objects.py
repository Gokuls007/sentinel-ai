"""Warehouse objects (YOLO-World, mocked: no weights or GPU in CI): the class-name cache, frame
skipping, running without the model, the overlay reasons and the class-list API."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.object_detector import OpenVocabDetector, cache_key


class FakeWorld:
    """Stands in for ultralytics.YOLOWorld: counts CLIP encodings, returns fixed boxes."""

    encodings = 0

    def __init__(self, path):
        head = SimpleNamespace(nc=0)
        self.model = SimpleNamespace(txt_feats=None, model=[head], names=None)
        self.predictor = None

    def set_classes(self, classes):
        FakeWorld.encodings += 1
        self.model.txt_feats = torch.ones(1, len(classes), 4)
        self.model.model[-1].nc = len(classes)
        self.model.names = list(classes)

    def predict(self, frame, **kw):
        boxes = SimpleNamespace(cls=torch.tensor([1.0, 0.0]), conf=torch.tensor([0.8, 0.4]),
                                xyxy=torch.tensor([[10.0, 20.0, 110.0, 220.0], [300.0, 300.0, 380.0, 360.0]]))
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture
def fake_world(monkeypatch, tmp_path):
    FakeWorld.encodings = 0
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLOWorld=FakeWorld))
    weights = tmp_path / "world.pt"
    weights.write_bytes(b"x")
    return str(weights)


def test_class_names_are_encoded_once_then_loaded_from_the_cache(fake_world, tmp_path):
    classes = ["cardboard box", "chair"]
    d1 = OpenVocabDetector(fake_world, classes, cache_dir=str(tmp_path / "cache"))
    assert d1.available and FakeWorld.encodings == 1
    d2 = OpenVocabDetector(fake_world, classes, cache_dir=str(tmp_path / "cache"))
    assert d2.available and FakeWorld.encodings == 1  # CLIP not run again
    assert d2._model.model.names == classes and d2._model.model.model[-1].nc == 2
    d2.set_classes(["chair", "ladder", "forklift"])  # a new list: encoded once more
    assert FakeWorld.encodings == 2 and cache_key(fake_world, classes) != cache_key(fake_world, d2.classes)


def test_detections_and_frame_skipping(fake_world, tmp_path):
    d = OpenVocabDetector(fake_world, ["cardboard box", "chair"], cache_dir=str(tmp_path), every_n_frames=3)
    frame = np.zeros((480, 640, 3), np.uint8)
    first = d.detect(frame)
    assert [(o.class_name, round(o.confidence, 1)) for o in first] == [("chair", 0.8), ("cardboard box", 0.4)]
    assert d.detect(frame) is first and d.detect(frame) is first  # skipped frames reuse the last result
    assert first[0].to_dict() == {"class_name": "chair", "confidence": 0.8, "bbox": [10.0, 20.0, 110.0, 220.0]}


def test_missing_weights_turn_objects_off_without_breaking_anything(tmp_path):
    d = OpenVocabDetector(str(tmp_path / "nope.pt"), ["chair"], cache_dir=str(tmp_path))
    assert d.detect(np.zeros((10, 10, 3), np.uint8)) == [] and not d.available
    assert "not found" in d.error


def test_reasons_show_next_to_the_person_and_expire():
    from core.pipeline import SentinelPipeline, _wrap

    p = SentinelPipeline.__new__(SentinelPipeline)
    p._alert_labels = {}
    det = SimpleNamespace(track_id=4, bbox=np.array([100, 100, 200, 400]))
    detections = SimpleNamespace(detections=[det])
    frame = np.zeros((480, 640, 3), np.uint8)
    alert = SimpleNamespace(track_id=4, message="Unsafe lift: back bent 62 deg, knees 168 deg")
    assert p._draw_alert_reasons(frame, detections, [alert], 10.0) == {4}
    assert frame[405:430, 100:200].any()  # drawn under the person's box
    assert p._draw_alert_reasons(frame, detections, [], 13.0) == {4}
    assert p._draw_alert_reasons(frame, detections, [], 15.0) == set()  # gone after ALERT_LABEL_S
    assert _wrap("a b c d", 3) == ["a b", "c d"]
    assert SentinelPipeline.object_color("chair") == SentinelPipeline.OBJECT_COLORS["chair"]
    assert SentinelPipeline.object_color("pallet jack") == SentinelPipeline.object_color("pallet jack")


def test_a_detected_box_counts_for_carrying():
    from activity.rules import CARRY_CLASSES

    assert "cardboard box" in CARRY_CLASSES


@pytest.fixture
def objects_api(tmp_config, monkeypatch):
    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True
    det = SimpleNamespace(classes=["cardboard box", "chair"], error=None, set_classes=None)
    det.set_classes = lambda classes: setattr(det, "classes", list(classes))
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "pipeline", SimpleNamespace(object_detector=det, config=tmp_config))
    monkeypatch.setattr(server, "_retention_worker", None)
    server._cameras.clear()
    with TestClient(server.app) as c:
        c.det = det
        yield c


def test_object_classes_api(objects_api):
    assert objects_api.get("/api/objects").json()["classes"] == ["cardboard box", "chair"]
    r = objects_api.put("/api/objects", json={"classes": ["Pallet  Jack", "chair", "chair"]})
    assert r.status_code == 200 and r.json()["classes"] == ["pallet jack", "chair"]
    assert objects_api.det.classes == ["pallet jack", "chair"]
    assert objects_api.put("/api/objects", json={"classes": ["<script>"]}).status_code == 422
    assert objects_api.put("/api/objects", json={"classes": []}).status_code == 422
