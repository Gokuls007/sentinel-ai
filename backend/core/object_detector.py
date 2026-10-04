"""Open-vocabulary objects (YOLO-World) for Warehouse mode.

Runs next to the YOLOv8 person tracker and the pose model; it adds objects only (people keep
their stable track ids from ByteTrack). The class list is plain text, e.g. "cardboard box".

Encoding the class names needs CLIP, which is slow to load (several seconds). The encoded
names are cached per model and class list in ``cache_dir``, so CLIP only runs when the list
changes, not at every start.

If the weights or CLIP are missing (e.g. in CI), the detector disables itself and says why;
nothing else stops working.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("sentinel.objects")


@dataclass
class ObjectDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": round(self.confidence, 3),
                "bbox": [round(float(v), 1) for v in self.bbox]}


def cache_key(model_path: str, classes: list[str]) -> str:
    text = os.path.basename(model_path) + "\n" + "\n".join(classes)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class OpenVocabDetector:
    def __init__(self, model_path: str, classes: list[str], confidence: float = 0.3, imgsz: int = 640,
                 device: str = "auto", cache_dir: str = "data/cache", every_n_frames: int = 1):
        self.model_path = model_path
        self.classes = list(classes)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = None if device == "auto" else device
        self.cache_dir = cache_dir
        self.every_n_frames = max(1, every_n_frames)
        self.error: str | None = None
        self.last: list[ObjectDetection] = []
        self._model = None
        self._frame = 0
        self._lock = threading.Lock()

    # --- model ------------------------------------------------------------------------------------

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"yoloworld_{cache_key(self.model_path, self.classes)}.pt")

    def _load(self):
        if self._model is not None or self.error:
            return self._model
        try:
            import torch
            from ultralytics import YOLOWorld

            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"{self.model_path} not found")
            model = YOLOWorld(self.model_path)
            self._apply_classes(model, torch)
            self._model = model
            logger.info("YOLO-World ready: %s", ", ".join(self.classes))
        except Exception as e:  # missing weights or CLIP: run without objects
            self.error = f"objects off: {e}"
            logger.warning("YOLO-World unavailable (%s); running without object detection", e)
        return self._model

    def _apply_classes(self, model, torch) -> None:
        path = self._cache_path()
        if os.path.isfile(path):
            feats = torch.load(path, map_location="cpu", weights_only=True)
            model.model.txt_feats = feats
            model.model.model[-1].nc = len(self.classes)
            model.model.names = list(self.classes)
            model.predictor = None
            logger.info("YOLO-World class names from cache (%s)", os.path.basename(path))
            return
        model.set_classes(self.classes)  # CLIP: a few seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(model.model.txt_feats.detach().cpu(), path)

    def set_classes(self, classes: list[str]) -> None:
        """Change the class list (re-encoded once, then cached)."""
        with self._lock:
            self.classes = [c.strip() for c in classes if c.strip()]
            self.last = []
            if self._model is not None:
                import torch

                self._apply_classes(self._model, torch)

    @property
    def available(self) -> bool:
        return self._load() is not None

    # --- every frame ------------------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Objects in this frame (on skipped frames, the last result)."""
        self._frame += 1
        if (self._frame - 1) % self.every_n_frames:
            return self.last
        model = self._load()
        if model is None or not self.classes:
            return []
        with self._lock:
            res = model.predict(frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            names = list(self.classes)
        out = []
        boxes = res.boxes
        for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True):
            if 0 <= int(cls) < len(names):
                out.append(ObjectDetection(names[int(cls)], float(conf), tuple(float(v) for v in box)))
        self.last = out
        return out
