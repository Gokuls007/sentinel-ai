"""Objects for Warehouse mode: a COCO detector plus open-vocabulary extras, merged.

Detection is split by class: the 80 COCO classes come from a COCO-trained detector (YOLO11m,
much better on small handheld things: a mouse no longer reads as a cell phone), and only the
classes COCO doesn't have (pillow, cardboard box, ladder, hard hat, safety vest, forklift) come
from YOLO-World. Both feed one tracker with class voting; where both find the same object
(IoU > 0.5), the higher-confidence label wins. People keep their own tracker (YOLOv8 +
ByteTrack) and are never part of this.

Three things make the open-vocabulary output steady enough to build rules on:

1. **Several prompts per class.** YOLO-World is sensitive to wording, so each canonical class
   has synonyms (``chair`` <- "chair", "office chair", "gaming chair", "wooden chair"). All
   prompts are detected together with class-agnostic NMS, so each object keeps its
   best-scoring prompt (the max across synonyms) and is reported under the canonical class.
2. **A confidence floor per class** (e.g. 0.25 for chair and box): below it, a soft bag no
   longer reads as a box.
3. **Voting per object.** Objects are tracked across frames by box overlap (IoU); the class
   shown is the one most often given over the last ``window`` frames, and an object shows only
   after ``min_hits`` sightings. That stops "chair -> backpack -> chair" flicker and one-frame
   false hits.

Encoding the prompts needs CLIP, which is slow to load. The encoded prompts are cached per
model and prompt list in ``cache_dir``, so CLIP only runs when the list changes. If the
weights or CLIP are missing (e.g. in CI), the detector disables itself and says why.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("sentinel.objects")

DEFAULT_SYNONYMS = {
    "chair": ["chair", "office chair", "gaming chair", "wooden chair"],
    "cardboard box": ["cardboard box", "shipping box", "carton"],
    "couch": ["couch", "sofa"],
}


@dataclass
class ObjectDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None
    prompt: str | None = None  # the synonym that scored best

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": round(self.confidence, 3),
                "bbox": [round(float(v), 1) for v in self.bbox], "track_id": self.track_id}


def cache_key(model_path: str, prompts: list[str]) -> str:
    text = os.path.basename(model_path) + "\n" + "\n".join(prompts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """Class voting per object across frames (IoU matching, class-agnostic)."""

    def __init__(self, window: int = 15, min_hits: int = 3, max_missed: int = 5, min_iou: float = 0.3):
        self.window = window
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.tracks: list[dict] = []
        self._next = 1

    def update(self, dets: list[ObjectDetection]) -> list[ObjectDetection]:
        pairs = sorted(((iou(t["bbox"], d.bbox), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.min_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t["bbox"], t["missed"], t["hits"] = d.bbox, 0, t["hits"] + 1
            t["votes"].append((d.class_name, d.confidence, d.prompt))
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1
                t["votes"].append(None)  # not seen this frame: the window still moves on
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append({"id": self._next, "bbox": d.bbox, "hits": 1, "missed": 0,
                                    "votes": deque([(d.class_name, d.confidence, d.prompt)], maxlen=self.window)})
                self._next += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        out = []
        for t in self.tracks:
            votes = [v for v in t["votes"] if v is not None]
            if t["missed"] or len(votes) < self.min_hits:
                continue
            counts = Counter(v[0] for v in votes)
            top = max(counts.values())
            # Most votes; a tie goes to the class with more total confidence.
            name = max((c for c in counts if counts[c] == top),
                       key=lambda c: sum(v[1] for v in votes if v[0] == c))
            confs = [v[1] for v in votes if v[0] == name]
            prompt = Counter(v[2] for v in votes if v[0] == name).most_common(1)[0][0]
            out.append(ObjectDetection(name, float(np.mean(confs)), t["bbox"], t["id"], prompt))
        return out


def merge(dets: list[ObjectDetection], min_iou: float = 0.5) -> list[ObjectDetection]:
    """One detection per object across sources: where boxes overlap (IoU > ``min_iou``), keep
    the higher-confidence one."""
    kept: list[ObjectDetection] = []
    for d in sorted(dets, key=lambda d: -d.confidence):
        if all(iou(d.bbox, k.bbox) <= min_iou for k in kept):
            kept.append(d)
    return kept


class CocoDetector:
    """The COCO classes from a COCO-trained ultralytics model (YOLO11m by default)."""

    def __init__(self, model_path: str, classes: list[str], confidence: float = 0.3, imgsz: int = 640,
                 device: str = "auto", floors: dict[str, float] | None = None):
        self.model_path = model_path
        self.classes = list(classes)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = None if device == "auto" else device
        self.floors = dict(floors or {})
        self.error: str | None = None
        self._model = None
        self._ids: list[int] = []

    def floor(self, class_name: str) -> float:
        return self.floors.get(class_name, self.confidence)

    def _load(self):
        if self._model is not None or self.error:
            return self._model
        try:
            from ultralytics import YOLO

            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"{self.model_path} not found")
            self._model = YOLO(self.model_path)
            self._set_ids()
            logger.info("COCO objects ready (%s): %d classes", os.path.basename(self.model_path), len(self._ids))
        except Exception as e:
            self.error = f"{os.path.basename(self.model_path)} off: {e}"
            logger.warning("COCO object detector unavailable (%s)", e)
        return self._model

    def _set_ids(self) -> None:
        names = self._model.names if self._model is not None else {}
        wanted = set(self.classes)
        self._ids = [i for i, n in names.items() if n in wanted]

    def set_classes(self, classes: list[str]) -> None:
        self.classes = list(classes)
        self._set_ids()

    @property
    def available(self) -> bool:
        return self._load() is not None

    def raw(self, frame: np.ndarray) -> list[ObjectDetection]:
        model = self._load()
        if model is None or not self._ids:
            return []
        lowest = min([self.confidence, *(self.floor(c) for c in self.classes)])
        res = model.predict(frame, conf=lowest, classes=self._ids, imgsz=self.imgsz, device=self.device,
                            agnostic_nms=True, verbose=False)[0]
        names = model.names
        out = []
        boxes = res.boxes
        for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True):
            name = names.get(int(cls))
            if name and conf >= self.floor(name):
                out.append(ObjectDetection(name, float(conf), tuple(float(v) for v in box), prompt=name))
        return out


class OpenVocabDetector:
    def __init__(self, model_path: str, classes: list[str], confidence: float = 0.3, imgsz: int = 640,
                 device: str = "auto", cache_dir: str = "data/cache", every_n_frames: int = 1,
                 synonyms: dict[str, list[str]] | None = None, floors: dict[str, float] | None = None,
                 vote_window: int = 15, min_hits: int = 3):
        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = None if device == "auto" else device
        self.cache_dir = cache_dir
        self.every_n_frames = max(1, every_n_frames)
        self.synonyms = {k: list(v) for k, v in (DEFAULT_SYNONYMS if synonyms is None else synonyms).items()}
        self.floors = dict(floors or {})
        self.error: str | None = None
        self.last: list[ObjectDetection] = []
        self.last_raw: list[ObjectDetection] = []
        self.tracker = ObjectTracker(window=vote_window, min_hits=min_hits)
        self._model = None
        self._frame = 0
        self._lock = threading.Lock()
        self._set_prompts(classes)

    def _set_prompts(self, classes: list[str]) -> None:
        self.classes = [c.strip() for c in classes if c.strip()]
        self.prompts: list[str] = []
        self.canonical: list[str] = []  # prompt index -> class
        for c in self.classes:
            for p in self.synonyms.get(c, [c]):
                if p not in self.prompts:
                    self.prompts.append(p)
                    self.canonical.append(c)

    def floor(self, class_name: str) -> float:
        return self.floors.get(class_name, self.confidence)

    # --- model ------------------------------------------------------------------------------------

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"yoloworld_{cache_key(self.model_path, self.prompts)}.pt")

    def _load(self):
        if self._model is not None or self.error:
            return self._model
        try:
            import torch
            from ultralytics import YOLOWorld

            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"{self.model_path} not found")
            model = YOLOWorld(self.model_path)
            self._apply_prompts(model, torch)
            self._model = model
            logger.info("YOLO-World ready: %s", ", ".join(self.classes))
        except Exception as e:  # missing weights or CLIP: run without objects
            self.error = f"objects off: {e}"
            logger.warning("YOLO-World unavailable (%s); running without object detection", e)
        return self._model

    def _apply_prompts(self, model, torch) -> None:
        path = self._cache_path()
        if os.path.isfile(path):
            feats = torch.load(path, map_location="cpu", weights_only=True)
            model.model.txt_feats = feats
            model.model.model[-1].nc = len(self.prompts)
            model.model.names = list(self.prompts)
            model.predictor = None
            logger.info("YOLO-World prompts from cache (%s)", os.path.basename(path))
            return
        model.set_classes(self.prompts)  # CLIP: a few seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(model.model.txt_feats.detach().cpu(), path)

    def set_classes(self, classes: list[str]) -> None:
        """Change the class list (its prompts are re-encoded once, then cached)."""
        with self._lock:
            self._set_prompts(classes)
            self.last, self.last_raw = [], []
            self.tracker = ObjectTracker(window=self.tracker.window, min_hits=self.tracker.min_hits)
            if self._model is not None:
                import torch

                self._apply_prompts(self._model, torch)

    @property
    def available(self) -> bool:
        return self._load() is not None

    # --- every frame ------------------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Steady objects (voted class, seen ``min_hits`` times); on skipped frames, the last."""
        self._frame += 1
        if (self._frame - 1) % self.every_n_frames:
            return self.last
        raw = self.raw(frame)
        self.last_raw = raw
        self.last = self.tracker.update(raw)
        return self.last

    def raw(self, frame: np.ndarray) -> list[ObjectDetection]:
        """This frame's detections (best prompt per object, floors applied), before voting."""
        model = self._load()
        if model is None or not self.classes:
            return []
        lowest = min([self.confidence, *(self.floor(c) for c in self.classes)])
        with self._lock:
            # Class-agnostic NMS: one box per object, under its best-scoring prompt.
            res = model.predict(frame, conf=lowest, imgsz=self.imgsz, device=self.device, agnostic_nms=True,
                                verbose=False)[0]
            canonical, prompts = list(self.canonical), list(self.prompts)
        raw = []
        boxes = res.boxes
        for cls, conf, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True):
            i = int(cls)
            if 0 <= i < len(canonical) and conf >= self.floor(canonical[i]):
                raw.append(ObjectDetection(canonical[i], float(conf), tuple(float(v) for v in box), prompt=prompts[i]))
        return raw


class ObjectDetector:
    """COCO classes from ``coco_model_path``, the rest from YOLO-World, one tracker for both."""

    def __init__(self, classes: list[str], world_model_path: str, coco_model_path: str,
                 coco_names: set[str] | list[str], confidence: float = 0.3, imgsz: int = 640, device: str = "auto",
                 cache_dir: str = "data/cache", every_n_frames: int = 1,
                 synonyms: dict[str, list[str]] | None = None, floors: dict[str, float] | None = None,
                 vote_window: int = 15, min_hits: int = 3):
        self.coco_names = set(coco_names)
        self.every_n_frames = max(1, every_n_frames)
        self.tracker = ObjectTracker(window=vote_window, min_hits=min_hits)
        self.last: list[ObjectDetection] = []
        self.last_raw: list[ObjectDetection] = []
        self._frame = 0
        self._lock = threading.Lock()
        self.floors = dict(floors or {})
        self.confidence = confidence
        coco, extra = self._split(classes)
        self.coco = CocoDetector(coco_model_path, coco, confidence=confidence, imgsz=imgsz, device=device,
                                 floors=floors)
        self.world = OpenVocabDetector(world_model_path, extra, confidence=confidence, imgsz=imgsz, device=device,
                                       cache_dir=cache_dir, synonyms=synonyms, floors=floors)

    def _split(self, classes) -> tuple[list[str], list[str]]:
        classes = [c.strip() for c in classes if c.strip()]
        self.classes = classes
        return [c for c in classes if c in self.coco_names], [c for c in classes if c not in self.coco_names]

    def floor(self, class_name: str) -> float:
        return self.floors.get(class_name, self.confidence)

    def set_classes(self, classes: list[str]) -> None:
        with self._lock:
            coco, extra = self._split(classes)
            self.coco.set_classes(coco)
            self.world.set_classes(extra)
            self.last, self.last_raw = [], []
            self.tracker = ObjectTracker(window=self.tracker.window, min_hits=self.tracker.min_hits)

    def _sources(self):
        return [s for s in (self.coco, self.world) if s.classes]

    @property
    def errors(self) -> list[str]:
        return [s.error for s in self._sources() if s.error]

    @property
    def error(self) -> str | None:
        """Set only when no source is running at all."""
        srcs = self._sources()
        if srcs and all(not s.available for s in srcs):
            return "; ".join(self.errors)
        return None

    @property
    def available(self) -> bool:
        return any(s.available for s in self._sources())

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        self._frame += 1
        if (self._frame - 1) % self.every_n_frames:
            return self.last
        with self._lock:
            raw = [d for s in self._sources() for d in s.raw(frame)]
            merged = merge(raw)
            self.last_raw = merged
            self.last = self.tracker.update(merged)
        return self.last
