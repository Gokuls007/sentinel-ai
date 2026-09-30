import time
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int | None
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    centroid: np.ndarray  # [cx, cy]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "centroid": self.centroid.tolist(),
            "width": float(self.width),
            "height": float(self.height),
            "aspect_ratio": float(self.aspect_ratio)
        }

@dataclass
class FrameDetections:
    detections: list[Detection] = field(default_factory=list)
    person_count: int = 0
    vehicle_count: int = 0
    inference_time_ms: float = 0.0

    @property
    def persons(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name == "person"]

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "person_count": self.person_count,
            "vehicle_count": self.vehicle_count,
            "inference_time_ms": self.inference_time_ms
        }

class Detector:
    COCO_NAMES: ClassVar[dict[int, str]] = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 
        5: "bus", 7: "truck", 24: "backpack", 26: "handbag", 27: "suitcase"
    }

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, 
                 iou_threshold: float = 0.45, device: str = "auto", classes: list[int] | None = None):
        
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0]  # default: people only
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing Detector on device: {self.device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Warm up
        print("Warming up detector...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.track(dummy_frame, persist=True, verbose=False)
        print("Detector ready.")

    def detect_and_track(self, frame: np.ndarray) -> FrameDetections:
        start_time = time.time()
        
        # Run tracking inference
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            classes=self.classes, 
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        detections = []
        person_count = 0
        vehicle_count = 0
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls, results[0].boxes.id
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Track ID can be None if not yet assigned
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else None
                
                class_name = self.COCO_NAMES.get(cls_id, "unknown")
                
                # Compute centroid
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                centroid = np.array([cx, cy])
                
                det = Detection(
                    track_id=track_id,
                    bbox=xyxy,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                    centroid=centroid
                )
                
                detections.append(det)
                
                if class_name == "person":
                    person_count += 1
                elif class_name in ["car", "bus", "truck", "motorcycle"]:
                    vehicle_count += 1
                    
        return FrameDetections(
            detections=detections,
            person_count=person_count,
            vehicle_count=vehicle_count,
            inference_time_ms=inference_time_ms
        )

    def reset_tracker(self):
        # Setting predictor to None clears the internal tracker state
        self.model.predictor = None
        print("Tracker state reset.")
