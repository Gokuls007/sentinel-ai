import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import cv2
import numpy as np

from activity import CARRY_CLASSES, ActivityTracker, ViewCheck
from activity.balance import BalanceTracker
from activity.object_rules import ObjectRules
from anomaly.engine import AnomalyAlert, AnomalyEngine
from anomaly.fall_recovery import recover_pose
from config.settings import ALL_OBJECT_CLASSES, COCO_OBJECTS, SentinelConfig
from core.object_detector import ObjectDetector
from events import Event, EventBus, EventStore
from exam import ExamMonitor
from notifications import NotificationDispatcher, build_notifiers
from output.clip_recorder import ClipRecorder
from output.skeleton_recorder import SkeletonRecorder
from posture import PostureCoach, select_main_person
from rules import PersonState, RuleEngine, SceneState
from rules.dsl import OBJECT_CLASSES
from rules.presets import builtin_event_type, builtin_rules
from rules.signals import head_turn, holding, looking_down
from rules.store import RuleStore

from .detector import Detector, FrameDetections
from .pose_estimator import PoseEstimator, PoseResult
from .utils import to_serializable
from .video_source import VideoSource

logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    frame: np.ndarray
    timestamp: float
    detections: FrameDetections
    poses: dict[int, PoseResult]
    alerts: list = field(default_factory=list) # Will hold AnomalyAlert later
    annotated_frame: np.ndarray | None = None
    annotated_frame_base64: str | None = None # For pre-optimized transmission
    processing_time_ms: float = 0.0
    total_alerts: int = 0
    frame_number: int = 0
    # Per-layer wall time for this frame (ms). Detection and tracking are one ultralytics call
    # (model.track), so they are timed together as "detect_track".
    timings_ms: dict[str, float] = field(default_factory=dict)
    events: list = field(default_factory=list)  # the Event published for each alert
    ergonomics: dict = field(default_factory=dict)  # track_id -> current REBA (TrackErgo.as_dict)
    activity: dict = field(default_factory=dict)  # track_id -> live activity label and history
    view: dict = field(default_factory=dict)  # the camera-view check (upper body only?)
    posture: dict | None = None  # desk posture coach snapshot (posture mode)
    exam: dict | None = None  # Exam Hall: seats, setup check, calibration (exam mode)
    objects: list = field(default_factory=list)  # open-vocabulary objects (warehouse mode)
    balance: dict = field(default_factory=dict)  # track id -> balance risk, centre of mass, base (warehouse)
    mode: str = "warehouse"

    def to_dict(self) -> dict:
        data = {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "processing_time_ms": self.processing_time_ms,
            "detections": self.detections.to_dict(),
            "alerts": [a.to_dict() if hasattr(a, 'to_dict') else str(a) for a in self.alerts],
            "stats": {
                "person_count": self.detections.person_count,
                "active_tracks": len(self.poses),
                "alert_count": self.total_alerts,
                "processing_time_ms": self.processing_time_ms
            },
            "timings_ms": self.timings_ms,
            "ergonomics": {str(k): v for k, v in self.ergonomics.items()},
            "activity": {str(k): v for k, v in self.activity.items()},
            "view": self.view,
            "posture": self.posture,
            "exam": self.exam,
            "objects": [o.to_dict() for o in self.objects],
            "balance": {str(k): v for k, v in self.balance.items()},
            "mode": self.mode,
        }
        # The JPEG is sent once, as the top-level "image" field of the WebSocket message.
        return to_serializable(data)

class SentinelPipeline:
    skeleton_only: bool = False  # set per instance in __init__; a default for partly built test pipelines
    SKELETON: ClassVar[list[tuple[int, int]]] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
        (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.frame_count = 0
        self.total_alerts = 0
        self.start_time = time.time()
        
        self._on_frame: Callable | None = None
        
        # Instantiate layers
        self.video_source = VideoSource(
            source=config.source,
            target_fps=config.target_fps,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
            loop=config.loop,
        )
        
        self.detector = Detector(
            model_path=config.detector.model_path,
            confidence_threshold=config.detector.confidence_threshold,
            iou_threshold=config.detector.iou_threshold,
            device=config.detector.device,
            classes=config.detector.classes
        )
        
        self.pose_estimator = PoseEstimator(
            model_path=config.detector.pose_model_path,
            sequence_length=config.pose.sequence_length,
            confidence_threshold=config.pose.confidence_threshold,
            device=config.detector.device
        )
        
        self.anomaly_engine = AnomalyEngine(config)
        # Plain-English rules (compiled once, checked here every frame in warehouse mode).
        self.rules = RuleEngine()
        self.rule_store: RuleStore | None = None
        # Live activity labels and the camera-view check (warehouse mode).
        self.activity = ActivityTracker()
        self.view_check = ViewCheck()
        
        # Output layer. Every alert becomes an Event published on the bus; the store
        # subscribes first so later subscribers (notifications, WebSocket) see its id.
        self.clip_recorder = ClipRecorder(
            clips_dir=config.output.clips_dir,
            buffer_seconds=config.output.clip_duration,
            fps=config.target_fps
        )
        self.skeleton_recorder = SkeletonRecorder(config.output.clips_dir)
        o = config.objects
        self.mode_classes = {m: list(v) for m, v in o.mode_classes.items()}  # each mode: its own objects
        self.show_all_objects = False  # Settings debug toggle: warehouse uses every class both detectors know
        self.object_detector = (ObjectDetector(self.mode_classes.get("warehouse", o.classes), o.model_path,
                                               o.coco_model_path, COCO_OBJECTS,
                                               confidence=o.confidence, imgsz=o.imgsz, device=config.detector.device,
                                               cache_dir=o.cache_dir, every_n_frames=o.every_n_frames,
                                               synonyms=o.synonyms, floors=o.floors, vote_window=o.vote_window,
                                               min_hits=o.min_hits)
                                if o.enabled else None)
        self._alert_labels: dict = {}  # track id -> (reason text, show until), drawn next to the person
        self.object_rules = ObjectRules(hazard_classes=o.hazards)  # unsafe lift, on a chair, hand on a hazard
        self.balance = BalanceTracker()  # centre of mass vs base of support: "Losing balance"
        self.event_store = EventStore(config.output.db_path)
        self.rule_store = RuleStore(config.output.db_path)
        self.reload_rules()
        self.event_bus = EventBus()
        self.event_bus.subscribe("store", self.event_store.emit)
        self.notifier = NotificationDispatcher(
            build_notifiers(config),
            debounce_s=config.notifications.debounce_s,
            min_severity=config.notifications.min_severity,
        )
        self.event_bus.subscribe("notifications", self.notifier.handle)

        # App mode (warehouse | posture | exam), switched live by the server.
        self.mode = getattr(config, "mode", "warehouse")
        data_dir = os.path.dirname(os.path.abspath(config.output.db_path))
        self.posture = PostureCoach(baseline_path=os.path.join(data_dir, f"posture_baseline_{config.camera_id}.json"))
        self.exam = ExamMonitor()  # Exam Hall: seats and calibration; sessions are loaded by the server
        self.save_clips = True  # forensic clips and snapshots (off for "Test with demo footage")
        # Skeleton-only (privacy): never write video or images; events keep keypoints instead.
        self.skeleton_only = False
        self.paused = False  # demo footage off: the loop idles without reading frames
        self._posture_id = None  # track id of the person the posture coach follows

        logger.info("Pipeline initialized successfully")

    def on_frame(self, callback: Callable):
        self._on_frame = callback
        return callback

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        self.frame_count += 1
        timings: dict[str, float] = {}
        start = mark = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal mark
            now = time.perf_counter()
            timings[name] = (now - mark) * 1000
            mark = now

        # 1. Detection & tracking (one ultralytics call)
        detections = self.detector.detect_and_track(frame)
        lap("detect_track")

        # 2. Pose estimation
        person_detections = [d for d in detections.detections if d.class_name == "person" and d.track_id is not None]
        track_ids = [d.track_id for d in person_detections]
        bboxes = [d.bbox for d in person_detections]
        poses = self.pose_estimator.estimate(frame, track_ids, bboxes, timestamp)
        if self.skeleton_only and self.save_clips:  # keypoints only, in memory until an event
            self.skeleton_recorder.add_frame(
                timestamp, SkeletonRecorder.frame_people(poses, frame.shape[1], frame.shape[0]))
        lap("pose")

        # 3. Analytics. The mode decides what runs: warehouse = falls, zones, loitering,
        #    ergonomics; posture = the desk posture coach only (no alerts); exam = nothing yet.
        all_features = self.pose_estimator.get_all_features()
        posture = exam = None
        objects = self._detect_objects(frame)
        if self.mode == "warehouse":
            recovered = self._recover_fallen(frame, poses, all_features, timestamp)
            alerts = self.anomaly_engine.process(poses, all_features, timestamp, recovered=recovered)
            if self.config.rules.builtins:  # the built-in rules replace these (no double alerts)
                alerts = [a for a in alerts if a.alert_type not in ("fall", "zone_intrusion")]
            alerts += self._evaluate_rules(poses, all_features, detections, timestamp)
            alerts += self.object_rules.update(poses, objects, timestamp)
            alerts += self.balance.update(poses, timestamp)
            timings["ergonomics"] = self.anomaly_engine.last_ergo_ms  # included in "analytics"
            ergonomics = self.anomaly_engine.ergonomics_snapshot
            self._persist_ergo_time(timestamp)
            activity, view = self._activity(poses, detections, timestamp, objects), self.view_check.snapshot()
        else:
            alerts, ergonomics = [], {}
            activity, view = {}, {}
            if self.mode == "exam":
                exam = self.exam.update(poses, timestamp, frame.shape[1], frame.shape[0], frame)
            if self.mode == "posture":
                # Only the person at the desk: the largest, most central face-and-shoulder area
                # (someone or something on the couch behind is ignored).
                self._posture_id = select_main_person(poses, frame.shape[1], prev_id=self._posture_id)
                main = poses.get(self._posture_id) if self._posture_id is not None else None
                # The tracker keeps a track alive on weak detections (confidence 0.1+). For the
                # coach that would keep an empty chair "occupied", so only a confident detection
                # counts as someone at the desk.
                if main is not None and not self.confident_person(
                        self._posture_id, detections, getattr(self.detector, "conf_threshold", 0.5)):
                    main = None
                self.posture.update(main.keypoints if main is not None else None, timestamp)
                posture = self.posture.snapshot()
        lap("analytics")

        # 4. Annotation (clips and snapshots use the annotated frame)
        if self.mode == "warehouse":
            annotated_frame = self._annotate_frame(frame.copy(), detections, poses, alerts, objects, timestamp)
            self._draw_person_tags(annotated_frame, detections, ergonomics, activity)
            self._draw_balance(annotated_frame, detections, self.balance.current)
        else:
            annotated_frame = frame.copy()
            shown = ([poses[self._posture_id]] if self.mode == "posture" and self._posture_id in poses
                     else [] if self.mode == "posture" else list(poses.values()))
            for pose in shown:
                self._draw_skeleton(annotated_frame, pose)
            if self.mode == "posture":
                self._draw_ghost(annotated_frame, self.posture.ghost())
            if getattr(self, "mode_classes", {}).get(self.mode):  # this mode looks for objects: draw them
                self._draw_objects(annotated_frame, objects)
                self._draw_status(annotated_frame, self.objects_status(objects))
        lap("annotate")

        # 5. Events: clip + snapshot, then publish (store -> notifications -> WebSocket)
        events = []
        for alert in alerts:
            self.total_alerts += 1
            # Forensic clip (pre + post alert) is encoded in the background.
            # A rule can leave the clip out; test footage never saves one; skeleton-only mode
            # saves keypoints instead of video and images.
            want_clip = self.save_clips and alert.details.get("record_clip", True) and not self.skeleton_only
            clip_path = (self.clip_recorder.save_clip(alert.alert_id, alert.timestamp, snapshot=annotated_frame)
                         if want_clip else None)
            clip_path = clip_path.replace("\\", "/") if clip_path else None
            alert.details = {**alert.details, "clip_path": clip_path}
            if self.skeleton_only and self.save_clips:
                skeleton = self.skeleton_recorder.save(
                    alert.alert_id, alert.timestamp, track_id=alert.track_id,
                    meta={"type": alert.alert_type, "camera_id": self.config.camera_id})
                alert.details["skeleton_path"] = skeleton.replace("\\", "/")
            event = Event.from_alert(
                alert,
                camera_id=self.config.camera_id,
                clip_path=clip_path,
                thumbnail_path=self.clip_recorder.snapshot_path(alert.alert_id) if want_clip else None,
            )
            events.append(self.event_bus.publish(event))
        self.clip_recorder.add_frame(annotated_frame, timestamp)
        lap("events_and_clips")

        processing_time_ms = (time.perf_counter() - start) * 1000
        
        result = FrameResult(
            frame=frame,
            timestamp=timestamp,
            detections=detections,
            poses=poses,
            alerts=alerts,
            annotated_frame=annotated_frame,
            processing_time_ms=processing_time_ms,
            total_alerts=self.total_alerts,
            frame_number=self.frame_count,
            timings_ms=timings,
            events=events,
            ergonomics=ergonomics,
            activity=activity,
            view=view,
            exam=exam,
            objects=objects,
            balance=dict(self.balance.current) if self.mode == "warehouse" else {},
            posture=posture,
            mode=self.mode,
        )

        # 6. Streaming (JPEG encode + serialise for the dashboard), timed separately
        if self._on_frame:
            stream_start = time.perf_counter()
            self._on_frame(result)
            timings["stream"] = (time.perf_counter() - stream_start) * 1000

        return result

    # --- rules ---------------------------------------------------------------------------------

    def reload_rules(self) -> None:
        """Load the confirmed rules (plus built-ins when RULES_BUILTINS is on) and tell the
        detector which object classes they need (none: people only, no extra cost)."""
        rules = self.rule_store.list() if self.rule_store else []
        if self.config.rules.builtins:
            zones = [{"id": z.id, "name": z.name, "zone_type": z.zone_type, "active": z.active}
                     for z in self.anomaly_engine.zone_monitor.zones]
            rules += builtin_rules(zones, fall_cooldown_s=self.config.fall.cooldown_seconds,
                                   zone_cooldown_s=self.config.zone.alert_cooldown)
        self.rules.set_rules(rules)
        needed = set().union(*(r.objects() for r in self.rules.rules)) if self.rules.rules else set()
        ids = [cid for cid, name in Detector.COCO_NAMES.items() if name in needed]
        base = [c for c in self.config.detector.classes if Detector.COCO_NAMES.get(c) not in OBJECT_CLASSES]
        # Objects people carry (for the "Carrying" activity label), from the same detector pass.
        carry = [cid for cid, name in Detector.COCO_NAMES.items() if name in CARRY_CLASSES]
        self.detector.classes = sorted(set(base) | set(ids) | set(carry))

    def _scene(self, poses, features, detections, timestamp) -> SceneState:
        engine = self.anomaly_engine
        fd, zm = engine.fall_detector, engine.zone_monitor
        ergo = engine.ergonomics_snapshot
        objects = [(d.class_name, tuple(float(v) for v in d.bbox)) for d in detections.detections
                   if d.class_name in OBJECT_CLASSES]
        persons = {}
        for tid, pose in poses.items():
            e = ergo.get(tid) or {}
            persons[tid] = PersonState(
                track_id=tid, point=(float(pose.mid_hip[0]), float(pose.mid_hip[1])),
                body_height=float(pose.body_height), zones={z.id for z in zm.zones_of(tid)},
                fallen=fd.state_of(tid) == fd.CONFIRMED,
                reba_level=e.get("level") if e.get("reliable") else None,
                holding=holding(pose.keypoints, float(pose.body_height), objects),
                head=head_turn(pose.keypoints), looking_down=looking_down(pose.keypoints))
        # A confirmed fall whose person the detector lost (lying down) is still a fall.
        for tid in features:
            if tid not in persons and fd.state_of(tid) == fd.CONFIRMED:
                box = fd.last_bbox(tid)
                if box is not None:
                    persons[tid] = PersonState(track_id=tid, point=(float(box[0] + box[2]) / 2, float(box[3])),
                                               body_height=float(box[3] - box[1]),
                                               zones={z.id for z in zm.zones_of(tid)}, fallen=True)
        return SceneState(ts=timestamp, camera_id=self.config.camera_id, persons=persons)

    def _evaluate_rules(self, poses, features, detections, timestamp) -> list[AnomalyAlert]:
        if not self.rules.rules:
            return []
        alerts = []
        for f in self.rules.evaluate(self._scene(poses, features, detections, timestamp)):
            r = f.rule
            alerts.append(AnomalyAlert(
                alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}",
                alert_type=builtin_event_type(r) or f"rule:{r.id}",
                track_id=f.track_id,  # None for scene rules (counts, time of day)
                timestamp=f.ts, confidence=1.0, severity=r.severity,
                message=r.name,
                details={"rule_id": r.id, "rule_name": r.name, "duration": f.held_s, "zone_id": f.zone_id,
                         "record_clip": "record_clip" in r.actions, "notify": "notify" in r.actions},
            ))
        return alerts

    def _recover_fallen(self, frame, poses, features, timestamp) -> dict:
        """Retry pose on the region around falling/fallen people the detector lost this frame
        (anomaly/fall_recovery.py). Off unless FALL recovery settings enable it."""
        f = self.config.fall
        if f.recovery_low_conf <= 0 and not f.recovery_rotated:
            return {}
        fd = self.anomaly_engine.fall_detector
        out = {}
        for tid in features:
            if tid in poses or not fd.is_down(tid):
                continue
            st = fd.tracks.get(tid)
            bbox = fd.last_bbox(tid)
            if (bbox is None or st is None or timestamp - st.falling_since > f.recovery_window_seconds
                    or (st.last_seen is not None and timestamp - st.last_seen > max(f.lost_hold_seconds, 1.0))):
                continue
            pose, _how = recover_pose(self.pose_estimator.model, frame, tid, bbox,
                                      low_conf=f.recovery_low_conf or self.config.pose.confidence_threshold,
                                      try_low=f.recovery_low_conf > 0, try_rotated=f.recovery_rotated)
            if pose is not None:
                out[tid] = pose
        return out

    OBJECT_COLORS: ClassVar[dict[str, tuple[int, int, int]]] = {  # BGR
        "cardboard box": (60, 140, 220), "chair": (200, 160, 60), "ladder": (40, 200, 240),
        "backpack": (180, 90, 200), "hard hat": (0, 215, 255), "safety vest": (0, 240, 160),
        "forklift": (50, 110, 255), "couch": (150, 150, 150),
    }
    ALERT_LABEL_S = 4.0  # how long a rule's reason stays next to the person

    @classmethod
    def object_color(cls, name: str) -> tuple[int, int, int]:
        if name in cls.OBJECT_COLORS:
            return cls.OBJECT_COLORS[name]
        h = sum(ord(c) * (i + 1) for i, c in enumerate(name))  # stable colour for custom classes
        return (60 + h % 180, 60 + (h // 7) % 180, 60 + (h // 49) % 180)

    def _draw_objects(self, frame, objects) -> None:
        for o in objects:
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            c = self.object_color(o.class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            text = f"{o.class_name} {o.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, y1 - h - 6)
            cv2.rectangle(frame, (x1, top), (x1 + w + 6, top + h + 6), c, -1)
            cv2.putText(frame, text, (x1 + 3, top + h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    def _draw_alert_reasons(self, frame, detections, alerts, ts) -> set:
        """Each alert's rule name and reason (with the measured values) under its person's box,
        for a few seconds. Returns the track ids that have one showing."""
        for a in alerts:
            tid = getattr(a, "track_id", None)
            if tid is not None:
                self._alert_labels[tid] = (a.message, ts + self.ALERT_LABEL_S)
        self._alert_labels = {k: v for k, v in self._alert_labels.items() if v[1] >= ts}
        boxes = {d.track_id: d.bbox for d in detections.detections if d.track_id is not None}
        for tid, (text, _until) in self._alert_labels.items():
            if tid not in boxes:
                continue
            x1, _y1, _x2, y2 = (int(v) for v in boxes[tid])
            for i, line in enumerate(_wrap(text, 48)[:3]):
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = min(frame.shape[0] - h - 6, y2 + 4 + i * (h + 8))
                cv2.rectangle(frame, (x1, top), (x1 + w + 8, top + h + 6), (0, 0, 200), -1)
                cv2.putText(frame, line, (x1 + 4, top + h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return set(self._alert_labels)

    def _detect_objects(self, frame) -> list:
        """This mode's objects only (its own class list; none in a mode without one)."""
        det = getattr(self, "object_detector", None)
        wanted = getattr(self, "mode_classes", {}).get(self.mode, [])
        if self.mode == "warehouse" and getattr(self, "show_all_objects", False):
            wanted = ALL_OBJECT_CLASSES
        if det is None or not wanted:
            return []
        if det.classes != wanted:
            det.set_classes(wanted)  # switching modes: instant (YOLO-World prompts are cached)
        return det.detect(frame)

    def objects_status(self, objects) -> str:
        """One line for the feed: is open-vocabulary detection running, and what it sees."""
        det = self.object_detector
        if det is None:
            return "Objects: off (OBJECTS_ENABLED=false)"
        if det.error:
            return f"Objects off: {det.error.removeprefix('objects off: ')}"
        partial = " · " + "; ".join(getattr(det, "errors", [])) if getattr(det, "errors", None) else ""
        if not objects:
            return "Objects: 0 detected" + partial
        names = sorted({o.class_name for o in objects})
        return f"Objects: {len(objects)} detected ({', '.join(names)})" + partial

    def _draw_status(self, frame, text: str) -> None:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (8, 8), (8 + w + 10, 8 + h + 10), (20, 20, 20), -1)
        cv2.putText(frame, text, (13, 8 + h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    def _annotate_frame(self, frame, detections, poses, alerts, objects=(), ts: float = 0.0) -> np.ndarray:
        # 1. Zones: every active zone (any of them can alert), none when there are none
        for zone in self.anomaly_engine.zone_overlay_data:
            poly = np.array(zone["polygon"])
            overlay = frame.copy()
            
            # Color mapping
            if zone["type"] == "restricted":
                color = (0, 0, 255) # Red
            elif zone["type"] == "one_way":
                color = (255, 0, 255) # Purple
            else:
                color = (0, 165, 255) # Orange
                
            cv2.fillPoly(overlay, [poly.astype(int)], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [poly.astype(int)], True, color, 2)
            cv2.putText(frame, zone["name"], (poly[0][0], poly[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Objects, then people (red while a rule's reason is showing for them)
        self._draw_objects(frame, objects)
        alerted_track_ids = self._draw_alert_reasons(frame, detections, alerts, ts)
        self._draw_status(frame, self.objects_status(objects))
        
        for det in detections.detections:
            tid = det.track_id
            bbox = det.bbox.astype(int)
            
            # Highlight red if alert active for this track
            is_alerted = tid in alerted_track_ids
            color = (0, 0, 255) if is_alerted else (0, 255, 0)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"ID:{tid} {det.confidence:.2f}", (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw skeleton if pose exists
            if tid in poses:
                self._draw_skeleton(frame, poses[tid])

        return frame

    ERGO_COLORS: ClassVar[dict[int, tuple[int, int, int]]] = {
        1: (80, 200, 80), 2: (80, 200, 80), 3: (0, 215, 255), 4: (0, 140, 255), 5: (0, 0, 230),
    }  # BGR: negligible/low green, medium yellow, high orange, very high red

    def _activity(self, poses, detections, timestamp: float, found: list | None = None) -> dict:
        """Live activity per tracked person, and the scene view check."""
        fd = self.anomaly_engine.fall_detector
        objects = [(d.class_name, tuple(float(v) for v in d.bbox)) for d in detections.detections
                   if d.class_name in CARRY_CLASSES]
        objects += [(o.class_name, o.bbox) for o in found or []]  # the activity rules decide what's carryable
        self.view_check.update({tid: p.keypoints for tid, p in poses.items()}, timestamp)
        self.activity.prune(poses.keys())
        return {tid: self.activity.update(tid, p.keypoints, timestamp, float(p.body_height),
                                          fallen=fd.state_of(tid) == fd.CONFIRMED, objects=objects)
                for tid, p in poses.items()}

    @staticmethod
    def person_tag(act: dict | None, info: dict | None) -> str:
        """e.g. "Bending · REBA 9 HIGH · back" (REBA only when it can be trusted)."""
        label = act["label"] if act else None
        if label == "Carrying" and act.get("detail"):
            label = f"Carrying {act['detail']}"
        parts = [label] if label else []
        if info and info.get("score") is not None and info.get("reliable"):
            parts.append(f"REBA {info['score']} {info['level_name'].replace('_', ' ').upper()}")
            if info.get("dominant"):
                parts.append(info["dominant"].replace("_", " "))
        return " · ".join(parts)

    def _draw_person_tags(self, frame: np.ndarray, detections, ergonomics: dict, activity: dict) -> None:
        """A tag above each person: what they're doing, plus REBA when it can be trusted."""
        for det in detections.detections:
            info, act = ergonomics.get(det.track_id), activity.get(det.track_id)
            text = self.person_tag(act, info).replace(" · ", " | ")  # OpenCV's font has no "·"
            if not text:
                continue
            x1, y1 = int(det.bbox[0]), int(det.bbox[1])
            reliable = bool(info and info.get("reliable") and info.get("level"))
            color = self.ERGO_COLORS.get(info["level"], (160, 160, 160)) if reliable else (200, 200, 200)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, y1 - 32 - h)  # above the balance bar
            cv2.rectangle(frame, (x1, top), (x1 + w + 8, top + h + 8), color, -1)
            cv2.putText(frame, text, (x1 + 4, top + h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    @staticmethod
    def balance_color(risk: float) -> tuple[int, int, int]:
        """BGR from green (steady) through yellow to red (losing balance)."""
        r = max(0.0, min(1.0, risk))
        return (40, int(200 * (1 - r) + 40 * r), int(60 + 195 * min(1.0, 2 * r)))

    def _draw_balance(self, frame: np.ndarray, detections, balance: dict) -> None:
        """A risk bar above each person (fills as the balance margin shrinks), the centre of mass
        as a dot and the base of support as a line at the ankles."""
        boxes = {d.track_id: d.bbox for d in detections.detections if d.track_id is not None}
        for tid, b in balance.items():
            if tid not in boxes:
                continue
            x1, y1, x2, _y2 = (int(v) for v in boxes[tid])
            risk = b.get("risk", b.get("raw_risk", 0.0))
            color = self.balance_color(risk)
            w = max(60, x2 - x1)
            top = max(0, y1 - 10)
            cv2.rectangle(frame, (x1, top), (x1 + w, top + 6), (60, 60, 60), -1)
            cv2.rectangle(frame, (x1, top), (x1 + int(w * risk), top + 6), color, -1)
            cv2.putText(frame, f"balance risk {risk:.0%}", (x1 + w + 4, top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1)
            cx, cy = (int(v) for v in b["com"])
            left, right = (int(v) for v in b["base"])
            cv2.circle(frame, (cx, cy), 5, color, -1)
            foot_y = int(_y2) - 4
            cv2.line(frame, (left, foot_y), (right, foot_y), color, 3)
            cv2.line(frame, (cx, cy), (cx, foot_y), color, 1)

    def _persist_ergo_time(self, timestamp: float, every_s: float = 10.0) -> None:
        """Flush accumulated time at risk to the store every ``every_s`` (and at stop)."""
        ergo = self.anomaly_engine.ergo
        if ergo is None:
            return
        last = getattr(self, "_ergo_flushed_at", None)
        if last is None:
            self._ergo_flushed_at = timestamp
        elif timestamp - last >= every_s or timestamp < last:
            self.event_store.add_ergo_time(self.config.camera_id, ergo.drain_time())
            self._ergo_flushed_at = timestamp

    def flush_ergo_time(self) -> None:
        if self.anomaly_engine.ergo is not None:
            self.event_store.add_ergo_time(self.config.camera_id, self.anomaly_engine.ergo.drain_time())

    @staticmethod
    def confident_person(track_id, detections, threshold: float) -> bool:
        """This frame has a confident detection for the track (not a weak box that only keeps
        an existing track alive)."""
        return any(d.track_id == track_id and d.class_name == "person" and d.confidence >= threshold
                   for d in detections.detections)

    @staticmethod
    def _draw_seats(frame: np.ndarray, exam: dict) -> None:
        """Seat outlines with their labels at the bottom-left corner (the desk), never over a
        face. Grey = empty, cyan = occupied, green once calibrated. Not drawn on the live feed
        (the Exam page overlays its own seat map); for saved exam clips (stage E2)."""
        h, w = frame.shape[:2]
        for s in exam.get("seats", []):
            x1, y1, x2, y2 = s["rect"]
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            colour = (120, 120, 120) if not s["occupied"] else (94, 197, 34) if s["calibrated"] else (238, 211, 34)
            cv2.rectangle(frame, p1, p2, colour, 2)
            cv2.putText(frame, s["label"], (p1[0] + 4, p2[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    @staticmethod
    def _draw_ghost(frame: np.ndarray, ghost: dict | None) -> None:
        """A faint dashed outline of your Good posture (posture coach fix guidance)."""
        if not ghost:
            return
        pts = ghost["points"]
        overlay = frame.copy()
        color = (255, 255, 255)
        for a, b in ghost["edges"]:
            if a not in pts or b not in pts:
                continue
            p, q = np.array(pts[a]), np.array(pts[b])
            n = max(1, int(np.hypot(*(q - p)) // 10))
            for i in range(0, n, 2):  # dashes
                s, e = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
                cv2.line(overlay, tuple(s.astype(int)), tuple(e.astype(int)), color, 3, cv2.LINE_AA)
        for i, (x, y) in pts.items():
            cv2.circle(overlay, (int(x), int(y)), 7 if i == 0 else 5, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    def _draw_skeleton(self, frame: np.ndarray, pose: PoseResult):
        for start_idx, end_idx in self.SKELETON:
            kp1, kp2 = pose.keypoints[start_idx], pose.keypoints[end_idx]
            if kp1[2] > 0.3 and kp2[2] > 0.3:
                cv2.line(frame, tuple(kp1[:2].astype(int)), tuple(kp2[:2].astype(int)), (0, 255, 255), 2)
        
        for kp in pose.keypoints:
            if kp[2] > 0.3:
                cv2.circle(frame, tuple(kp[:2].astype(int)), 4, (255, 255, 0), -1)
            
        return frame

    def run(self):
        self.video_source.start()
        logger.info(f"Pipeline running on source: {self.config.source}")
        clip_fps_set = False
        try:
            while self.video_source.is_running:
                if self.paused:
                    self.video_source.paused = True
                    time.sleep(0.1)
                    continue
                self.video_source.paused = False
                result = self.video_source.read()
                if result is None:
                    time.sleep(0.005)
                    continue

                if not clip_fps_set and self.video_source.source_fps:
                    # Clips play back at the source's real frame rate.
                    self.clip_recorder.set_fps(self.video_source.source_fps)
                    clip_fps_set = True

                frame, ts = result
                _ = self.process_frame(frame, ts)
                
                if self.frame_count % 100 == 0:
                    logger.info(f"Frame {self.frame_count} | Stats: {self.stats}")
                    
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.stop()

    def stop(self):
        self.video_source.stop()
        self.clip_recorder.flush()
        self.skeleton_recorder.flush()
        self.flush_ergo_time()
        self.notifier.close()
        uptime = time.time() - self.start_time
        logger.info(
            f"Pipeline stopped. Uptime: {uptime:.1f}s | Total Frames: {self.frame_count} "
            f"| Total Alerts: {self.total_alerts}"
        )

    @property
    def stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "frames_processed": self.frame_count,
            "total_alerts": self.total_alerts,
            "uptime_seconds": round(uptime, 1),
            "avg_fps": round(self.frame_count / uptime, 1) if uptime > 0 else 0,
            "source_stats": self.video_source.stats,
            "active_tracks": len(self.pose_estimator.get_all_features()),
            "notifications": dict(self.notifier.stats),
        }


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line] if line else lines
