"""Safe path (home mode): prevent the trip, not just catch the fall.

- **Walking paths**: where people's feet have been while standing or walking (ankles in view), on
  a coarse grid, learned live. Nothing about who walked is kept, only where feet touched the floor.
- **Trip hazards** (``trip_hazard``): a loose object (a bag, pillow, box, shoe, clothes) lying on a
  walking path, still for ``hazard_hold_s`` and not on a bed, couch or chair. Once per object.
- **Watch your step** (``watch_step``): someone walking whose feet, carried on at their current
  speed for ``lookahead_s``, would reach a trip hazard. Spoken, so the person hears it in time.
- **Dark room** (``dark_room``): someone gets out of bed or stands up while the picture is dark
  (mean brightness under ``dark_level``): poor light is a common cause of night-time falls.

Each alert says what and where. Live only.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from anomaly.engine import AnomalyAlert

L_ANK, R_ANK = 15, 16
LOOSE = ("backpack", "handbag", "suitcase", "pillow", "cardboard box", "shoe", "clothes", "towel", "sports ball",
         "teddy bear", "book", "bottle", "umbrella")
FURNITURE = ("bed", "couch", "chair", "dining table", "bench")


@dataclass
class TripConfig:
    min_conf: float = 0.4
    cell: int = 40                 # px of the picture per path cell
    path_min: float = 3.0          # foot sightings before a cell counts as walked
    near_cells: int = 1            # an object within this many cells of a walked cell is on the path
    hazard_hold_s: float = 10.0    # lying still this long before it's a hazard (not something being carried)
    same_iou: float = 0.3          # a box overlapping a known object this much is that object (boxes wobble)
    person_overlap: float = 0.5    # an object mostly inside a person's box is on them (clothes, a bag held)
    max_tall: float = 1.6          # height / width: things lying on the floor are low (a standing fan isn't)
    forget_s: float = 30.0         # an object unseen this long is gone
    report_again_s: float = 600.0  # a hazard in the same place isn't reported again for this long
    lookahead_s: float = 1.0
    approach_margin: float = 0.25  # x body height around the hazard box
    step_cooldown_s: float = 6.0
    dark_level: float = 45.0       # mean grey level 0-255
    dark_cooldown_s: float = 120.0


@dataclass
class _Obj:
    name: str
    box: tuple
    first: float
    last: float
    reported: bool = False


@dataclass
class _Walker:
    feet: deque = field(default_factory=lambda: deque(maxlen=15))  # (ts, x, y, height)
    warned: dict = field(default_factory=dict)  # hazard key -> last time


def _feet(kp, c):
    pts = [kp[i, :2] for i in (L_ANK, R_ANK) if kp[i, 2] >= c]
    return np.mean(pts, axis=0) if pts else None


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _overlap(a, b) -> float:
    """Share of box ``a`` inside box ``b``."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1.0)


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1.0)


class TripHazards:
    def __init__(self, width: int = 1280, height: int = 720, cfg: TripConfig | None = None):
        self.cfg = cfg or TripConfig()
        self.width, self.height = width, height
        self.walked = np.zeros((-(-height // self.cfg.cell), -(-width // self.cfg.cell)), np.float32)
        self.objects: dict[tuple, _Obj] = {}
        self.walkers: dict[int, _Walker] = {}
        self.hazards: list[dict] = []  # current hazards (for the overlay and the API)
        self._dark_fired = -math.inf
        self._reported: list[tuple[tuple, float]] = []  # (box, time) of hazards already reported

    # -- learning where people walk -------------------------------------------------------------

    def _learn(self, poses: dict, activity: dict) -> None:
        c = self.cfg
        for tid, pose in poses.items():
            if (activity.get(tid) or {}).get("label") not in ("Standing", "Walking"):
                continue
            f = _feet(np.asarray(pose.keypoints, float), c.min_conf)
            if f is not None:
                r, col = int(f[1] // c.cell), int(f[0] // c.cell)
                if 0 <= r < self.walked.shape[0] and 0 <= col < self.walked.shape[1]:
                    self.walked[r, col] = min(1000.0, self.walked[r, col] + 1.0)

    def on_path(self, box) -> bool:
        """The bottom edge of the box (where it rests on the floor) is on or next to a walked cell."""
        c = self.cfg
        x1, _y1, x2, y2 = (float(v) for v in box[:4])
        r = int(y2 // c.cell)
        rows = range(max(0, r - c.near_cells), min(self.walked.shape[0], r + c.near_cells + 1))
        c1, c2 = int(x1 // c.cell) - c.near_cells, int(x2 // c.cell) + c.near_cells
        cols = range(max(0, c1), min(self.walked.shape[1], c2 + 1))
        return any(self.walked[rr, cc] >= c.path_min for rr in rows for cc in cols)

    # -- one frame ----------------------------------------------------------------------------------

    def update(self, poses: dict, activity: dict, objects: list, ts: float, brightness: float | None = None,
               risky_moves: list | None = None) -> list[AnomalyAlert]:
        """``objects``: detected objects; ``brightness``: the frame's mean grey level; ``risky_moves``:
        (track id, what) for people getting out of bed or standing up this frame."""
        self._learn(poses, activity)
        people = [tuple(float(v) for v in p.bbox[:4]) for p in poses.values() if getattr(p, "bbox", None) is not None]
        alerts = self._update_hazards(objects, ts, people)
        alerts += self._watch_steps(poses, activity, ts)
        if brightness is not None and risky_moves and brightness < self.cfg.dark_level \
                and ts - self._dark_fired >= self.cfg.dark_cooldown_s:
            self._dark_fired = ts
            tid, what = risky_moves[0]
            alerts.append(self._alert("dark_room", tid, ts, "medium",
                                      f"It's dark: turn on a light before {what}",
                                      {"rule": "Dark room", "brightness": round(float(brightness), 1)}))
        return alerts

    def _update_hazards(self, objects: list, ts: float, people: list | None = None) -> list:
        c = self.cfg
        furniture = [o.bbox for o in objects if o.class_name in FURNITURE]
        seen = set()
        for o in objects:
            if o.class_name not in LOOSE:
                continue
            box = tuple(float(v) for v in o.bbox[:4])
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if any(_inside(centre, f) for f in furniture):
                continue  # a pillow on the couch, a bag on a chair
            if (box[3] - box[1]) > c.max_tall * max(box[2] - box[0], 1.0):
                continue  # tall: standing up, not lying on the floor
            if any(_overlap(box, p) >= c.person_overlap for p in people or []):
                continue  # on someone: their own clothes, a bag they carry, or a person read as "clothes"
            key = max((k for k in self.objects if k not in seen and _iou(self.objects[k].box, box) >= c.same_iou),
                      key=lambda k: _iou(self.objects[k].box, box), default=None)
            if key is None:
                key = (o.class_name, round(centre[0]), round(centre[1]), ts)
                self.objects[key] = _Obj(o.class_name, box, ts, ts)
            else:
                self.objects[key].last = ts
            seen.add(key)
        # Forget objects not seen for a while (picked up, or the detector lost them).
        # Forget an object unseen this long: picked up (people walking in front hide it for a while).
        self.objects = {k: v for k, v in self.objects.items() if ts - v.last <= c.forget_s}
        self._reported = [(b, t) for b, t in self._reported if ts - t <= c.report_again_s]
        alerts, self.hazards = [], []
        for k, v in self.objects.items():
            if ts - v.first < c.hazard_hold_s or not self.on_path(v.box):
                continue
            self.hazards.append({"key": k, "name": v.name, "box": list(v.box), "since": round(v.first, 1)})
            if not v.reported and any(_iou(b, v.box) >= c.same_iou or max(_overlap(v.box, b), _overlap(b, v.box)) >= 0.5
                                      for b, _t in self._reported):
                v.reported = True  # the same spot was reported a little while ago: shown, not reported again
            if not v.reported:
                v.reported = True
                self._reported.append((v.box, ts))
                alerts.append(self._alert("trip_hazard", None, ts, "low",
                                          f"Trip hazard: {v.name} on the floor where people walk",
                                          {"rule": "Trip hazard", "object": v.name, "box": list(v.box)}))
        return alerts

    def _watch_steps(self, poses: dict, activity: dict, ts: float) -> list:
        c = self.cfg
        for tid in [t for t in self.walkers if t not in poses]:
            del self.walkers[tid]
        alerts = []
        for tid, pose in poses.items():
            warned_now = False
            w = self.walkers.setdefault(tid, _Walker())
            kp = np.asarray(pose.keypoints, float)
            f = _feet(kp, c.min_conf)
            box = getattr(pose, "bbox", None)
            if f is None or box is None:
                continue
            h = float(box[3] - box[1])
            w.feet.append((ts, float(f[0]), float(f[1]), h))
            if (activity.get(tid) or {}).get("label") != "Walking" or len(w.feet) < 5 or not self.hazards:
                continue
            (t0, x0, y0, _), (t1, x1, y1, _) = w.feet[0], w.feet[-1]
            if t1 - t0 < 0.3:
                continue
            vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
            ahead = (x1 + vx * c.lookahead_s, y1 + vy * c.lookahead_s)
            pad = c.approach_margin * h
            for hz in self.hazards:
                b = hz["box"]
                grown = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
                now_d = math.hypot(x1 - (b[0] + b[2]) / 2, y1 - (b[1] + b[3]) / 2)
                then_d = math.hypot(ahead[0] - (b[0] + b[2]) / 2, ahead[1] - (b[1] + b[3]) / 2)
                if (not warned_now and (_inside(ahead, grown) or _inside((x1, y1), grown)) and then_d < now_d
                        and ts - max(w.warned.values(), default=-math.inf) >= c.step_cooldown_s):
                    warned_now = True
                    w.warned[hz["key"]] = ts
                    alerts.append(self._alert("watch_step", tid, ts, "high", f"Watch your step: {hz['name']} ahead",
                                              {"rule": "Watch your step", "object": hz["name"]}))
        return alerts

    @staticmethod
    def _alert(kind, tid, ts, severity, message, details) -> AnomalyAlert:
        return AnomalyAlert(alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}", alert_type=kind,
                            track_id=None if tid is None else int(tid), timestamp=ts, confidence=0.7,
                            severity=severity, message=message, details=details)
