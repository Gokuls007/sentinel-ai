"""Warehouse rules that combine a person's pose with detected objects.

- **Unsafe lift**: a wrist inside a box's bounding box (grown by a margin), the back bent
  (trunk more than ``trunk_deg`` from vertical, or, when bending toward the camera hides the
  angle, the torso shorter than ``foreshortened`` of that person's upright length), and the
  legs straight (knee angle over ``knee_straight_deg``). Knees are only judged when hip, knee
  and ankle are confidently visible; if they aren't (a box often hides the legs), the alert
  says "knees not visible" instead of guessing.
- **Standing on a chair**: both ankles horizontally inside a chair's box and above its vertical
  midpoint (about seat height; the box top is the backrest), and the person's feet (box
  bottom) well above the chair's base. The horizontal check stops a person standing behind a
  chair (further back = higher in the image) from counting.
- **Hand on a hazard**: a wrist inside the box of an object marked as a hazard (grown by a small
  margin), e.g. "Hand on knife (hazard)".

Distances scale with torso length (shoulder midpoint to hip midpoint), which barely changes when
someone bends, unlike their bounding box. "Bent toward the camera" compares the torso with the
thigh (both shrink equally with distance, so walking away isn't a bend), against that person's
own upright ratio. A hazard only counts while the person is nearly still: walking past an object
can put a hand over it in the image without touching it. Each rule must hold for a short time and then has a
per-person cooldown. Thresholds are first guesses, to be tuned on recordings.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from anomaly.engine import AnomalyAlert

L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
DEFAULT_HAZARDS = ["tv", "knife", "scissors", "oven", "laptop"]


@dataclass
class ObjectRulesConfig:
    lift_classes: tuple[str, ...] = ("cardboard box",)
    lift_margin: float = 0.25          # x torso length around the box
    trunk_deg: float = 45.0
    foreshortened: float = 0.70        # torso this short vs upright: bent toward the camera
    knee_straight_deg: float = 150.0
    leg_conf: float = 0.5              # hip, knee and ankle all this confident to judge a knee
    lift_hold_s: float = 0.5
    chair_classes: tuple[str, ...] = ("chair",)
    ankle_conf: float = 0.5
    feet_above_base: float = 0.15      # x chair height: feet clearly off the floor
    chair_hold_s: float = 0.5
    hazard_margin: float = 0.08        # x torso length (at least 6 px)
    hazard_hold_s: float = 0.3
    cooldown_s: float = 10.0
    wrist_conf: float = 0.3
    still_speed: float = 0.35          # body heights per second: faster = walking (no hazard contact)


def _mid(kp, a, b, conf=0.3):
    pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= conf]
    return np.mean(pts, axis=0) if pts else None


def _angle_from_vertical(top, bottom) -> float:
    dx, dy = top[0] - bottom[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), dy)) if (dx or dy) else 0.0


def knee_angle(kp, hip, knee, ankle) -> float:
    """Angle at the knee (degrees, 180 = straight)."""
    a, b, c = kp[hip, :2], kp[knee, :2], kp[ankle, :2]
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def box_gap(point, box) -> float:
    """Pixels from a point to a box (0 inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return math.hypot(dx, dy)


class ObjectRules:
    def __init__(self, cfg: ObjectRulesConfig | None = None, hazard_classes: list[str] | None = None):
        self.cfg = cfg or ObjectRulesConfig()
        self.hazard_classes = list(DEFAULT_HAZARDS if hazard_classes is None else hazard_classes)
        self._upright: dict[int, float] = {}   # track id -> upright torso / thigh ratio
        self._hips: dict[int, deque] = {}       # track id -> (ts, hip midpoint, box height)
        self._since: dict[tuple, float] = {}   # (track, rule, object) -> condition true since
        self._fired: dict[tuple, float] = {}   # (track, rule) -> last alert time

    def _hold(self, key: tuple, ok: bool, ts: float, hold_s: float) -> bool:
        if not ok:
            self._since.pop(key, None)
            return False
        start = self._since.setdefault(key, ts)
        if ts - start < hold_s:
            return False
        rule_key = key[:2]
        if ts - self._fired.get(rule_key, -1e9) < self.cfg.cooldown_s:
            return False
        self._fired[rule_key] = ts
        self._since.pop(key, None)
        return True

    def update(self, poses: dict, objects: list, ts: float, labels: dict | None = None) -> list[AnomalyAlert]:
        """``labels``: activity per track; the unsafe-lift rule only checks people on their feet."""
        c = self.cfg
        labels = labels or {}
        for tid in [t for t in self._upright if t not in poses]:
            del self._upright[tid]
        for tid in [t for t in self._hips if t not in poses]:
            del self._hips[tid]
        self._since = {k: v for k, v in self._since.items() if k[0] in poses}
        alerts = []
        for tid, pose in poses.items():
            kp = np.asarray(pose.keypoints, float)
            sh, hip = _mid(kp, L_SH, R_SH), _mid(kp, L_HIP, R_HIP)
            if sh is None or hip is None:
                continue
            torso = float(np.hypot(*(sh - hip)))
            trunk = _angle_from_vertical(sh, hip)
            thighs = [float(np.hypot(*(kp[k, :2] - kp[h, :2]))) for h, k in ((L_HIP, L_KNEE), (R_HIP, R_KNEE))
                      if min(kp[h, 2], kp[k, 2]) >= 0.4]
            thigh = float(np.mean(thighs)) if thighs else 0.0
            rel = torso / thigh if thigh > 0.2 * torso else None
            up = self._upright.get(tid)
            # Upright torso/thigh ratio: only from frames that look upright AND aren't foreshortened
            # (a bend toward the camera also looks vertical; letting it in would shrink the reference).
            if rel is not None and trunk <= 15 and (up is None or rel >= 0.85 * up):
                self._upright[tid] = rel if up is None else max(rel, 0.97 * up + 0.03 * rel)
            up = self._upright.get(tid)
            ratio = rel / up if rel is not None and up else None   # torso vs its upright length, same depth
            scale = max(torso, up * thigh) if up and thigh else torso  # the upright torso at this distance
            wrists = [kp[i, :2] for i in (L_WR, R_WR) if kp[i, 2] >= c.wrist_conf]
            moving = self._speed(tid, hip, pose.bbox, ts) > c.still_speed
            if labels.get(tid) not in ("Sitting", "Lying down", "Fallen"):
                alerts += self._unsafe_lift(tid, kp, wrists, trunk, ratio, scale, objects, ts)
            alerts += self._on_chair(tid, kp, pose.bbox, objects, ts)
            alerts += self._hazards(tid, wrists, scale, objects, ts, moving)
        return alerts

    def _speed(self, tid, hip, box, ts, window_s: float = 0.6) -> float:
        """Hip speed in body (box) heights per second over the last ``window_s``."""
        q = self._hips.setdefault(tid, deque())
        q.append((ts, hip, max(float(box[3] - box[1]), 1.0)))
        while q and ts - q[0][0] > window_s:
            q.popleft()
        if len(q) < 2 or q[-1][0] - q[0][0] < 0.2:
            return 0.0
        (t0, p0, h0), (t1, p1, _h1) = q[0], q[-1]
        return float(np.hypot(*(p1 - p0))) / h0 / (t1 - t0)

    # --- rules -------------------------------------------------------------------------------------

    def _unsafe_lift(self, tid, kp, wrists, trunk, ratio, scale, objects, ts) -> list:
        c = self.cfg
        boxes = [o for o in objects if o.class_name in c.lift_classes]
        near = None
        for o in boxes:
            m = c.lift_margin * scale
            x1, y1, x2, y2 = o.bbox
            for w in wrists:
                if x1 - m <= w[0] <= x2 + m and y1 - m <= w[1] <= y2 + m:
                    gap = box_gap(w, o.bbox)
                    if near is None or gap < near[1]:
                        near = (o, gap)
        foreshort = ratio is not None and ratio < c.foreshortened
        bent = trunk > c.trunk_deg or foreshort
        knees = []
        for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
            if min(kp[hip, 2], kp[knee, 2], kp[ankle, 2]) >= c.leg_conf:
                knees.append(knee_angle(kp, hip, knee, ankle))
        straight = bool(knees) and min(knees) > c.knee_straight_deg
        ok = near is not None and bent and (straight or not knees)
        if not self._hold((tid, "unsafe_lift", near[0].class_name if near else ""), ok, ts, c.lift_hold_s):
            return []
        o, gap = near
        back = (f"back bent {trunk:.0f}°" if trunk > c.trunk_deg
                else f"bent toward the camera (torso {ratio:.0%} of upright)")
        legs = f"knees {min(knees):.0f}° (straight)" if knees else "knees not visible"
        hand = "hand on" if gap == 0 else f"hand {gap:.0f} px from"
        msg = f"Unsafe lift: {back}, {legs}, {hand} {o.class_name}"
        details = {"rule": "Unsafe lift", "trunk_deg": round(trunk, 1),
                   "torso_ratio": round(ratio, 2) if ratio is not None else None,
                   "knee_deg": round(min(knees), 1) if knees else None, "knees_visible": bool(knees),
                   "hand_gap_px": round(gap, 1), "object": o.class_name}
        return [self._alert("unsafe_lift", tid, ts, "medium", msg, details, o.confidence)]

    def _on_chair(self, tid, kp, pbox, objects, ts) -> list:
        c = self.cfg
        if min(kp[L_ANK, 2], kp[R_ANK, 2]) < c.ankle_conf:
            self._since.pop((tid, "standing_on_chair", "chair"), None)
            return []
        ankles = [kp[L_ANK, :2], kp[R_ANK, :2]]
        feet = float(pbox[3])
        hit = None
        for o in objects:
            if o.class_name not in c.chair_classes:
                continue
            x1, y1, x2, y2 = o.bbox
            seat = (y1 + y2) / 2
            inside = all(x1 <= a[0] <= x2 for a in ankles)
            above_seat = all(a[1] < seat for a in ankles)
            off_floor = feet < y2 - c.feet_above_base * (y2 - y1)
            if inside and above_seat and off_floor:
                hit = (o, [seat - a[1] for a in ankles], y2 - feet)
                break
        if not self._hold((tid, "standing_on_chair", "chair"), hit is not None, ts, c.chair_hold_s):
            return []
        o, above, base = hit
        msg = (f"Standing on chair: ankles {above[0]:.0f} px and {above[1]:.0f} px above the seat line, "
               f"feet {base:.0f} px above the chair's base")
        details = {"rule": "Standing on chair", "ankles_above_seat_px": [round(a, 1) for a in above],
                   "feet_above_base_px": round(base, 1), "object": o.class_name}
        return [self._alert("standing_on_chair", tid, ts, "high", msg, details, o.confidence)]

    def _hazards(self, tid, wrists, scale, objects, ts, moving: bool = False) -> list:
        c = self.cfg
        out = []
        if moving:  # walking past: a hand over an object in the image isn't a touch
            self._since = {k: v for k, v in self._since.items() if not (k[0] == tid and k[1] == "hazard_contact")}
            return out
        m = max(6.0, c.hazard_margin * scale)
        for o in objects:
            if o.class_name not in self.hazard_classes:
                continue
            x1, y1, x2, y2 = o.bbox
            touching = any(x1 - m <= w[0] <= x2 + m and y1 - m <= w[1] <= y2 + m for w in wrists)
            if self._hold((tid, "hazard_contact", o.class_name), touching, ts, c.hazard_hold_s):
                msg = f"Hand on {o.class_name} (hazard)"
                out.append(self._alert("hazard_contact", tid, ts, "high", msg,
                                       {"rule": "Hand on hazard", "object": o.class_name}, o.confidence))
        return out

    @staticmethod
    def _alert(kind, tid, ts, severity, msg, details, conf) -> AnomalyAlert:
        return AnomalyAlert(alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}", alert_type=kind, track_id=int(tid),
                            timestamp=ts, confidence=float(conf), severity=severity, message=msg, details=details)
