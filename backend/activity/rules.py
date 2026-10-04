"""Live activity labels from pose rules: Standing, Walking, Sitting, Bending, Lifting, Carrying,
Reaching overhead, Lying down, Fallen.

Rules first (each needs only the body parts it uses, and isn't offered without them):
- **Fallen**: the fall detector's confirmed state.
- **Lying down**: on the ground per the fall detector (not yet confirmed), or the torso more
  than ``lying_deg`` from vertical, or the head as low as the hips, or a wide, flat box.
- **Reaching overhead**: a wrist above the head.
- **Bending**: trunk flexion of ``bend_deg`` or more, or (bending toward a front camera, which
  hides the angle) the torso foreshortened to ``bend_torso_ratio`` of its upright length, or
  the hands down at knee height with the legs extended (a squat pick-up with an upright back).
- **Lifting**: bending with the wrists low (near or below the knees) while the hips stay up
  (not sitting or kneeling), then rising to upright within ``lift_window_s``; shown for
  ``lift_show_s``.
- **Carrying**: upright, wrists between hips and shoulders in front of the body, holding a
  detected COCO object (backpack, handbag, suitcase). Without a detected object (COCO has no
  "box"), only when walking with the hands together and raised above the hips (hanging arms
  overlap in a side view), and marked low confidence.
- **Sitting**: thighs (hip -> knee) well off vertical, or the hips dropped by ``sit_drop`` of the
  upright torso length, with an upright trunk.
- **Walking / Standing**: upright, hip speed over / under ``walk_speed`` body heights per second.

Labels are smoothed (majority over ``smooth_s``) and each person keeps a short history
(``history_s``), live only: nothing is stored per person.

Tuned only on CAUCAFall subjects 1-5 (subjects 6-10 are the held-out fall test set). Where a
learned model could help later: the ActionLSTM for Lifting vs Bending timing and Carrying with
no visible object.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from activity.visibility import visible_parts

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

STANDING, WALKING, SITTING, BENDING = "Standing", "Walking", "Sitting", "Bending"
LIFTING, CARRYING, REACHING, LYING, FALLEN = "Lifting", "Carrying", "Reaching overhead", "Lying down", "Fallen"
UPPER_ONLY = "Upper body only"
CARRY_CLASSES = ("backpack", "handbag", "suitcase")


@dataclass
class ActivityConfig:
    min_conf: float = 0.3
    lying_deg: float = 75.0           # a deep stoop is ~70 degrees: still bending
    bend_deg: float = 40.0
    upright_deg: float = 25.0
    sit_thigh_deg: float = 50.0
    sit_drop: float = 0.35            # hips this far below their standing height (x upright torso)
    bend_torso_ratio: float = 0.72    # torso this short vs upright: bending toward the camera
    lying_aspect: float = 1.3         # box width / height
    lift_hip_drop: float = 0.25       # hips dropped more than this (x upright torso): not a lift
    legs_extended: float = 0.5        # knees this far below the hips (x torso): not sitting
    knee_margin: float = 0.0          # wrists at or below knee height (x torso): reaching low
    low_hold_s: float = 0.5           # hands low this long (not walking) before a rise counts as a lift
    walk_speed: float = 0.35          # body heights per second
    speed_window_s: float = 1.0
    reach_margin: float = 0.03        # body heights above the head
    lift_window_s: float = 4.0
    lift_show_s: float = 2.0
    smooth_s: float = 1.0
    history_s: float = 120.0


@dataclass
class _Track:
    raw: deque = field(default_factory=deque)        # (ts, label)
    hips: deque = field(default_factory=deque)       # (ts, (x, y), body height)
    label: str | None = None
    since: float = 0.0
    low_bend_at: float | None = None                 # last time: bending with wrists low
    low_since: float | None = None                   # start of the current hands-low stretch
    lift_until: float = 0.0
    history: list = field(default_factory=list)      # [label, start, end]
    detail: str | None = None
    torso_up: float | None = None                    # upright torso length (px), slow average
    hip_up: float | None = None                      # upright hip height (image y)


def _mid(kp, a, b, min_conf):
    pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= min_conf]
    return np.mean(pts, axis=0) if pts else None


def _angle_from_vertical(top, bottom) -> float:
    """Degrees between the segment bottom->top and straight up (image y grows downward)."""
    dx, dy = top[0] - bottom[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), dy)) if (dx or dy) else 0.0


class ActivityTracker:
    def __init__(self, cfg: ActivityConfig | None = None):
        self.cfg = cfg or ActivityConfig()
        self.tracks: dict[int, _Track] = {}

    def prune(self, active) -> None:
        for tid in [t for t in self.tracks if t not in active]:
            del self.tracks[tid]

    # --- one frame, one person -----------------------------------------------------------------

    def classify(self, tid: int, kp, ts: float, body_height: float, fallen: bool = False,
                 objects: list[tuple[str, tuple]] | None = None, fall_state: str | None = None,
                 box: tuple | None = None) -> tuple[str, str | None, str]:
        """(label, detail, confidence "high"/"low") for this frame, before smoothing."""
        c = self.cfg
        kp = np.asarray(kp, float)
        st = self.tracks.setdefault(tid, _Track())
        parts = visible_parts(kp, c.min_conf)
        bh = max(body_height, 1.0)
        if fallen:
            return FALLEN, None, "high"
        sh, hip = _mid(kp, L_SH, R_SH, c.min_conf), _mid(kp, L_HIP, R_HIP, c.min_conf)
        head = min((kp[i, 1] for i in (NOSE, L_EYE, R_EYE) if kp[i, 2] >= c.min_conf), default=None)
        wrists = [kp[i, :2] for i in (L_WR, R_WR) if kp[i, 2] >= c.min_conf]
        if hip is not None:
            st.hips.append((ts, tuple(hip), bh))
        while st.hips and ts - st.hips[0][0] > c.speed_window_s:
            st.hips.popleft()

        if fall_state == "fallen":
            return LYING, None, "high"  # on the ground; "Fallen" once the fall detector confirms
        if box is not None and (box[2] - box[0]) >= c.lying_aspect * max(box[3] - box[1], 1.0):
            return LYING, None, "high"
        if sh is None or hip is None:
            if head is not None and wrists and min(w[1] for w in wrists) < head - c.reach_margin * bh:
                return REACHING, None, "high"
            return UPPER_ONLY, None, "low"
        trunk = _angle_from_vertical(sh, hip)
        torso = float(math.hypot(*(sh - hip)))
        if trunk >= c.lying_deg or (head is not None and head >= hip[1] - 0.05 * bh):
            return LYING, None, "high"
        if head is not None and wrists and min(w[1] for w in wrists) < head - c.reach_margin * bh:
            return REACHING, None, "high"
        knee = _mid(kp, L_KNEE, R_KNEE, c.min_conf)
        thigh = None
        if knee is not None and parts["knees"]:
            thigh = _angle_from_vertical(hip, knee)  # 0 = straight down (standing)
            thigh = 180 - thigh if thigh > 90 else thigh
        speed = 0.0
        if len(st.hips) >= 2 and st.hips[-1][0] - st.hips[0][0] > 0.3:
            (t0, p0, b0), (t1, p1, _b1) = st.hips[0], st.hips[-1]
            speed = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(b0, 1.0) / (t1 - t0)
        # Upright reference: torso length and hip height while clearly standing.
        if trunk <= 15 and (thigh is None or thigh <= 25):
            st.torso_up = torso if st.torso_up is None else max(torso, 0.97 * st.torso_up + 0.03 * torso)
            st.hip_up = hip[1] if st.hip_up is None else 0.9 * st.hip_up + 0.1 * hip[1]
        hip_drop = ((hip[1] - st.hip_up) / st.torso_up) if st.torso_up and st.hip_up is not None else 0.0
        foreshortened = st.torso_up is not None and torso <= c.bend_torso_ratio * st.torso_up and hip_drop < c.sit_drop
        # Hands down at knee height with the legs extended (not sitting): reaching low, e.g. a
        # squat pick-up with an upright back (CAUCAFall "Pick up object" looks like this).
        legs_out = knee is not None and (knee[1] - hip[1]) >= c.legs_extended * torso
        low_reach = legs_out and len(wrists) > 0 and max(w[1] for w in wrists) >= knee[1] - c.knee_margin * torso
        if trunk >= c.bend_deg or foreshortened or low_reach:
            low_line = knee[1] - 0.1 * bh if knee is not None else hip[1] + 0.35 * bh
            hips_up = hip_drop < c.lift_hip_drop and (thigh is None or thigh < c.sit_thigh_deg)
            hands_low = bool(wrists) and max(w[1] for w in wrists) >= low_line
            if hands_low and hips_up and speed < c.walk_speed:
                st.low_since = ts if st.low_since is None else st.low_since
                if ts - st.low_since >= c.low_hold_s:
                    st.low_bend_at = ts
            else:
                st.low_since = None
            return BENDING, None, "high"
        # Lifting: bent with hands low, now upright again.
        st.low_since = None
        if st.low_bend_at is not None and ts - st.low_bend_at <= c.lift_window_s and trunk <= c.upright_deg:
            st.lift_until, st.low_bend_at = ts + c.lift_show_s, None
        if ts < st.lift_until:
            return LIFTING, None, "high"

        if trunk <= c.bend_deg and ((thigh is not None and thigh >= c.sit_thigh_deg)
                                     or (parts["knees"] and hip_drop >= c.sit_drop)):
            return SITTING, None, "high"
        walking = speed >= c.walk_speed and (parts["knees"] or parts["ankles"])

        if wrists and len(wrists) == 2 and trunk <= c.upright_deg:
            between = all(sh[1] - 0.05 * bh <= w[1] <= hip[1] + 0.1 * bh for w in wrists)
            held = [name for name, (x1, y1, x2, y2) in (objects or [])
                    if name in CARRY_CLASSES and any(x1 - 0.1 * bh <= w[0] <= x2 + 0.1 * bh
                                                     and y1 - 0.1 * bh <= w[1] <= y2 + 0.1 * bh for w in wrists)]
            if between and held:
                return CARRYING, held[0], "high"
            together = math.hypot(*(wrists[0] - wrists[1])) <= 0.3 * bh
            raised = all(w[1] <= hip[1] - 0.1 * bh for w in wrists)  # hanging arms (side view) overlap too
            if between and together and raised and walking:
                return CARRYING, None, "low"
        if not (parts["knees"] or parts["ankles"]):
            return UPPER_ONLY, None, "low"
        return (WALKING if walking else STANDING), None, "high"

    def update(self, tid: int, kp, ts: float, body_height: float, fallen: bool = False,
               objects: list | None = None, fall_state: str | None = None, box: tuple | None = None) -> dict:
        """Classify, smooth, keep the history; returns the person's live activity."""
        c = self.cfg
        label, detail, conf = self.classify(tid, kp, ts, body_height, fallen, objects, fall_state, box)
        st = self.tracks[tid]
        st.raw.append((ts, label))
        while st.raw and ts - st.raw[0][0] > c.smooth_s:
            st.raw.popleft()
        # Short, important moments (a fall, a bend, a lift) show at once; the rest by majority
        # over smooth_s, so keypoint jitter doesn't flicker between Standing and Walking.
        smoothed = label if label in (FALLEN, LIFTING, BENDING) else Counter(
            lbl for _t, lbl in st.raw).most_common(1)[0][0]
        if smoothed != st.label:
            st.label, st.since = smoothed, ts
            st.history.append([smoothed, ts, ts])
        elif st.history:
            st.history[-1][2] = ts
        st.detail = detail if smoothed == label else st.detail
        while st.history and ts - st.history[0][2] > c.history_s:
            st.history.pop(0)
        return {"label": st.label, "since": st.since, "detail": st.detail,
                "confidence": conf if smoothed == label else "high",
                "history": [{"label": lbl, "start": round(a, 2), "end": round(b, 2)} for lbl, a, b in st.history]}
