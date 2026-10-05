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
  Torso length is measured against the thigh, so walking away from the camera (everything
  shrinks) isn't mistaken for a bend.
- **Lifting**: bending with the wrists low (below mid-thigh) while the hips stay up (not
  sitting or kneeling), then rising to upright within ``lift_window_s`` **with the hands up in
  front** (holding something; after a bend without a load the arms hang); shown for
  ``lift_show_s``.
- **Carrying**: after a lift, while the hands stay up in front; it ends when both arms hang for
  ``hang_s`` or the person bends (putting it down). Also: upright, wrists between hips and
  shoulders, holding a detected COCO object (backpack, handbag, suitcase); or walking with the
  hands together and raised (low confidence).
- **Sitting**: thighs (hip -> knee) well off vertical, or the hips dropped by ``sit_drop`` of the
  upright torso length, with an upright trunk. Also, found automatically: the hips on a detected
  chair, couch or bed (leaning forward up to ``seated_max_trunk`` still counts: reading), or,
  facing the camera, a thigh foreshortened toward the lens. The cue is the shin against the thigh
  (thigh length against the torso didn't separate them: both ~0.65 with RTMPose on a webcam):
  seated facing the camera the shin looks ``seat_shank_ratio``-``facing_shank_ratio`` x the thigh
  or more; standing, about the same length (p90 1.1-1.3 on own recordings). On a bed with the
  trunk tilted past 45 degrees: Lying down.
- **Walking / Standing**: upright, hip speed over / under ``walk_speed`` body heights per second,
  or growing/shrinking in the image (walking toward or away from the camera) faster than
  ``approach_rate`` per second.

**Transitions** are reported next to the label (they're the risky moments for falls):
"sit-to-stand" for someone seated in the last ``transition_window_s`` who either leans the trunk
forward by ``lean_deg`` while still seated, the usual first stage of standing up, or whose hips
have risen cumulatively by ``rise_torso`` x torso (slow risers stand in 2-4 s, often in stages).
Both are measured from how the person sat (medians over the seated frames of the last
``rise_window_s``) and must hold for ``sustain_s``: shifting about on a seat moves single frames
past those amounts, standing up keeps them there;
"lying-to-sitting" when someone lying has their trunk coming upright by ``sit_up_deg``. A
transition lasts ``transition_hold_s`` after the last frame that showed it. Balance
checks stay on during transitions; only settled sitting or lying turns them off.

Labels are smoothed (majority over ``smooth_s``) and each person keeps a short history
(``history_s``), live only: nothing is stored per person.

Tuned only on CAUCAFall subjects 1-5 (subjects 6-10 are the held-out fall test set) and on
the author's own side-view lift-and-carry recording. Where a
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
CARRY_CLASSES = ("backpack", "handbag", "suitcase", "cardboard box")  # COCO, plus YOLO-World's box
SEATS = ("chair", "couch", "bed", "bench", "toilet")  # detected objects someone can sit or lie on
# Detected objects nobody carries around: touching them isn't "Carrying".
NOT_CARRIED = frozenset({
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "microwave", "oven", "toaster",
    "sink", "refrigerator", "forklift", "ladder",
})


def carryable(name: str) -> bool:
    return name not in NOT_CARRIED


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
    seated_max_trunk: float = 60.0    # on a seat, leaning forward up to this still counts as sitting
    seat_shank_ratio: float = 1.3     # on a detected seat: shin >= this x thigh (thigh foreshortened) = seated
    facing_shank_ratio: float = 1.5   # no seat detected: shin >= this x thigh, upright trunk = seated
    settled_share: float = 0.6        # a transition needs this share of Sitting frames over 1 s first
    transition_window_s: float = 3.0  # seated / lying this recently, then rising: a transition
    rise_window_s: float = 3.0        # the hip rise is measured cumulatively over up to this long
    rise_torso: float = 0.20          # hips up by this x torso from their lowest point: standing up
    lean_deg: float = 15.0            # trunk tilting forward this much while still seated: about to stand
    sit_up_deg: float = 20.0          # trunk this much more upright (from reclined): sitting up
    transition_hold_s: float = 1.5
    sustain_s: float = 0.3            # a lean or rise must hold this long (one jittery frame isn't standing up)
    stable_window_s: float = 0.7      # the transition logic sees labels smoothed over this window...
    stable_share: float = 0.6         # ...switching only when a new label holds this share (hysteresis)
    legs_extended: float = 0.5        # knees this far below the hips (x torso): not sitting
    knee_margin: float = 0.0          # wrists at or below knee height (x torso): reaching low
    low_hold_s: float = 0.5           # hands low this long (not walking) before a rise counts as a lift
    low_gap_s: float = 0.3            # hands-low gaps shorter than this don't restart that timer
    walk_speed: float = 0.25          # body heights per second
    carry_walk_speed: float = 0.35    # the no-object, hands-raised Carrying needs a brisker walk
    approach_rate: float = 0.22       # |d ln(torso length)| per second: walking toward/away from the camera
    lift_hands_low: float = 0.4       # wrists below this far from hip to knee (0 = hips, 1 = knees)
    held_margin: float = 0.15         # a wrist this far above the hips (x torso) is "up", holding
    hang_s: float = 0.5               # both arms down this long: no longer carrying
    speed_window_s: float = 1.0
    reach_margin: float = 0.03        # body heights above the head
    lift_window_s: float = 4.0
    lift_show_s: float = 1.0
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
    low_last: float | None = None                    # last frame with the hands low
    lift_until: float = 0.0
    history: list = field(default_factory=list)      # [label, start, end]
    detail: str | None = None
    torso_up: float | None = None                    # upright torso length (px), slow average
    torso_rel_up: float | None = None                # upright torso / thigh length (scale-free)
    carrying: bool = False                           # lifted something and still holding it
    hang_since: float | None = None                  # both arms down since (while carrying)
    hip_up: float | None = None                      # upright hip height (image y)
    posture: deque = field(default_factory=deque)    # (ts, hip y, torso, trunk deg, stable label)
    stable: deque = field(default_factory=deque)     # (ts, raw label) over stable_window_s
    stable_label: str | None = None                  # smoothed with hysteresis, for transitions
    transition: str | None = None                    # "sit-to-stand" / "lying-to-sitting"
    transition_until: float = 0.0
    transition_why: str | None = None                # what started it: "leaning forward", "hips rising"...


def _shank_ratio(kp, min_conf) -> float | None:
    """Longest shin/thigh ratio over the legs with hip, knee and ankle visible (None if none)."""
    best = None
    for h, k, a in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if min(kp[h, 2], kp[k, 2], kp[a, 2]) >= min_conf:
            thigh = math.hypot(*(kp[k, :2] - kp[h, :2]))
            if thigh > 1:
                r = math.hypot(*(kp[a, :2] - kp[k, :2])) / thigh
                best = r if best is None else max(best, r)
    return best


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
        if body_height <= 1.0 and box is not None:  # no ankles in view: use the box
            body_height = box[3] - box[1]
        bh = max(body_height, 1.0)
        if fallen:
            st.carrying = False
            return FALLEN, None, "high"
        sh, hip = _mid(kp, L_SH, R_SH, c.min_conf), _mid(kp, L_HIP, R_HIP, c.min_conf)
        head = min((kp[i, 1] for i in (NOSE, L_EYE, R_EYE) if kp[i, 2] >= c.min_conf), default=None)
        wrists = [kp[i, :2] for i in (L_WR, R_WR) if kp[i, 2] >= c.min_conf]
        if hip is not None:
            torso_px = float(math.hypot(*(sh - hip))) if sh is not None else None
            st.hips.append((ts, tuple(hip), bh, torso_px))
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
        speed = approach = 0.0
        if len(st.hips) >= 2 and st.hips[-1][0] - st.hips[0][0] > 0.3:
            (t0, p0, b0, s0), (t1, p1, _b1, s1) = st.hips[0], st.hips[-1]
            speed = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(b0, 1.0) / (t1 - t0)
            if s0 and s1:
                approach = abs(math.log(s1 / s0)) / (t1 - t0)
        thigh_len = float(math.hypot(*(knee - hip))) if thigh is not None else 0.0
        torso_rel = torso / thigh_len if thigh_len > 0.2 * torso else None
        # Upright reference: torso length and hip height while clearly standing (not from a seated
        # person facing the camera, whose thigh is foreshortened: that would make standing look bent).
        shank = _shank_ratio(kp, c.min_conf)
        if trunk <= 15 and (thigh is None or (thigh <= 25 and (shank is None or shank < c.seat_shank_ratio))):
            st.torso_up = torso if st.torso_up is None else max(torso, 0.97 * st.torso_up + 0.03 * torso)
            st.hip_up = hip[1] if st.hip_up is None else 0.9 * st.hip_up + 0.1 * hip[1]
            if torso_rel is not None:
                st.torso_rel_up = (torso_rel if st.torso_rel_up is None
                                   else max(torso_rel, 0.97 * st.torso_rel_up + 0.03 * torso_rel))
        hip_drop = ((hip[1] - st.hip_up) / st.torso_up) if st.torso_up and st.hip_up is not None else 0.0
        if torso_rel is not None and st.torso_rel_up:  # scale-free: walking away shrinks both
            foreshortened = torso_rel <= c.bend_torso_ratio * st.torso_rel_up and hip_drop < c.sit_drop
        else:
            foreshortened = (st.torso_up is not None and torso <= c.bend_torso_ratio * st.torso_up
                             and hip_drop < c.sit_drop)
        # A wrist up in front (above the hips): holding something.
        held = any(w[1] <= hip[1] - c.held_margin * torso for w in wrists)
        # Hands down at knee height with the legs extended (not sitting): reaching low, e.g. a
        # squat pick-up with an upright back (CAUCAFall "Pick up object" looks like this).
        legs_out = knee is not None and (knee[1] - hip[1]) >= c.legs_extended * torso
        low_reach = legs_out and len(wrists) > 0 and max(w[1] for w in wrists) >= knee[1] - c.knee_margin * torso
        # Seated, found automatically: hips on a detected chair/couch/bed, or a thigh pointing at the
        # camera. (Standing in front of a chair: long, vertical thighs, no hip drop: not seated.)
        seat = next((name for name, (x1, y1, x2, y2) in (objects or [])
                     if name in SEATS and x1 <= hip[0] <= x2 and y1 <= hip[1] <= y2), None)
        foreshort_thigh = shank is not None and shank >= c.seat_shank_ratio
        if seat and (foreshort_thigh or thigh is None or thigh >= 30 or hip_drop >= 0.2):
            if seat == "bed" and trunk >= 45:
                return LYING, "bed", "high"
            if trunk < c.seated_max_trunk:
                st.carrying = False
                return SITTING, seat, "high"
        if shank is not None and shank >= c.facing_shank_ratio and trunk <= c.bend_deg:
            return SITTING, None, "high"
        if trunk >= c.bend_deg or foreshortened or low_reach:
            st.carrying = False  # bending puts it down (or picks something else up)
            low_line = (hip[1] + c.lift_hands_low * (knee[1] - hip[1]) if knee is not None
                        else hip[1] + 0.35 * bh)
            hips_up = hip_drop < c.lift_hip_drop and (thigh is None or thigh < c.sit_thigh_deg)
            hands_low = bool(wrists) and max(w[1] for w in wrists) >= low_line
            if hands_low and hips_up and speed < c.walk_speed:
                st.low_since = ts if st.low_since is None else st.low_since
                st.low_last = ts
                if ts - st.low_since >= c.low_hold_s:
                    st.low_bend_at = ts
            elif st.low_last is None or ts - st.low_last > c.low_gap_s:  # brief keypoint jitter is ignored
                st.low_since = None
            return BENDING, None, "high"
        # Lifting: bent with hands low, now upright again.
        st.low_since = None
        if st.low_bend_at is not None and ts - st.low_bend_at > c.lift_window_s:
            st.low_bend_at = None
        if st.low_bend_at is not None and trunk <= c.upright_deg and held:
            st.lift_until, st.low_bend_at, st.carrying, st.hang_since = ts + c.lift_show_s, None, True, None
        if ts < st.lift_until:
            return LIFTING, None, "high"
        if st.carrying:
            if held:
                st.hang_since = None
            else:
                st.hang_since = ts if st.hang_since is None else st.hang_since
                if ts - st.hang_since >= c.hang_s:
                    st.carrying = False

        if trunk <= c.bend_deg and ((thigh is not None and thigh >= c.sit_thigh_deg)
                                     or (parts["knees"] and hip_drop >= c.sit_drop)):
            return SITTING, None, "high"
        walking = (speed >= c.walk_speed or approach >= c.approach_rate) and (parts["knees"] or parts["ankles"])
        # A carryable object at one of the hands, e.g. "Carrying pillow".
        in_hand = next((name for name, (x1, y1, x2, y2) in (objects or [])
                        if carryable(name) and any(x1 - 0.1 * bh <= w[0] <= x2 + 0.1 * bh
                                                   and y1 - 0.1 * bh <= w[1] <= y2 + 0.1 * bh for w in wrists)), None)
        if st.carrying and trunk <= c.upright_deg:
            return CARRYING, in_hand, "high"

        if wrists and trunk <= c.upright_deg:
            at_body = any(sh[1] - 0.05 * bh <= w[1] <= hip[1] + 0.25 * bh for w in wrists)
            if in_hand and at_body:
                return CARRYING, in_hand, "high"
        if wrists and len(wrists) == 2 and trunk <= c.upright_deg:
            between = all(sh[1] - 0.05 * bh <= w[1] <= hip[1] + 0.1 * bh for w in wrists)
            together = math.hypot(*(wrists[0] - wrists[1])) <= 0.3 * bh
            raised = all(w[1] <= hip[1] - 0.1 * bh for w in wrists)  # hanging arms (side view) overlap too
            if between and together and raised and walking and speed >= c.carry_walk_speed:
                return CARRYING, None, "low"
        if not (parts["knees"] or parts["ankles"]):
            return UPPER_ONLY, None, "low"
        return (WALKING if walking else STANDING), None, "high"

    def _stable_label(self, st: _Track, ts: float, label: str) -> str:
        """Majority over stable_window_s with hysteresis: a new label takes over only when it holds
        stable_share of the window, so single flickery frames (Sitting <-> Bending) don't count."""
        c = self.cfg
        st.stable.append((ts, label))
        while st.stable and ts - st.stable[0][0] > c.stable_window_s:
            st.stable.popleft()
        counts = Counter(lbl for _t, lbl in st.stable)
        top, n = counts.most_common(1)[0]
        if st.stable_label is None or (top != st.stable_label and n >= c.stable_share * len(st.stable)):
            st.stable_label = top
        return st.stable_label

    def _transition(self, st: _Track, kp, ts: float, label: str) -> str | None:
        """Sit-to-stand / lying-to-sitting, from the last few seconds of hips and trunk."""
        c = self.cfg
        kp = np.asarray(kp, float)
        sh, hip = _mid(kp, L_SH, R_SH, c.min_conf), _mid(kp, L_HIP, R_HIP, c.min_conf)
        if sh is not None and hip is not None:
            st.posture.append((ts, float(hip[1]), float(math.hypot(*(sh - hip))), _angle_from_vertical(sh, hip),
                               self._stable_label(st, ts, label)))
        while st.posture and ts - st.posture[0][0] > c.transition_window_s + c.rise_window_s:
            st.posture.popleft()
        if len(st.posture) < 2 or label in (FALLEN,):
            return st.transition if ts < st.transition_until else None
        now = st.posture[-1]
        recent = [p for p in st.posture if now[0] - p[0] <= c.transition_window_s]
        # Settled sitting first: at least settled_share of the frames in some 1 s stretch of the window
        # (flickery single "Sitting" frames while bending or shifting don't count).
        seated = []
        for start in recent:
            span = [p for p in recent if 0 <= p[0] - start[0] <= 1.0]
            if (span and span[-1][0] - span[0][0] >= 0.8
                    and sum(p[4] == SITTING for p in span) >= c.settled_share * len(span)):
                seated = span
                break
        # Settled: a second upright (standing or walking, trunk near vertical) with the hips no longer
        # rising. Then the stand-up is over.
        last_s = [p for p in st.posture if now[0] - p[0] <= 1.0]
        still_rising = bool(last_s) and last_s[0][1] - now[1] >= 0.05 * max(now[2], 1.0)
        settled = (now[0] - st.posture[0][0] >= 1.0 and not still_rising
                   and all(p[4] in (STANDING, WALKING) and p[3] <= 15 for p in last_s))
        kind = None
        if settled:
            st.transition_until = 0.0
        elif st.transition == "sit-to-stand" and still_rising and ts < st.transition_until + 1.0:
            kind, why = "sit-to-stand", "rise in progress"  # keeps going, however long ago the seat was
        elif seated and now[4] != LYING:
            # Compared with how the person sat (medians over the seated frames, so shifting about on the
            # seat doesn't move the baseline), and only once it has held for sustain_s.
            sat = [p for p in st.posture if p[4] == SITTING and now[0] - p[0] <= c.rise_window_s] or seated
            hip0, torso0 = float(np.median([p[1] for p in sat])), max(float(np.median([p[2] for p in sat])), 1.0)
            trunk0 = float(np.median([p[3] for p in sat]))
            held = [p for p in st.posture if now[0] - p[0] <= c.sustain_s]
            # Cumulative rise: hips up from where they sat (the torso while seated sets the scale: a
            # torso leaning toward the camera looks short).
            rise = min(hip0 - p[1] for p in held) / torso0
            if rise >= c.rise_torso:
                kind, why = "sit-to-stand", f"hips rising {rise:.2f} torso"
            # Leaning forward while still seated: the first stage of standing up.
            lean = min(p[3] for p in held) - trunk0
            if kind is None and all(p[4] == SITTING for p in held) and lean >= c.lean_deg:
                kind, why = "sit-to-stand", f"leaning forward {lean:.0f} deg"
        if kind is None and any(p[4] == LYING for p in recent):
            reclined = max(p[3] for p in recent if p[4] == LYING)
            if reclined >= 45 and now[3] < 45 and reclined - now[3] >= c.sit_up_deg:
                kind, why = "lying-to-sitting", f"trunk up {reclined - now[3]:.0f} deg"
        if kind:
            st.transition, st.transition_until, st.transition_why = kind, ts + c.transition_hold_s, why
            return kind
        return st.transition if ts < st.transition_until else None

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
        st.transition = self._transition(st, kp, ts, label)
        return {"label": st.label, "since": st.since, "detail": st.detail, "transition": st.transition,
                "transition_why": st.transition_why if st.transition else None,
                "confidence": conf if smoothed == label else "high",
                "history": [{"label": lbl, "start": round(a, 2), "end": round(b, 2)} for lbl, a, b in st.history]}
