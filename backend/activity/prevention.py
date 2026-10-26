"""Fall prevention (home mode): warnings for the moments before a fall, while there is still time
for someone to help.

- **Getting out of bed** (``bed_exit``): someone who has been settled in bed (hips on a detected
  bed, lying or sitting, for ``settled_s``) puts a foot over the side (an ankle outside the bed's
  box while the hips are still on it) or starts to stand up. Hospital bed-exit alarms fire at the
  same moment: most falls of frail people at home and in care happen getting out of bed. Once per
  bed stay: it re-arms only after the person has settled in bed again.
- **Slipping out of a chair** (``chair_slip``): someone settled on a chair or couch whose hips sink
  ``slip_drop`` x torso below where they sat, for ``slip_hold_s``, without standing up: sliding
  forward off the seat. Own recording (two real slides): settled sitting moved the hips at most
  0.29 torso, lying on a bed 0.19, the slides 1.4-2.1. Fires before the person reaches the floor.
- **On the floor** (``on_the_floor``): off any seat and down on the floor for ``floor_hold_s``:
  hips near the feet (``leg_extension`` under ``floor_legs``), or the person's box under
  ``floor_height`` of the height a standing person has at that spot (the room map learns it from
  the people it sees standing; this works without legs in view), or labelled lying off any bed or
  couch: kneeling, crawling, sitting or lying on the floor. Lower severity than a fall: an older person on
  hands and knees who isn't getting up may have fallen and can't get up; someone who just lay
  down on the rug probably hasn't. Not raised while the fall detector already has them down.

Each alert carries the values that triggered it. Live only: nothing is stored per person.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from activity.balance import leg_extension
from anomaly.engine import AnomalyAlert
from anomaly.geometry import box_too_close

L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
SEATS = ("chair", "couch", "bench")
BEDS = ("bed",)
LYING = ("Lying down", "Fallen")


@dataclass
class PreventionConfig:
    min_conf: float = 0.4
    settled_s: float = 2.0       # in bed / on a seat this long before a change counts
    exit_hold_s: float = 0.5     # a foot over the side of the bed this long
    slip_drop: float = 0.5       # hips this far below where they sat (x seated torso)
    slip_hold_s: float = 0.5
    floor_legs: float = 0.35     # hips this close to the feet (x the leg's own length): down on the floor
    floor_height: float = 0.6    # or the box under this x a standing person's height at that spot (room map)
    floor_hold_s: float = 3.0
    lying_hold_s: float = 6.0    # "lying" from the label alone (no low hips, no short box): a slow pick-up
                                 # with the head down reads as lying for a few seconds
    cooldown_s: float = 10.0     # duplicate guard per person and warning (each fires once per bed stay,
                                 # seat stay or time on the floor anyway)
    gap_s: float = 1.0           # frames without the person (or an odd pose) shorter than this are bridged
    keep_s: float = 5.0          # a person missing this long is forgotten


@dataclass
class _Person:
    bed_since: float | None = None      # settled in bed since
    bed_armed: bool = False             # settled in bed, a bed exit not yet reported
    exit_since: float | None = None
    seat_since: float | None = None
    seat_hips: deque = field(default_factory=lambda: deque(maxlen=90))  # (ts, hip y, torso) while seated
    slip_since: float | None = None
    floor_since: float | None = None
    floor_last: float | None = None     # last frame that looked down (short gaps don't restart the timer)
    floor_reported: bool = False
    last_seen: float = 0.0
    fired: dict = field(default_factory=dict)  # alert type -> last time


def _hips(kp, c):
    pts = [kp[i, :2] for i in (L_HIP, R_HIP) if kp[i, 2] >= c]
    return np.mean(pts, axis=0) if pts else None


def _torso(kp, c):
    if min(kp[L_SH, 2], kp[R_SH, 2], kp[L_HIP, 2], kp[R_HIP, 2]) < c:
        return None
    return float(np.linalg.norm(kp[[L_SH, R_SH], :2].mean(0) - kp[[L_HIP, R_HIP], :2].mean(0)))


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


class PreventionTracker:
    def __init__(self, cfg: PreventionConfig | None = None):
        self.cfg = cfg or PreventionConfig()
        self.people: dict[int, _Person] = {}
        self.current: dict[int, dict] = {}  # track id -> what each rule sees now (for the overlay / API)

    def update(self, poses: dict, activity: dict, objects: list, ts: float,
               down_ids: set | frozenset = frozenset(), room=None) -> list[AnomalyAlert]:
        """``objects``: detected objects (class_name, bbox); ``down_ids``: people the fall detector
        already has falling or on the ground; ``room``: the room map (standing height by position)."""
        for tid in [t for t, p in self.people.items() if t not in poses and ts - p.last_seen > self.cfg.keep_s]:
            del self.people[tid]
        beds = [o.bbox for o in objects if o.class_name in BEDS]
        seats = [o.bbox for o in objects if o.class_name in SEATS]
        alerts, self.current = [], {}
        for tid, pose in poses.items():
            st = self.people.setdefault(tid, _Person())
            st.last_seen = ts
            kp = np.asarray(pose.keypoints, float)
            act = activity.get(tid) or {}
            view: dict = {}
            alerts += self._bed_exit(tid, st, kp, act, beds, ts, view)
            alerts += self._chair_slip(tid, st, kp, act, seats, ts, view)
            alerts += self._on_floor(tid, st, kp, act, beds + seats, ts, view, tid in down_ids,
                                     getattr(pose, "bbox", None), room)
            self.current[tid] = view
        return alerts

    # --- getting out of bed ----------------------------------------------------------------------

    def _bed_exit(self, tid, st, kp, act, beds, ts, view) -> list:
        c = self.cfg
        hip = _hips(kp, c.min_conf)
        bed = next((b for b in beds if hip is not None and _inside(hip, b)), None)
        label = act.get("label")
        in_bed = bed is not None and label in ("Lying down", "Sitting")
        if in_bed:
            st.bed_since = ts if st.bed_since is None else st.bed_since
            if ts - st.bed_since >= c.settled_s:
                st.bed_armed = True
        elif bed is None:
            st.bed_since = None
        if not st.bed_armed:
            st.exit_since = None
            return []
        view["in_bed"] = True
        ankles = [kp[i, :2] for i in (L_ANK, R_ANK) if kp[i, 2] >= c.min_conf]
        foot_out = bed is not None and any(not _inside(a, bed) for a in ankles)
        # Up off the bed: hips no longer on it and upright. (The "standing up" transition alone isn't
        # used: lying along the camera's line of sight it fires while still in bed.)
        standing_up = bed is None and hip is not None and label in ("Standing", "Walking", "Bending")
        if not (foot_out or standing_up):
            st.exit_since = None
            return []
        st.exit_since = ts if st.exit_since is None else st.exit_since
        if ts - st.exit_since < c.exit_hold_s and not standing_up:
            return []
        st.bed_armed, st.exit_since, st.bed_since = False, None, None
        how = "starting to stand up" if standing_up else "a foot over the side of the bed"
        return self._fire(tid, st, ts, "bed_exit", "high", f"Getting out of bed: {how}",
                          {"rule": "Getting out of bed", "how": how})

    # --- slipping out of a chair -----------------------------------------------------------------

    def _chair_slip(self, tid, st, kp, act, seats, ts, view) -> list:
        c = self.cfg
        hip, torso = _hips(kp, c.min_conf), _torso(kp, c.min_conf)
        label = act.get("label")
        # A seat stay starts on a detected seat (a "Sitting" label alone also comes from walking under a
        # high camera); once started, plain "Sitting" frames continue it (the seat's box comes and goes).
        seated = label == "Sitting" and (act.get("detail") in SEATS or st.seat_since is not None)
        if seated and hip is not None and torso and not act.get("transition"):
            st.seat_since = ts if st.seat_since is None else st.seat_since
            st.seat_hips.append((ts, float(hip[1]), torso))
        settled = st.seat_since is not None and ts - st.seat_since >= c.settled_s and len(st.seat_hips) >= 8
        if st.seat_hips and ts - st.seat_hips[-1][0] > 10.0:  # off the seat a while: that stay is over
            st.seat_since = None
            st.seat_hips.clear()
            settled = False
        if not settled:
            if not seated:
                st.seat_since = None
                st.seat_hips.clear()
            st.slip_since = None
            return []
        recent = [h for h in st.seat_hips if ts - h[0] <= 4.0] or list(st.seat_hips)
        hip0, torso0 = float(np.median([h[1] for h in recent])), float(np.median([h[2] for h in recent]))
        if hip is None:
            return []
        drop = (float(hip[1]) - hip0) / max(torso0, 1.0)
        view["seat_drop"] = round(drop, 2)
        # Standing up raises the hips well above the seat. (Not the label: sliding forward fast reads
        # as "Walking" for a moment.)
        rising = drop <= -c.slip_drop
        if rising or drop < c.slip_drop:
            st.slip_since = None
            if rising:  # stood up: the seat stay is over
                st.seat_since = None
                st.seat_hips.clear()
            return []
        st.slip_since = ts if st.slip_since is None else st.slip_since
        if ts - st.slip_since < c.slip_hold_s:
            return []
        st.slip_since, st.seat_since = None, None
        st.seat_hips.clear()
        return self._fire(tid, st, ts, "chair_slip", "high",
                          f"Slipping out of a seat: hips {drop:.1f} torso lengths below where they sat",
                          {"rule": "Slipping out of a seat", "hip_drop_torso": round(drop, 2)})

    # --- on the floor ----------------------------------------------------------------------------

    def _on_floor(self, tid, st, kp, act, furniture, ts, view, already_down, box, room) -> list:
        c = self.cfg
        hip = _hips(kp, c.min_conf)
        on_furniture = hip is not None and any(_inside(hip, b) for b in furniture)
        legs = leg_extension(kp, c.min_conf)
        label = act.get("label")
        low = legs is not None and legs < c.floor_legs
        # Much shorter than a person standing at that spot would look (the room map learns that from
        # the people it sees standing): kneeling, crawling, sitting or lying on the floor. Works
        # without legs in view, where the leg measure can't.
        short = self._height_ratio(box, room)
        if short is not None:
            view["height_ratio"] = round(short, 2)
        low_box = short is not None and short < c.floor_height
        lying_off_bed = label in LYING and act.get("detail") not in ("bed", "couch")
        seated_on_furniture = label == "Sitting" and act.get("detail") in (*SEATS, *BEDS)
        down = (low or low_box or lying_off_bed) and not on_furniture and not seated_on_furniture
        if already_down or (not down and (st.floor_last is None or ts - st.floor_last > c.gap_s)):
            st.floor_since = st.floor_last = None
            st.floor_reported = False
            return []
        if not down:
            return []
        if st.floor_since is None or ts - st.floor_last > c.gap_s:
            st.floor_since = ts
        st.floor_last = ts
        view["on_floor_s"] = round(ts - st.floor_since, 1)
        hold = c.floor_hold_s if (low or low_box) else c.lying_hold_s
        if ts - st.floor_since < hold or st.floor_reported:
            return []
        st.floor_reported = True
        how = (f"hips near the feet ({legs:.2f} of leg length)" if low
               else f"{short:.0%} of standing height there" if low_box else "lying")
        return self._fire(tid, st, ts, "on_the_floor", "medium",
                          f"On the floor for {ts - st.floor_since:.0f} s: {how}",
                          {"rule": "On the floor", "how": how, "seconds": round(ts - st.floor_since, 1)})

    @staticmethod
    def _height_ratio(box, room) -> float | None:
        """Box height over the room map's standing height for feet at the box bottom (None without a
        good fit, or with the box cut off by the picture's edge or too close to the camera)."""
        if room is None or box is None or getattr(room, "fit", None) is None:
            return None
        _x1, y1, _x2, y2 = (float(v) for v in box[:4])
        if y2 >= 0.97 * room.height or y1 <= 0.01 * room.height or box_too_close(box, room.width, room.height):
            return None
        h = room.standing_height(y2)
        return (y2 - y1) / h if h else None

    # ----------------------------------------------------------------------------------------------

    def _fire(self, tid, st, ts, kind, severity, message, details) -> list:
        if ts - st.fired.get(kind, -math.inf) < self.cfg.cooldown_s:
            return []
        st.fired[kind] = ts
        return [AnomalyAlert(alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}", alert_type=kind, track_id=int(tid),
                             timestamp=ts, confidence=0.8, severity=severity, message=message, details=details)]
