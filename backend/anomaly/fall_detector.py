"""Rule-based fall detection: a per-person state machine over pose signals.

    UPRIGHT --rapid descent--> FALLING --lying / head low--> FALLEN  ("possible fall" here)
    FALLEN --still for stillness_seconds--> CONFIRMED  ("confirmed fall" alert here)
    FALLING/FALLEN --stands up or times out--> UPRIGHT
    CONFIRMED --stands up--> UPRIGHT

All speeds are normalised by the person's calibrated standing height and by real
elapsed time, so thresholds work regardless of resolution, camera distance or FPS.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.pose_estimator import PoseResult, TrackFeatures


@dataclass
class FallEvent:
    track_id: int
    timestamp: float
    confidence: float
    stage: str  # "possible" (reached the ground) | "confirmed" (stayed down, still)
    signals: dict[str, float]
    head_y: float
    hip_y: float
    velocity: float  # descent speed that started the fall (body heights / s)
    aspect_ratio: float


@dataclass
class _TrackState:
    state: str = "upright"
    prev_hip_y: float | None = None
    prev_time: float | None = None
    upright_head_y: float | None = None  # running estimate while standing
    falling_since: float = 0.0
    fallen_since: float = 0.0
    still_since: float | None = None
    peak_descent: float = 0.0
    last_alert_time: float = field(default=-1e18)
    last_possible_time: float = field(default=-1e18)
    hip_history: deque = field(default_factory=deque)  # (timestamp, hip_y) for stillness
    last_seen: float | None = None  # when check() last had a pose for this track
    last_pose: PoseResult | None = None
    last_signals: dict | None = None
    upright_since: float | None = None  # start of the current run of "looks upright" frames
    anchor: object = None  # box centre where a recovered (re-found) person was lying
    floor_y: float | None = None  # where the feet were while standing (image y)
    box_samples: list = field(default_factory=list)  # upright box heights (box calibration)
    box_scale: float = 0.0


class FallDetector:
    UPRIGHT = "upright"
    FALLING = "falling"
    FALLEN = "fallen"
    CONFIRMED = "confirmed"

    FALLING_WINDOW_S = 1.5  # a fall must reach the ground within this time
    LYING_TORSO_DEG = 60.0  # torso this far from vertical = lying
    TILTED_TORSO_DEG = 35.0  # a wide box counts as lying only with the torso at least this tilted
    STILLNESS_WINDOW_S = 0.5
    POSSIBLE_COOLDOWN_S = 10.0  # at most one "possible fall" per person per 10 s

    def __init__(self, descent_speed_threshold: float = 1.2, aspect_ratio_threshold: float = 1.2,
                 head_drop_ratio: float = 0.5, stillness_seconds: float = 1.0,
                 stillness_speed_threshold: float = 0.15, fallen_timeout_seconds: float = 5.0,
                 cooldown_seconds: float = 30.0, lost_hold_seconds: float = 0.0,
                 upright_hold_seconds: float = 0.0, ground_mode: str = "torso",
                 box_calibration: bool = False):
        # ground_mode: how "on the ground" is judged. "torso": torso near horizontal and head
        #   dropped (this class's default, kept so older replays reproduce). "combined": also
        #   direction-independent signals (hips near the floor, the skeleton collapsing), with a
        #   veto when the hips are still high (bending to pick something up). The app uses
        #   "combined" by default via FallDetectorConfig.ground_mode.
        # box_calibration: without a measurable skeleton, estimate standing height from an
        #   upright bounding box so fall detection can start.
        self.ground_mode = ground_mode
        self.box_calibration = box_calibration
        # lost_hold_seconds: a person who vanishes while falling / on the ground (detector
        #   misses many lying people) is held at their last position, motionless, this long.
        # upright_hold_seconds: "looks upright" must last this long before leaving the
        #   ground state (one odd pose on the floor no longer cancels a fall).
        self.lost_hold_seconds = lost_hold_seconds
        self.upright_hold_seconds = upright_hold_seconds
        self.descent_speed_threshold = descent_speed_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.head_drop_ratio = head_drop_ratio
        self.stillness_seconds = stillness_seconds
        self.stillness_speed_threshold = stillness_speed_threshold
        self.fallen_timeout_seconds = fallen_timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self.tracks: dict[int, _TrackState] = {}
        self.possible_events: list[FallEvent] = []  # drained by drain_possible()

    # -- public API ---------------------------------------------------------------------

    def state_of(self, track_id: int) -> str:
        st = self.tracks.get(track_id)
        return st.state if st else self.UPRIGHT

    def check(self, track_id: int, pose: PoseResult, features: TrackFeatures,
              timestamp: float) -> FallEvent | None:
        """Advance the state machine for one person; returns an event on a confirmed fall."""
        st = self.tracks.setdefault(track_id, _TrackState())
        body_h = features.initial_standing_height
        if body_h <= 0 and self.box_calibration:
            body_h = self._box_scale(st, pose)
        if body_h <= 0:  # not calibrated yet: we don't know how tall this person is
            self._remember(st, pose, timestamp)
            return None

        signals = self._compute_signals(st, pose, body_h, timestamp)
        event = None

        if st.state == self.UPRIGHT:
            if not signals["horizontal_pose"] and pose.head_valid:
                # Track the standing head height slowly, so a fall can't drag it down.
                st.upright_head_y = (pose.head_y if st.upright_head_y is None
                                     else 0.9 * st.upright_head_y + 0.1 * pose.head_y)
            if signals["descent_speed"] > self.descent_speed_threshold:
                st.state = self.FALLING
                st.falling_since = timestamp
                st.peak_descent = signals["descent_speed"]
                st.anchor = None

        elif st.state == self.FALLING:
            st.peak_descent = max(st.peak_descent, signals["descent_speed"])
            if self._on_ground(signals):
                self._enter_fallen(st, track_id, timestamp, signals, pose)
            elif timestamp - st.falling_since > self.FALLING_WINDOW_S:
                st.state = self.UPRIGHT  # e.g. sat down or crouched quickly

        elif st.state == self.FALLEN:
            if self._upright_held(st, signals, timestamp):
                st.state = self.UPRIGHT
            else:
                if signals["is_still"]:
                    st.still_since = st.still_since if st.still_since is not None else timestamp
                else:
                    st.still_since = None
                still_long_enough = (st.still_since is not None and
                                     timestamp - st.still_since >= self.stillness_seconds)
                if still_long_enough and timestamp - st.last_alert_time >= self.cooldown_seconds:
                    st.state = self.CONFIRMED
                    st.last_alert_time = timestamp
                    event = self._make_event(track_id, timestamp, st, signals, pose)
                elif timestamp - st.fallen_since > self.fallen_timeout_seconds and not still_long_enough:
                    st.state = self.UPRIGHT

        elif st.state == self.CONFIRMED:
            if self._upright_held(st, signals, timestamp):
                st.state = self.UPRIGHT

        self._remember(st, pose, timestamp)
        st.last_seen, st.last_pose, st.last_signals = timestamp, pose, signals
        return event

    def is_down(self, track_id: int) -> bool:
        """Falling or on the ground (where losing the person must not lose the fall)."""
        return self.state_of(track_id) in (self.FALLING, self.FALLEN)

    def last_bbox(self, track_id: int):
        st = self.tracks.get(track_id)
        return None if st is None or st.last_pose is None else st.last_pose.bbox

    def check_missing(self, track_id: int, timestamp: float) -> FallEvent | None:
        """No pose this frame for a tracked person. If they vanished while falling or on the
        ground, near the floor, hold them at the last position as motionless for up to
        ``lost_hold_seconds`` so the fall can still be confirmed."""
        st = self.tracks.get(track_id)
        if (st is None or self.lost_hold_seconds <= 0 or st.last_seen is None
                or timestamp - st.last_seen > self.lost_hold_seconds):
            return None
        if not self._lying(st.last_signals or {}):
            return None  # last seen upright, crouching or bent over, not lying on the floor
        if st.state == self.FALLING:
            if not (st.last_signals or {}).get("head_dropped"):
                return None  # vanished mid-descent but not near the floor (e.g. left the frame)
            self._enter_fallen(st, track_id, timestamp, st.last_signals or {}, st.last_pose)
        if st.state != self.FALLEN:
            return None
        st.upright_since = None
        st.still_since = st.still_since if st.still_since is not None else timestamp
        if (timestamp - st.still_since >= self.stillness_seconds
                and timestamp - st.last_alert_time >= self.cooldown_seconds):
            st.state = self.CONFIRMED
            st.last_alert_time = timestamp
            signals = {**(st.last_signals or {}), "held_while_lost": True}
            event = self._make_event(track_id, timestamp, st, signals, st.last_pose)
            event.confidence = round(max(0.5, event.confidence - 0.1), 2)  # not seen at the moment
            return event
        return None

    def drain_possible(self) -> list[FallEvent]:
        """"Possible fall" events raised since the last call (stage "possible")."""
        events, self.possible_events = self.possible_events, []
        return events

    def _enter_fallen(self, st: _TrackState, track_id: int, timestamp: float, signals, pose) -> None:
        """FALLING -> FALLEN (the on-the-ground stage). Raises one "possible fall" event per
        fall (at most one per ``POSSIBLE_COOLDOWN_S`` per person); confirmation comes later."""
        st.state = self.FALLEN
        st.fallen_since = timestamp
        st.still_since = None
        if pose is not None and timestamp - st.last_possible_time >= self.POSSIBLE_COOLDOWN_S:
            st.last_possible_time = timestamp
            event = self._make_event(track_id, timestamp, st, signals, pose)
            event.stage = "possible"
            event.confidence = round(min(event.confidence, 0.7), 2)
            self.possible_events.append(event)

    def reset_track(self, track_id: int):
        self.tracks.pop(track_id, None)

    def prune(self, active_track_ids):
        """Forget people who are no longer tracked (keeps memory bounded)."""
        for tid in [t for t in self.tracks if t not in active_track_ids]:
            del self.tracks[tid]

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _remember(st: _TrackState, pose: PoseResult, timestamp: float):
        st.prev_hip_y = float(pose.mid_hip[1])
        st.prev_time = timestamp

    def _compute_signals(self, st: _TrackState, pose: PoseResult, body_h: float,
                         timestamp: float) -> dict[str, float]:
        hip_y = float(pose.mid_hip[1])
        descent_speed = 0.0
        if st.prev_hip_y is not None and st.prev_time is not None:
            dt = timestamp - st.prev_time
            if dt > 0:
                descent_speed = (hip_y - st.prev_hip_y) / dt / body_h  # + means moving down

        bbox_w = float(pose.bbox[2] - pose.bbox[0])
        bbox_h = float(pose.bbox[3] - pose.bbox[1])
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0

        # Lying = the torso is near horizontal. Without a visible torso, fall back to a wide
        # box. (A wide box alone also fits someone sitting close to a webcam, so when the
        # torso is visible it must be tilted too.)
        torso = getattr(pose, "torso_angle", None)
        wide = aspect_ratio > self.aspect_ratio_threshold
        horizontal = wide if torso is None else (
            torso > self.LYING_TORSO_DEG or (wide and torso > self.TILTED_TORSO_DEG))

        head_known = pose.head_valid and st.upright_head_y is not None
        head_drop = (pose.head_y - st.upright_head_y) / body_h if head_known else 0.0

        # Stillness over a short window, not frame to frame: keypoints jitter by a few
        # pixels per frame, which at 30 fps alone exceeds the speed threshold.
        st.hip_history.append((timestamp, hip_y))
        while st.hip_history and timestamp - st.hip_history[0][0] > self.STILLNESS_WINDOW_S:
            st.hip_history.popleft()
        span = st.hip_history[-1][0] - st.hip_history[0][0]
        if span >= 0.6 * self.STILLNESS_WINDOW_S:
            ys = [y for _, y in st.hip_history]
            is_still = (max(ys) - min(ys)) / span / body_h < self.stillness_speed_threshold
        else:
            is_still = abs(descent_speed) < self.stillness_speed_threshold

        # Direction-independent: hip height above the feet (visible ankles, else where the feet
        # were while standing), and the vertical spread of the visible keypoints.
        kp = pose.keypoints
        visible = kp[kp[:, 2] >= 0.3]
        ankles = kp[[15, 16]]
        ankles = ankles[ankles[:, 2] >= 0.3]
        hips = kp[[11, 12]]
        hips = hips[hips[:, 2] >= 0.3]
        if len(ankles) and not horizontal:
            foot_y = float(ankles[:, 1].mean())
            st.floor_y = foot_y if st.floor_y is None else 0.8 * st.floor_y + 0.2 * foot_y
        ref = float(ankles[:, 1].mean()) if len(ankles) else st.floor_y
        hip_height = (ref - float(hips[:, 1].mean())) / body_h if (len(hips) and ref is not None) else None
        spread = (float(visible[:, 1].max() - visible[:, 1].min()) / body_h) if len(visible) >= 5 else None

        return {
            "hip_height": hip_height,
            "spread": spread,
            "descent_speed": descent_speed,
            "aspect_ratio": aspect_ratio,
            "torso_angle": torso if torso is not None else -1.0,
            "horizontal_pose": horizontal,
            "head_known": head_known,
            "head_drop": head_drop,
            "head_dropped": head_known and head_drop > self.head_drop_ratio,
            "is_still": is_still,
        }

    def check_recovered(self, track_id: int, pose: PoseResult, features: TrackFeatures,
                        timestamp: float, mode: str = "pose") -> FallEvent | None:
        """A falling/fallen person the detector lost, found again by the recovery retries
        (anomaly/fall_recovery.py).

        ``mode="pose"`` (strict): the re-found pose goes through the normal check, like any
        tracked pose. It never confirms more than a real pose would, but those keypoints come
        from a crop, a lower threshold or a rotated image and jitter a lot, so stillness is
        rarely met.

        ``mode="presence"``: the re-found person is used as evidence that they are still there:
        - clearly upright (torso near vertical, held for ``upright_hold_seconds``): back to
          upright;
        - moved more than half a body height from where they were: not still;
        - otherwise: motionless in place, which counts toward confirmation like a hold.
        """
        if mode == "pose":
            return self.check(track_id, pose, features, timestamp)
        st = self.tracks.get(track_id)
        body_h = features.initial_standing_height or (st.box_scale if st is not None else 0.0)
        if st is None or st.state not in (self.FALLING, self.FALLEN) or body_h <= 0:
            return None
        if not self._lying(st.last_signals or {}):
            return None  # same rule as the hold: only someone last seen lying
        torso = pose.torso_angle
        upright = torso is not None and torso <= 30.0
        if self._upright_held(st, {"horizontal_pose": not upright, "head_dropped": False}, timestamp):
            st.state = self.UPRIGHT
            return None
        if upright:
            return None
        centre = np.array([(pose.bbox[0] + pose.bbox[2]) / 2, (pose.bbox[1] + pose.bbox[3]) / 2])
        if st.anchor is None:
            st.anchor = centre
        moved = float(np.hypot(*(centre - st.anchor))) / body_h > 0.5
        st.last_seen = timestamp
        if st.state == self.FALLING:
            self._enter_fallen(st, track_id, timestamp, st.last_signals or {}, st.last_pose or pose)
        if moved:
            st.anchor = centre
            st.still_since = None
            return None
        st.still_since = st.still_since if st.still_since is not None else timestamp
        if (timestamp - st.still_since >= self.stillness_seconds
                and timestamp - st.last_alert_time >= self.cooldown_seconds):
            st.state = self.CONFIRMED
            st.last_alert_time = timestamp
            signals = {**(st.last_signals or {}), "recovered": True}
            event = self._make_event(track_id, timestamp, st, signals, st.last_pose or pose)
            event.confidence = round(max(0.5, event.confidence - 0.1), 2)
            return event
        return None

    def _lying(self, signals) -> bool:
        """Last seen lying: torso within 30 degrees of horizontal (not bent over with the head
        below the hips), or, without a visible torso, a wide box."""
        torso = signals.get("torso_angle", -1.0)
        if torso is None or torso < 0:
            return bool(signals.get("horizontal_pose"))
        return self.LYING_TORSO_DEG <= torso <= 180.0 - self.LYING_TORSO_DEG

    # Direction-independent thresholds (fractions of the head-to-ankle height). Standing, the
    # hips are ~0.6 above the ankles and the keypoints span ~1.0; lying, both drop near 0.
    HIP_LOW = 0.25       # hips this close to the floor: down
    HIP_HIGH = 0.35      # hips at least this high: not on the ground (e.g. bending over)
    SPREAD_COLLAPSED = 0.5

    def _on_ground(self, signals) -> bool:
        torso_rule = signals["horizontal_pose"] and (signals["head_dropped"] or not signals["head_known"])
        if self.ground_mode != "combined":
            return torso_rule
        hip_h = signals.get("hip_height")
        spread = signals.get("spread")
        low = hip_h is not None and hip_h < self.HIP_LOW
        collapsed = spread is not None and spread < self.SPREAD_COLLAPSED
        high_hips = hip_h is not None and hip_h > self.HIP_HIGH
        return (torso_rule or (low and collapsed) or (collapsed and signals["head_dropped"])) and not high_hips

    def _box_scale(self, st: _TrackState, pose: PoseResult) -> float:
        """Head-to-ankle scale from upright boxes (taller than 1.6x their width) when the
        skeleton can't be measured: median of 10 samples, x0.9 (the box includes the crown and
        feet)."""
        if st.box_scale > 0:
            return st.box_scale
        w = float(pose.bbox[2] - pose.bbox[0])
        h = float(pose.bbox[3] - pose.bbox[1])
        if w > 0 and h / w >= 1.6:
            st.box_samples.append(0.9 * h)
            if len(st.box_samples) >= 10:
                st.box_scale = float(np.median(st.box_samples))
        return st.box_scale

    def _stood_up(self, signals) -> bool:
        up = not signals["horizontal_pose"] and not signals["head_dropped"]
        if self.ground_mode == "combined" and signals.get("hip_height") is not None:
            up = up and signals["hip_height"] >= self.HIP_LOW
        return up

    def _upright_held(self, st: _TrackState, signals, timestamp: float) -> bool:
        """Upright now, and (with hysteresis) for at least ``upright_hold_seconds``."""
        if not self._stood_up(signals):
            st.upright_since = None
            return False
        if st.upright_since is None:
            st.upright_since = timestamp
        if timestamp - st.upright_since >= self.upright_hold_seconds:
            st.upright_since = None
            return True
        return False

    def _make_event(self, track_id: int, timestamp: float, st: _TrackState, signals,
                    pose: PoseResult) -> FallEvent:
        # Confidence grows with how many independent signals agreed.
        agreeing = 1 + int(signals["head_dropped"]) + int(st.peak_descent > 2 * self.descent_speed_threshold)
        confidence = round(min(0.99, 0.7 + 0.1 * agreeing), 2)
        return FallEvent(
            track_id=track_id,
            timestamp=timestamp,
            confidence=confidence,
            stage=self.CONFIRMED,
            signals={k: (v if v is None or isinstance(v, bool) else round(float(v), 3)) for k, v in signals.items()},
            head_y=float(pose.head_y),
            hip_y=float(pose.mid_hip[1]),
            velocity=round(float(st.peak_descent), 3),
            aspect_ratio=round(float(signals["aspect_ratio"]), 3),
        )
