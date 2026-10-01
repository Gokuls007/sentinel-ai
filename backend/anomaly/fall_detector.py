"""Rule-based fall detection: a per-person state machine over pose signals.

    UPRIGHT --rapid descent--> FALLING --lying / head low--> FALLEN
    FALLEN --still for stillness_seconds--> CONFIRMED  (one alert is raised here)
    FALLING/FALLEN --stands up or times out--> UPRIGHT
    CONFIRMED --stands up--> UPRIGHT

All speeds are normalised by the person's calibrated standing height and by real
elapsed time, so thresholds work regardless of resolution, camera distance or FPS.
"""

from collections import deque
from dataclasses import dataclass, field

from core.pose_estimator import PoseResult, TrackFeatures


@dataclass
class FallEvent:
    track_id: int
    timestamp: float
    confidence: float
    stage: str  # "confirmed"
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
    hip_history: deque = field(default_factory=deque)  # (timestamp, hip_y) for stillness


class FallDetector:
    UPRIGHT = "upright"
    FALLING = "falling"
    FALLEN = "fallen"
    CONFIRMED = "confirmed"

    FALLING_WINDOW_S = 1.5  # a fall must reach the ground within this time
    LYING_TORSO_DEG = 60.0  # torso this far from vertical = lying
    TILTED_TORSO_DEG = 35.0  # a wide box counts as lying only with the torso at least this tilted
    STILLNESS_WINDOW_S = 0.5

    def __init__(self, descent_speed_threshold: float = 1.2, aspect_ratio_threshold: float = 1.2,
                 head_drop_ratio: float = 0.5, stillness_seconds: float = 1.0,
                 stillness_speed_threshold: float = 0.15, fallen_timeout_seconds: float = 5.0,
                 cooldown_seconds: float = 30.0):
        self.descent_speed_threshold = descent_speed_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.head_drop_ratio = head_drop_ratio
        self.stillness_seconds = stillness_seconds
        self.stillness_speed_threshold = stillness_speed_threshold
        self.fallen_timeout_seconds = fallen_timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self.tracks: dict[int, _TrackState] = {}

    # -- public API ---------------------------------------------------------------------

    def state_of(self, track_id: int) -> str:
        st = self.tracks.get(track_id)
        return st.state if st else self.UPRIGHT

    def check(self, track_id: int, pose: PoseResult, features: TrackFeatures,
              timestamp: float) -> FallEvent | None:
        """Advance the state machine for one person; returns an event on a confirmed fall."""
        st = self.tracks.setdefault(track_id, _TrackState())
        body_h = features.initial_standing_height
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

        elif st.state == self.FALLING:
            st.peak_descent = max(st.peak_descent, signals["descent_speed"])
            on_ground = signals["horizontal_pose"] and (
                signals["head_dropped"] or not signals["head_known"])
            if on_ground:
                st.state = self.FALLEN
                st.fallen_since = timestamp
                st.still_since = None
            elif timestamp - st.falling_since > self.FALLING_WINDOW_S:
                st.state = self.UPRIGHT  # e.g. sat down or crouched quickly

        elif st.state == self.FALLEN:
            if self._stood_up(signals):
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
            if self._stood_up(signals):
                st.state = self.UPRIGHT

        self._remember(st, pose, timestamp)
        return event

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

        return {
            "descent_speed": descent_speed,
            "aspect_ratio": aspect_ratio,
            "torso_angle": torso if torso is not None else -1.0,
            "horizontal_pose": horizontal,
            "head_known": head_known,
            "head_drop": head_drop,
            "head_dropped": head_known and head_drop > self.head_drop_ratio,
            "is_still": is_still,
        }

    @staticmethod
    def _stood_up(signals) -> bool:
        return not signals["horizontal_pose"] and not signals["head_dropped"]

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
            signals={k: (round(float(v), 3) if not isinstance(v, bool) else v) for k, v in signals.items()},
            head_y=float(pose.head_y),
            hip_y=float(pose.mid_hip[1]),
            velocity=round(float(st.peak_descent), 3),
            aspect_ratio=round(float(signals["aspect_ratio"]), 3),
        )
