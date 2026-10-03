"""Desk posture coach: compares a seated person, seen from the front by a laptop webcam, with
their own upright baseline.

REBA isn't used here: a frontal webcam can't measure trunk or neck flexion reliably (see
ergonomics/). Every posture measurement is **relative to your own shoulders**, never to where you
sit in the frame, so moving your chair or sitting off-centre isn't a posture problem:

- head height: nose above the shoulder line / shoulder width. It shrinks when the head drops
  or juts forward (slouching).
- face-to-shoulder ratio: eye span / shoulder width. It grows when the shoulders roll in
  (hunching).
- head offset: ear midpoint (else eye midpoint; never the nose, which swings sideways whenever
  you turn your head) minus shoulder midpoint, / shoulder width. With shoulder tilt, it shows
  leaning.
- face size vs baseline: sitting too close to the screen (distance is the point here).

If your whole body has moved a lot since the baseline (shoulders shifted by more than a shoulder
width, or shoulder width changed by over 25% without your face growing to match), no posture is
judged; a "You've moved" hint suggests resetting the baseline instead.

Per-frame values are smoothed (median over ``smooth_s``), and the displayed status changes only
after a new status has held for ``switch_s``, so it doesn't flicker. Everything runs locally;
nothing is sent anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from posture.breaks import SAFETY_NOTE, BreakConfig, BreakRoutine
from posture.classifier import (
    POSTURE_LABELS,
    POSTURES,
    RAW_FEATURES,
    PostureModel,
    StableStatus,
    evaluate,
    head_turn,
    raw_features,
    train,
)
from posture.guidance import (
    GHOST_EDGES,
    CorrectionTracker,
    median_skeleton,
    normalise,
    place_ghost,
    to_json,
)
from posture.history import PostureHistory
from posture.holds import HEAD_DOWN, INSTRUCTIONS, LEAN, HoldConfig, HoldTracker, extremes
from posture.movement import AWAY as MOVE_AWAY
from posture.movement import MovementConfig, MovementTracker

logger = logging.getLogger("sentinel.posture")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6

GOOD, SLOUCHING, LEANING, TOO_CLOSE = "good", "slouching", "leaning", "too_close"
AWAY, NO_BASELINE, CALIBRATING, MOVED = "away", "no_baseline", "calibrating", "moved"
UNCLEAR, SLUMPED, LOOKING_DOWN, CHECKING = "unclear", "slumped", "looking_down", "checking"
LOOKING_AWAY, NOT_SURE = "looking_away", "not_sure"
BAD = (SLOUCHING, LEANING, TOO_CLOSE, SLUMPED)
# Counted as neither good nor poor posture. Looking down is reminded about if held a long time.
NEUTRAL = (LOOKING_DOWN, LOOKING_AWAY, NOT_SURE)
LABELS = {GOOD: "Good", SLOUCHING: "Slouching", LEANING: "Leaning", TOO_CLOSE: "Too close", SLUMPED: "Slumped",
          LOOKING_DOWN: "Looking down",
          AWAY: "Away", NO_BASELINE: "Press Set baseline to start", CALIBRATING: "Hold still...",
          MOVED: "You've moved", UNCLEAR: "Can't see your shoulders", CHECKING: "Checking your posture...",
          LOOKING_AWAY: "Looking away", NOT_SURE: "Not sure"}


@dataclass
class PostureConfig:
    min_kp_conf: float = 0.4
    baseline_delay_s: float = 3.0   # "get ready" countdown after pressing Set baseline
    baseline_seconds: float = 3.0   # then recording
    baseline_min_frames: int = 10
    # A baseline is rejected if you moved while it recorded (spread of the measurements) or it
    # doesn't look like sitting upright.
    baseline_max_std_head: float = 0.06
    baseline_max_std_lateral: float = 0.08
    baseline_max_std_tilt: float = 4.0
    baseline_max_abs_tilt: float = 12.0
    baseline_min_head_ratio: float = 0.2
    smooth_s: float = 1.0          # median window for per-frame measurements
    switch_s: float = 2.0          # a new status must hold this long before it is shown
    away_after_s: float = 2.0      # no usable pose this long: "away"
    # Deviations from baseline that count as poor posture.
    head_drop_ratio: float = 0.80  # head height falls below 80% of baseline
    hunch_ratio: float = 1.15      # face/shoulder ratio grows 15%
    tilt_deg: float = 8.0          # shoulder line tilts 8 degrees more than baseline
    lateral_shift: float = 0.30    # head offset changes by 30% of shoulder width
    too_close_ratio: float = 1.25  # face 25% bigger than at baseline
    # "You've moved": shoulder midpoint shifted by more than this many baseline shoulder widths,
    # or shoulder width changed by more than this fraction (unless the face grew to match:
    # that's sitting too close, which *is* judged).
    moved_shift_widths: float = 1.0
    moved_width_change: float = 0.25
    # Body turned: shoulders this much narrower relative to the face than at baseline
    # (face/shoulder ratio up 40%). Then hunch is ignored and the head-offset limit doubles.
    rotation_face_ratio: float = 1.4
    # Shoulder quality. Frames with a shoulder keypoint below this confidence, or a shoulder
    # width outside this range of eye spans (inter-eye distance; ~6 for adults, 6.4-7.1 on the
    # first real webcam session), are not used. Not checked per side: leaning moves the head
    # over one shoulder.
    shoulder_min_conf: float = 0.5
    shoulder_width_eyes: tuple = (3.0, 9.5)
    unclear_fraction: float = 0.5   # this share of recent frames unclear: show the lighting hint
    # Guided calibration / test: countdown, recording length and minimum usable frames per posture.
    calibration_get_ready_s: float = 3.0
    calibration_record_s: float = 20.0
    calibration_min_frames: int = 30
    # Personal classifier: checks before it. Head yaw beyond this (eye spans, vs your Good
    # recording; 0.5 = nose past one eye) is "looking away"; held this long it counts as away.
    looking_away_yaw: float = 0.5
    looking_away_to_away_s: float = 120.0
    # Frames unlike any calibration frame (out of distribution) for this long: "not sure".
    not_sure_after_s: float = 3.0
    # Quick "back to good" check while the ghost is shown (classifier: this frame's Good probability).
    quick_good_prob: float = 0.6
    timeline_bucket_s: float = 10.0


@dataclass
class Measurement:
    shoulder_width: float
    head_ratio: float              # (shoulder line y - nose y) / shoulder width
    tilt_deg: float                # shoulder line angle from horizontal (signed)
    lateral: float | None          # (ear midpoint, else eye midpoint, x - shoulder midpoint x) / shoulder width
    face_size: float | None        # eye span in px
    face_ratio: float | None       # eye span / shoulder width
    mid_x: float | None = None     # shoulder midpoint in the frame (px): only for "you've moved"
    mid_y: float | None = None
    head_ref: str | None = None    # "ears" | "eyes": what the head offset was measured from


@dataclass
class Baseline:
    shoulder_width: float
    head_ratio: float
    tilt_deg: float
    lateral: float | None
    face_size: float | None
    face_ratio: float | None
    recorded_at: float = 0.0
    frames: int = 0
    mid_x: float | None = None
    mid_y: float | None = None
    yaw: float | None = None  # head yaw while recording (the looking-away reference); None = 0
    skeleton: dict | None = None  # median face + shoulder points (the fix-guidance ghost)

    @classmethod
    def from_dict(cls, data: dict) -> Baseline:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _visible(kp: np.ndarray, *idx: int, conf: float) -> bool:
    return all(kp[i, 2] >= conf for i in idx)


def measure(keypoints: np.ndarray, min_conf: float = 0.4) -> Measurement | None:
    """Posture measurements from 17 COCO keypoints (x, y, conf), or None if the shoulders and
    nose aren't all visible."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or not _visible(kp, NOSE, L_SH, R_SH, conf=min_conf):
        return None
    ls, rs, nose = kp[L_SH, :2], kp[R_SH, :2], kp[NOSE, :2]
    width = float(np.hypot(*(ls - rs)))
    if width < 10:
        return None
    mid = (ls + rs) / 2
    # Image left/right: order the shoulders by x so the sign doesn't depend on mirroring.
    a, b = (ls, rs) if ls[0] <= rs[0] else (rs, ls)
    tilt = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    face = None
    if _visible(kp, L_EYE, R_EYE, conf=min_conf):
        face = float(np.hypot(*(kp[L_EYE, :2] - kp[R_EYE, :2])))
    # Head offset from the ear midpoint (barely moves when you turn your head), else the eye
    # midpoint. Never the nose.
    head_x, head_ref = None, None
    if _visible(kp, L_EAR, R_EAR, conf=min_conf):
        head_x, head_ref = float((kp[L_EAR, 0] + kp[R_EAR, 0]) / 2), "ears"
    elif _visible(kp, L_EYE, R_EYE, conf=min_conf):
        head_x, head_ref = float((kp[L_EYE, 0] + kp[R_EYE, 0]) / 2), "eyes"
    return Measurement(
        shoulder_width=width,
        head_ratio=float(mid[1] - nose[1]) / width,
        tilt_deg=tilt,
        lateral=(head_x - float(mid[0])) / width if head_x is not None else None,
        face_size=face,
        face_ratio=face / width if face else None,
        mid_x=float(mid[0]),
        mid_y=float(mid[1]),
        head_ref=head_ref,
    )


UNCLEAR_MESSAGE = "Can't see your shoulders clearly. Try better lighting or a lighter background."


def shoulder_problem(keypoints: np.ndarray, cfg: PostureConfig) -> str | None:
    """Why this frame's shoulders can't be trusted, or None. Dark clothes against a dark chair
    put a shoulder keypoint on the arm or the chair: its confidence drops, or the shoulder width
    stops matching the face size (judged in eye spans, a scale that doesn't depend on where you
    sit)."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or kp[NOSE, 2] < cfg.min_kp_conf:
        return None  # no face: that's "away", not unclear shoulders
    if min(kp[L_SH, 2], kp[R_SH, 2]) < cfg.shoulder_min_conf:
        return "low_confidence"
    if not _visible(kp, L_EYE, R_EYE, conf=cfg.min_kp_conf):
        return None  # can't check proportions without the eyes
    eye_span = float(np.hypot(*(kp[L_EYE, :2] - kp[R_EYE, :2])))
    if eye_span < 3:
        return None
    width = float(np.hypot(*(kp[L_SH, :2] - kp[R_SH, :2]))) / eye_span
    if not cfg.shoulder_width_eyes[0] <= width <= cfg.shoulder_width_eyes[1]:
        return "implausible"
    return None


def select_main_person(poses: dict, frame_width: float, prev_id=None, keep_ratio: float = 1.5):
    """The person the coach is for: the largest face-and-shoulder area, favouring the middle of
    the frame. The current person is kept unless someone else is clearly more prominent
    (``keep_ratio``), so the coach doesn't jump between people. ``poses`` maps id -> PoseResult."""
    scores = {}
    for pid, pose in poses.items():
        kp = np.asarray(pose.keypoints, float)
        upper = kp[:7][kp[:7, 2] >= 0.3]
        if len(upper) < 3:
            continue
        w = float(upper[:, 0].max() - upper[:, 0].min())
        h = float(upper[:, 1].max() - upper[:, 1].min())
        cx = float(upper[:, 0].mean())
        centrality = 1.0 - min(1.0, abs(cx - frame_width / 2) / (frame_width / 2)) if frame_width else 1.0
        scores[pid] = w * h * (0.5 + 0.5 * centrality)
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if prev_id in scores and scores[prev_id] * keep_ratio >= scores[best]:
        return prev_id
    return best


def _ratio(a, b):
    return a / b if a is not None and b else None


def explain(m: Measurement, base: Baseline, cfg: PostureConfig) -> list[dict]:
    """Every measurement next to its baseline and limit (the debug panel and ``assess``).

    Shoulder tilt (an angle) and head offset / head height / face-to-shoulder (ratios of
    shoulder width) don't depend on how far you sit, so moving never switches them off. Only
    "too close" depends on distance, and it is not judged while the body has moved.

    Turning your body narrows the shoulders on screen, which inflates head offset and
    face-to-shoulder. When the shoulders look much narrower relative to the face than at
    baseline (``rotation_face_ratio``), the hunch check is ignored and the head-offset limit is
    doubled; shoulder tilt is still fully trusted.
    """
    rows = []

    def row(key, label, now, baseline, change, limit, triggered, status, note=""):
        rows.append({"key": key, "label": label, "now": now, "baseline": baseline, "change": change,
                     "limit": limit, "triggered": bool(triggered), "status": status, "note": note})

    face_change = _ratio(m.face_size, base.face_size)
    width_change = _ratio(m.shoulder_width, base.shoulder_width)
    head_change = _ratio(m.head_ratio, base.head_ratio)
    hunch_change = _ratio(m.face_ratio, base.face_ratio)
    tilt_change = m.tilt_deg - base.tilt_deg
    offset_change = m.lateral - base.lateral if m.lateral is not None and base.lateral is not None else None
    shift = None
    if None not in (m.mid_x, m.mid_y, base.mid_x, base.mid_y):
        shift = math.hypot(m.mid_x - base.mid_x, m.mid_y - base.mid_y) / base.shoulder_width

    rotated = hunch_change is not None and hunch_change >= cfg.rotation_face_ratio
    offset_limit = cfg.lateral_shift * (2.0 if rotated else 1.0)
    grew = (face_change is not None and face_change > cfg.too_close_ratio) or (
        face_change is None and width_change is not None and width_change > cfg.too_close_ratio)
    moved_width = width_change is not None and abs(width_change - 1) > cfg.moved_width_change and not grew
    moved = (shift is not None and shift > cfg.moved_shift_widths) or moved_width

    row("face_size", "Face size vs baseline", m.face_size, base.face_size, face_change,
        f"> {cfg.too_close_ratio:.2f}x", grew and not moved, TOO_CLOSE,
        "not judged after moving" if moved else "")
    row("head_ratio", "Head height (/ shoulder width)", m.head_ratio, base.head_ratio, head_change,
        f"< {cfg.head_drop_ratio:.2f}x", head_change is not None and head_change < cfg.head_drop_ratio, SLOUCHING)
    row("face_ratio", "Face / shoulder width", m.face_ratio, base.face_ratio, hunch_change,
        f"> {cfg.hunch_ratio:.2f}x",
        hunch_change is not None and hunch_change > cfg.hunch_ratio and not rotated, SLOUCHING,
        "ignored: body turned" if rotated else "")
    row("tilt_deg", "Shoulder tilt (deg)", m.tilt_deg, base.tilt_deg, tilt_change,
        f"> {cfg.tilt_deg:g} deg", abs(tilt_change) > cfg.tilt_deg, LEANING)
    row("lateral", f"Head offset ({m.head_ref or 'n/a'}, / shoulder width)", m.lateral, base.lateral,
        offset_change, f"> {offset_limit:.2f}",
        offset_change is not None and abs(offset_change) > offset_limit, LEANING,
        "limit doubled: body turned" if rotated else "")
    row("moved", "Body moved (shoulder shift / width change)", shift, None, width_change,
        f"> {cfg.moved_shift_widths:g} widths or {cfg.moved_width_change:.0%}", moved, MOVED)
    return rows


MOVED_HINT = "You've moved since your baseline. Reset baseline?"


def assess(m: Measurement, base: Baseline, cfg: PostureConfig) -> dict:
    """Status, reasons and an optional hint, from (smoothed) measurements vs the baseline.

    Posture comes first: if any posture measure is over its limit, that's the status ("Slumped"
    when both a lean measure and a slouch measure fire), and "you've moved" is only a hint next
    to it. "You've moved" is the status on its own only when the posture measures look normal.
    """
    rows = {r["key"]: r for r in explain(m, base, cfg)}
    moved = rows["moved"]["triggered"]
    lean: list[str] = []
    slouch: list[str] = []
    if rows["tilt_deg"]["triggered"]:
        lean.append(f"Shoulders tilted {abs(rows['tilt_deg']['change']):.0f}° more than at baseline: "
                    "level your shoulders")
    if rows["lateral"]["triggered"]:
        side = "right" if rows["lateral"]["change"] > 0 else "left"  # image side
        lean.append(f"Head leaning to the {side} of your shoulders: bring it back over them")
    if rows["head_ratio"]["triggered"]:
        slouch.append("Head has dropped toward the shoulders: sit tall, chin back")
    if rows["face_ratio"]["triggered"]:
        slouch.append("Shoulders are hunched forward: roll them back and down")
    hint = MOVED_HINT if moved else None

    if lean and slouch:
        return {"status": SLUMPED, "reasons": slouch + lean, "hint": hint}
    if slouch:
        return {"status": SLOUCHING, "reasons": slouch, "hint": hint}
    if lean:
        return {"status": LEANING, "reasons": lean, "hint": hint}
    if moved:
        return {"status": MOVED, "reasons": [MOVED_HINT], "hint": None}
    if rows["face_size"]["triggered"]:
        change = rows["face_size"]["change"]
        return {"status": TOO_CLOSE, "hint": None,
                "reasons": [f"Face {change:.0%} of its baseline size: move back from the screen"
                            if change else "Much closer than at baseline: move back from the screen"]}
    return {"status": GOOD, "reasons": [], "hint": None}


def classify(m: Measurement, base: Baseline, cfg: PostureConfig) -> tuple[str, list[str]]:
    """(status, reasons); see ``assess``."""
    a = assess(m, base, cfg)
    return a["status"], a["reasons"]


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    seconds: dict = field(default_factory=lambda: {s: 0.0 for s in (GOOD, SLOUCHING, LEANING, TOO_CLOSE, SLUMPED,
                                                                     LOOKING_DOWN, LOOKING_AWAY, NOT_SURE, AWAY, MOVED,
                                                                     UNCLEAR)})
    timeline: list = field(default_factory=list)  # [{"t": offset_s, "status": ...}] one per bucket
    reminders: int = 0


@dataclass
class CoachSettings:
    """What the person can change on the coach page (saved with the history)."""
    reminder_min: float = 30.0
    head_down_enabled: bool = True
    head_down_min: float = 20.0
    lean_enabled: bool = True
    lean_min: float = 20.0
    # Shortened timings for testing and recording a demo: reminder after 1 min, long holds 2 min.
    demo_timings: bool = False

    DEMO_REMINDER_S = 60.0
    DEMO_HOLD_S = 120.0


class PostureCoach:
    def __init__(self, cfg: PostureConfig | None = None, baseline_path: str | None = None,
                 clock=time.time):
        self.cfg = cfg or PostureConfig()
        self.baseline_path = baseline_path
        self.clock = clock
        self.baseline: Baseline | None = self._load()
        self.session = Session(started_at=clock())
        self._window: deque[tuple[float, Measurement]] = deque()
        self._calib: list[Measurement] | None = None
        self._calib_start = 0.0
        self._calib_until = 0.0
        self._calib_error: str | None = None
        self.status = NO_BASELINE if self.baseline is None else AWAY
        self.reasons: list[str] = []
        self._candidate = self.status
        self._candidate_since = 0.0
        self._status_since = 0.0
        self.episode = 0  # increments every time the shown status changes
        self._last_ts: float | None = None
        self._last_seen = -1e9
        self._bucket: dict[str, float] = {}
        self._bucket_start: float | None = None
        self._origin: float | None = None  # first frame time of the session (timeline offsets)
        self.last: Measurement | None = None
        self.smoothed: Measurement | None = None  # what the last classification used
        self.last_problem: str | None = None  # this frame's shoulder problem, if any
        self.hint: str | None = None  # shown next to the status, e.g. "you've moved, reset baseline?"
        self._quality: deque[tuple[float, bool]] = deque()  # (ts, shoulders unclear) recent frames
        self._calib_unclear = 0
        self._calib_yaw: list[float] = []
        # Personal classifier (trained from the guided calibration); thresholds are the fallback.
        stem = os.path.splitext(baseline_path)[0].replace("posture_baseline", "posture_{}") if baseline_path else None
        self.model_path = stem.format("model") + ".pkl" if stem else None
        self.calibration_path = stem.format("calibration") + ".json" if stem else None
        self.model: PostureModel | None = PostureModel.load(self.model_path) if self.model_path else None
        self.stable = StableStatus()
        self.probs: dict = {}
        self.rec: dict | None = None  # an active calibration step or test run
        self.last_test: dict | None = None
        self.rec_error: str | None = None
        self.last_raw: dict | None = None  # the classifier's inputs for the last usable frame
        self.yaw: float | None = None  # head yaw vs the Good recording (classifier mode)
        self.turn_reason: str | None = None
        self.ood: float | None = None  # distance to the calibration data (classifier mode)
        self._away_look_since: float | None = None  # start of the current looking-away stretch
        self._ood_since: float | None = None  # start of the current run of unfamiliar frames
        # Coaching: fix guidance (ghost + quick "back to good"), local history, stretch breaks.
        self.guide = CorrectionTracker()
        self.instruction: str | None = None
        self.lean_side: str | None = None  # "left" | "right" (your side) while leaning, if known
        self._kp = None  # this frame's keypoints (the ghost is placed on them)
        self._frame_good: float | None = None  # this frame's Good probability (classifier)
        self._calib_skel: list = []
        self.history: PostureHistory | None = None
        if stem:
            os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
            self.history = PostureHistory(stem.format("history") + ".db", clock=clock)
        # The movement coach: time since you last moved, breaks, static time, the movement
        # reminder (which offers the stretch break), and long-hold warnings for sustained extremes.
        self.break_cfg = BreakConfig()
        self.movement = MovementTracker(MovementConfig(), history=self.history, now=clock())
        self.holds = HoldTracker(HoldConfig())
        self._hold_window: deque = deque()  # (ts, head_ratio, tilt, lateral) for the classifier path
        # Shared by every camera's coach (Settings can change them while the webcam is off).
        self.settings_path = (os.path.join(os.path.dirname(baseline_path) or ".", "coach_settings.json")
                              if baseline_path else None)
        self.settings = self._load_settings()
        self._apply_settings()
        self.routine: BreakRoutine | None = None
        self.break_result: dict | None = None
        self.ghost_good = self._calibration_ghost()
        self.calibration_counts = {p: len(v) for p, v in self.recordings().items()}
        if self.model is not None and self.status == NO_BASELINE:
            self.status = AWAY

    # --- calibration and test recordings ----------------------------------------------------

    def recordings(self) -> dict:
        if not self.calibration_path or not os.path.isfile(self.calibration_path):
            return {}
        try:
            with open(self.calibration_path, encoding="utf-8") as f:
                return json.load(f).get("postures", {})
        except (OSError, ValueError):
            return {}

    def _save_recordings(self, data: dict) -> None:
        if not self.calibration_path:
            return
        os.makedirs(os.path.dirname(self.calibration_path) or ".", exist_ok=True)
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump({"postures": data, "features": list(RAW_FEATURES)}, f)
        self.calibration_counts = {p: len(v) for p, v in data.items()}
        self.ghost_good = self._calibration_ghost(data)

    def _calibration_ghost(self, data: dict | None = None) -> dict | None:
        """Your Good posture from the calibration (None if recorded before the ghost existed)."""
        frames = (data if data is not None else self.recordings()).get("good", [])
        return median_skeleton([f.get("skeleton") for f in frames])

    def start_recording(self, kind: str, postures: list[str], now: float | None = None,
                        get_ready_s: float | None = None, record_s: float | None = None) -> None:
        """Record ``postures`` in order. kind "calibrate" saves each one as calibration data;
        kind "test" scores the saved model on them at the end."""
        unknown = [p for p in postures if p not in POSTURES]
        if unknown:
            raise ValueError(f"unknown posture(s): {unknown}")
        if kind == "test" and self.model is None:
            raise ValueError("train the model first")
        if now is None:
            now = self._last_ts if self._last_ts is not None else self.clock()
        self.rec = {
            "kind": kind, "postures": list(postures), "step": 0, "frames": [], "labelled": [],
            "get_ready_s": self.cfg.calibration_get_ready_s if get_ready_s is None else get_ready_s,
            "record_s": self.cfg.calibration_record_s if record_s is None else record_s,
        }
        self.rec_error = None
        self._begin_step(now)

    def cancel_recording(self) -> None:
        self.rec = None

    def _begin_step(self, now: float) -> None:
        r = self.rec
        r["record_from"] = now + r["get_ready_s"]
        r["record_until"] = r["record_from"] + r["record_s"]
        r["frames"] = []
        r["skipped"] = 0

    def _record(self, kp, problem, ts: float) -> None:
        r = self.rec
        posture = r["postures"][r["step"]]
        if ts >= r["record_from"]:
            raw = raw_features(kp, self.cfg.min_kp_conf) if kp is not None and problem is None else None
            if raw is not None:
                frame = {"t": round(ts, 3), "features": raw}
                if posture == "good" and r["kind"] == "calibrate":
                    frame["skeleton"] = to_json(normalise(kp, self.cfg.min_kp_conf))
                r["frames"].append(frame)
            else:
                r["skipped"] += 1
        if ts < r["record_until"]:
            return
        # Step finished.
        if r["kind"] == "calibrate":
            if len(r["frames"]) < self.cfg.calibration_min_frames:
                self.rec_error = (f"Only {len(r['frames'])} usable frames for {POSTURE_LABELS[posture]}: keep "
                                  "your face and both shoulders in view (better light helps), then redo it.")
            else:
                data = self.recordings()
                data[posture] = r["frames"]
                self._save_recordings(data)
        else:
            r["labelled"] += [(posture, f["features"]) for f in r["frames"]]
        r["step"] += 1
        if r["step"] < len(r["postures"]):
            self._begin_step(ts)
            return
        if r["kind"] == "test":
            if r["labelled"]:
                self.last_test = {**evaluate(self.model, r["labelled"]), "postures": r["postures"],
                                  "tested_at": self.clock()}
                self.model.test_report = self.last_test
                if self.model_path:
                    self.model.save(self.model_path)
            else:
                self.rec_error = "No usable frames in the test: keep your face and shoulders in view."
        self.rec = None

    def train_model(self) -> dict:
        """Train the personal classifier on the saved calibration and start using it."""
        data = self.recordings()
        self.calibration_counts = {p: len(v) for p, v in data.items()}
        model = train(data, min_frames=self.cfg.calibration_min_frames)
        self.rec_error = None
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            model.save(self.model_path)
        self.model = model
        self.last_test = None
        self.stable.reset()
        self._set_status(CHECKING, ["Learning how you're sitting right now (a few seconds)"],
                         self._last_ts if self._last_ts is not None else self.clock())
        return model.report

    def delete_model(self) -> None:
        self.model = None
        self.stable.reset()
        if self.model_path and os.path.isfile(self.model_path):
            os.remove(self.model_path)

    def clear_calibration(self) -> None:
        """Delete the recorded calibration (the trained model, if any, is kept)."""
        if self.calibration_path and os.path.isfile(self.calibration_path):
            os.remove(self.calibration_path)
        self.calibration_counts = {}
        self.ghost_good = None

    def _classify_with_model(self, kp, ts: float) -> None:
        raw = raw_features(kp, self.cfg.min_kp_conf) if kp is not None else None
        if raw is None:
            return
        self.last_raw = raw
        self.ood = self.model.ood_distance(raw)
        if self.ood is not None and self.ood > self.model.ood_threshold:
            # Unlike anything recorded in calibration: don't let the model guess. Short runs are
            # ignored (the previous status holds); a sustained one shows "not sure".
            if self._ood_since is None:
                self._ood_since = ts
            if ts - self._ood_since >= self.cfg.not_sure_after_s:
                self.hint = None
                self._set_status(NOT_SURE, ["This doesn't look like any posture you calibrated. If it's how you "
                                            "often sit, record it again (closest posture) and retrain."], ts)
            return
        self._ood_since = None
        frame_probs = self.model.predict_proba(raw)
        self._frame_good = frame_probs.get("good")
        current = self.stable.update(frame_probs, ts)
        self.probs = dict(self.stable.mean)
        if current is None:  # not confident in anything yet
            self._set_status(CHECKING, ["Learning how you're sitting right now (a few seconds)"], ts)
            return
        status = {"leaning_left": LEANING, "leaning_right": LEANING}.get(current, current)
        self.lean_side = {"leaning_left": "left", "leaning_right": "right"}.get(current)
        reasons = {
            "slouching": ["Sitting the way you slouched in calibration: sit tall, chin back"],
            "leaning_left": ["Leaning to your left: centre yourself over your hips"],
            "leaning_right": ["Leaning to your right: centre yourself over your hips"],
            "too_close": ["Closer to the screen than your good posture: move back"],
            "looking_down": ["Looking down (keyboard or phone): fine for a while, look up now and then"],
        }.get(current, [])
        self.hint = None
        self._set_status(status, reasons, ts)

    # --- baseline -------------------------------------------------------------------------

    def start_baseline(self, now: float | None = None, delay: float | None = None) -> None:
        """Count down ``baseline_delay_s`` (time to sit back after clicking), then record for
        ``baseline_seconds``."""
        now = self.clock() if now is None else now
        delay = self.cfg.baseline_delay_s if delay is None else delay
        self._calib = []
        self._calib_yaw = []
        self._calib_skel = []
        self._calib_unclear = 0
        self._calib_start = now + delay
        self._calib_until = self._calib_start + self.cfg.baseline_seconds
        self._calib_error = None

    def baseline_problem(self, samples: list[Measurement]) -> str | None:
        """Why these samples don't make a usable upright baseline, or None if they do."""
        cfg = self.cfg
        if len(samples) < cfg.baseline_min_frames:
            return ("Couldn't see your face and shoulders clearly enough. Sit facing the camera with both "
                    "shoulders in view, then try again.")
        head = np.array([s.head_ratio for s in samples])
        lateral = np.array([s.lateral for s in samples if s.lateral is not None] or [0.0])
        tilt = np.array([s.tilt_deg for s in samples])
        if (head.std() > cfg.baseline_max_std_head or lateral.std() > cfg.baseline_max_std_lateral
                or tilt.std() > cfg.baseline_max_std_tilt):
            return "You moved while the baseline was recording. Sit upright, keep still, and try again."
        if abs(float(np.median(tilt))) > cfg.baseline_max_abs_tilt:
            return ("Your shoulders looked tilted, so this doesn't look like an upright baseline. Sit level "
                    "and try again.")
        if float(np.median(head)) < cfg.baseline_min_head_ratio:
            return ("Your head looked very low (slouched or looking down). Sit tall, look at the screen, "
                    "and try again.")
        return None

    def _finish_baseline(self) -> None:
        samples, self._calib = self._calib or [], None
        unclear, self._calib_unclear = self._calib_unclear, 0
        problem = self.baseline_problem(samples)
        if unclear >= max(1, len(samples)):  # at least half the frames had unusable shoulders
            problem = UNCLEAR_MESSAGE
        yaws, self._calib_yaw = self._calib_yaw, []
        if not problem and yaws and abs(float(np.median(yaws))) > self.cfg.looking_away_yaw:
            problem = "Your head looked turned away. Face the screen and try again."
        if problem:
            self._calib_error = problem
            logger.info("posture baseline rejected (%d frames): %s", len(samples), problem)
            return

        def med(values):
            values = [v for v in values if v is not None]
            return float(np.median(values)) if values else None

        self.baseline = Baseline(
            shoulder_width=med([s.shoulder_width for s in samples]),
            head_ratio=med([s.head_ratio for s in samples]),
            tilt_deg=med([s.tilt_deg for s in samples]),
            lateral=med([s.lateral for s in samples]),
            face_size=med([s.face_size for s in samples]),
            face_ratio=med([s.face_ratio for s in samples]),
            recorded_at=self.clock(),
            frames=len(samples),
            mid_x=med([s.mid_x for s in samples]),
            mid_y=med([s.mid_y for s in samples]),
            yaw=float(np.median(yaws)) if yaws else None,
            skeleton=to_json(median_skeleton(self._calib_skel)),
        )
        self._calib_skel = []
        self._save()
        self._window.clear()
        self._set_status(GOOD, [], self._last_ts or self.clock())
        logger.info("posture baseline recorded from %d frames", len(samples))

    def clear_baseline(self) -> None:
        self.baseline = None
        if self.baseline_path and os.path.isfile(self.baseline_path):
            os.remove(self.baseline_path)
        self._set_status(NO_BASELINE, [], self._last_ts or self.clock())

    # --- per frame ------------------------------------------------------------------------

    def update(self, keypoints: np.ndarray | None, ts: float) -> None:
        """Feed the main person's keypoints for one frame (None if nobody is visible)."""
        dt = 0.0 if self._last_ts is None else min(1.0, max(0.0, ts - self._last_ts))
        self._last_ts = ts
        kp = np.asarray(keypoints, float) if keypoints is not None else None
        face_seen = kp is not None and kp.shape[0] >= 7 and kp[NOSE, 2] >= self.cfg.min_kp_conf
        problem = shoulder_problem(kp, self.cfg) if face_seen else None
        m = measure(kp, self.cfg.min_kp_conf) if face_seen and problem is None else None
        if face_seen:
            self._last_seen = ts  # someone is there, even if their shoulders are unclear
        if m is not None:
            self.last = m
        self.last_problem = problem
        if self.routine is None:
            self.movement.update(kp, ts, self.cfg.min_kp_conf, seen=face_seen)
        if face_seen:
            self._quality.append((ts, problem is not None))
        while self._quality and ts - self._quality[0][0] > self.cfg.smooth_s:
            self._quality.popleft()

        if self._calib is not None:
            if ts >= self._calib_start:  # not during the get-ready countdown
                if m is not None:
                    self._calib.append(m)
                    _turned, yaw, _why = head_turn(kp, self.cfg.min_kp_conf)
                    if yaw is not None:
                        self._calib_yaw.append(yaw)
                    self._calib_skel.append(normalise(kp, self.cfg.min_kp_conf))
                elif problem is not None:
                    self._calib_unclear += 1
            if ts >= self._calib_until:
                self._finish_baseline()
            self._account(dt, ts, count=False)
            return

        if self.rec is not None:  # a calibration step or test run: record, don't coach
            self._record(kp if face_seen else None, problem, ts)
            self._account(dt, ts, count=False)
            return

        if self.routine is not None:  # a stretch break: count reps, don't coach
            self.routine.update(kp, ts)
            if self.routine.done:
                self._finish_break(ts)
            self._account(dt, ts, count=False)
            return

        self._frame_good = None

        unclear = (bool(self._quality) and
                   sum(bad for _t, bad in self._quality) / len(self._quality) >= self.cfg.unclear_fraction)
        if self.model is not None:
            self._update_with_model(kp if face_seen else None, unclear, ts)
        elif self.baseline is None:
            self._set_status(NO_BASELINE, [], ts)
        elif ts - self._last_seen > self.cfg.away_after_s:
            self._set_status(AWAY, ["Nobody in front of the camera"], ts)
            self._window.clear()
            self._away_look_since = None
        elif self._looking_away(kp if face_seen else None, self.baseline.yaw, ts):
            self._window.clear()  # a turned head says nothing about posture (and skews head offset)
            self.smoothed = None
            self.hint = None
        elif unclear:
            self._window.clear()  # don't judge posture from untrustworthy shoulders
            self.smoothed = None
            self._propose(UNCLEAR, [UNCLEAR_MESSAGE], ts)
        elif m is not None:
            self._window.append((ts, m))
            while self._window and ts - self._window[0][0] > self.cfg.smooth_s:
                self._window.popleft()
            self.smoothed = self._smoothed()
            a = assess(self.smoothed, self.baseline, self.cfg)
            self.hint = a["hint"]
            self.lean_side = self._threshold_lean_side() if a["status"] in (LEANING, SLUMPED) else None
            self._propose(a["status"], a["reasons"], ts)
        self._coach_frame(kp if face_seen else None, m, dt, ts)
        self._account(dt, ts)

    def _threshold_lean_side(self) -> str | None:
        """Your side (the webcam image is not mirrored: image right = your left)."""
        s, b = self.smoothed, self.baseline
        if s.lateral is not None and b.lateral is not None and abs(s.lateral - b.lateral) > self.cfg.lateral_shift:
            return "left" if s.lateral > b.lateral else "right"
        tilt = s.tilt_deg - b.tilt_deg  # positive: the shoulder on the image's right is lower
        return ("left" if tilt > 0 else "right") if abs(tilt) > self.cfg.tilt_deg else None

    # --- coaching: fix guidance and breaks ---------------------------------------------------

    def _coach_frame(self, kp, m, dt: float, ts: float) -> None:
        """Long holds and their fix guidance. Short-term posture is never flagged."""
        self._kp = kp
        smoothed, frame = self._extreme_flags(kp, m, ts)
        if self.movement.state(ts) == MOVE_AWAY:  # got up: whatever was held is over
            self.holds.reset(HEAD_DOWN)
            self.holds.reset(LEAN)
        for kind in self.holds.update(smoothed, ts):
            if self.history:
                self.history.add_event(ts, "hold_warning", {"kind": kind})
        warned = [k for k in (HEAD_DOWN, LEAN) if k in self.holds.warned]
        kind = warned[0] if warned else None
        guide_status = {HEAD_DOWN: SLOUCHING, LEAN: LEANING}.get(kind, GOOD)
        ok = None if kind is None or frame is None else not frame[kind]
        corrected = self.guide.update(guide_status, ts, ok)
        if corrected:
            self.holds.reset(kind)  # corrected on purpose: the hold starts over
            if self.history:
                self.history.add_correction(ts, kind, corrected["seconds"], corrected["after_reminder"])
        self.instruction = INSTRUCTIONS[kind] if kind else None

    def _hold_reference(self) -> dict | None:
        if self.model is not None:
            return self.model.ref
        if self.baseline is not None:
            b = self.baseline
            return {"head_ratio": b.head_ratio, "tilt_deg": b.tilt_deg, "lateral": b.lateral}
        return None

    def _extreme_flags(self, kp, m, ts: float) -> tuple[dict | None, dict | None]:
        """(smoothed flags, this frame's flags) for the long-hold checks; None when this frame
        can't be judged (away, turned head, unclear shoulders, no reference)."""
        ref = self._hold_reference()
        if ref is None or kp is None or self.turn_reason is not None or self.status == UNCLEAR:
            return None, None
        cfg = self.holds.cfg
        if self.model is not None:
            raw = self.last_raw if self.last_raw is not None else None
            if raw is None:
                return None, None
            self._hold_window.append((ts, raw["head_ratio"], raw["tilt_deg"], raw["lateral"]))
            while self._hold_window and ts - self._hold_window[0][0] > self.cfg.smooth_s:
                self._hold_window.popleft()
            med = [float(np.median([row[i] for row in self._hold_window])) for i in (1, 2, 3)]
            cls = self.stable.current
            return (extremes(*med, ref, cfg, cls),
                    extremes(raw["head_ratio"], raw["tilt_deg"], raw["lateral"], ref, cfg, cls))
        if m is None or self.smoothed is None:
            return None, None
        s = self.smoothed
        return (extremes(s.head_ratio, s.tilt_deg, s.lateral, ref, cfg),
                extremes(m.head_ratio, m.tilt_deg, m.lateral, ref, cfg))

    # --- settings ------------------------------------------------------------------------------

    def _apply_settings(self) -> None:
        s = self.settings
        demo = s.demo_timings
        self.movement.cfg.reminder_s = CoachSettings.DEMO_REMINDER_S if demo else s.reminder_min * 60
        self.holds.cfg.head_down_enabled, self.holds.cfg.lean_enabled = s.head_down_enabled, s.lean_enabled
        self.holds.cfg.head_down_s = CoachSettings.DEMO_HOLD_S if demo else s.head_down_min * 60
        self.holds.cfg.lean_s = CoachSettings.DEMO_HOLD_S if demo else s.lean_min * 60

    def _load_settings(self) -> CoachSettings:
        saved = {}
        if self.settings_path and os.path.isfile(self.settings_path):
            try:
                with open(self.settings_path, encoding="utf-8") as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                saved = {}
        known = {f.name for f in fields(CoachSettings)}
        try:
            return CoachSettings(**{k: v for k, v in saved.items() if k in known})
        except TypeError:
            return CoachSettings()

    def reload_settings(self) -> None:
        """Pick up settings another camera's coach saved."""
        self.settings = self._load_settings()
        self._apply_settings()

    def update_settings(self, **changes) -> dict:
        known = {f.name for f in fields(CoachSettings)}
        unknown = set(changes) - known
        if unknown:
            raise ValueError(f"unknown setting(s): {sorted(unknown)}")
        self.settings = CoachSettings(**{**asdict(self.settings), **changes})
        self._apply_settings()
        if self.settings_path:
            os.makedirs(os.path.dirname(self.settings_path) or ".", exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=2)
        return asdict(self.settings)

    def _ghost_ref(self) -> tuple[dict | None, float | None]:
        """(Good skeleton, Good shoulder width px): the calibration's with a model, else the baseline's."""
        if self.model is not None and self.ghost_good:
            return self.ghost_good, self.model.ref.get("shoulder_width")
        if self.baseline is not None and self.baseline.skeleton:
            return self.baseline.skeleton, self.baseline.shoulder_width
        return None, None

    def ghost(self) -> dict | None:
        """The ghost of your Good posture in image pixels, while a poor posture is being corrected."""
        if not self.guide.ghost_visible or self._kp is None:
            return None
        ref, _good_width = self._ghost_ref()
        if not ref:
            return None
        pts = place_ghost(ref, self._kp, None, self.cfg.min_kp_conf)
        return {"points": pts, "edges": GHOST_EDGES} if pts else None

    def _now(self, now: float | None) -> float:
        if now is not None:
            return now
        return self._last_ts if self._last_ts is not None else self.clock()

    def start_break(self, now: float | None = None) -> None:
        if self.rec is not None or self._calib is not None:
            raise ValueError("finish the calibration first")
        now = self._now(now)
        self.routine = BreakRoutine(now, self.break_cfg, self.cfg.min_kp_conf)
        self.movement.start_break()
        self.break_result = None

    def snooze_break(self, now: float | None = None) -> None:
        now = self._now(now)
        self.movement.snooze(now)
        if self.history:
            self.history.add_event(now, "break_snoozed")

    def skip_break(self, now: float | None = None) -> None:
        now = self._now(now)
        self.movement.dismiss(now)
        if self.history:
            self.history.add_event(now, "break_skipped")

    def stood_up(self, now: float | None = None) -> None:
        if self.routine is None:
            return
        now = self._now(now)
        self.routine.stood_up(now)
        if self.routine.done:
            self._finish_break(now)

    def cancel_break(self, now: float | None = None) -> None:
        if self.routine is not None:
            self._finish_break(self._now(now), cancelled=True)

    def _finish_break(self, ts: float, cancelled: bool = False) -> None:
        result = self.routine.result()
        if cancelled:
            result["outcome"] = "cancelled"
        self.break_result = {**result, "at": ts}
        self.routine = None
        self.movement.finish_break(ts, completed=result["outcome"] == "completed")
        self.holds.reset(HEAD_DOWN)
        self.holds.reset(LEAN)
        self.guide.update(AWAY, ts, None)  # a fresh start after the break
        if self.history:
            self.history.add_event(ts, "break_" + result["outcome"], result)

    def _break_snapshot(self, ts: float) -> dict:
        return {"routine": self.routine.snapshot(ts) if self.routine else None,
                "result": self.break_result if self.break_result and ts - self.break_result["at"] <= 10 else None,
                "note": SAFETY_NOTE}

    def _looking_away(self, kp, ref_yaw: float, ts: float) -> bool:
        """Head turned away from the screen (both methods): "looking away", neutral, with the usual
        switch delay; held over ``looking_away_to_away_s`` it counts as away. True if turned."""
        turned, self.yaw, self.turn_reason = (head_turn(kp, self.cfg.min_kp_conf, self.cfg.looking_away_yaw,
                                                        ref_yaw or 0.0)
                                              if kp is not None else (False, None, None))
        if not turned:
            self._away_look_since = None
            return False
        if self._away_look_since is None:
            self._away_look_since = ts
        if ts - self._away_look_since >= self.cfg.looking_away_to_away_s:
            self._set_status(AWAY, ["Looking away from the screen for over "
                                    f"{self.cfg.looking_away_to_away_s / 60:g} min"], ts)
        else:
            self._propose(LOOKING_AWAY, ["Head turned away from the screen"], ts)
        return True

    def _update_with_model(self, kp, unclear: bool, ts: float) -> None:
        if ts - self._last_seen > self.cfg.away_after_s:
            self.stable.reset()
            self._away_look_since = None
            self._set_status(AWAY, ["Nobody in front of the camera"], ts)
            return
        if self._looking_away(kp, self.model.ref.get("yaw", 0.0), ts):
            return  # before the classifier: it only knows the postures it was shown
        if unclear:
            self.stable.reset()
            self._propose(UNCLEAR, [UNCLEAR_MESSAGE], ts)
            return
        self._classify_with_model(kp, ts)

    def _smoothed(self) -> Measurement:
        ms = [m for _t, m in self._window]

        def med(attr):
            values = [getattr(m, attr) for m in ms if getattr(m, attr) is not None]
            return float(np.median(values)) if values else None

        refs = [m.head_ref for m in ms if m.head_ref]
        return Measurement(med("shoulder_width"), med("head_ratio"), med("tilt_deg"), med("lateral"),
                           med("face_size"), med("face_ratio"), med("mid_x"), med("mid_y"),
                           max(set(refs), key=refs.count) if refs else None)

    def _propose(self, status: str, reasons: list[str], ts: float) -> None:
        if status == self.status:
            self.reasons = reasons
            self._candidate = status
            return
        if status != self._candidate:
            self._candidate, self._candidate_since = status, ts
        if ts - self._candidate_since >= self.cfg.switch_s:
            self._set_status(status, reasons, ts)

    def _set_status(self, status: str, reasons: list[str], ts: float) -> None:
        if status != self.status:
            self.status = status
            self._status_since = ts
            self.episode += 1
        self.reasons = reasons
        self._candidate, self._candidate_since = status, ts

    def _account(self, dt: float, ts: float, count: bool = True) -> None:
        key = self.status if self.status in self.session.seconds else None
        if count and key and dt > 0:
            self.session.seconds[key] += dt
            if self.history:
                side = self.lean_side if key in (LEANING, SLUMPED) else None
                self.history.add_seconds(ts, f"leaning_{side}" if key == LEANING and side else key, dt)
        # Timeline: the most common status in each bucket.
        if self._bucket_start is None:
            self._bucket_start = ts
        if self._origin is None:
            self._origin = ts
        if key and dt > 0:
            self._bucket[key] = self._bucket.get(key, 0.0) + dt
        if ts - self._bucket_start >= self.cfg.timeline_bucket_s:
            if self._bucket:
                self.session.timeline.append({
                    "t": round(self._bucket_start - self._origin, 1),
                    "status": max(self._bucket, key=self._bucket.get),
                })
                self.session.timeline = self.session.timeline[-720:]  # 2 h at 10 s
            self._bucket, self._bucket_start = {}, ts

    def note_reminder(self) -> None:
        self.session.reminders += 1
        self.guide.note_reminder()
        if self.history:
            self.history.add_event(self._now(None), "reminder", {"status": self.status})

    def reset_session(self) -> None:
        self.session = Session(started_at=self.clock())
        self._bucket, self._bucket_start = {}, None
        self._origin = None

    # --- output ---------------------------------------------------------------------------

    def snapshot(self) -> dict:
        ts = self._last_ts or 0.0
        calibrating = self._calib is not None
        getting_ready = calibrating and ts < self._calib_start
        status = CALIBRATING if calibrating else self.status
        held = ts - self._status_since if self.status in BAD + NEUTRAL and not calibrating else 0.0
        good = self.session.seconds[GOOD]
        poor = sum(self.session.seconds[s] for s in BAD)
        if getting_ready:
            label, reasons = "Get ready", ["Sit upright, shoulders relaxed and level, and look at the screen"]
            left = self._calib_start - ts
        elif calibrating:
            label, reasons = LABELS[CALIBRATING], ["Recording your baseline: keep still"]
            left = self._calib_until - ts
        else:
            label, reasons, left = LABELS.get(status, status), self.reasons, 0.0
        return {
            "status": status,
            "label": label,
            "reasons": reasons,
            "held_s": round(held, 1),
            "episode": self.episode,
            "has_baseline": self.baseline is not None,
            "calibrating": calibrating,
            "calibration_phase": ("get_ready" if getting_ready else "recording") if calibrating else None,
            "calibration_left_s": round(max(0.0, left), 1),
            "calibration_error": self._calib_error,
            "baseline_recorded_at": self.baseline.recorded_at if self.baseline else None,
            "session": {
                "seconds": {k: round(v, 1) for k, v in self.session.seconds.items()},
                "good_s": round(good, 1),
                "poor_s": round(poor, 1),
                "good_fraction": round(good / (good + poor), 3) if good + poor > 0 else None,
                "started_at": self.session.started_at,
                "reminders": self.session.reminders,
            },
            "measurements": asdict(self.last) if self.last else None,
            "shoulder_problem": self.last_problem,
            "hint": self.hint if self.status in BAD and not calibrating else None,
            # Every value behind the status (smoothed, vs baseline, with its limit): the debug panel.
            "debug": explain(self.smoothed, self.baseline, self.cfg)
            if self.smoothed is not None and self.baseline is not None and not calibrating and self.model is None
            else None,
            "method": "classifier" if self.model is not None else "thresholds",
            "probabilities": dict(self.probs) if self.model is not None else None,
            "features": self.last_raw if self.model is not None else None,
            "yaw": self.yaw,
            "turn_reason": self.turn_reason,
            "ood": {"distance": self.ood, "threshold": self.model.ood_threshold}
            if self.model is not None and self.model.ood_threshold is not None else None,
            "recording": self._recording_snapshot(ts),
            "recording_error": self.rec_error,
            "model": self.model_summary(),
            "calibration_counts": dict(self.calibration_counts),
            "guidance": {
                "active": self.guide.ghost_visible,
                "instruction": self.instruction if self.status in BAD else None,
                "ghost_available": self._ghost_ref()[0] is not None,
                "since_s": round(ts - self.guide.active["start"], 1) if self.guide.ghost_visible else None,
            },
            "back_to_good": self.guide.recent(ts),
            "break": self._break_snapshot(ts),
            "movement": {**self.movement.snapshot(ts), "demo_timings": self.settings.demo_timings},
            "holds": self.holds.snapshot(
                ts, available=self._hold_reference() is not None,
                reason=None if self._hold_reference() is not None else
                "Set a baseline or calibrate to turn on long-hold warnings (head down, strong lean)"),
            "settings": asdict(self.settings),
        }

    def model_summary(self) -> dict | None:
        if self.model is None:
            return None
        r, t = self.model.report, self.model.test_report
        return {"name": self.model.name, "trained_at": self.model.trained_at, "accuracy": r.get("accuracy"),
                "balanced_accuracy": r.get("balanced_accuracy"),
                "test_accuracy": t.get("accuracy") if t else None,
                "test_balanced_accuracy": t.get("balanced_accuracy") if t else None}

    def _recording_snapshot(self, ts: float) -> dict | None:
        r = self.rec
        if r is None:
            return None
        posture = r["postures"][r["step"]]
        ready = ts < r["record_from"]
        return {
            "kind": r["kind"], "posture": posture, "label": POSTURE_LABELS[posture],
            "step": r["step"] + 1, "steps": len(r["postures"]),
            "phase": "get_ready" if ready else "recording",
            "left_s": round(max(0.0, (r["record_from"] if ready else r["record_until"]) - ts), 1),
            "record_s": r["record_s"], "frames": len(r["frames"]), "skipped": r["skipped"],
        }

    def timeline(self) -> list[dict]:
        return list(self.session.timeline)

    # --- persistence ----------------------------------------------------------------------

    def _load(self) -> Baseline | None:
        if not self.baseline_path or not os.path.isfile(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, encoding="utf-8") as f:
                return Baseline.from_dict(json.load(f))
        except (OSError, ValueError, TypeError):
            logger.warning("ignoring unreadable posture baseline %s", self.baseline_path)
            return None

    def _save(self) -> None:
        if not self.baseline_path or self.baseline is None:
            return
        os.makedirs(os.path.dirname(self.baseline_path) or ".", exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.baseline), f, indent=2)
