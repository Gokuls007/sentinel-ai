"""Fix guidance: a faint ghost of your own Good posture over the live video, a one-line
instruction, and a quick "back to good" check that times the correction.

The ghost is your Good skeleton (nose, eyes, ears, shoulders) stored relative to the shoulder
midpoint in shoulder widths. It is drawn on your *current* shoulder midpoint at your current
shoulder width, so a slouch shows the head where it would be, and a lean shows level
shoulders. For "too close" it is drawn at your Good size instead, so it looks smaller: move back.

The official status changes only after its stability delay (2 s thresholds, 5 s classifier).
That is too slow to feel like feedback, so while the ghost is shown a quicker check runs on
every frame: once ~1 s of frames look Good, "back to good" shows, the ghost hides and the
time-to-correct is measured to that moment. The official status keeps its own delay.
"""

from __future__ import annotations

from collections import deque

import numpy as np

GHOST_POINTS = (0, 1, 2, 3, 4, 5, 6)  # nose, eyes, ears, shoulders
GHOST_EDGES = ((5, 6), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4))
L_SH, R_SH = 5, 6

POOR = ("slouching", "leaning", "too_close", "slumped")

STATUS_INSTRUCTIONS = {
    "slouching": "Sit back and lift your head",
    "slumped": "Sit back, lift your head and level your shoulders",
    "leaning": "Centre yourself: level your shoulders, head over them",
    "too_close": "Move back from the screen",
}
MEASURE_INSTRUCTIONS = {
    "head_ratio": "Sit back and lift your head",
    "face_ratio": "Roll your shoulders back and down",
    "tilt_deg": "Level your shoulders",
    "lateral": "Bring your head back over your shoulders",
    "face_size": "Move back from the screen",
}


def normalise(keypoints, min_conf: float = 0.4) -> dict[int, tuple[float, float]] | None:
    """Visible face and shoulder points relative to the shoulder midpoint, in shoulder widths."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or min(kp[L_SH, 2], kp[R_SH, 2]) < min_conf:
        return None
    mid = (kp[L_SH, :2] + kp[R_SH, :2]) / 2
    width = float(np.hypot(*(kp[L_SH, :2] - kp[R_SH, :2])))
    if width < 10:
        return None
    return {i: (float((kp[i, 0] - mid[0]) / width), float((kp[i, 1] - mid[1]) / width))
            for i in GHOST_POINTS if kp[i, 2] >= min_conf}


def median_skeleton(skeletons) -> dict[int, tuple[float, float]] | None:
    """Median of each point seen in at least half the frames (keys may be JSON strings)."""
    skeletons = [{int(k): v for k, v in s.items()} for s in skeletons if s]
    if not skeletons:
        return None
    out = {}
    for i in GHOST_POINTS:
        pts = [s[i] for s in skeletons if i in s]
        if len(pts) * 2 >= len(skeletons):
            out[i] = (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
    return out if L_SH in out and R_SH in out else None


def to_json(skeleton: dict | None) -> dict | None:
    return {str(i): [round(x, 4), round(y, 4)] for i, (x, y) in skeleton.items()} if skeleton else None


def place_ghost(ghost: dict, keypoints, scale_px: float | None = None,
                min_conf: float = 0.4) -> dict[int, tuple[float, float]] | None:
    """The ghost in image pixels, on the current shoulder midpoint. Scaled to the current shoulder
    width, or to ``scale_px`` (your Good shoulder width) when given."""
    kp = np.asarray(keypoints, float)
    if ghost is None or kp.shape[0] < 7 or min(kp[L_SH, 2], kp[R_SH, 2]) < min_conf:
        return None
    mid = (kp[L_SH, :2] + kp[R_SH, :2]) / 2
    width = scale_px or float(np.hypot(*(kp[L_SH, :2] - kp[R_SH, :2])))
    return {int(i): (float(mid[0] + x * width), float(mid[1] + y * width)) for i, (x, y) in ghost.items()}


def instruction(status: str, rows: list[dict] | None = None, cfg=None) -> str | None:
    """One line telling you what to do. With the threshold method's measurement rows, it follows
    the measure that is furthest past its limit; otherwise it follows the status."""
    if status not in POOR:
        return None
    if rows and cfg is not None:
        excess = {}
        for r in rows:
            if not r["triggered"] or r["change"] is None:
                continue
            c, k = r["change"], r["key"]
            if k == "head_ratio":
                excess[k] = (cfg.head_drop_ratio - c) / max(1e-6, 1 - cfg.head_drop_ratio)
            elif k == "face_ratio":
                excess[k] = (c - cfg.hunch_ratio) / max(1e-6, cfg.hunch_ratio - 1)
            elif k == "tilt_deg":
                excess[k] = (abs(c) - cfg.tilt_deg) / cfg.tilt_deg
            elif k == "lateral":
                excess[k] = (abs(c) - cfg.lateral_shift) / cfg.lateral_shift
            elif k == "face_size":
                excess[k] = (c - cfg.too_close_ratio) / max(1e-6, cfg.too_close_ratio - 1)
        if excess:
            return MEASURE_INSTRUCTIONS[max(excess, key=excess.get)]
    return STATUS_INSTRUCTIONS[status]


class CorrectionTracker:
    """Follows one poor-posture episode from the moment it is shown until you're back to good.

    ``update`` is called every frame with the official status and ``ok``: whether this frame on
    its own looks Good (None when it can't be judged). Once the share of Good frames over the
    last ``quick_s`` reaches ``good_share``, the correction is recorded (time from when the poor
    status was shown) and the ghost hides. If the official status is still poor and the frames
    look poor again for ``relapse_s``, the ghost comes back as a new correction."""

    def __init__(self, quick_s: float = 1.0, good_share: float = 0.75, relapse_s: float = 2.0,
                 relapse_share: float = 0.25, show_s: float = 5.0):
        self.quick_s, self.good_share = quick_s, good_share
        self.relapse_s, self.relapse_share, self.show_s = relapse_s, relapse_share, show_s
        self.active: dict | None = None
        self.last: dict | None = None  # the latest correction
        self._frames: deque[tuple[float, bool]] = deque()
        self._poor_since: float | None = None
        self._count = 0

    @property
    def ghost_visible(self) -> bool:
        return self.active is not None and self.active["corrected_at"] is None

    def note_reminder(self) -> None:
        if self.ghost_visible:
            self.active["after_reminder"] = True

    def _share(self, ts: float) -> float | None:
        while self._frames and ts - self._frames[0][0] > self.quick_s:
            self._frames.popleft()
        # Needs (almost) a full window of judged frames.
        if not self._frames or ts - self._frames[0][0] < 0.9 * self.quick_s:
            return None
        return sum(ok for _t, ok in self._frames) / len(self._frames)

    def _start(self, posture: str, ts: float) -> None:
        self.active = {"posture": posture, "start": ts, "after_reminder": False, "corrected_at": None}
        self._frames.clear()
        self._poor_since = None

    def _finish(self, ts: float) -> dict:
        a = self.active
        a["corrected_at"] = ts
        self._count += 1
        self.last = {"id": self._count, "posture": a["posture"], "seconds": round(ts - a["start"], 1),
                     "at": ts, "after_reminder": a["after_reminder"]}
        return self.last

    def update(self, status: str, ts: float, ok: bool | None) -> dict | None:
        """Returns the correction when one is recorded on this frame."""
        if status not in POOR:
            done = None
            if status == "good" and self.ghost_visible:
                done = self._finish(ts)  # the official status got there before the quick check
            self.active = None
            self._frames.clear()
            return done
        if self.active is None:
            self._start(status, ts)
        self.active["posture"] = status
        if ok is not None:
            self._frames.append((ts, ok))
        share = self._share(ts)
        if self.active["corrected_at"] is None:
            if share is not None and share >= self.good_share:
                return self._finish(ts)
            return None
        # Already corrected, but the official status hasn't caught up: watch for a relapse.
        if share is not None and share <= self.relapse_share:
            self._poor_since = self._poor_since if self._poor_since is not None else ts
            if ts - self._poor_since >= self.relapse_s:
                self._start(status, ts)
        else:
            self._poor_since = None
        return None

    def recent(self, ts: float) -> dict | None:
        """The latest correction while it should still be shown ("Back to good")."""
        if self.last and ts - self.last["at"] <= self.show_s:
            return self.last
        return None
