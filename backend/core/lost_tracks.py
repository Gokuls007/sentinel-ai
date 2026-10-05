"""Never silently lose a person who may have fallen.

When a person's track disappears, the tracker just drops it. If they were lying down (activity
"Lying down"/"Fallen", the fall detector past "upright", or a wide box), or on/next to a bed,
couch or chair, the most likely reason is that the detector stopped seeing them lying there,
not that they left. Such a track is kept at its last position for up to ``keep_s`` and marked
"lost while lying"; after ``alert_after_s`` an alert asks someone to check. Walking out of the
frame (the last box at an edge, moving toward it) is a normal exit and is dropped as before.
The mark clears when a person is seen again where they were lost.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from anomaly.engine import AnomalyAlert

LYING_LABELS = ("Lying down", "Fallen")
DOWN_STATES = ("falling", "fallen", "confirmed")
FURNITURE = ("bed", "couch", "chair")


def _overlap(a, b) -> float:
    """Share of box a covered by box b."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return inter / area


@dataclass
class _Seen:
    box: tuple
    ts: float
    label: str | None = None
    fall: str | None = None
    prev_box: tuple | None = None


@dataclass
class LostTrack:
    track_id: int
    box: tuple
    since: float
    reason: str
    alerted: bool = False
    history: list = field(default_factory=list)

    def to_dict(self, now: float) -> dict:
        return {"track_id": self.track_id, "box": [round(float(v), 1) for v in self.box],
                "seconds": round(now - self.since, 1), "reason": self.reason}


class LostTracks:
    def __init__(self, keep_s: float = 30.0, alert_after_s: float = 3.0, edge_frac: float = 0.03):
        self.keep_s = keep_s
        self.alert_after_s = alert_after_s
        self.edge_frac = edge_frac
        self.seen: dict[int, _Seen] = {}
        self.lost: dict[int, LostTrack] = {}

    def _why_lying(self, s: _Seen, furniture: list) -> str | None:
        if s.label in LYING_LABELS:
            return s.label.lower()
        if s.fall in DOWN_STATES:
            return f"fall detector: {s.fall}"
        w, h = s.box[2] - s.box[0], s.box[3] - s.box[1]
        if w > 1.2 * h:
            return "lying (wide box)"
        for name, box in furniture:
            if _overlap(s.box, box) > 0.3:
                return f"on/near {name}"
        return None

    def _walked_out(self, s: _Seen, width: int, height: int) -> bool:
        x1, y1, x2, y2 = s.box
        m = self.edge_frac
        at_left, at_right = x1 <= m * width, x2 >= (1 - m) * width
        at_top, at_bottom = y1 <= m * height, y2 >= (1 - m) * height
        if not (at_left or at_right or at_top or at_bottom):
            return False
        if s.prev_box is None:
            return True
        cx, pcx = (x1 + x2) / 2, (s.prev_box[0] + s.prev_box[2]) / 2
        moving_left, moving_right = cx < pcx - 1, cx > pcx + 1
        return (at_left and moving_left) or (at_right and moving_right) or at_top or (at_bottom and not s.label)

    def update(self, ts: float, people: dict, labels: dict, falls: dict, furniture: list,
               width: int, height: int) -> list[AnomalyAlert]:
        """``people``: {track id: box} seen this frame. Returns new "lost while lying" alerts."""
        alerts = []
        # Seen again where they were lost: no longer lost (the tracker may give a new id).
        for tid, lt in list(self.lost.items()):
            if tid in people or any(_overlap(lt.box, b) > 0.3 or _overlap(b, lt.box) > 0.3 for b in people.values()):
                del self.lost[tid]
        for tid, s in list(self.seen.items()):
            if tid in people:
                continue
            del self.seen[tid]
            reason = self._why_lying(s, furniture)
            if reason and not self._walked_out(s, width, height):
                self.lost[tid] = LostTrack(tid, s.box, s.ts, reason)
        for tid, box in people.items():
            prev = self.seen.get(tid)
            self.seen[tid] = _Seen(tuple(float(v) for v in box), ts, labels.get(tid), falls.get(tid),
                                   prev.box if prev else None)
        for tid, lt in list(self.lost.items()):
            age = ts - lt.since
            if age > self.keep_s:
                del self.lost[tid]
            elif age >= self.alert_after_s and not lt.alerted:
                lt.alerted = True
                alerts.append(AnomalyAlert(
                    alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}", alert_type="person_lost_lying", track_id=int(tid),
                    timestamp=ts, confidence=0.5, severity="high",
                    message=f"Lost sight of a person while lying ({lt.reason}): check on them",
                    details={"rule": "Lost while lying", "reason": lt.reason, "last_box": list(lt.box)}))
        return alerts

    def snapshot(self, ts: float) -> list[dict]:
        return [lt.to_dict(ts) for lt in self.lost.values()]
