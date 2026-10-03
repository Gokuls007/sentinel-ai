"""Long-hold warnings: only sustained extremes are flagged, never short-term posture.

- **Head far down** held ``head_down_s`` (20 min): head height (nose above the shoulder line,
  in shoulder widths) below ``head_drop_ratio`` (70%) of your reference, or the personal
  model saying slouching / looking down while the head is clearly dropped (below
  ``model_head_drop_ratio``).
- **Strong lean** held ``lean_s`` (20 min): the shoulder line tilted ``tilt_deg`` (15°) more
  than your reference, or the head offset past ``lateral_limit`` shoulder widths; or the model
  saying leaning while tilted at least ``model_tilt_deg``.

The reference is the Good calibration (with a trained model) or the baseline; without either,
long holds are off. A dip out of the extreme shorter than ``interrupt_s`` (1 min) doesn't reset
the hold; longer does. Correcting after a warning ("back to good") resets it.
"""

from __future__ import annotations

from dataclasses import dataclass

HEAD_DOWN, LEAN = "head_down", "lean"
LABELS = {HEAD_DOWN: "Head down for a long time", LEAN: "Leaning to one side for a long time"}
INSTRUCTIONS = {HEAD_DOWN: "Lift your head and sit back", LEAN: "Centre yourself: level your shoulders"}


@dataclass
class HoldConfig:
    head_down_enabled: bool = True
    head_down_s: float = 1200.0
    lean_enabled: bool = True
    lean_s: float = 1200.0
    interrupt_s: float = 60.0
    head_drop_ratio: float = 0.70
    model_head_drop_ratio: float = 0.85
    tilt_deg: float = 15.0
    lateral_limit: float = 0.45
    model_tilt_deg: float = 10.0


def extremes(head_ratio: float | None, tilt_deg: float | None, lateral: float | None, ref: dict,
             cfg: HoldConfig, model_class: str | None = None) -> dict[str, bool]:
    """Which extremes this (smoothed) frame shows. ``ref`` has head_ratio, tilt_deg and lateral
    from the Good calibration or the baseline."""
    out = {HEAD_DOWN: False, LEAN: False}
    if head_ratio is not None and ref.get("head_ratio"):
        drop = head_ratio / ref["head_ratio"]
        out[HEAD_DOWN] = drop < cfg.head_drop_ratio or (
            model_class in ("slouching", "looking_down") and drop < cfg.model_head_drop_ratio)
    tilt = abs(tilt_deg - ref.get("tilt_deg", 0.0)) if tilt_deg is not None else None
    offset = abs(lateral - (ref.get("lateral") or 0.0)) if lateral is not None else None
    out[LEAN] = bool((tilt is not None and tilt >= cfg.tilt_deg)
                     or (offset is not None and offset >= cfg.lateral_limit)
                     or (model_class in ("leaning_left", "leaning_right", "leaning") and tilt is not None
                         and tilt >= cfg.model_tilt_deg))
    return out


class HoldTracker:
    def __init__(self, cfg: HoldConfig | None = None):
        self.cfg = cfg or HoldConfig()
        self.since: dict[str, float | None] = {HEAD_DOWN: None, LEAN: None}
        self.last_true: dict[str, float] = {}
        self.warned: dict[str, int] = {}  # kind -> warning id while it's showing
        self._next_id = 0

    def limit(self, kind: str) -> float:
        return self.cfg.head_down_s if kind == HEAD_DOWN else self.cfg.lean_s

    def enabled(self, kind: str) -> bool:
        return self.cfg.head_down_enabled if kind == HEAD_DOWN else self.cfg.lean_enabled

    def update(self, flags: dict[str, bool] | None, ts: float) -> list[str]:
        """``flags`` from ``extremes`` (None = can't judge this frame: nothing changes).
        Returns the kinds whose warning starts on this frame."""
        started = []
        for kind in (HEAD_DOWN, LEAN):
            if not self.enabled(kind):
                self.reset(kind)
                continue
            if flags is None:
                continue
            if flags.get(kind):
                if self.since[kind] is None:
                    self.since[kind] = ts
                self.last_true[kind] = ts
                if kind not in self.warned and ts - self.since[kind] >= self.limit(kind):
                    self._next_id += 1
                    self.warned[kind] = self._next_id
                    started.append(kind)
            elif self.since[kind] is not None and ts - self.last_true.get(kind, ts) > self.cfg.interrupt_s:
                self.reset(kind)
        return started

    def reset(self, kind: str) -> None:
        self.since[kind] = None
        self.warned.pop(kind, None)

    def held_s(self, kind: str, ts: float) -> float:
        return 0.0 if self.since[kind] is None else ts - self.since[kind]

    def snapshot(self, ts: float, available: bool, reason: str | None) -> dict:
        return {
            "available": available, "reason": reason,
            "active": [{"kind": k, "id": i, "label": LABELS[k], "instruction": INSTRUCTIONS[k],
                        "held_s": round(self.held_s(k, ts), 1)} for k, i in self.warned.items()],
            "progress": {k: {"held_s": round(self.held_s(k, ts), 1), "limit_s": self.limit(k),
                             "enabled": self.enabled(k)} for k in (HEAD_DOWN, LEAN)},
        }
