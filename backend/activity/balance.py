"""Balance prediction: centre of mass vs base of support, per person, every frame.

- **Centre of mass (COM)**: a weighted mean of body-segment midpoints from the keypoints,
  using standard segment mass fractions (Dempster/Winter: trunk ~50%, thighs 10% each, head 8%,
  shanks+feet 6%, upper arms 3%, forearms+hands 2%). Segments that aren't visible are left out
  and the rest re-weighted; trunk and both ankles are required.
- **Base of support (BoS)**: the horizontal span between the ankles, widened by a foot margin
  on each side (feet extend past the ankle point).
- **Margin**: how far the COM sits inside the BoS (negative = outside), scaled by body size.
  Risk = 1 - margin / ``safe_margin``, clipped to 0-1 and smoothed. "Losing balance" when risk
  stays above ``warn_risk`` for ``warn_hold_s`` with the COM clearly outside the base, and only
  while the person is nearly still: in walking the COM leaves the base every step (dynamic
  balance), so the bar still shows but nobody is warned mid-stride.

- **On the feet**: only checked for someone standing on their legs: the hips must be at least
  ``on_feet_ratio`` of the leg's own length (hip-knee + knee-ankle, same frame) above the ankles.
  Seated (even on a couch edge, labelled Bending) or on the floor, the leg is folded and the
  hips are far lower than that, and the feet aren't what holds the person up. Measured as a
  ratio within one frame, so camera distance and height don't matter: own recordings standing
  p10 0.99, couch edge 0.67, on the floor 0.19; CAUCAFall (camera high in a corner) walking 95-98%
  of frames >= 0.85, seated 85-100% below. (The knee angle doesn't work here: a shin pointing at
  the camera looks straight.)

This is a 2D, image-plane estimate: it sees sideways (left/right) balance in the camera's view,
not forward/back balance toward the camera. A first version, to tune on recordings.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from anomaly.engine import AnomalyAlert

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# (mass fraction, keypoint a, keypoint b): the segment's midpoint stands for its mass.
SEGMENTS = (
    (0.081, NOSE, NOSE),                                  # head
    (0.028, L_SH, L_EL), (0.028, R_SH, R_EL),             # upper arms
    (0.022, L_EL, L_WR), (0.022, R_EL, R_WR),             # forearms + hands
    (0.100, L_HIP, L_KNEE), (0.100, R_HIP, R_KNEE),       # thighs
    (0.061, L_KNEE, L_ANK), (0.061, R_KNEE, R_ANK),       # shanks + feet
)
TRUNK = 0.497
NOT_ON_FEET = ("Sitting", "Lying down", "Fallen", "Upper body only")


@dataclass
class BalanceConfig:
    min_conf: float = 0.4
    ankle_conf: float = 0.5
    foot_margin: float = 0.08    # x body height, added to each side of the ankle span
    safe_margin: float = 0.10    # x body height: COM this far inside the BoS = no risk
    smooth_s: float = 0.3
    warn_risk: float = 0.85
    warn_hold_s: float = 0.4
    cooldown_s: float = 10.0
    warn_outside: float = 0.02   # x body height: the COM must be at least this far outside the base
    still_speed: float = 0.35    # body heights per second; faster = walking, no warning
    on_feet_ratio: float = 0.8   # hips this far above the ankles (x the leg's own length) = standing on the legs


def leg_extension(kp: np.ndarray, min_conf: float = 0.4) -> float | None:
    """Height of the hips above the ankle over the leg's own length (thigh + shin), the straighter
    leg of the two: ~1.0 standing, ~0.4-0.7 seated, ~0 or below on the floor. None without a leg."""
    kp = np.asarray(kp, float)
    best = None
    for h, k, a in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if min(kp[h, 2], kp[k, 2], kp[a, 2]) < min_conf:
            continue
        length = float(np.hypot(*(kp[k, :2] - kp[h, :2])) + np.hypot(*(kp[a, :2] - kp[k, :2])))
        if length > 1.0:
            r = float(kp[a, 1] - kp[h, 1]) / length
            best = r if best is None else max(best, r)
    return best


def centre_of_mass(kp: np.ndarray, min_conf: float = 0.4) -> np.ndarray | None:
    """(x, y) centre of mass from COCO keypoints, or None without a visible trunk."""
    kp = np.asarray(kp, float)
    ok = lambda i: kp[i, 2] >= min_conf  # noqa: E731
    if not (ok(L_SH) and ok(R_SH) and ok(L_HIP) and ok(R_HIP)):
        return None
    trunk = (kp[L_SH, :2] + kp[R_SH, :2] + kp[L_HIP, :2] + kp[R_HIP, :2]) / 4
    total, acc = TRUNK, TRUNK * trunk
    for w, a, b in SEGMENTS:
        if ok(a) and ok(b):
            acc = acc + w * (kp[a, :2] + kp[b, :2]) / 2
            total += w
    return acc / total


class BalanceTracker:
    def __init__(self, cfg: BalanceConfig | None = None):
        self.cfg = cfg or BalanceConfig()
        self._risk: dict[int, deque] = {}       # track id -> (ts, risk)
        self._high_since: dict[int, float] = {}
        self._fired: dict[int, float] = {}
        self.current: dict[int, dict] = {}       # track id -> latest view (for the overlay)
        self._com: dict[int, deque] = {}         # track id -> (ts, centre of mass, body height)

    def update(self, poses: dict, ts: float, labels: dict | None = None) -> list[AnomalyAlert]:
        """``labels``: activity per track; seated or lying people aren't on their feet, so there is no
        standing balance to estimate (no bar, no warning)."""
        c = self.cfg
        labels = labels or {}
        for tid in [t for t in self._risk if t not in poses]:
            self._risk.pop(tid, None)
            self._high_since.pop(tid, None)
            self._com.pop(tid, None)
        self.current = {}
        alerts = []
        for tid, pose in poses.items():
            view = (None if labels.get(tid) in NOT_ON_FEET
                    else self.measure(pose.keypoints, float(getattr(pose, "body_height", 0) or 0), pose.bbox))
            if view is None:
                self._risk.pop(tid, None)
                self._high_since.pop(tid, None)
                continue
            q = self._risk.setdefault(tid, deque())
            q.append((ts, view["raw_risk"]))
            while q and ts - q[0][0] > c.smooth_s:
                q.popleft()
            risk = float(np.mean([r for _t, r in q]))
            view["risk"] = round(risk, 2)
            speed = self._speed(tid, view, ts)
            view["moving"] = speed > c.still_speed
            self.current[tid] = view
            outside = view["margin"] <= -c.warn_outside
            if risk >= c.warn_risk and outside and not view["moving"]:
                start = self._high_since.setdefault(tid, ts)
                if ts - start >= c.warn_hold_s and ts - self._fired.get(tid, -1e9) >= c.cooldown_s:
                    self._fired[tid] = ts
                    self._high_since.pop(tid, None)
                    alerts.append(self._alert(tid, ts, view))
            else:
                self._high_since.pop(tid, None)
        return alerts

    def _speed(self, tid, view, ts, window_s: float = 0.6) -> float:
        """Centre-of-mass speed in body heights per second over the last ``window_s``."""
        q = self._com.setdefault(tid, deque())
        q.append((ts, np.array(view["com"]), view["body_height"]))
        while q and ts - q[0][0] > window_s:
            q.popleft()
        if len(q) < 2 or q[-1][0] - q[0][0] < 0.2:
            return 0.0
        (t0, p0, h0), (t1, p1, _h) = q[0], q[-1]
        return float(np.hypot(*(p1 - p0))) / max(h0, 1.0) / (t1 - t0)

    def measure(self, kp, body_height: float, box) -> dict | None:
        c = self.cfg
        kp = np.asarray(kp, float)
        if kp[L_ANK, 2] < c.ankle_conf or kp[R_ANK, 2] < c.ankle_conf:
            return None
        legs = leg_extension(kp, c.min_conf)
        if legs is None or legs < c.on_feet_ratio:  # not standing on the legs: seated or on the floor
            return None
        com = centre_of_mass(kp, c.min_conf)
        if com is None:
            return None
        bh = body_height if body_height > 1 else float(box[3] - box[1])
        bh = max(bh, 1.0)
        pad = c.foot_margin * bh
        left = min(kp[L_ANK, 0], kp[R_ANK, 0]) - pad
        right = max(kp[L_ANK, 0], kp[R_ANK, 0]) + pad
        margin = min(com[0] - left, right - com[0])  # px; negative = COM outside the base
        raw = float(np.clip(1.0 - margin / (c.safe_margin * bh), 0.0, 1.0))
        return {"com": [round(float(com[0]), 1), round(float(com[1]), 1)],
                "base": [round(float(left), 1), round(float(right), 1)],
                "margin_px": round(float(margin), 1), "margin": round(float(margin / bh), 3), "raw_risk": raw,
                "body_height": round(bh, 1), "legs": round(legs, 2)}

    @staticmethod
    def _alert(tid, ts, view) -> AnomalyAlert:
        m = view["margin_px"]
        where = f"{-m:.0f} px outside" if m < 0 else f"only {m:.0f} px inside"
        msg = f"Losing balance: centre of mass {where} the base of support (risk {view['risk']:.0%})"
        return AnomalyAlert(alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}", alert_type="losing_balance",
                            track_id=int(tid), timestamp=ts, confidence=view["risk"], severity="high", message=msg,
                            details={"rule": "Losing balance", "margin_px": m, "risk": view["risk"],
                                     "com": view["com"], "base": view["base"]})
