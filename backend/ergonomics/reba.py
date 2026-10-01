"""REBA (Rapid Entire Body Assessment) from 2D posture angles.

Method: Hignett S, McAtamney L. "Rapid Entire Body Assessment (REBA)". Applied Ergonomics
31 (2000) 201-205. The tables below were checked cell by cell against the REBA Employee
Assessment Worksheet (Dr. Alan Hedge, distributed by Ergonomics Plus).

This is a **2D approximation**, not a certified assessment:
- Wrist angle can't be measured from COCO keypoints, so it gets a fixed sub-score
  (``wrist_default_score``).
- Twisting, side-bending, shoulder raise, arm abduction and arm support can't be measured
  reliably from one camera, so they get no adjustment.
- Load/force and coupling are unknown: both default to 0, and load can be set per zone
  (e.g. a loading dock).
- Legs: weight bearing can't be judged reliably, so the base score is 1 (bilateral), plus
  the knee-flexion adjustment.
- Activity: +1 when a posture is held static for more than a minute (the caller decides).
"""

from __future__ import annotations

from dataclasses import dataclass

from ergonomics.angles import PostureAngles

# --- Lookup tables (Hignett & McAtamney 2000) -----------------------------------------

# TABLE_A[trunk 1-5][neck 1-3][legs 1-4]
TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# TABLE_B[upper arm 1-6][lower arm 1-2][wrist 1-3]
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]

# TABLE_C[score A 1-12][score B 1-12]
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]

# Final score -> (level index, name). 1 negligible, 2-3 low, 4-7 medium, 8-10 high, 11+ very high.
RISK_LEVELS = ["negligible", "low", "medium", "high", "very_high"]


def risk_level(score: int) -> tuple[int, str]:
    if score <= 1:
        idx = 1
    elif score <= 3:
        idx = 2
    elif score <= 7:
        idx = 3
    elif score <= 10:
        idx = 4
    else:
        idx = 5
    return idx, RISK_LEVELS[idx - 1]


# --- Posture sub-scores ---------------------------------------------------------------

def trunk_score(angle: float) -> int:
    """Upright 1; 0-20 flexion or extension 2; 20-60 flexion or >20 extension 3; >60 flexion 4.
    Within 5 degrees of vertical counts as upright (2D keypoints are never exactly vertical)."""
    if abs(angle) <= 5:
        return 1
    if angle < 0:  # extension
        return 2 if -angle <= 20 else 3
    if angle <= 20:
        return 2
    return 3 if angle <= 60 else 4


def neck_score(angle: float) -> int:
    """0-20 flexion 1; >20 flexion or extension 2. Like the trunk, the first 5 degrees of
    apparent extension count as neutral: 2D keypoints jitter by a few degrees."""
    return 1 if -5 <= angle <= 20 else 2


def legs_score(knee_flexion: float | None) -> int:
    """Base 1 (bilateral support assumed); +1 knees flexed 30-60, +2 more than 60."""
    if knee_flexion is None or knee_flexion < 30:
        return 1
    return 2 if knee_flexion <= 60 else 3


def upper_arm_score(angle: float) -> int:
    """20 extension to 20 flexion 1; >20 extension or 20-45 flexion 2; 45-90 flexion 3; >90 flexion 4."""
    if -20 <= angle <= 20:
        return 1
    if angle < -20 or angle <= 45:
        return 2
    return 3 if angle <= 90 else 4


def lower_arm_score(flexion: float) -> int:
    """Elbow flexion 60-100 is 1, anything else 2 (flexion = 180 - elbow interior angle)."""
    return 1 if 60 <= flexion <= 100 else 2


# --- Assessment -----------------------------------------------------------------------

@dataclass
class RebaResult:
    score: int
    level: int          # 1..5
    level_name: str
    table_a: int
    table_b: int
    table_c: int
    trunk: int
    neck: int
    legs: int
    upper_arm: int
    lower_arm: int
    wrist: int
    load: int
    coupling: int
    activity: int
    dominant: str       # body part contributing most, relative to its maximum
    confidence: float
    estimated_parts: list[str]  # parts scored with defaults because they weren't visible

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _worst(*values: float | None) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals, key=abs) if vals else None


def assess(angles: PostureAngles, *, load: int = 0, coupling: int = 0, static: bool = False,
           wrist_score: int = 1) -> RebaResult | None:
    """REBA for one posture, or None if the trunk isn't visible."""
    if angles.trunk is None:
        return None
    estimated = []
    t = trunk_score(angles.trunk)
    if angles.neck is None:
        n = 1
        estimated.append("neck")
    else:
        n = neck_score(angles.neck)
    knee = _worst(angles.knee_left, angles.knee_right)
    if knee is None:
        estimated.append("legs")
    legs = legs_score(knee)

    # REBA scores one arm: take the side with the higher (worse) score.
    arm_scores = []
    for side in ("left", "right"):
        ua = getattr(angles, f"upper_arm_{side}")
        la = getattr(angles, f"lower_arm_{side}")
        if ua is None:
            continue
        arm_scores.append((upper_arm_score(ua), lower_arm_score(la) if la is not None else 1, la is None))
    if arm_scores:
        ua_s, la_s, la_missing = max(arm_scores)
        if la_missing:
            estimated.append("lower_arm")
    else:
        ua_s, la_s = 1, 1
        estimated += ["upper_arm", "lower_arm"]

    load = max(0, min(3, int(load)))
    coupling = max(0, min(3, int(coupling)))
    a = TABLE_A[t - 1][n - 1][legs - 1]
    b = TABLE_B[ua_s - 1][la_s - 1][wrist_score - 1]
    score_a = min(12, a + load)
    score_b = min(12, b + coupling)
    c = TABLE_C[score_a - 1][score_b - 1]
    activity = 1 if static else 0
    final = c + activity
    level, name = risk_level(final)

    # Dominant part: the highest sub-score relative to that part's maximum.
    shares = {"trunk": (t - 1) / 3, "neck": (n - 1) / 1, "legs": (legs - 1) / 2,
              "upper_arm": (ua_s - 1) / 3, "lower_arm": (la_s - 1) / 1}
    dominant = max(shares, key=lambda k: (shares[k], k == "trunk"))
    return RebaResult(
        score=final, level=level, level_name=name, table_a=a, table_b=b, table_c=c,
        trunk=t, neck=n, legs=legs, upper_arm=ua_s, lower_arm=la_s, wrist=wrist_score,
        load=load, coupling=coupling, activity=activity, dominant=dominant,
        confidence=angles.confidence, estimated_parts=estimated,
    )
