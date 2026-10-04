"""Live activity labels, the camera-view check and the "why not scored" reasons (synthetic poses)."""

import numpy as np
import pytest

from activity import ActivityTracker, ViewCheck, ergo_reason, visible_parts
from activity.rules import BENDING, FALLEN, LIFTING, LYING, REACHING, SITTING, STANDING, UPPER_ONLY, WALKING

BH = 300.0


def o(p, dx, dy):
    return p + np.array([dx, dy])


def person(x=300.0, top=100.0, trunk_deg=0.0, sitting=False, wrists="side", conf=0.9, legs=True):
    """17 COCO keypoints of a standing person (side view), optionally bent, sitting, arms up."""
    k = np.zeros((17, 3), np.float32)
    hip = np.array([x, top + 150.0])
    torso = 90.0
    a = np.radians(trunk_deg)
    sh = hip + np.array([np.sin(a) * torso, -np.cos(a) * torso])
    head = sh + np.array([np.sin(a) * 30, -np.cos(a) * 30])
    k[0] = (*head, conf)
    k[1] = (*o(head, -5, -3), conf)
    k[2] = (*o(head, 5, -3), conf)
    k[5] = (*o(sh, -10, 0), conf)
    k[6] = (*o(sh, 10, 0), conf)
    k[11] = (*o(hip, -10, 0), conf)
    k[12] = (*o(hip, 10, 0), conf)
    if legs:
        knee = o(hip, 70, 5) if sitting else o(hip, 0, 75)
        ankle = o(knee, 0, 70)
        k[13], k[14] = (*o(knee, -8, 0), conf), (*o(knee, 8, 0), conf)
        k[15], k[16] = (*o(ankle, -8, 0), conf), (*o(ankle, 8, 0), conf)
    if wrists == "side":
        w = o(hip, 15, 10)
    elif wrists == "up":
        w = o(head, 0, -60)
    elif wrists == "low":  # down at the feet
        w = o(hip, np.sin(a) * 150, 120)
    elif wrists == "held":  # holding something in front, at chest height
        w = o(sh, 25, 35)
    else:
        w = None
    if w is not None:
        k[9], k[10] = (*o(w, -6, 0), conf), (*o(w, 6, 0), conf)
    return k


def scaled(k, s, x=300.0, y=400.0):
    """The same pose s times as big around (x, y): walking toward (s > 1) or away (s < 1)."""
    k = k.copy()
    k[:, :2] = (k[:, :2] - (x, y)) * s + (x, y)
    return k


def run(tracker, make, t, seconds, fps=10, tid=1, body_height=BH, **kw):
    out = None
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        out = tracker.update(tid, make(t), t, body_height, **kw)
    return t, out


def test_standing_walking_sitting():
    a = ActivityTracker()
    t, out = run(a, lambda _t: person(), 0.0, 2)
    assert out["label"] == STANDING
    t, out = run(a, lambda tt: person(x=300 + 200 * tt), t, 2)  # 200 px/s = 0.67 body heights/s
    assert out["label"] == WALKING
    t, out = run(a, lambda _t: person(sitting=True), t, 2)
    assert out["label"] == SITTING


def test_lift_then_carry_then_put_down():
    from activity.rules import CARRYING

    a = ActivityTracker()
    t, _ = run(a, lambda _t: person(), 0.0, 2)
    t, out = run(a, lambda _t: person(trunk_deg=70, wrists="low"), t, 1.5)
    assert out["label"] == BENDING
    t, out = run(a, lambda _t: person(wrists="held"), t, 0.5)
    assert out["label"] == LIFTING
    t, out = run(a, lambda _t: person(wrists="held"), t, 3)  # standing still, holding it
    assert out["label"] == CARRYING
    t, out = run(a, lambda _t: person(trunk_deg=60, wrists="low"), t, 1)  # bends to put it down
    assert out["label"] == BENDING
    t, out = run(a, lambda _t: person(), t, 2)
    assert out["label"] == STANDING
    labels = [h["label"] for h in out["history"]]
    assert labels[-5:] == [BENDING, LIFTING, CARRYING, BENDING, STANDING]


def test_carrying_ends_when_both_arms_hang():
    from activity.rules import CARRYING

    a = ActivityTracker()
    t, _ = run(a, lambda _t: person(), 0.0, 2)
    t, _ = run(a, lambda _t: person(trunk_deg=70, wrists="low"), t, 1.5)
    t, out = run(a, lambda _t: person(wrists="held"), t, 2)
    assert out["label"] == CARRYING
    t, out = run(a, lambda _t: person(), t, 2)  # dropped it: arms by the sides
    assert out["label"] == STANDING


def test_a_bend_without_lifting_anything_is_not_a_lift():
    a = ActivityTracker()
    t, _ = run(a, lambda _t: person(), 0.0, 2)
    t, out = run(a, lambda _t: person(trunk_deg=60, wrists="side"), t, 1.5)  # hands not low
    t, out = run(a, lambda _t: person(), t, 1)
    assert out["label"] == STANDING and LIFTING not in [h["label"] for h in out["history"]]
    t, out = run(a, lambda _t: person(trunk_deg=70, wrists="low"), t, 1.5)  # hands low, then arms hang
    t, out = run(a, lambda _t: person(), t, 2)
    labels = [h["label"] for h in out["history"]]
    assert out["label"] == STANDING and LIFTING not in labels and "Carrying" not in labels


def test_walking_away_from_the_camera_is_not_bending():
    a = ActivityTracker()
    t, _ = run(a, lambda _t: person(), 0.0, 2)
    # Shrinks to 55% over 2 s (walking straight away), the hips barely move sideways.
    t0 = t
    t, out = run(a, lambda tt: scaled(person(), 1 - 0.225 * (tt - t0)), t, 2)
    labels = {h["label"] for h in out["history"]}
    assert BENDING not in labels and out["label"] == WALKING


def test_no_ankles_uses_the_box_for_body_height():
    a = ActivityTracker()
    box = (270, 70, 330, 400)  # 330 px tall
    # 50 px/s sideways is 0.15 body heights per second: standing (swaying), not walking.
    _, out = run(a, lambda tt: person(x=300 + 50 * tt), 0.0, 2, body_height=0.0, box=box)
    assert out["label"] == STANDING


def test_reaching_lying_fallen_and_upper_body_only():
    a = ActivityTracker()
    _, out = run(a, lambda _t: person(wrists="up"), 0.0, 1.5)
    assert out["label"] == REACHING
    _, out = run(ActivityTracker(), lambda _t: person(trunk_deg=85), 0.0, 1.5)
    assert out["label"] == LYING
    _, out = run(ActivityTracker(), lambda _t: person(), 0.0, 0.5, fallen=True)
    assert out["label"] == FALLEN
    _, out = run(ActivityTracker(), lambda _t: person(), 0.0, 0.5, fall_state="fallen")
    assert out["label"] == LYING  # on the ground, not yet confirmed
    _, out = run(ActivityTracker(), lambda _t: person(legs=False), 0.0, 1.5)
    assert out["label"] == UPPER_ONLY  # no Standing/Walking guess without legs


def test_carrying_needs_a_detected_object_or_is_marked_low():
    from activity.rules import CARRYING

    a = ActivityTracker()

    def holding(tt):
        k = person(x=300 + 200 * tt, wrists=None)
        hip_y = k[11, 1]
        k[9] = (k[11, 0] + 25, hip_y - 40, 0.9)
        k[10] = (k[12, 0] + 25, hip_y - 40, 0.9)
        return k

    t, out = run(a, holding, 0.0, 1.5)
    assert out["label"] == CARRYING and out["confidence"] == "low"
    box = lambda tt: [("suitcase", (300 + 200 * tt, 150, 360 + 200 * tt, 230))]  # noqa: E731
    out = None
    for _ in range(15):
        t += 0.1
        out = a.update(1, holding(t), t, BH, objects=box(t))
    assert out["label"] == CARRYING and out["detail"] == "suitcase"


def test_short_jitter_does_not_flicker():
    a = ActivityTracker()
    t, _ = run(a, lambda _t: person(), 0.0, 2)
    t, out = run(a, lambda tt: person(x=300 + (40 if int(tt * 10) % 7 == 0 else 0)), t, 2)
    assert out["label"] == STANDING


# --- the camera-view check and REBA reasons ------------------------------------------------------

def test_upper_body_only_banner_after_5_seconds():
    v = ViewCheck(window_s=5)
    t = 0.0
    for _ in range(40):
        t += 0.1
        v.update({1: person(legs=False)}, t)
    assert not v.upper_body_only  # not yet
    for _ in range(20):
        t += 0.1
        v.update({1: person(legs=False)}, t)
    assert v.upper_body_only and "2–4 m away" in v.snapshot()["message"]
    v.update({1: person(legs=False), 2: person(x=600)}, t + 0.1)  # someone full-body appears
    assert not v.upper_body_only
    empty = ViewCheck(window_s=5)
    for i in range(80):
        empty.update({}, i * 0.1)
    assert not empty.upper_body_only  # nobody there is not "upper body only"


@pytest.mark.parametrize(("mutate", "conf", "box", "reason"), [
    (lambda k: k.__setitem__((slice(11, 17), 2), 0), 0.9, 0.6, "hips not visible"),
    (lambda k: k.__setitem__((slice(13, 17), 2), 0), 0.9, 0.6, "legs not visible"),
    (lambda k: None, 0.9, 0.1, "too small or far away"),
    (lambda k: None, 0.2, 0.6, "facing the camera (REBA needs a side view)"),
    (lambda k: None, 0.9, 0.6, "keypoints unclear"),
])
def test_ergo_reasons(mutate, conf, box, reason):
    k = person()
    mutate(k)
    assert ergo_reason(k, conf, 0.5, box) == reason


def test_visible_parts():
    p = visible_parts(person(legs=False))
    assert p["hips"] and p["shoulders"] and not p["knees"] and not p["ankles"]
