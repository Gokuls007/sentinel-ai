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


# --- seated, found automatically (furniture or a thigh pointing at the camera) ----------------

def facing(thigh_px=80.0, trunk_deg=0.0, top=100.0):
    """Front view: knees straight below the hips by ``thigh_px`` (short = the thigh points at the camera)."""
    k = person(trunk_deg=trunk_deg, top=top)
    for hip_i, knee_i, ank_i in ((11, 13, 15), (12, 14, 16)):
        k[knee_i, :2] = (k[hip_i, 0], k[hip_i, 1] + thigh_px)
        k[ank_i, :2] = (k[hip_i, 0], k[hip_i, 1] + thigh_px + 60)
    return k


CHAIR = ("chair", (250.0, 150.0, 360.0, 320.0))  # the hips (300, 250) are on its seat


def test_seated_on_a_detected_chair_even_facing_the_camera():
    a = ActivityTracker()
    _, out = run(a, lambda _t: facing(thigh_px=40), 0.0, 1.5, objects=[CHAIR])
    assert out["label"] == SITTING and out["detail"] == "chair"
    _, out = run(ActivityTracker(), lambda _t: facing(thigh_px=40, trunk_deg=45), 0.0, 1.5, objects=[CHAIR])
    assert out["label"] == SITTING  # leaning forward to read is still sitting


def test_standing_in_front_of_a_chair_is_not_sitting():
    _, out = run(ActivityTracker(), lambda _t: person(), 0.0, 1.5, objects=[CHAIR])
    assert out["label"] == STANDING  # long, vertical thighs


def test_a_thigh_pointing_at_the_camera_reads_as_sitting_without_furniture():
    _, out = run(ActivityTracker(), lambda _t: facing(thigh_px=35), 0.0, 1.5)
    assert out["label"] == SITTING
    _, out = run(ActivityTracker(), lambda _t: facing(thigh_px=85), 0.0, 1.5)
    assert out["label"] == STANDING


def test_reclined_on_a_bed_is_lying_on_the_bed():
    bed = ("bed", (150.0, 150.0, 600.0, 400.0))
    _, out = run(ActivityTracker(), lambda _t: facing(thigh_px=40, trunk_deg=55), 0.0, 1.5, objects=[bed])
    assert out["label"] == LYING and out["detail"] == "bed"


def test_balance_and_lift_skip_people_who_are_not_on_their_feet():
    from types import SimpleNamespace

    from activity.balance import BalanceTracker
    from activity.object_rules import ObjectRules
    from core.object_detector import ObjectDetection
    from core.pipeline import SentinelPipeline

    pose = SimpleNamespace(keypoints=person(), bbox=(250, 80, 350, 400), body_height=300.0)
    b = BalanceTracker()
    b.update({1: pose}, 1.0, {1: "Sitting"})
    assert b.current == {}
    b.update({1: pose}, 1.1, {1: "Standing"})
    assert 1 in b.current
    bent = SimpleNamespace(keypoints=person(trunk_deg=70, wrists="low"), bbox=(250, 80, 420, 400))
    box = ObjectDetection("cardboard box", 0.9, (300.0, 300.0, 500.0, 420.0))
    r = ObjectRules()
    alerts = []
    for i in range(10):
        alerts += r.update({1: bent}, [box], i / 10, {1: "Sitting"})
    assert alerts == []
    assert SentinelPipeline.person_tag({"label": "Sitting", "detail": "chair"}, None) == "Sitting on chair"
    assert SentinelPipeline.person_tag({"label": "Lying down", "detail": "bed"}, None) == "Lying on bed"


# --- transitions: balance stays on while getting up -----------------------------------------------

def test_sit_to_stand_is_a_transition_and_keeps_balance_checks_on():
    from types import SimpleNamespace

    from core.pipeline import SentinelPipeline

    a = ActivityTracker()
    t, out = run(a, lambda _t: facing(thigh_px=40), 0.0, 2, objects=[CHAIR])
    assert out["label"] == SITTING and out["transition"] is None
    # Hips rise off the seat (y up by 70 px in 0.5 s, torso 90 px) and the legs straighten.
    t0 = t
    t, out = run(a, lambda tt: person(top=100 - 140 * (tt - t0)), t, 0.5)
    assert out["transition"] == "sit-to-stand"
    p = SentinelPipeline.__new__(SentinelPipeline)
    p.activity = a
    assert p._activity_labels()[1] == "Standing up"  # not a "Sitting" label: balance stays on
    assert SentinelPipeline.person_tag({"label": "Sitting", "transition": "sit-to-stand"}, None) == "Standing up"
    t, out = run(a, lambda _t: person(top=30), t, 3)
    assert out["transition"] is None and out["label"] == STANDING  # settled: plain standing

    from activity.balance import BalanceTracker
    b = BalanceTracker()
    pose = SimpleNamespace(keypoints=person(), bbox=(250, 80, 350, 400), body_height=300.0)
    b.update({1: pose}, 1.0, {1: "Standing up"})
    assert 1 in b.current


def test_settled_sitting_is_not_a_transition():
    a = ActivityTracker()
    _, out = run(a, lambda _t: facing(thigh_px=40), 0.0, 4, objects=[CHAIR])
    assert out["transition"] is None


def test_sitting_up_from_lying_is_a_transition():
    bed = ("bed", (150.0, 150.0, 600.0, 400.0))
    a = ActivityTracker()
    t, out = run(a, lambda _t: facing(thigh_px=40, trunk_deg=70), 0.0, 1.5, objects=[bed])
    assert out["label"] == LYING
    t0 = t
    _, out = run(a, lambda tt: facing(thigh_px=40, trunk_deg=max(10.0, 70 - 120 * (tt - t0))), t, 0.6,
                 objects=[bed])
    assert out["transition"] == "lying-to-sitting"


def test_slow_staged_stand_up_starts_at_the_forward_lean_and_lasts_through_the_rise():
    """An elderly-style stand: lean forward ~1 s, pause, then the hips rise slowly in two stages
    over ~3 s. The transition starts at the lean and stays on until standing has settled."""
    a = ActivityTracker()
    t, out = run(a, lambda _t: facing(thigh_px=40), 0.0, 2, objects=[CHAIR])
    assert out["label"] == SITTING and out["transition"] is None
    t0 = t
    t, out = run(a, lambda tt: facing(thigh_px=40, trunk_deg=min(30.0, 30 * (tt - t0))), t, 1.0, objects=[CHAIR])
    assert out["label"] == SITTING and out["transition"] == "sit-to-stand"  # on before the hips move
    states = []
    for stage in range(2):  # two slow stages, 1.5 s each, a pause between
        t1 = t
        for _ in range(15):
            t += 0.1
            lift = 35 * stage + 35 * (t - t1) / 1.5  # hips rise 70 px in all, slowly
            states.append(a.update(1, facing(thigh_px=40 + lift, trunk_deg=30 - 10 * stage, top=100 - lift),
                                   t, BH, objects=[CHAIR])["transition"])
        for _ in range(5):
            t += 0.1
            lift = 35 * (stage + 1)
            states.append(a.update(1, facing(thigh_px=40 + lift, trunk_deg=20 - 10 * stage, top=100 - lift),
                                   t, BH, objects=[CHAIR])["transition"])
    assert all(s == "sit-to-stand" for s in states)  # no gap during the slow rise
    t, out = run(a, lambda _t: person(), t, 4)
    assert out["transition"] is None  # settled standing

