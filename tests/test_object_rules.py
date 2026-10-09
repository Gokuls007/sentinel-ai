"""Unsafe lift, standing on a chair and hand on a hazard: synthetic keypoints and object boxes."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from activity.object_rules import ObjectRules, knee_angle
from core.object_detector import ObjectDetection

C = 0.9


def body(hip=(400.0, 400.0), torso=100.0, trunk_deg=0.0, knee_bend=0.0, legs_conf=C, wrist=None, ankles=None):
    """A side-view person: hips at ``hip``, torso tilted ``trunk_deg`` forward (toward +x),
    thighs straight down; ``knee_bend`` folds the shins back (0 = straight legs)."""
    k = np.zeros((17, 3), np.float32)
    hx, hy = hip
    a = math.radians(trunk_deg)
    sh = (hx + math.sin(a) * torso, hy - math.cos(a) * torso)
    k[0] = (sh[0] + math.sin(a) * 30, sh[1] - math.cos(a) * 30, C)
    k[5], k[6] = (sh[0] - 5, sh[1], C), (sh[0] + 5, sh[1], C)
    k[11], k[12] = (hx - 5, hy, C), (hx + 5, hy, C)
    knee = (hx, hy + 90)
    b = math.radians(knee_bend)
    ankle = (knee[0] - math.sin(b) * 90, knee[1] + math.cos(b) * 90)
    k[13], k[14] = (*knee, legs_conf), (knee[0] + 4, knee[1], legs_conf)
    k[15], k[16] = (*ankle, legs_conf), (ankle[0] + 4, ankle[1], legs_conf)
    if ankles is not None:
        k[15], k[16] = (*ankles[0], C), (*ankles[1], C)
    w = wrist or (sh[0] + 20, sh[1] + 60)
    k[9], k[10] = (*w, C), (w[0] + 5, w[1], C)
    ys = [p[1] for p in k if p[2] > 0]
    xs = [p[0] for p in k if p[2] > 0]
    return SimpleNamespace(keypoints=k, bbox=(min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 5))


def run(rules, make, objects, seconds, t=0.0, fps=10):
    alerts = []
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        alerts += rules.update({1: make()}, objects, t)
    return t, alerts


BOX = ObjectDetection("cardboard box", 0.8, (480.0, 540.0, 560.0, 590.0))


def test_knee_angle():
    k = body()
    assert knee_angle(k.keypoints, 11, 13, 15) == pytest.approx(177, abs=1)  # the hip sits 5 px to one side
    assert knee_angle(body(knee_bend=60).keypoints, 11, 13, 15) == pytest.approx(117, abs=2)


def test_unsafe_lift_bent_back_straight_legs_hands_at_the_box():
    r = ObjectRules()
    t, _ = run(r, lambda: body(), [BOX], 1)  # upright first: the torso reference
    stoop = lambda: body(trunk_deg=70, wrist=(500.0, 545.0))  # noqa: E731
    t, alerts = run(r, stoop, [BOX], 0.3, t)
    assert alerts == []  # must hold 0.5 s
    t, alerts = run(r, stoop, [BOX], 0.5, t)
    assert len(alerts) == 1 and alerts[0].alert_type == "unsafe_lift"
    msg = alerts[0].message
    assert msg == "Unsafe lift: back bent 70°, knees 177° (straight), hand on cardboard box"
    assert alerts[0].details["knees_visible"] and alerts[0].details["hand_gap_px"] == 0
    _, again = run(r, stoop, [BOX], 3, t)
    assert again == []  # cooldown


def test_a_squat_lift_or_no_box_is_not_unsafe():
    r = ObjectRules()
    _, alerts = run(r, lambda: body(trunk_deg=70, knee_bend=80, wrist=(500.0, 545.0)), [BOX], 2)
    assert alerts == []  # knees bent: a proper lift
    r = ObjectRules()
    _, alerts = run(r, lambda: body(trunk_deg=70, wrist=(300.0, 545.0)), [BOX], 2)
    assert alerts == []  # hands nowhere near the box


def test_knees_hidden_by_the_box_are_reported_not_guessed():
    r = ObjectRules()
    _, alerts = run(r, lambda: body(trunk_deg=70, legs_conf=0.2, wrist=(500.0, 545.0)), [BOX], 1)
    assert len(alerts) == 1 and "knees not visible" in alerts[0].message
    assert alerts[0].details["knee_deg"] is None


def test_bending_toward_the_camera_counts_by_torso_length():
    r = ObjectRules()
    t, _ = run(r, lambda: body(torso=100), [BOX], 1)
    # Facing the camera: the trunk looks vertical but the torso is 60% of its upright length.
    _, alerts = run(r, lambda: body(torso=60, wrist=(500.0, 545.0)), [BOX], 1, t)
    assert len(alerts) == 1 and "torso 60% of upright" in alerts[0].message


CHAIR = ObjectDetection("chair", 0.7, (350.0, 300.0, 470.0, 560.0))  # backrest top 300, seat line 430, base 560


def test_standing_on_the_seat_of_a_chair():
    r = ObjectRules()
    on = lambda: body(hip=(410.0, 220.0), ankles=[(400.0, 400.0), (420.0, 402.0)])  # noqa: E731
    _, alerts = run(r, on, [CHAIR], 1)
    assert len(alerts) == 1 and alerts[0].alert_type == "standing_on_chair"
    assert alerts[0].message.startswith("Standing on chair: ankles 30 px and 28 px above the seat line")


def test_beside_or_in_front_of_a_chair_is_not_standing_on_it():
    r = ObjectRules()
    beside = lambda: body(hip=(560.0, 220.0), ankles=[(550.0, 400.0), (570.0, 402.0)])  # noqa: E731
    _, alerts = run(r, beside, [CHAIR], 1)
    assert alerts == []  # ankles outside the chair horizontally
    floor = lambda: body(hip=(410.0, 380.0), ankles=[(400.0, 556.0), (420.0, 558.0)])  # noqa: E731
    _, alerts = run(r, floor, [CHAIR], 1)
    assert alerts == []  # feet at the chair's base, below the seat line


def test_hand_on_a_hazard():
    knife = ObjectDetection("knife", 0.6, (470.0, 440.0, 520.0, 470.0))
    r = ObjectRules()
    _, alerts = run(r, lambda: body(wrist=(480.0, 455.0)), [knife], 0.5)
    assert [a.message for a in alerts] == ["Hand on knife (hazard)"]
    r = ObjectRules(hazard_classes=["oven"])
    _, alerts = run(r, lambda: body(wrist=(480.0, 455.0)), [knife], 1)
    assert alerts == []  # knife not marked as a hazard


def test_carrying_names_the_object():
    from activity import ActivityTracker

    a = ActivityTracker()
    k = body().keypoints
    pillow = [("pillow", (390.0, 300.0, 470.0, 380.0))]
    out = None
    for i in range(15):
        out = a.update(1, k, i / 10, 300.0, objects=pillow, box=(380, 250, 480, 590))
    assert out["label"] == "Carrying" and out["detail"] == "pillow"
    chair = [("chair", (390.0, 300.0, 470.0, 380.0))]
    out = a.update(2, k, 0.1, 300.0, objects=chair, box=(380, 250, 480, 590))
    assert out["detail"] != "chair"  # furniture isn't carried


def test_person_tag_reads_carrying_pillow():
    from core.pipeline import SentinelPipeline

    assert SentinelPipeline.person_tag({"label": "Carrying", "detail": "pillow"}, None) == "Carrying pillow"


def test_walking_past_a_hazard_is_not_touching_it():
    oven = ObjectDetection("oven", 0.9, (300.0, 400.0, 700.0, 480.0))
    r = ObjectRules()
    alerts, t = [], 0.0
    for i in range(20):  # 2 s walking across, a hand passing over the TV in the image
        t += 0.1
        x = 300 + 20 * i
        alerts += r.update({1: body(hip=(x, 400.0), wrist=(x + 10, 440.0))}, [oven], t)
    assert alerts == []


def test_walking_away_from_the_camera_is_not_a_bend_toward_it():
    r = ObjectRules()
    t, _ = run(r, lambda: body(torso=100), [BOX], 1)
    # Everything shrinks together (torso and thighs): further away, not bent.
    def far():
        k = body(torso=60, wrist=(500.0, 545.0))
        kp = k.keypoints
        kp[:, :2] = (kp[:, :2] - (400, 400)) * 1.0 + (400, 400)
        kp[13, 1] = kp[11, 1] + 54  # thigh shrinks with the torso (90 * 0.6)
        kp[14, 1] = kp[12, 1] + 54
        return k
    _, alerts = run(r, far, [BOX], 1, t)
    assert alerts == []


def test_monitors_and_laptops_are_not_hazards_by_default():
    from config.settings import DEFAULT_HAZARD_CLASSES

    assert "tv" not in DEFAULT_HAZARD_CLASSES and "laptop" not in DEFAULT_HAZARD_CLASSES
    tv = ObjectDetection("tv", 0.9, (300.0, 300.0, 700.0, 480.0))
    r = ObjectRules()
    alerts, t = [], 0.0
    for _ in range(30):  # 3 s with a hand resting on the monitor
        t += 0.1
        alerts += r.update({1: body(hip=(500.0, 400.0), wrist=(500.0, 380.0))}, [tv], t)
    assert alerts == []

