"""Fall prevention (home mode): getting out of bed, slipping out of a seat, on the floor."""

from types import SimpleNamespace

import numpy as np
import pytest

from activity.prevention import PreventionConfig, PreventionTracker

C = 0.9
BED = SimpleNamespace(class_name="bed", bbox=(100.0, 300.0, 700.0, 520.0))
CHAIR = SimpleNamespace(class_name="chair", bbox=(380.0, 300.0, 520.0, 560.0))


def body(hip=(450.0, 400.0), torso=100.0, trunk_deg=0.0, ankles=((430.0, 600.0), (470.0, 600.0)),
         knees=((430.0, 500.0), (470.0, 500.0)), legs=True):
    k = np.zeros((17, 3), np.float32)
    hx, hy = hip
    a = np.radians(trunk_deg)
    sx, sy = hx + np.sin(a) * torso, hy - np.cos(a) * torso
    k[0] = (sx, sy - 30, C)
    k[5], k[6] = (sx - 25, sy, C), (sx + 25, sy, C)
    k[11], k[12] = (hx - 20, hy, C), (hx + 20, hy, C)
    if legs:
        k[13], k[14] = (*knees[0], C), (*knees[1], C)
        k[15], k[16] = (*ankles[0], C), (*ankles[1], C)
    xs, ys = k[k[:, 2] > 0, 0], k[k[:, 2] > 0, 1]
    return SimpleNamespace(keypoints=k, bbox=(float(xs.min() - 20), float(ys.min() - 20), float(xs.max() + 20),
                                              float(ys.max() + 10)))


def run(tracker, frames, objects, t=0.0, fps=10, down=()):
    """frames: [(seconds, pose, activity dict)] -> alerts."""
    alerts = []
    for seconds, pose, act in frames:
        for _ in range(round(seconds * fps)):
            t += 1 / fps
            alerts += tracker.update({1: pose}, {1: act}, objects, t, set(down))
    return t, alerts


SIT = {"label": "Sitting", "detail": "chair", "transition": None}


def test_slipping_out_of_a_chair_warns_before_the_floor():
    seated = body(hip=(450, 400), knees=((440, 400), (480, 400)), ankles=((440, 520), (480, 520)))
    sliding = body(hip=(470, 480), trunk_deg=-30, knees=((470, 470), (510, 470)), ankles=((480, 560), (520, 560)))
    _, alerts = run(PreventionTracker(), [(4, seated, SIT), (1, sliding, {"label": "Bending", "detail": None})],
                    [CHAIR])
    assert [a.alert_type for a in alerts] == ["chair_slip"]
    assert alerts[0].severity == "high" and "below where they sat" in alerts[0].message


def test_shifting_on_a_chair_or_standing_up_is_not_a_slip():
    seated = body(hip=(450, 400), knees=((440, 400), (480, 400)), ankles=((440, 520), (480, 520)))
    shifted = body(hip=(455, 425), knees=((440, 400), (480, 400)), ankles=((440, 520), (480, 520)))  # +0.25 torso
    standing = body(hip=(450, 250))
    _, alerts = run(PreventionTracker(), [(4, seated, SIT), (3, shifted, SIT),
                                          (2, standing, {"label": "Walking", "detail": None})], [CHAIR])
    assert alerts == []


def test_a_slide_that_reads_as_walking_for_a_moment_is_still_a_slip():
    """Sliding forward fast makes the hips move: the label says Walking for a few frames."""
    seated = body(hip=(450, 400), knees=((440, 400), (480, 400)), ankles=((440, 520), (480, 520)))
    sliding = body(hip=(470, 480), knees=((470, 470), (510, 470)), ankles=((480, 560), (520, 560)))
    _, alerts = run(PreventionTracker(), [(4, seated, SIT), (1, sliding, {"label": "Walking", "detail": None})],
                    [CHAIR])
    assert [a.alert_type for a in alerts] == ["chair_slip"]


LYING_IN_BED = {"label": "Lying down", "detail": "bed", "transition": None}


def test_a_foot_over_the_side_of_the_bed_warns_once_per_bed_stay():
    lying = body(hip=(400, 420), trunk_deg=80, knees=((480, 420), (480, 430)), ankles=((560, 425), (560, 435)))
    foot_out = body(hip=(400, 420), trunk_deg=60, knees=((450, 480), (460, 480)), ankles=((450, 600), (460, 600)))
    tr = PreventionTracker()
    t, alerts = run(tr, [(3, lying, LYING_IN_BED), (1, foot_out, {"label": "Sitting", "detail": "bed"})], [BED])
    assert [a.alert_type for a in alerts] == ["bed_exit"] and "foot over the side" in alerts[0].message
    _, again = run(tr, [(2, foot_out, {"label": "Sitting", "detail": "bed"})], [BED], t)
    assert again == []  # re-arms only after settling in bed again


def test_no_bed_exit_without_settling_in_bed_first_or_while_lying_still():
    lying = body(hip=(400, 420), trunk_deg=80, knees=((480, 420), (480, 430)), ankles=((560, 425), (560, 435)))
    foot_out = body(hip=(400, 420), trunk_deg=60, knees=((450, 480), (460, 480)), ankles=((450, 600), (460, 600)))
    sitting_in_bed = {"label": "Sitting", "detail": "bed"}
    _, brief = run(PreventionTracker(), [(1, lying, LYING_IN_BED), (1, foot_out, sitting_in_bed)], [BED])
    _, still = run(PreventionTracker(), [(10, lying, LYING_IN_BED)], [BED])
    assert brief == [] and still == []


# Sitting on the floor, knees up: the hips are level with the feet. (Kneeling upright scores ~0.6 of a
# leg's length on this measure, like sitting on a chair; the room map's height catches kneeling.)
KNEEL = body(hip=(300, 640), knees=((340, 590), (360, 590)), ankles=((390, 645), (410, 645)))


def test_sitting_on_the_floor_for_a_while_is_on_the_floor_once():
    tr = PreventionTracker()
    t, alerts = run(tr, [(2, KNEEL, {"label": "Bending", "detail": None})], [])
    assert alerts == []  # not yet: under floor_hold_s
    t, alerts = run(tr, [(5, KNEEL, {"label": "Bending", "detail": None})], [], t)
    assert [a.alert_type for a in alerts] == ["on_the_floor"] and alerts[0].severity == "medium"


def test_short_gaps_do_not_restart_the_floor_timer():
    tr = PreventionTracker(PreventionConfig(floor_hold_s=3.0))
    t, alerts = 0.0, []
    for _ in range(4):  # 0.8 s down, then 0.3 s with no pose, repeated
        t, a = run(tr, [(0.8, KNEEL, {"label": "Bending", "detail": None})], [], t)
        alerts += a
        t += 0.3
    assert [a.alert_type for a in alerts] == ["on_the_floor"]


@pytest.mark.parametrize("case", ["on a chair", "fall detector has them", "standing"])
def test_not_on_the_floor(case):
    if case == "on a chair":
        frames, objs, down = [(5, body(hip=(450, 400), knees=((440, 400), (480, 400)), ankles=((440, 440), (480, 440))),
                               SIT)], [CHAIR], ()
    elif case == "fall detector has them":
        frames, objs, down = [(5, KNEEL, {"label": "Fallen", "detail": None})], [], (1,)
    else:
        frames, objs, down = [(5, body(), {"label": "Standing", "detail": None})], [], ()
    _, alerts = run(PreventionTracker(), frames, objs, down=down)
    assert alerts == []


def test_short_box_on_the_floor_by_the_room_maps_standing_height():
    """No legs in view (crawling behind a bag): the box is far shorter than a person standing at that
    spot would be, from the room map."""
    room = SimpleNamespace(width=1280, height=720, fit=(1.0, 0.0, 0.03), standing_height=lambda y: 400.0)
    crawl = body(hip=(300, 560), trunk_deg=80, legs=False)
    tr = PreventionTracker()
    t, alerts = 0.0, []
    for _ in range(50):
        t += 0.1
        alerts += tr.update({1: crawl}, {1: {"label": "Upper body only", "detail": None}}, [], t, set(), room)
    assert [a.alert_type for a in alerts] == ["on_the_floor"] and "standing height" in alerts[0].message


def test_a_seat_stay_starts_only_on_a_detected_seat_but_survives_the_seat_box_dropping_out():
    seated = body(hip=(450, 400), knees=((440, 400), (480, 400)), ankles=((440, 520), (480, 520)))
    sliding = body(hip=(470, 480), knees=((470, 470), (510, 470)), ankles=((480, 560), (520, 560)))
    no_seat = {"label": "Sitting", "detail": None, "transition": None}
    # "Sitting" with no seat under the hips (walking misread under a high camera): never armed.
    _, alerts = run(PreventionTracker(), [(4, seated, no_seat), (1, sliding, {"label": "Bending", "detail": None})],
                    [])
    assert alerts == []
    # On the chair for 1 s, then the chair's box drops out: still the same stay, and the slide counts.
    _, alerts = run(PreventionTracker(), [(1, seated, SIT), (2, seated, no_seat),
                                          (1, sliding, {"label": "Bending", "detail": None})], [CHAIR])
    assert [a.alert_type for a in alerts] == ["chair_slip"]
