"""Safe path (home mode): trip hazards on walking paths, "watch your step", dark room."""

from types import SimpleNamespace

import numpy as np

from activity.trip_hazards import TripHazards

C = 0.9


def person(x, y=600.0, h=400.0):
    """Feet at (x, y), standing; box h tall."""
    k = np.zeros((17, 3), np.float32)
    k[15], k[16] = (x - 15, y, C), (x + 15, y, C)
    return SimpleNamespace(keypoints=k, bbox=(x - 60, y - h, x + 60, y + 5))


def obj(name, box):
    return SimpleNamespace(class_name=name, bbox=box)


BAG = obj("backpack", (500.0, 560.0, 580.0, 610.0))
WALK = {"label": "Walking", "detail": None, "transition": None}


def walk_across(tr, t, objects, x0=200, x1=1000, seconds=4.0, fps=10):
    alerts, n = [], round(seconds * fps)
    for i in range(n):
        t += 1 / fps
        x = x0 + (x1 - x0) * i / n
        alerts += tr.update({1: person(x)}, {1: WALK}, objects, t, brightness=120.0)
    return t, alerts


def test_a_bag_on_the_walking_path_is_a_trip_hazard_once_and_walking_at_it_warns():
    tr = TripHazards()
    t, _ = walk_across(tr, 0.0, [], seconds=8.0)        # learn where people walk (3+ steps per cell)
    alerts = []
    for _ in range(120):                                # the bag lies there 12 s, nobody around
        t += 0.1
        alerts += tr.update({}, {}, [BAG], t, brightness=120.0)
    assert [a.alert_type for a in alerts] == ["trip_hazard"] and "backpack" in alerts[0].message
    t, alerts = walk_across(tr, t, [BAG], x0=200, x1=700, seconds=3.0)
    assert [a.alert_type for a in alerts] == ["watch_step"] and alerts[0].message == "Watch your step: backpack ahead"


def test_not_a_hazard_off_the_path_on_furniture_on_a_person_or_standing_tall():
    tr = TripHazards()
    t, _ = walk_across(tr, 0.0, [], seconds=8.0)
    off_path = obj("backpack", (500.0, 100.0, 580.0, 150.0))
    on_couch = [obj("pillow", (500.0, 560.0, 580.0, 600.0)), obj("couch", (400.0, 450.0, 800.0, 650.0))]
    tall = obj("clothes", (700.0, 300.0, 760.0, 610.0))  # a standing fan read as "clothes"
    worn = obj("clothes", (840.0, 300.0, 960.0, 600.0))  # inside the person standing there
    alerts = []
    for _ in range(150):
        t += 0.1
        alerts += tr.update({2: person(900)}, {2: {"label": "Standing"}}, [off_path, tall, worn, *on_couch], t)
    assert [a for a in alerts if a.alert_type == "trip_hazard"] == []


def test_a_wobbling_or_briefly_hidden_object_is_reported_once():
    tr = TripHazards()
    t, _ = walk_across(tr, 0.0, [], seconds=8.0)
    alerts = []
    for i in range(300):
        t += 0.1
        if 150 <= i < 250:
            seen = []  # hidden behind people for 10 s
        else:
            d = 6 * np.sin(i)  # the box wobbles a few pixels
            seen = [obj("clothes", (500 + d, 560.0, 600 + d, 610.0))]
        alerts += tr.update({}, {}, seen, t)
    assert [a.alert_type for a in alerts] == ["trip_hazard"]


def test_standing_up_in_the_dark_says_turn_on_a_light():
    tr = TripHazards()
    dark = tr.update({1: person(500)}, {1: {"label": "Sitting", "transition": "sit-to-stand"}}, [], 1.0,
                     brightness=20.0, risky_moves=[(1, "standing up")])
    lit = TripHazards().update({1: person(500)}, {1: {}}, [], 1.0, brightness=150.0, risky_moves=[(1, "standing up")])
    assert [a.alert_type for a in dark] == ["dark_room"]
    assert dark[0].message == "It's dark: turn on a light before standing up"
    assert lit == []
