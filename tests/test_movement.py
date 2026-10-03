"""Movement coach: what counts as moving, breaks, static time, the reminder, and today's totals."""

import json
import math
import os
from datetime import datetime

import numpy as np
import pytest

from posture.history import PostureHistory
from posture.holds import HEAD_DOWN, LEAN, HoldConfig, HoldTracker, extremes
from posture.movement import AWAY, LONG_STILL, MOVING, STILL, MovementConfig, MovementTracker


def body(x=320.0, y=300.0, width=120.0, tilt_deg=0.0, conf=0.9):
    """Keypoints with the shoulders placed by midpoint, width and tilt (plus a nose)."""
    k = np.zeros((17, 3), np.float32)
    dx = width / 2 * math.cos(math.radians(tilt_deg))
    dy = width / 2 * math.sin(math.radians(tilt_deg))
    k[0] = (x, y - 100, conf)
    k[5] = (x - dx, y - dy, conf)
    k[6] = (x + dx, y + dy, conf)
    return k


def run(m, make, t, seconds, fps=10, rng=None, jitter=0.0):
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        k = make()
        if rng is not None and jitter:
            k[:, :2] += rng.normal(0, jitter, (17, 2)).astype(np.float32)
        m.update(k, t)
    return t


T0 = datetime(2026, 10, 5, 9, 0).timestamp()


def test_typing_fidgeting_and_jitter_are_not_movement():
    m = MovementTracker(now=T0)
    rng = np.random.default_rng(0)
    t = run(m, body, T0, 120, rng=rng, jitter=4)  # keypoint jitter
    t = run(m, lambda: body(x=335), t, 30)  # a small shift: 0.125 shoulder widths
    t = run(m, lambda: body(tilt_deg=6), t, 30)  # a slight lean
    # Ordinary seated posture changes (these counted as movement with the first thresholds):
    t = run(m, lambda: body(x=370), t, 30)  # 0.42 shoulder widths
    t = run(m, lambda: body(tilt_deg=12), t, 30)
    t = run(m, lambda: body(width=145), t, 30)  # leaned in a little (21%)
    assert m.still_s(t) == pytest.approx(270, abs=1) and m.state(t) == STILL


@pytest.mark.parametrize("moved", [
    lambda: body(x=392),          # torso shift of 0.6 shoulder widths (moved the chair, stood partly)
    lambda: body(tilt_deg=25),    # a lean change of 25 degrees
    lambda: body(width=170),      # leaned far in (shoulders 42% wider on screen)
])
def test_a_large_position_change_held_8_s_is_movement(moved):
    m = MovementTracker(now=T0)
    t = run(m, body, T0, 100)
    assert m.still_s(t) > 99 and m.state(t) == STILL  # never "moving" just because the session started
    t = run(m, moved, t, 6)  # not yet: under 8 s
    assert m.still_s(t) > 99
    t = run(m, moved, t, 6)
    assert m.still_s(t) < 12 and m.state(t) == MOVING  # dated from when the change started
    t = run(m, moved, t, 15)  # staying in the new position is stillness again
    assert m.state(t) == STILL


def test_shifting_back_and_forth_quickly_is_not_movement():
    m = MovementTracker(now=T0)
    t = run(m, body, T0, 60)
    for _ in range(10):  # leaning over for 1 s at a time (reaching for something)
        t = run(m, lambda: body(x=380), t, 1)
        t = run(m, body, t, 2)
    assert m.still_s(t) == pytest.approx(90, abs=1)


def test_walking_around_in_view_is_movement():
    """Up and walking around in front of the camera: big changes that keep crossing back near
    the starting spot. Not held in one place, but clearly moving."""
    m = MovementTracker(now=T0)
    t = run(m, body, T0, 100)
    xs = [320, 420, 520, 420, 320, 260, 320, 420, 520, 420]
    for x in xs * 2:  # about 2 s per stride position
        t = run(m, lambda x=x: body(x=x, width=100), t, 2)
    assert m.still_s(t) < 40 and m.last_change is not None


def test_a_detection_dropout_is_not_getting_up_but_leaving_is():
    m = MovementTracker(now=T0)
    t = run(m, body, T0, 100)
    t = run(m, lambda: None, t, 5)  # lost for 5 s
    t = run(m, body, t, 1)
    assert m.still_s(t) > 100
    t = run(m, lambda: None, t, 25)  # away 25 s: got up
    assert m.state(t) == AWAY and m.still_s(t) == 0
    t = run(m, body, t, 1)
    assert m.still_s(t) < 1.1 and m.today.breaks == 0  # moved, but too short for a break
    t = run(m, lambda: None, t, 90)
    t = run(m, body, t, 1)
    assert m.today.breaks == 1


def test_face_visible_but_shoulders_unclear_is_still_present():
    m = MovementTracker(now=T0)
    t = run(m, body, T0, 50)
    for _ in range(300):  # 30 s of dim light: someone is there
        t += 0.1
        m.update(None, t, seen=True)
    assert m.state(t) != AWAY and m.still_s(t) == pytest.approx(80, abs=1)


def test_static_time_counts_only_stretches_over_10_minutes():
    m = MovementTracker(MovementConfig(static_min_s=600), now=T0)
    t = run(m, body, T0, 300, fps=2)  # 5 min still
    t = run(m, lambda: body(x=400), t, 14, fps=2)  # move
    assert m.snapshot(t)["static_today_s"] == 0
    t = run(m, lambda: body(x=400), t, 700, fps=2)  # 11+ min still
    snap = m.snapshot(t)
    # Ongoing, and counted from when the shift began (some seconds before the 700 s).
    assert m.state(t) == LONG_STILL and snap["static_today_s"] == pytest.approx(712, abs=8)
    t = run(m, body, t, 14, fps=2)  # move: the stretch is counted for good
    snap = m.snapshot(t)
    assert snap["static_today_s"] == pytest.approx(714, abs=8)
    assert snap["longest_still_s"] == pytest.approx(snap["static_today_s"], abs=8)


def test_reminder_after_the_still_time_and_moving_resets_it():
    m = MovementTracker(MovementConfig(reminder_s=1800, snooze_s=600), now=T0)
    t = run(m, body, T0, 1799, fps=1)
    assert not m.reminder_offered
    t = run(m, body, t, 2, fps=1)
    assert m.reminder_offered and m.reminder_id == 1
    t = run(m, lambda: body(x=400), t, 14)
    assert not m.reminder_offered  # moved
    t = run(m, lambda: body(x=400), t, 1800, fps=1)
    assert m.reminder_offered and m.reminder_id == 2
    m.snooze(t)
    t = run(m, lambda: body(x=400), t, 599, fps=1)
    assert not m.reminder_offered
    t = run(m, lambda: body(x=400), t, 2, fps=1)
    assert m.reminder_offered
    m.dismiss(t)  # skip: nothing until you've moved and been still for the full time again
    t = run(m, lambda: body(x=400), t, 3600, fps=1)
    assert not m.reminder_offered
    t = run(m, body, t, 14)
    t = run(m, body, t, 1801, fps=1)
    assert m.reminder_offered


def test_today_survives_a_restart_and_rolls_over_at_midnight(tmp_path):
    h = PostureHistory(str(tmp_path / "h.db"))
    m = MovementTracker(MovementConfig(static_min_s=60), history=h, now=T0)
    t = run(m, body, T0, 90, fps=2)
    t = run(m, lambda: body(x=400), t, 14, fps=2)  # a ~90 s stretch: static
    m.finish_break(t, completed=True)
    again = MovementTracker(MovementConfig(static_min_s=60), history=h, now=t)
    assert again.today.breaks == 1 and again.today.static_s == pytest.approx(92, abs=4)
    tomorrow = MovementTracker(history=h, now=T0 + 86400)
    assert tomorrow.today.breaks == 0 and tomorrow.today.static_s == 0
    h.close()


def test_movement_time_is_kept_apart_from_posture_totals(tmp_path):
    h = PostureHistory(str(tmp_path / "h.db"))
    m = MovementTracker(history=h, now=T0)
    run(m, body, T0, 30)
    h.add_seconds(T0, "good", 30)
    h.flush()
    assert set(h.totals(T0 - 60)) == {"good"}
    h.close()


# --- long holds ---------------------------------------------------------------------------------

REF = {"head_ratio": 0.8, "tilt_deg": 0.0, "lateral": 0.0}


@pytest.mark.parametrize(("args", "head", "lean"), [
    ((0.75, 2.0, 0.05, None), False, False),          # ordinary sitting
    ((0.50, 2.0, 0.05, None), True, False),           # head far down (62%)
    ((0.65, 2.0, 0.05, None), False, False),          # a normal slouch (81%) is not an extreme
    ((0.65, 2.0, 0.05, "slouching"), True, False),    # ...unless the model also says slouching
    ((0.75, 16.0, 0.05, None), False, True),          # strong tilt
    ((0.75, 9.0, 0.50, None), False, True),           # head far off to one side
    ((0.75, 11.0, 0.05, "leaning_left"), False, True),
    ((0.75, 11.0, 0.05, None), False, False),         # a moderate lean alone is fine
])
def test_extremes(args, head, lean):
    f = extremes(*args[:3], REF, HoldConfig(), args[3])
    assert (f[HEAD_DOWN], f[LEAN]) == (head, lean)


def test_hold_tracker_needs_the_full_time_and_tolerates_short_dips():
    h = HoldTracker(HoldConfig(head_down_s=1200, interrupt_s=60))
    t = 0.0
    down, up = {HEAD_DOWN: True, LEAN: False}, {HEAD_DOWN: False, LEAN: False}
    for _ in range(1100):
        t += 1
        assert h.update(down, t) == []
    for _ in range(50):  # 50 s up: under a minute
        t += 1
        h.update(up, t)
    for _ in range(100):
        t += 1
        h.update(down, t)
    assert HEAD_DOWN in h.warned
    h.reset(HEAD_DOWN)
    for _ in range(61):
        t += 1
        h.update(up, t)
    assert h.since[HEAD_DOWN] is None
    for _ in range(10):  # can't judge (away from view): nothing changes
        t += 1
        h.update(None, t)
    assert h.since[HEAD_DOWN] is None


def test_real_seated_session_counts_as_still():
    """Real laptop-webcam measures (one person, seated: lean right, upright, slouching, moving
    naturally), played twice: about two minutes of ordinary sitting. The first thresholds found
    five "movements" here, so the still timer kept resetting and the reminder never came."""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "real_seated_shoulders.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    span = rows[-1][0] + 0.05
    series = [(r * span + ts, w, tilt) for r in range(2) for ts, w, tilt in rows]

    def replay(cfg):
        m = MovementTracker(cfg, now=T0)
        moves, last = 0, None
        for ts, w, tilt in series:
            m.update(body(width=400 * w, tilt_deg=tilt), T0 + ts)
            if m.last_change is not last:
                moves, last = moves + 1, m.last_change
        return moves, m.still_s(T0 + series[-1][0])

    moves, still = replay(MovementConfig())
    assert moves == 0 and still > 115
    first = MovementConfig(shift_widths=0.35, lean_change_deg=10, width_change=0.15, hold_s=3, smooth_s=1)
    assert replay(first)[0] >= 3  # the bug, reproduced
