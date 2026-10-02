"""Posture coaching: fix guidance (ghost alignment, quick "back to good", time-to-correct), local
history and root-cause tips, and camera-checked stretch breaks (rep counting)."""

import math
from datetime import datetime

import numpy as np
import pytest

from posture.breaks import BreakConfig, BreakRoutine, BreakScheduler, RepCounter
from posture.coach import GOOD, SLOUCHING, PostureCoach, PostureConfig
from posture.guidance import CorrectionTracker, instruction, median_skeleton, normalise, place_ghost
from posture.history import PostureHistory, current_tip, find_tips
from test_posture import LEAN, SLOUCH, UPRIGHT, coach_with_baseline, kp, run

# --- ghost alignment ----------------------------------------------------------------------------


def test_ghost_is_your_good_skeleton_on_your_current_shoulders():
    ghost = normalise(UPRIGHT)
    assert ghost[5] == pytest.approx((-0.5, 0)) and ghost[0] == pytest.approx((0, -100 / 120))
    # Moved and further away: the ghost follows the current shoulder midpoint and width.
    far = kp(nose=(450, 300), shoulders=((420, 350), (480, 350)), eyes=((442.5, 295), (457.5, 295)))
    pts = place_ghost(ghost, far)
    assert pts[5] == pytest.approx((420, 350)) and pts[6] == pytest.approx((480, 350))
    assert pts[0] == pytest.approx((450, 300))  # same proportions as Good: lines up exactly


def test_ghost_shows_where_the_head_should_be_when_slouching():
    pts = place_ghost(normalise(UPRIGHT), SLOUCH)
    assert pts[0][1] < SLOUCH[0, 1] - 40  # ghost nose well above the slouched nose
    assert pts[5] == pytest.approx(tuple(SLOUCH[5, :2]))


def test_ghost_shoulders_are_level_when_you_lean():
    pts = place_ghost(normalise(UPRIGHT), LEAN)
    assert pts[5][1] == pytest.approx(pts[6][1])  # Good is level even though you're tilted
    mid = (LEAN[5, :2] + LEAN[6, :2]) / 2
    assert ((pts[5][0] + pts[6][0]) / 2, pts[5][1]) == pytest.approx(tuple(mid))


def test_ghost_is_smaller_when_too_close():
    close = kp(nose=(320, 180), shoulders=((230, 310), (410, 310)), eyes=((298, 168), (342, 168)))
    pts = place_ghost(normalise(UPRIGHT), close, scale_px=120)
    assert pts[6][0] - pts[5][0] == pytest.approx(120)  # your Good size, smaller than now (180)


def test_median_skeleton_ignores_rare_points_and_json_keys():
    a, b = normalise(UPRIGHT), normalise(kp(ears=((290, 205), (350, 205))))
    sk = median_skeleton([{str(k): v for k, v in a.items()}, a, b])
    assert 3 not in sk and sk[0] == pytest.approx(a[0])  # ears seen in 1 of 3 frames
    assert median_skeleton([None, {}]) is None


def test_instruction_follows_the_measure_furthest_off():
    cfg = PostureConfig()
    rows = [{"key": "head_ratio", "triggered": True, "change": 0.5},
            {"key": "tilt_deg", "triggered": True, "change": cfg.tilt_deg + 0.5}]
    assert instruction("slumped", rows, cfg) == "Sit back and lift your head"
    rows[1]["change"] = cfg.tilt_deg * 3
    assert instruction("slumped", rows, cfg) == "Level your shoulders"
    assert instruction("too_close") == "Move back from the screen"
    assert instruction("good") is None


# --- quick "back to good" and time-to-correct ------------------------------------------------------

def test_tracker_times_the_correction_from_when_poor_was_shown():
    tr = CorrectionTracker()
    t = 100.0
    for _ in range(30):  # shown as slouching, frames still poor
        t += 0.1
        assert tr.update("slouching", t, False) is None
    start = 100.1
    tr.note_reminder()
    done = None
    for _ in range(15):  # sitting up: the official status is still slouching
        t += 0.1
        done = done or tr.update("slouching", t, True)
    assert done["seconds"] == pytest.approx(t - start - 0.5, abs=0.3) and done["after_reminder"] is True
    assert not tr.ghost_visible and tr.recent(t)["id"] == 1
    assert tr.update("good", t + 1, None) is None  # the official status catching up adds nothing
    assert tr.recent(t + 10) is None


def test_tracker_brings_the_ghost_back_on_a_relapse():
    tr = CorrectionTracker()
    t = 0.0
    for ok in [*[False] * 10, *[True] * 12]:
        t += 0.1
        tr.update("slouching", t, ok)
    assert not tr.ghost_visible
    for _ in range(35):  # slumps again before the official status changed
        t += 0.1
        tr.update("slouching", t, False)
    assert tr.ghost_visible


def test_jittery_frames_do_not_count_as_corrected():
    tr = CorrectionTracker()
    t = 0.0
    for i in range(40):
        t += 0.1
        assert tr.update("slouching", t, i % 2 == 0) is None  # half good, half poor


def test_coach_back_to_good_comes_before_the_official_status(tmp_path):
    c, t = coach_with_baseline()
    assert c.baseline.skeleton
    t = run(c, SLOUCH, t, 4)
    snap = c.snapshot()
    assert snap["status"] == SLOUCHING and snap["guidance"]["active"]
    assert snap["guidance"]["instruction"] == "Sit back and lift your head"
    ghost = c.ghost()
    assert ghost and ghost["points"][0][1] < SLOUCH[0, 1]
    shown_at = c._status_since
    corrected_at = None
    for _ in range(30):
        t += 0.1
        c.update(UPRIGHT, t)
        if corrected_at is None and c.snapshot()["back_to_good"]:
            corrected_at = t
            assert c.status == SLOUCHING and c.ghost() is None  # official status not yet changed
    assert corrected_at is not None
    btg = c.snapshot()["back_to_good"]
    assert btg["seconds"] == pytest.approx(corrected_at - shown_at, abs=0.05) and btg["posture"] == SLOUCHING
    assert c.status == GOOD and btg["id"] == 1  # the official switch didn't record a second correction


def test_no_ghost_without_a_recorded_good_skeleton():
    c, t = coach_with_baseline()
    c.baseline.skeleton = None  # an old baseline, recorded before the ghost existed
    run(c, SLOUCH, t, 4)
    snap = c.snapshot()
    assert snap["guidance"]["active"] and not snap["guidance"]["ghost_available"] and c.ghost() is None
    assert snap["guidance"]["instruction"]


def test_classifier_mode_quick_check_and_calibration_ghost(tmp_path):
    from test_posture_classifier import calibrated_coach, feed

    rng = np.random.default_rng(21)
    c, t = calibrated_coach(tmp_path, rng)
    c.train_model()
    assert c.ghost_good is not None
    t = feed(c, "good", t, 10, rng)
    t = feed(c, "slouching", t, 16, rng)
    assert c.status == SLOUCHING and c.ghost() is not None
    t = feed(c, "good", t, 1.5, rng)
    assert c.snapshot()["back_to_good"] and c.status == SLOUCHING  # 5 s stability still running


# --- history and tips --------------------------------------------------------------------------

def at(day, hour, minute=30):
    return datetime(2026, 9, day, hour, minute).timestamp()


def fill(h, day, hour, **seconds):
    for status, s in seconds.items():
        h.add_seconds(at(day, hour), status, s)


@pytest.fixture
def hist(tmp_path):
    h = PostureHistory(str(tmp_path / "h.db"), clock=lambda: at(28, 20))
    yield h
    h.close()


def rules(h):
    since = at(28, 20) - 7 * 86400
    return [t["rule"] for t in find_tips(h.totals(since), h.by_hour(since))]


def test_no_tips_without_two_hours_of_data(hist):
    fill(hist, 27, 10, looking_down=3000, good=1000)
    assert rules(hist) == [] and current_tip(hist) is None


def test_screen_low_tip_with_its_evidence(hist):
    fill(hist, 27, 10, good=6480, looking_down=2520, slouching=1800)  # 3 h, 40% low
    tip = current_tip(hist)
    assert tip["rule"] == "screen_low" and "40%" in tip["evidence"] and "3.0 h" in tip["evidence"]
    assert "keyboard" in tip["advice"]


def test_screen_low_quiet_below_its_share(hist):
    fill(hist, 27, 10, good=9000, looking_down=1800)
    assert "screen_low" not in rules(hist)


def test_too_close_tip(hist):
    fill(hist, 27, 10, good=8000, too_close=2000)
    assert rules(hist) == ["too_close"] and "font" in current_tip(hist)["advice"]


def test_lean_side_tip_needs_mostly_one_side(hist):
    fill(hist, 27, 10, good=8820, leaning_left=1800, leaning_right=180)
    tip = current_tip(hist)
    assert tip["rule"] == "lean_side" and "left" in tip["title"] and "91%" in tip["evidence"]


def test_lean_balanced_sides_gives_no_tip(hist):
    fill(hist, 27, 10, good=8000, leaning_left=1000, leaning_right=1000)
    assert "lean_side" not in rules(hist)


def test_time_of_day_tip(hist):
    for day in (22, 23, 24, 25):
        for hour in range(9, 15):
            fill(hist, day, hour, good=2160, slouching=240)  # 10% poor
        for hour in (15, 16):
            fill(hist, day, hour, good=1200, slouching=1200)  # 50% poor
    tip = current_tip(hist)
    assert tip["rule"] == "time_of_day" and "between 3pm and 5pm" in tip["title"]
    assert "50%" in tip["evidence"] and "20%" in tip["evidence"]


def test_time_of_day_needs_three_days(hist):
    for day in (24, 25):
        for hour in range(9, 15):
            fill(hist, day, hour, good=6480, slouching=720)
        fill(hist, day, 15, good=1200, slouching=1200)
    assert "time_of_day" not in rules(hist)


def test_strongest_tip_first_and_dismissal(hist):
    fill(hist, 27, 10, good=4000, too_close=3000, looking_down=3800)  # too close 28%, low 35%
    assert rules(hist) == ["too_close", "screen_low"]
    hist.dismiss("too_close", at(28, 20) + 7 * 86400)
    assert current_tip(hist)["rule"] == "screen_low"
    hist.dismiss("too_close", at(28, 19))  # expired
    assert current_tip(hist)["rule"] == "too_close"


def test_history_persists_across_restarts_with_lean_side(tmp_path):
    path = str(tmp_path / "posture_baseline_laptop.json")
    c = PostureCoach(PostureConfig(), baseline_path=path)
    t = 1_790_000_000.0
    c.update(UPRIGHT, t)
    c.start_baseline(now=t)
    t = run(c, UPRIGHT, t, 7)
    t = run(c, SLOUCH, t, 8)
    t = run(c, UPRIGHT, t, 4)  # corrected
    t = run(c, LEAN, t, 8)
    c.history.close()
    again = PostureHistory(str(tmp_path / "posture_history_laptop.db"))
    totals = again.totals(t - 3600)
    assert totals["slouching"] > 3 and totals["good"] > 0
    assert any(k.startswith("leaning_") for k in totals)  # stored with your side
    (fix,) = again.corrections(t - 3600)
    # Shown ~2 s into the 8 s slouch, corrected ~1 s after sitting up: about 7 s.
    assert fix["posture"] == SLOUCHING and 6 < fix["seconds"] < 8
    again.close()


# --- stretch breaks ------------------------------------------------------------------------------

def test_rep_counter_ignores_jitter_around_one_threshold():
    r = RepCounter(15, 6)
    for v in (14.5, 15.5, 14.8, 15.2, 14.9, 16, 8, 7):  # past the line and wobbling, not back to centre
        r.update(v)
    assert r.count == 0
    r.update(3)
    assert r.count == 1
    for v in (10, 12, 14, 9, 7):  # never far enough
        r.update(v)
    assert r.count == 1


def tilted(deg, base=UPRIGHT):
    k = base.copy()
    c = (k[1, :2] + k[2, :2]) / 2
    a = math.radians(deg)
    for i in (1, 2):
        d = k[i, :2] - c
        k[i, :2] = c + np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
    return k


def shifted_up(k, dy):
    out = k.copy()
    out[:7, 1] -= dy
    return out


def shrug(k=UPRIGHT, rise=20):
    out = k.copy()
    out[5:7, 1] -= rise
    return out


def play(r, frames, t, seconds, fps=10):
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        r.update(frames, t)
    return t


def test_full_break_counts_reps_and_confirms_each_step():
    r = BreakRoutine(0.0)
    t = play(r, UPRIGHT, 0.0, 3.1)  # get ready: the reference
    assert r.ref["roll"] == pytest.approx(0)
    for _ in range(3):
        t = play(r, tilted(20), t, 1)
        t = play(r, UPRIGHT, t, 1)
        t = play(r, tilted(-20), t, 1)
        t = play(r, UPRIGHT, t, 1)
    assert r.tilt_a.count == 3 and r.tilt_b.count == 3 and r.step == 1
    snap = r.snapshot(t)
    assert snap["label"] == "Shoulder shrugs" and snap["counts"] == {"reps": [0, 5]}
    for _ in range(5):
        t = play(r, shrug(), t, 1)
        t = play(r, UPRIGHT, t, 1)
    assert r.shrug.count == 5 and r.step == 2
    t = play(r, shifted_up(UPRIGHT, 110), t, 1)
    assert r.done and r.result()["outcome"] == "completed"


def test_small_tilts_and_shrugs_are_not_counted():
    r = BreakRoutine(0.0)
    t = play(r, UPRIGHT, 0.0, 3.1)
    for _ in range(4):
        t = play(r, tilted(9), t, 1)
        t = play(r, UPRIGHT, t, 1)
    assert r.tilt_a.count == 0


def test_standing_out_of_view_or_the_manual_button():
    r = BreakRoutine(0.0)
    t = play(r, UPRIGHT, 0.0, 3.1)
    r.step, r.step_start = 2, t  # straight to "stand up"
    t = play(r, None, t, 2.5)  # left the picture
    assert r.done and r.stood
    m = BreakRoutine(0.0)
    t = play(m, UPRIGHT, 0.0, 3.1)
    m.step, m.step_start = 2, t
    m.stood_up(t)
    assert m.done and m.result()["steps"]["stand_up"]


def test_a_step_left_unfinished_moves_on_and_is_partial():
    r = BreakRoutine(0.0, BreakConfig(step_max_s=5))
    t = play(r, UPRIGHT, 0.0, 3.1)
    t = play(r, UPRIGHT, t, 5.5)
    assert r.step == 1 and r.completed["neck_tilts"] is False
    play(r, UPRIGHT, t, 30)
    assert r.done and r.result()["outcome"] == "partial"


def test_scheduler_offer_snooze_skip_and_away_reset():
    s = BreakScheduler(BreakConfig(sit_minutes=50, away_reset_s=300, snooze_s=600))
    t = 0.0
    offered = False
    for _ in range(50 * 60):
        t += 1
        offered = s.update(True, 1, t) or offered
    assert offered and s.offered
    s.snooze(t)
    for _ in range(599):
        t += 1
        s.update(True, 1, t)
    assert not s.offered
    t += 2
    s.update(True, 1, t)
    assert s.offered
    s.reset()  # skipped
    assert s.sit_s == 0 and not s.offered
    for _ in range(40 * 60):
        t += 1
        s.update(True, 1, t)
    for _ in range(301):  # a real break away from the desk
        t += 1
        s.update(False, 1, t)
    assert s.sit_s == 0


def test_coach_offers_runs_and_logs_a_break(tmp_path):
    path = str(tmp_path / "posture_baseline_laptop.json")
    c = PostureCoach(PostureConfig(), baseline_path=path)
    t = 1_790_000_000.0
    c.update(UPRIGHT, t)
    c.start_baseline(now=t)
    t = run(c, UPRIGHT, t, 7)
    c.set_break_minutes(0.2)  # 12 s at the desk (the baseline recording itself isn't counted)
    t = run(c, UPRIGHT, t, 11)
    assert not c.snapshot()["break"]["offered"]
    t = run(c, UPRIGHT, t, 2)
    assert c.snapshot()["break"]["offered"]
    c.start_break(now=t)
    t = run(c, UPRIGHT, t, 3.1)
    for _ in range(3):
        t = run(c, tilted(20), t, 1)
        t = run(c, UPRIGHT, t, 1)
        t = run(c, tilted(-20), t, 1)
        t = run(c, UPRIGHT, t, 1)
    snap = c.snapshot()
    assert snap["break"]["routine"]["label"] == "Shoulder shrugs"
    good_before = snap["session"]["seconds"]["good"]
    for _ in range(5):
        t = run(c, shrug(), t, 1)
        t = run(c, UPRIGHT, t, 1)
    c.stood_up(now=t)
    snap = c.snapshot()
    assert snap["break"]["routine"] is None and snap["break"]["result"]["outcome"] == "completed"
    assert snap["break"]["sit_min"] == 0 and not snap["break"]["offered"]
    assert snap["session"]["seconds"]["good"] == good_before  # break time isn't posture time
    # The setting is saved with the history.
    again = PostureCoach(PostureConfig(), baseline_path=path)
    assert again.breaks.cfg.sit_minutes == 0.2


@pytest.fixture
def coaching_api(tmp_config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import server

    coach = PostureCoach(PostureConfig(), baseline_path=str(tmp_path / "posture_baseline_laptop.json"))
    primary = SimpleNamespace(config=tmp_config, mode="posture", paused=False, posture=coach)
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.allow_remote_camera_control = True
    server._cameras.clear()
    with TestClient(server.app) as client:
        client.coach = coach
        yield client


def test_coaching_endpoints(coaching_api):
    api, coach = coaching_api, coaching_api.coach
    assert api.get("/api/posture/tips").json() == {"tip": None}
    now = datetime.now().timestamp()
    coach.history.add_seconds(now - 600, "good", 6000)
    coach.history.add_seconds(now - 600, "too_close", 3000)
    assert api.get("/api/posture/tips").json()["tip"]["rule"] == "too_close"
    assert api.post("/api/posture/tips/too_close/dismiss", json={}).json() == {"tip": None}
    days = api.get("/api/posture/history?days=7").json()["days"]
    assert days and days[-1]["seconds"]["too_close"] == 3000
    assert api.put("/api/posture/break/settings", json={"sit_minutes": 45}).json()["sit_minutes"] == 45
    assert api.put("/api/posture/break/settings", json={"sit_minutes": 1}).status_code == 422
    coach.update(UPRIGHT, now)
    snap = api.post("/api/posture/break/start", json={}).json()
    assert snap["break"]["routine"]["phase"] == "get_ready" and "not medical advice" in snap["break"]["note"]
    assert api.post("/api/posture/break/cancel", json={}).json()["break"]["result"]["outcome"] == "cancelled"
    assert api.post("/api/posture/break/snooze", json={}).json()["break"]["snoozed"] is True
    assert api.post("/api/posture/break/skip", json={}).status_code == 200
    assert api.post("/api/posture/break/dance", json={}).status_code == 404


def test_pipeline_draws_the_ghost_faintly():
    from core.pipeline import SentinelPipeline
    from posture.guidance import GHOST_EDGES

    frame = np.zeros((480, 640, 3), np.uint8)
    SentinelPipeline._draw_ghost(frame, None)
    assert frame.max() == 0
    SentinelPipeline._draw_ghost(frame, {"points": place_ghost(normalise(UPRIGHT), SLOUCH), "edges": GHOST_EDGES})
    assert 0 < frame.max() < 255  # drawn, but blended (faint)
