"""Desk posture coach: measurements, classification vs baseline, smoothing/hysteresis, sessions,
and the API (synthetic keypoints, fake pipeline)."""

import numpy as np
import pytest

from posture.coach import (
    AWAY,
    GOOD,
    LEANING,
    NO_BASELINE,
    SLOUCHING,
    TOO_CLOSE,
    PostureCoach,
    PostureConfig,
    measure,
)


def kp(nose=(320, 200), shoulders=((260, 300), (380, 300)), eyes=((305, 190), (335, 190)), conf=0.9, ears=None):
    k = np.zeros((17, 3), np.float32)
    k[0] = (*nose, conf)
    k[1], k[2] = (*eyes[0], conf), (*eyes[1], conf)
    if ears:
        k[3], k[4] = (*ears[0], conf), (*ears[1], conf)
    k[5], k[6] = (*shoulders[0], conf), (*shoulders[1], conf)
    return k


def shifted(points, dx=0.0, dy=0.0):
    """The same pose moved in the frame (chair moved / sitting off-centre)."""
    k = points.copy()
    seen = k[:, 2] > 0
    k[seen, 0] += dx
    k[seen, 1] += dy
    return k


UPRIGHT = kp()
SLOUCH = kp(nose=(320, 255), eyes=((305, 245), (335, 245)))             # head dropped toward shoulders
HUNCH = kp(shoulders=((272, 300), (368, 300)))                         # shoulders 20% narrower vs face
LEAN = kp(nose=(320, 200), shoulders=((260, 285), (380, 315)))         # shoulder line tilted ~14 deg
SHIFT = kp(nose=(360, 200), eyes=((345, 190), (375, 190)))             # head moved sideways
CLOSE = kp(nose=(320, 180), shoulders=((230, 310), (410, 310)), eyes=((298, 168), (342, 168)))  # all bigger


def test_measurements_are_ratios_of_shoulder_width():
    m = measure(UPRIGHT)
    assert m.shoulder_width == 120 and m.head_ratio == pytest.approx(100 / 120)
    assert m.tilt_deg == pytest.approx(0) and m.lateral == pytest.approx(0) and m.face_size == 30
    far = measure(kp(nose=(320, 250), shoulders=((290, 300), (350, 300)), eyes=((312.5, 245), (327.5, 245))))
    assert far.head_ratio == pytest.approx(m.head_ratio) and far.face_ratio == pytest.approx(m.face_ratio)


def test_missing_shoulders_or_nose_gives_no_measurement():
    k = UPRIGHT.copy()
    k[6, 2] = 0.1
    assert measure(k) is None


def coach_with_baseline(frames=UPRIGHT, cfg=None):
    c = PostureCoach(cfg or PostureConfig())
    t = 0.0
    c.update(frames, t)
    c.start_baseline(now=t)
    for _ in range(70):  # 3 s get-ready + 3 s recording + margin, at 10 fps
        t += 0.1
        c.update(frames, t)
    assert c.baseline is not None
    return c, t


def run(c, frames, t, seconds, fps=10):
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        c.update(frames, t)
    return t


@pytest.mark.parametrize(("pose", "status"), [(UPRIGHT, GOOD), (SLOUCH, SLOUCHING), (HUNCH, SLOUCHING),
                                              (LEAN, LEANING), (SHIFT, LEANING), (CLOSE, TOO_CLOSE)])
def test_each_posture_is_recognised(pose, status):
    c, t = coach_with_baseline()
    run(c, pose, t, 4)
    snap = c.snapshot()
    assert snap["status"] == status, snap
    assert (snap["reasons"] == []) == (status == GOOD)


def test_status_needs_to_hold_before_it_changes():
    c, t = coach_with_baseline()
    t = run(c, SLOUCH, t, 1.5)  # 1 s smoothing + < 2 s hold
    assert c.status == GOOD
    t = run(c, SLOUCH, t, 2.0)
    assert c.status == SLOUCHING
    t = run(c, UPRIGHT, t, 0.5)  # a brief correction doesn't flip it back immediately
    assert c.status == SLOUCHING


def test_held_time_session_totals_and_timeline():
    c, t = coach_with_baseline()
    c.reset_session()
    t = run(c, UPRIGHT, t, 20)
    t = run(c, SLOUCH, t, 30)
    snap = c.snapshot()
    assert snap["status"] == SLOUCHING and 25 <= snap["held_s"] <= 29
    s = snap["session"]
    assert 20 <= s["good_s"] <= 24 and 25 <= s["poor_s"] <= 30
    assert 0.4 < s["good_fraction"] < 0.5
    statuses = [b["status"] for b in c.timeline()]
    assert statuses[:2] == [GOOD, GOOD] and statuses[-1] == SLOUCHING and len(statuses) >= 4


def test_away_and_no_baseline():
    c = PostureCoach()
    c.update(UPRIGHT, 0.0)
    assert c.snapshot()["status"] == NO_BASELINE
    c, t = coach_with_baseline()
    t = run(c, None, t, 3)
    assert c.snapshot()["status"] == AWAY


def test_baseline_needs_a_visible_person():
    c = PostureCoach()
    c.start_baseline(now=0.0)
    for i in range(70):
        c.update(None, i * 0.1)
    snap = c.snapshot()
    assert c.baseline is None and "shoulders" in snap["calibration_error"]


def test_baseline_is_saved_and_reloaded(tmp_path):
    path = str(tmp_path / "posture_baseline.json")
    c = PostureCoach(baseline_path=path)
    c.start_baseline(now=0.0)
    for i in range(70):
        c.update(UPRIGHT, i * 0.1)
    again = PostureCoach(baseline_path=path)
    assert again.baseline is not None and again.baseline.shoulder_width == 120
    again.clear_baseline()
    assert PostureCoach(baseline_path=path).baseline is None


def test_calibration_snapshot_counts_down():
    c = PostureCoach()
    c.update(UPRIGHT, 0.0)
    c.start_baseline(now=0.0)
    c.update(UPRIGHT, 1.0)
    snap = c.snapshot()
    assert snap["status"] == "calibrating" and snap["calibration_left_s"] == pytest.approx(2.0)


# --- pipeline + API -------------------------------------------------------------------------------

def test_posture_mode_runs_the_coach_and_no_alerts(tmp_config):
    """The pipeline's mode switch, with the real classes but no models (process_frame stubbed)."""
    from types import SimpleNamespace

    from core.pipeline import FrameResult, SentinelPipeline
    from core.pose_estimator import PoseResult

    p = SentinelPipeline.__new__(SentinelPipeline)
    p.mode = "posture"
    p.posture = PostureCoach()
    p.frame_count = 0
    p.total_alerts = 0
    p._on_frame = None
    p.config = tmp_config
    pose = PoseResult(1, UPRIGHT, np.array([250, 150, 390, 400], np.float32))
    p.detector = SimpleNamespace(detect_and_track=lambda f: SimpleNamespace(detections=[], person_count=1,
                                                                              to_dict=lambda: {}))
    p.pose_estimator = SimpleNamespace(estimate=lambda *a: {1: pose}, get_all_features=dict)
    p.anomaly_engine = SimpleNamespace(process=lambda *a, **k: pytest.fail("engine must not run in posture mode"))
    p.clip_recorder = SimpleNamespace(add_frame=lambda *a: None)
    p._draw_skeleton = lambda frame, pose: frame
    r = p.process_frame(np.zeros((480, 640, 3), np.uint8), 1.0)
    assert isinstance(r, FrameResult) and r.alerts == [] and r.posture["status"] == NO_BASELINE
    assert r.to_dict()["mode"] == "posture"


@pytest.fixture
def posture_api(tmp_config, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import server

    primary = SimpleNamespace(config=tmp_config, mode="warehouse", paused=False, posture=PostureCoach())
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.allow_remote_camera_control = True
    server._cameras.clear()
    with TestClient(server.app) as c:
        c.primary = primary
        yield c


def test_app_settings_default_switch_and_persist(posture_api):
    assert posture_api.get("/api/app").json() == {"mode": "warehouse", "demo_footage": False}
    r = posture_api.put("/api/app", json={"mode": "posture"})
    assert r.json() == {"mode": "posture", "demo_footage": False}
    assert posture_api.primary.mode == "posture" and posture_api.primary.paused is True  # demo off = paused
    posture_api.put("/api/app", json={"demo_footage": True})
    assert posture_api.primary.paused is False
    assert posture_api.get("/api/app").json() == {"mode": "posture", "demo_footage": True}
    assert posture_api.put("/api/app", json={"mode": "karaoke"}).status_code == 422


def test_posture_endpoints(posture_api):
    coach = posture_api.primary.posture
    coach.update(UPRIGHT, 0.0)
    assert posture_api.get("/api/posture").json()["status"] == NO_BASELINE
    snap = posture_api.post("/api/posture/baseline", json={}).json()
    assert snap["calibrating"] is True
    for i in range(1, 70):
        coach.update(UPRIGHT, i * 0.1)
    data = posture_api.get("/api/posture").json()
    assert data["has_baseline"] and data["status"] == GOOD and "timeline" in data
    assert posture_api.post("/api/posture/reminder", json={}).json() == {"reminders": 1}
    reset = posture_api.post("/api/posture/session/reset", json={}).json()
    assert reset["session"]["good_s"] == 0 and reset["timeline"] == []
    assert posture_api.request("DELETE", "/api/posture/baseline", json={}).json()["has_baseline"] is False


# --- baseline quality and glancing (from the first real-webcam session) ----------------------------

def record(c, frames_fn, seconds=7.0, fps=10):
    c.start_baseline(now=0.0)
    for i in range(int(seconds * fps)):
        c.update(frames_fn(i), i / fps)
    return c.snapshot()


def test_countdown_before_recording_ignores_the_click_movement():
    """The first real baseline caught the user still moving after clicking: the first 3 s are
    a get-ready countdown and are not recorded."""
    c = PostureCoach()
    reaching = kp(nose=(420, 280), shoulders=((300, 260), (420, 330)), eyes=((405, 270), (435, 270)))
    snap = record(c, lambda i: reaching if i < 30 else UPRIGHT)  # moving for the first 3 s only
    assert c.baseline is not None and snap["status"] == GOOD
    assert c.baseline.tilt_deg == pytest.approx(0) and c.baseline.head_ratio == pytest.approx(100 / 120)


def test_get_ready_then_recording_phases():
    c = PostureCoach()
    c.start_baseline(now=0.0)
    c.update(UPRIGHT, 1.0)
    assert c.snapshot()["calibration_phase"] == "get_ready" and c.snapshot()["label"] == "Get ready"
    c.update(UPRIGHT, 4.0)
    s = c.snapshot()
    assert s["calibration_phase"] == "recording" and s["calibration_left_s"] == pytest.approx(2.0)


def test_baseline_rejected_if_you_move_while_it_records():
    rng = np.random.default_rng(0)

    def wobble(i):
        dx, dy = rng.normal(0, 25, 2)
        return kp(nose=(320 + dx, 200 + dy), eyes=((305 + dx, 190 + dy), (335 + dx, 190 + dy)))

    c = PostureCoach()
    snap = record(c, wobble)
    assert c.baseline is None and "moved" in snap["calibration_error"]


def test_baseline_rejected_if_it_does_not_look_upright():
    crooked = kp(shoulders=((260, 280), (380, 320)))  # ~18 degrees
    c = PostureCoach()
    snap = record(c, lambda i: crooked)
    assert c.baseline is None and "tilted" in snap["calibration_error"]
    slumped = kp(nose=(320, 285), eyes=((305, 278), (335, 278)))  # head almost at shoulder level
    c2 = PostureCoach()
    snap2 = record(c2, lambda i: slumped)
    assert c2.baseline is None and "very low" in snap2["calibration_error"]


def test_glancing_sideways_is_not_leaning():
    """Turning the head moves the nose a lot but the eye midpoint only a little."""
    glance = kp(nose=(350, 200), eyes=((310, 190), (340, 190)))
    c, t = coach_with_baseline()
    run(c, glance, t, 4)
    assert c.snapshot()["status"] == GOOD


# --- lean is measured against your own body, never the frame (requested after the webcam test) --

WITH_EARS = kp(ears=((290, 205), (350, 205)))


def baseline_then(pose, baseline_pose=UPRIGHT, seconds=4):
    c, t = coach_with_baseline(baseline_pose)
    run(c, pose, t, seconds)
    return c.snapshot()


def test_sitting_upright_but_shifted_sideways_in_the_frame_is_good():
    for dx in (-50, 50):  # ~0.4 shoulder widths off-centre
        assert baseline_then(shifted(UPRIGHT, dx=dx))["status"] == GOOD
        assert baseline_then(shifted(WITH_EARS, dx=dx), WITH_EARS)["status"] == GOOD


def test_turning_the_head_to_look_sideways_is_good():
    # Nose swings 35 px, eyes 12 px, ears 8 px: a glance, not a lean.
    turned = kp(nose=(355, 200), eyes=((317, 190), (347, 190)), ears=((298, 205), (358, 205)))
    assert baseline_then(turned, WITH_EARS)["status"] == GOOD
    turned_no_ears = kp(nose=(355, 200), eyes=((317, 190), (347, 190)))
    assert baseline_then(turned_no_ears)["status"] == GOOD


def test_real_lean_head_and_shoulders_tilted_is_leaning():
    # Upper body tilted to one side: shoulder line ~12 deg, head well off the shoulder midpoint.
    lean = kp(nose=(370, 205), eyes=((355, 195), (385, 195)), ears=((340, 210), (400, 210)),
              shoulders=((262, 288), (378, 312)))
    snap = baseline_then(lean, WITH_EARS)
    assert snap["status"] == LEANING
    assert any("tilted" in r for r in snap["reasons"]) and any("Head leaning" in r for r in snap["reasons"])


def test_head_offset_uses_ears_then_eyes_never_the_nose():
    m = measure(kp(nose=(380, 200), ears=((290, 205), (350, 205))))
    assert m.head_ref == "ears" and m.lateral == pytest.approx((320 - 320) / 120)
    m = measure(kp(nose=(380, 200)))
    assert m.head_ref == "eyes" and m.lateral == pytest.approx(0.0)
    k = kp(nose=(380, 200))
    k[1, 2] = k[2, 2] = 0.0
    m = measure(k)
    assert m.head_ref is None and m.lateral is None  # no ears or eyes: no head offset at all


def test_moving_a_lot_shows_a_hint_instead_of_a_posture():
    far_aside = baseline_then(shifted(UPRIGHT, dx=150))  # 1.25 shoulder widths
    assert far_aside["status"] == "moved" and "Reset baseline" in far_aside["reasons"][0]
    leaned_back = kp(nose=(320, 225), shoulders=((278, 300), (362, 300)), eyes=((309.5, 218), (330.5, 218)))
    assert baseline_then(leaned_back)["status"] == "moved"  # everything 30% smaller: moved back
    # Closer, with the face growing too: that's "too close", a real posture issue.
    assert baseline_then(CLOSE)["status"] == TOO_CLOSE


def test_moved_time_is_not_counted_as_poor_posture():
    c, t = coach_with_baseline()
    c.reset_session()
    run(c, shifted(UPRIGHT, dx=150), t, 20)
    s = c.snapshot()["session"]
    assert s["poor_s"] == 0 and s["seconds"]["moved"] > 10


def test_debug_rows_show_what_drives_the_status():
    snap = baseline_then(SLOUCH)
    rows = {r["key"]: r for r in snap["debug"]}
    assert set(rows) == {"face_size", "head_ratio", "face_ratio", "tilt_deg", "lateral", "moved"}
    assert rows["head_ratio"]["triggered"] and rows["head_ratio"]["status"] == SLOUCHING
    assert rows["head_ratio"]["change"] == pytest.approx((45 / 120) / (100 / 120), abs=0.02)
    assert not rows["tilt_deg"]["triggered"] and not rows["moved"]["triggered"]


def test_old_baseline_files_still_load(tmp_path):
    import json

    path = tmp_path / "posture_baseline.json"
    path.write_text(json.dumps({"shoulder_width": 120, "head_ratio": 0.8, "tilt_deg": 0, "lateral": 0,
                                "face_size": 30, "face_ratio": 0.25, "recorded_at": 1, "frames": 30}))
    c = PostureCoach(baseline_path=str(path))
    assert c.baseline is not None and c.baseline.mid_x is None
