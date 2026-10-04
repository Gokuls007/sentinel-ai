"""Exam Hall E1: head pose behind an interface, seats (labels, neighbours, assignment), the setup
check, per-seat calibration, the monitor, storage (schema v5) and the session API."""

import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exam import (
    Baseline,
    ExamConfig,
    ExamMonitor,
    HeadPose,
    KeypointHeadPose,
    SeatCalibration,
    SetupCheck,
    assign,
    label_seats,
    relabel,
    seat_rect_for,
)

W, H = 1280, 720


def student(x, y, sw=80.0, yaw_px=0.0, conf=0.9, ears=True, wrists=True, nose_drop=12.0):
    """A seated student facing the camera: shoulders at (x +- sw/2, y), head above."""
    k = np.zeros((17, 3), np.float32)
    head_y = y - 1.1 * sw
    k[0] = (x + yaw_px, head_y + nose_drop, conf)          # nose
    k[1] = (x - 0.18 * sw + yaw_px, head_y, conf)          # eyes
    k[2] = (x + 0.18 * sw + yaw_px, head_y, conf)
    if ears:
        k[3] = (x - 0.4 * sw, head_y + 3, conf)
        k[4] = (x + 0.4 * sw, head_y + 3, conf)
    k[5] = (x - sw / 2, y, conf)
    k[6] = (x + sw / 2, y, conf)
    if wrists:
        k[9] = (x - 0.3 * sw, y + 0.9 * sw, conf)
        k[10] = (x + 0.3 * sw, y + 0.9 * sw, conf)
    return k


def pose(kp):
    return SimpleNamespace(keypoints=kp, bbox=(0, 0, 10, 10))


# --- head pose ----------------------------------------------------------------------------------


def test_head_pose_yaw_sign_profile_and_low_confidence():
    est = KeypointHeadPose()
    assert abs(est.estimate(student(400, 400)).yaw) < 1
    right = est.estimate(student(400, 400, yaw_px=20))
    assert right.yaw > 20 and right.method == "ears" and right.confidence > 0.8  # face toward image right
    assert est.estimate(student(400, 400, yaw_px=-20)).yaw < -20
    prof = student(400, 400)
    prof[[2, 4], 2] = 0.0  # a profile: the far eye and ear are hidden
    p = est.estimate(prof)
    assert abs(p.yaw) == 75 and p.pitch is None
    assert est.estimate(student(400, 400, conf=0.2)) is None  # low confidence: no estimate
    down = est.estimate(student(400, 400, nose_drop=40)).pitch
    assert down > est.estimate(student(400, 400)).pitch  # head tilted down -> larger pitch


def test_a_different_head_pose_estimator_can_be_plugged_in():
    class Fixed:
        name = "fixed"

        def estimate(self, keypoints, frame=None, box=None):
            return HeadPose(33.0, 4.0, 0.99, "fixed")

    m = ExamMonitor(ExamConfig(calibration_s=1.0), estimator=Fixed())
    seats = label_seats([seat_rect_for(student(400, 400), W, H)])
    m.load({"id": 1, "status": "calibrating", "calibration_s": 1.0}, seats)
    for i in range(40):
        m.update({1: pose(student(400, 400))}, i * 0.05, W, H)
    assert m.calib["A1"].baseline.yaw_med == pytest.approx(33.0)


# --- seats --------------------------------------------------------------------------------------


def grid(rows=2, cols=3):
    """Seat rects for a grid of students; row 0 is nearest the camera (lowest in the image)."""
    rects = []
    for r in range(rows):
        for c in range(cols):
            rects.append(seat_rect_for(student(250 + c * 380, 650 - r * 260, sw=70), W, H))
    return rects


def test_seats_are_labelled_in_reading_order_with_neighbours():
    seats = {s.label: s for s in label_seats(list(reversed(grid())))}
    assert sorted(seats) == ["A1", "A2", "A3", "B1", "B2", "B3"]
    a1, b1 = seats["A1"], seats["B1"]
    assert a1.centre[1] > b1.centre[1]  # row A is the front row: lowest in the image
    assert a1.centre[0] < seats["A2"].centre[0]  # 1 is leftmost
    assert seats["A2"].neighbours == {"left": "A1", "right": "A3", "front": None, "back": "B2"}
    assert seats["B3"].neighbours == {"left": "B2", "right": None, "front": "A3", "back": None}


def test_relabel_keeps_custom_labels_and_recomputes_neighbours():
    seats = label_seats(grid(1, 3))
    seats[1].label = "Z9"
    fresh = {s.label: s for s in relabel(seats)}
    assert set(fresh) == {"A1", "Z9", "A3"}
    assert fresh["A1"].neighbours["right"] == "Z9"


def test_people_are_assigned_by_shoulder_midpoint_and_staff_are_left_out():
    seats = label_seats(grid(1, 2))
    people = {7: student(250, 650, sw=70), 9: student(630, 650, sw=70), 11: student(1100, 200, sw=60)}
    assert assign(people, seats, W, H) == {"A1": 7, "A2": 9}  # 11 (walking at the back) is in no seat


# --- setup check --------------------------------------------------------------------------------


def checks(snapshot):
    return {c["id"]: c for c in snapshot["checks"]}


def test_setup_check_warns_when_students_are_too_small_or_unclear():
    ok = SetupCheck()
    for i in range(10):
        ok.update([student(300, 400), student(700, 400)], i * 0.1)
    snap = ok.snapshot()
    assert snap["ok"] and snap["people"] == 2
    small = SetupCheck()
    for i in range(10):
        small.update([student(300, 400, sw=25)], i * 0.1)
    assert "too small" in checks(small.snapshot())["size"]["message"]
    dark = SetupCheck()
    for i in range(10):
        dark.update([student(300, 400, conf=0.35)], i * 0.1)
    c = checks(dark.snapshot())
    assert not c["lighting"]["ok"] and c["view"]["ok"]


# --- calibration --------------------------------------------------------------------------------


def test_seat_baseline_after_the_calibration_period():
    cal = SeatCalibration(duration_s=10, min_samples=20)
    est = KeypointHeadPose()
    b = None
    for i in range(300):  # 30 s at 10 fps; the student is turned a little to the right
        kp = student(400, 400, yaw_px=8 + (i % 5))
        b = cal.add(i / 10, est.estimate(kp), 0.9) or b
        if i == 50:
            assert cal.baseline is None and 0.4 < cal.progress(i / 10) < 0.6
    assert b is not None and b.yaw_med > 10 and b.yaw_spread >= 6 and b.hand_height_med == pytest.approx(0.9)
    assert cal.progress(99) == 1.0


def test_monitor_calibrates_then_goes_live_and_late_arrivals_calibrate_alone():
    seats = label_seats(grid(1, 2))
    statuses, baselines = [], {}
    m = ExamMonitor(ExamConfig(calibration_s=10))
    m.load({"id": 5, "status": "calibrating", "calibration_s": 10}, seats)
    m.on_status = lambda sid, s: statuses.append((sid, s))
    m.on_baseline = lambda sid, label, b: baselines.setdefault(label, (sid, b))
    early = student(250, 650, sw=70)
    t = 0.0
    for _ in range(120):  # 12 s: only A1 is taken
        t += 0.1
        snap = m.update({1: pose(early)}, t, W, H)
    assert statuses == [(5, "live")] and snap["status"] == "live" and snap["calibration_remaining_s"] is None
    assert "A1" in baselines and "A2" not in baselines
    late = student(630, 650, sw=70)
    for _ in range(50):  # A2's student arrives late: 5 s in, not yet calibrated
        t += 0.1
        snap = m.update({1: pose(early), 2: pose(late)}, t, W, H)
    a2 = next(s for s in snap["seats"] if s["label"] == "A2")
    assert a2["occupied"] and not a2["calibrated"] and 0.4 < a2["calibration_progress"] < 0.6
    for _ in range(60):
        t += 0.1
        m.update({1: pose(early), 2: pose(late)}, t, W, H)
    assert "A2" in baselines


def test_no_calibration_before_the_exam_starts_and_staff_are_counted_apart():
    seats = label_seats(grid(1, 1))
    m = ExamMonitor(ExamConfig(calibration_s=1))
    m.load({"id": 1, "status": "setup", "calibration_s": 1}, seats)
    for i in range(40):
        snap = m.update({1: pose(student(250, 650, sw=70)), 2: pose(student(1000, 150, sw=60))}, i * 0.1, W, H)
    assert snap["status"] == "setup" and not snap["seats"][0]["calibrated"] and snap["staff"] == 1


def test_detect_seats_offers_people_sitting_still_but_not_someone_walking():
    m = ExamMonitor(ExamConfig(stable_s=5))
    m.load({"id": 1, "status": "setup"}, [])
    for i in range(70):  # 7 s
        t = i / 10
        m.update({1: pose(student(250, 650, sw=70)), 2: pose(student(630, 650, sw=70)),
                  3: pose(student(100 + 60 * t, 300, sw=60))}, t, W, H)
    seats = m.detect_seats()
    assert [s.label for s in seats] == ["A1", "A2"]


# --- storage and API ----------------------------------------------------------------------------


def test_v5_adds_the_exam_tables_and_the_store_round_trips(tmp_path):
    from exam.store import ExamStore

    store = ExamStore(str(tmp_path / "events.db"))
    with sqlite3.connect(tmp_path / "events.db") as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    assert {"exam_sessions", "exam_seats", "exam_seat_baselines", "exam_flags", "exam_unblur_log"} <= tables
    s = store.create_session("Maths mock", "laptop", room="R101")
    assert store.open_session_for("laptop")["id"] == s["id"] and s["status"] == "setup"
    seats = store.save_seats(s["id"], label_seats(grid(1, 2)))
    store.save_baseline(s["id"], "A1", Baseline(3.0, 6.0, 20.0, 5.0, 0.9, 40, 1.0))
    assert [x.label for x in store.seats(s["id"])] == ["A1", "A2"]
    assert store.seats(s["id"])[1].neighbours["left"] == "A1"
    assert store.baselines(s["id"])["A1"].yaw_med == 3.0
    store.save_seats(s["id"], seats[:1])  # replaced: the old baselines go with the old seats
    assert store.baselines(s["id"]) == {}
    store.update_session(s["id"], status="ended")
    assert store.open_session_for("laptop") is None


@pytest.fixture
def exam_api(tmp_config, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import server

    tmp_config.allow_remote_camera_control = True
    monkeypatch.setattr(server, "config", tmp_config)
    monkeypatch.setattr(server, "pipeline", None)
    monkeypatch.setattr(server, "_exam_store_obj", None)
    monkeypatch.setattr(server, "_retention_worker", None)
    server._cameras.clear()
    with TestClient(server.app) as c:
        yield c


def test_session_api_setup_seats_start_and_end(exam_api):
    r = exam_api.post("/api/exam/sessions", json={"name": "Mock", "camera_id": "laptop", "calibration_s": 60})
    assert r.status_code == 201
    sid = r.json()["id"]
    assert exam_api.post("/api/exam/sessions", json={"name": "Two", "camera_id": "laptop"}).status_code == 409
    assert exam_api.post(f"/api/exam/sessions/{sid}/start", json={}).status_code == 409  # no seats yet
    bad = {"seats": [{"label": "A1", "rect": [0.1, 0.1, 0.3, 0.4]}, {"label": "a1", "rect": [0.5, 0.1, 0.7, 0.4]}]}
    assert exam_api.put(f"/api/exam/sessions/{sid}/seats", json=bad).status_code == 422  # duplicate labels
    assert exam_api.put(f"/api/exam/sessions/{sid}/seats",
                        json={"seats": [{"label": "A1", "rect": [0.5, 0.1, 0.3, 0.4]}]}).status_code == 422
    good = {"seats": [{"label": "A1", "rect": [0.1, 0.5, 0.3, 0.9]}, {"label": "A2", "rect": [0.5, 0.5, 0.7, 0.9]}]}
    body = exam_api.put(f"/api/exam/sessions/{sid}/seats", json=good).json()
    assert [s["label"] for s in body["seats"]] == ["A1", "A2"] and body["seats"][0]["neighbours"]["right"] == "A2"
    assert exam_api.post(f"/api/exam/sessions/{sid}/start", json={}).json()["status"] == "calibrating"
    assert exam_api.put(f"/api/exam/sessions/{sid}/seats", json=good).status_code == 409  # locked once started
    assert exam_api.post(f"/api/exam/sessions/{sid}/detect-seats", json={}).status_code == 409  # camera not running
    assert exam_api.post(f"/api/exam/sessions/{sid}/end", json={}).json()["status"] == "ended"
    two = exam_api.post("/api/exam/sessions", json={"name": "Two", "camera_id": "laptop"}).json()
    copied = exam_api.post(f"/api/exam/sessions/{two['id']}/copy-seats", json={"from_session_id": sid}).json()
    assert [s["label"] for s in copied["seats"]] == ["A1", "A2"]
    assert [s["id"] for s in exam_api.get("/api/exam/sessions").json()["sessions"]] == [two["id"], sid]


# --- wording ------------------------------------------------------------------------------------

FORBIDDEN = re.compile(r"cheat|suspicious (student|person|behaviou?r)", re.IGNORECASE)


def test_ui_and_exam_code_never_use_accusatory_words():
    """exam-hall.md 1.1: flags are for human review; nothing in the UI or the exam code may say
    anyone cheated (the rule compiler's prompt, which tells the LLM to refuse such rules, is not UI)."""
    root = Path(__file__).resolve().parents[1]
    files = [*root.glob("frontend/src/**/*.js"), *root.glob("frontend/src/**/*.jsx"),
             *root.glob("backend/exam/**/*.py")]
    assert files
    hits = [f"{f.relative_to(root)}:{i}" for f in files
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert hits == []
