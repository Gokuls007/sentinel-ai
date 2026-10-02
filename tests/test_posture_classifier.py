"""Personal posture classifier: features, time hold-out, training report, the stable-status rule,
the guided calibration / test flows (fake timestamps) and the API."""

import numpy as np
import pytest

from posture.classifier import (
    POSTURES,
    PostureModel,
    StableStatus,
    raw_features,
    score_report,
    split_by_time,
    train,
)
from posture.coach import (
    AWAY,
    CHECKING,
    GOOD,
    LEANING,
    LOOKING_AWAY,
    LOOKING_DOWN,
    NOT_SURE,
    SLOUCHING,
    PostureCoach,
    PostureConfig,
)


def pose(posture, rng, jitter=2.0):
    """Synthetic keypoints for each posture (front webcam), with keypoint jitter."""
    k = np.zeros((17, 3), np.float32)
    nose, eyes, ears = [320, 200], [[305, 190], [335, 190]], [[290, 195], [350, 195]]
    shoulders = [[260, 300], [380, 300]]
    if posture == "slouching":  # head drops toward the shoulders, shoulders round in
        nose, eyes, ears = [320, 240], [[305, 230], [335, 230]], [[290, 235], [350, 235]]
        shoulders = [[265, 300], [375, 300]]
    elif posture in ("leaning_left", "leaning_right"):
        s = 1 if posture == "leaning_left" else -1
        dx = 28 * s
        nose, eyes, ears = [320 + dx, 205], [[305 + dx, 195], [335 + dx, 195]], [[290 + dx, 200], [350 + dx, 200]]
        shoulders = [[260, 300 - 14 * s], [380, 300 + 14 * s]]
    elif posture == "too_close":  # everything bigger around the centre
        c = np.array([320, 250])
        scale = lambda p: list(c + (np.array(p) - c) * 1.45)  # noqa: E731
        nose, eyes, ears, shoulders = scale(nose), [scale(e) for e in eyes], [scale(e) for e in ears], \
            [scale(s) for s in shoulders]
    elif posture == "looking_down":  # nose drops below the ear line, head a little lower
        nose, eyes, ears = [320, 222], [[305, 208], [335, 208]], [[290, 200], [350, 200]]
    pts = [nose, *eyes, *ears, *shoulders]
    for i, p in enumerate(pts):
        k[i] = (*(np.array(p, float) + rng.normal(0, jitter, 2)), 0.9)
    return k


def recordings(rng, n=80, fps=10):
    return {p: [{"t": i / fps, "features": raw_features(pose(p, rng))} for i in range(n)] for p in POSTURES}


def test_raw_features_need_face_and_shoulders():
    rng = np.random.default_rng(0)
    k = pose("good", rng, jitter=0)
    f = raw_features(k)
    assert f["lateral"] == pytest.approx(0) and f["yaw"] == pytest.approx(0) and f["shoulder_width"] == 120
    down = raw_features(pose("looking_down", rng, jitter=0))
    assert down["pitch"] > f["pitch"]  # nose drops relative to the ears
    k[6, 2] = 0.1
    assert raw_features(k) is None


def test_hold_out_is_the_last_part_of_each_recording_by_time():
    recs = {"good": [{"t": t, "features": {"x": t}} for t in (3, 1, 2, 0)]}
    train_rows, test_rows = split_by_time(recs, holdout=0.25)
    assert [r["x"] for _p, r in train_rows] == [0, 1, 2] and [r["x"] for _p, r in test_rows] == [3]


def test_score_report_confusion_and_balanced_accuracy():
    r = score_report(("a", "b"), ["a", "a", "a", "b"], ["a", "a", "b", "b"])
    assert r["accuracy"] == 0.75 and r["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert r["confusion"]["matrix"] == [[2, 1], [0, 1]]


def test_train_reports_hold_out_accuracy_and_both_models(tmp_path):
    model = train(recordings(np.random.default_rng(1)))
    rep = model.report
    assert rep["model"] in ("logistic regression", "gradient boosting")
    assert set(rep["compared"]) == {"logistic regression", "gradient boosting"}
    assert rep["balanced_accuracy"] > 0.9 and rep["frames"] == 6 * 20
    assert rep["confusion"]["classes"] == list(POSTURES)
    probs = model.predict_proba(raw_features(pose("looking_down", np.random.default_rng(5))))
    assert max(probs, key=probs.get) == "looking_down" and sum(probs.values()) == pytest.approx(1)
    model.save(tmp_path / "m.pkl")
    assert PostureModel.load(tmp_path / "m.pkl").name == model.name
    assert PostureModel.load(tmp_path / "missing.pkl") is None


def test_train_needs_every_posture():
    recs = recordings(np.random.default_rng(2))
    del recs["looking_down"]
    with pytest.raises(ValueError, match="looking_down"):
        train(recs)


def test_stable_status_needs_a_confident_posture_held_for_5_s():
    s = StableStatus(window_s=10, hold_s=5, min_prob=0.7)
    good, bad = {"good": 0.9, "slouching": 0.1}, {"good": 0.1, "slouching": 0.9}
    t = 0.0
    for _ in range(60):
        t += 0.1
        s.update(good, t)
    assert s.current == "good"
    for _ in range(30):  # a 3 s glance: not enough
        t += 0.1
        s.update(bad, t)
    for _ in range(30):
        t += 0.1
        s.update(good, t)
    assert s.current == "good"
    for _ in range(150):  # a real slouch: the average crosses 0.7, then holds 5 s
        t += 0.1
        s.update(bad, t)
    assert s.current == "slouching"


def test_stable_status_stays_put_when_nothing_is_confident():
    s = StableStatus()
    for i in range(100):
        s.update({"good": 0.55, "slouching": 0.45}, i * 0.1)
    assert s.current is None


# --- the coach: guided calibration, training, classification, test ------------------------------

def feed(c, posture, t, seconds, rng, fps=10):
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        c.update(pose(posture, rng), t)
    return t


def calibrated_coach(tmp_path, rng):
    cfg = PostureConfig(calibration_record_s=8.0)
    c = PostureCoach(cfg, baseline_path=str(tmp_path / "posture_baseline_laptop.json"))
    t = 0.0
    c.update(pose("good", rng), t)
    for p in POSTURES:
        c.start_recording("calibrate", [p], now=t)
        snap = c.snapshot()
        assert snap["recording"]["phase"] == "get_ready" and snap["recording"]["label"]
        t = feed(c, p, t, 11.5, rng)
        assert c.rec is None and c.rec_error is None
    return c, t


def test_calibration_records_each_posture_then_trains_and_classifies(tmp_path):
    rng = np.random.default_rng(3)
    c, t = calibrated_coach(tmp_path, rng)
    counts = c.snapshot()["calibration_counts"]
    assert set(counts) == set(POSTURES) and min(counts.values()) >= 75
    report = c.train_model()
    assert report["balanced_accuracy"] > 0.9
    assert (tmp_path / "posture_model_laptop.pkl").exists()
    assert (tmp_path / "posture_calibration_laptop.json").exists()
    t = feed(c, "good", t, 3, rng)
    assert c.status == CHECKING  # not yet confident for 5 s
    t = feed(c, "good", t, 5, rng)
    assert c.status == GOOD and c.snapshot()["method"] == "classifier"
    t = feed(c, "slouching", t, 4, rng)  # a few seconds is ignored
    assert c.status == GOOD
    t = feed(c, "slouching", t, 12, rng)
    assert c.status == SLOUCHING
    t = feed(c, "leaning_right", t, 16, rng)
    assert c.status == LEANING and "right" in c.reasons[0]
    t = feed(c, "looking_down", t, 16, rng)
    snap = c.snapshot()
    assert snap["status"] == LOOKING_DOWN and snap["held_s"] > 0
    assert snap["session"]["seconds"][LOOKING_DOWN] > 0
    assert snap["probabilities"]["looking_down"] > 0.7

    # The model is reloaded by a new coach (same files).
    again = PostureCoach(PostureConfig(), baseline_path=str(tmp_path / "posture_baseline_laptop.json"))
    assert again.model is not None and again.snapshot()["calibration_counts"] == counts


def test_short_recording_is_rejected_with_a_redo_message(tmp_path):
    c = PostureCoach(PostureConfig(calibration_record_s=8.0),
                     baseline_path=str(tmp_path / "posture_baseline_laptop.json"))
    c.start_recording("calibrate", ["good"], now=0.0)
    t = 0.0
    for _ in range(115):  # nobody visible
        t += 0.1
        c.update(None, t)
    assert c.rec is None and "redo" in c.rec_error and c.calibration_counts == {}


def test_test_my_calibration_scores_new_data_separately(tmp_path):
    rng = np.random.default_rng(6)
    c, t = calibrated_coach(tmp_path, rng)
    with pytest.raises(ValueError):
        PostureCoach().start_recording("test", ["good"])  # no model yet
    c.train_model()
    order = ["looking_down", "good", "too_close", "slouching", "leaning_left", "leaning_right"]
    c.start_recording("test", order, now=t, record_s=6.0)
    for i, p in enumerate(order):
        assert c.snapshot()["recording"]["step"] == i + 1 and c.snapshot()["recording"]["posture"] == p
        while c.rec is not None and c.rec["step"] == i:
            t = feed(c, p, t, 0.1, rng)
    assert c.rec is None
    rep = c.model.test_report
    assert abs(rep["frames"] - 6 * 60) <= 12 and rep["accuracy"] > 0.85 and rep["postures"] == order
    summary = c.snapshot()["model"]
    assert summary["test_accuracy"] == rep["accuracy"] and summary["accuracy"] == c.model.report["accuracy"]
    # Saved with the model, so it survives a restart.
    assert PostureModel.load(str(tmp_path / "posture_model_laptop.pkl")).test_report["frames"] == rep["frames"]


def test_delete_model_falls_back_to_thresholds(tmp_path):
    rng = np.random.default_rng(7)
    c, _t = calibrated_coach(tmp_path, rng)
    c.train_model()
    c.delete_model()
    assert c.model is None and c.snapshot()["method"] == "thresholds"
    assert not (tmp_path / "posture_model_laptop.pkl").exists()


@pytest.fixture
def classifier_api(tmp_config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from api import server

    coach = PostureCoach(PostureConfig(calibration_record_s=8.0),
                         baseline_path=str(tmp_path / "posture_baseline_laptop.json"))
    primary = SimpleNamespace(config=tmp_config, mode="posture", paused=False, posture=coach)
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "config", tmp_config)
    tmp_config.allow_remote_camera_control = True
    server._cameras.clear()
    with TestClient(server.app) as c:
        c.coach = coach
        yield c


def test_classifier_endpoints(classifier_api):
    api, coach = classifier_api, classifier_api.coach
    rng = np.random.default_rng(8)
    assert api.post("/api/posture/calibration/record", json={"posture": "sideways"}).status_code == 400
    assert api.post("/api/posture/calibration/train", json={}).status_code == 400  # nothing recorded
    assert api.post("/api/posture/test", json={}).status_code == 400  # no model
    t = 0.0
    coach.update(pose("good", rng), t)
    for p in POSTURES:
        snap = api.post("/api/posture/calibration/record", json={"posture": p}).json()
        assert snap["recording"]["posture"] == p
        t = feed(coach, p, t, 11.5, rng)
    trained = api.post("/api/posture/calibration/train", json={}).json()
    assert trained["report"]["balanced_accuracy"] > 0.9 and trained["method"] == "classifier"
    model = api.get("/api/posture/model").json()
    assert model["report"]["confusion"]["classes"] == list(POSTURES) and model["test_report"] is None
    snap = api.post("/api/posture/test", json={}).json()
    assert snap["recording"]["kind"] == "test" and snap["recording"]["steps"] == 6
    assert api.post("/api/posture/recording/cancel", json={}).json()["recording"] is None
    assert api.request("DELETE", "/api/posture/model", json={}).json()["method"] == "thresholds"
    assert api.request("DELETE", "/api/posture/calibration", json={}).json()["calibration_counts"] == {}


# --- looking away and unfamiliar poses (the classifier only knows the six calibrated postures) ----

def profile(rng, jitter=1.5):
    """Head turned fully sideways: eyes bunched together, nose well past them, one ear hidden."""
    k = np.zeros((17, 3), np.float32)
    pts = {0: (282, 205), 1: (300, 192), 2: (311, 192), 4: (334, 196), 5: (262, 300), 6: (378, 300)}
    for i, p in pts.items():
        k[i] = (*(np.array(p, float) + rng.normal(0, jitter, 2)), 0.9)
    return k


def odd(rng, jitter=2.0):
    """Face the screen normally, but shoulders tilted ~37 degrees: nothing like any calibration."""
    k = pose("good", rng, jitter)
    k[5, :2] = (260 + rng.normal(0, jitter), 255 + rng.normal(0, jitter))
    k[6, :2] = (380 + rng.normal(0, jitter), 345 + rng.normal(0, jitter))
    return k


def feed_fn(c, make, t, seconds, rng, fps=10):
    seen = []
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        c.update(make(rng), t)
        seen.append(c.status)
    return t, seen


def test_head_turn_rules():
    from posture.classifier import head_turn

    rng = np.random.default_rng(10)
    assert head_turn(pose("good", rng, jitter=0))[0] is False
    assert head_turn(pose("leaning_left", rng, jitter=0))[0] is False  # head moves with the body
    turned, yaw, why = head_turn(profile(rng, jitter=0))
    assert turned and yaw < -1 and why
    one_eye = pose("good", rng, jitter=0)
    one_eye[2, 2] = 0.1
    assert head_turn(one_eye) == (True, None, "one eye hidden")
    outside = pose("good", rng, jitter=0)
    outside[0, 0] = 360  # nose beyond the ear
    assert head_turn(outside)[0] is True


def test_profile_view_is_looking_away_not_good(tmp_path):
    rng = np.random.default_rng(11)
    c, t = calibrated_coach(tmp_path, rng)
    c.train_model()
    t = feed(c, "good", t, 10, rng)
    assert c.status == GOOD
    t, seen = feed_fn(c, profile, t, 5, rng)
    assert c.status == LOOKING_AWAY and seen[-1] == LOOKING_AWAY
    snap = c.snapshot()
    assert snap["yaw"] < -1 and snap["turn_reason"]
    good_before = snap["session"]["seconds"][GOOD]
    t, _ = feed_fn(c, profile, t, 30, rng)
    snap = c.snapshot()
    assert snap["session"]["seconds"][LOOKING_AWAY] > 25 and snap["session"]["seconds"][GOOD] == good_before
    assert snap["session"]["poor_s"] == 0  # neither good nor poor
    # Held over 2 minutes: treated as away.
    t, _ = feed_fn(c, profile, t, 90, rng)
    assert c.status == AWAY
    # Facing the screen again: back to the posture.
    t = feed(c, "good", t, 8, rng)
    assert c.status == GOOD


def test_unfamiliar_pose_is_not_sure_never_good(tmp_path):
    rng = np.random.default_rng(12)
    c, t = calibrated_coach(tmp_path, rng)
    c.train_model()
    assert c.model.ood_threshold > 0
    # A fresh start in an unfamiliar pose never becomes Good (or any posture).
    t, seen = feed_fn(c, odd, t, 15, rng)
    assert GOOD not in seen and set(seen) <= {CHECKING, NOT_SURE, AWAY} and c.status == NOT_SURE
    snap = c.snapshot()
    assert snap["ood"]["distance"] > snap["ood"]["threshold"]
    # From Good: a short unfamiliar moment holds Good; a sustained one becomes Not sure.
    t = feed(c, "good", t, 10, rng)
    assert c.status == GOOD
    t, _ = feed_fn(c, odd, t, 2, rng)
    assert c.status == GOOD
    t, seen = feed_fn(c, odd, t, 4, rng)
    assert c.status == NOT_SURE and set(seen) <= {GOOD, NOT_SURE}
    assert c.snapshot()["session"]["seconds"][NOT_SURE] > 0


def test_familiar_postures_are_rarely_flagged(tmp_path):
    rng = np.random.default_rng(13)
    c, _t = calibrated_coach(tmp_path, rng)
    c.train_model()
    labelled = [(p, raw_features(pose(p, rng))) for p in POSTURES for _ in range(50)]
    from posture.classifier import evaluate

    rep = evaluate(c.model, labelled)
    assert rep["unfamiliar_fraction"] < 0.05
    assert evaluate(c.model, [("good", raw_features(odd(rng))) for _ in range(20)])["unfamiliar_fraction"] > 0.9
