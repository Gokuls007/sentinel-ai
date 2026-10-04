"""REBA from synthetic 2D skeletons with known joint angles.

Expected scores are worked out by hand from the published tables (Hignett & McAtamney 2000).
"""

import math

import numpy as np
import pytest

from ergonomics import ErgonomicsConfig, ErgoTracker, assess, compute_angles, risk_level
from ergonomics.reba import TABLE_A, TABLE_B, TABLE_C, lower_arm_score, neck_score, trunk_score, upper_arm_score

TORSO = 100.0


def rot(v, deg):
    """Rotate an image-space vector (y down) by deg; positive turns 'up' toward image-right."""
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def skeleton(trunk=0.0, neck=0.0, arm=None, elbow_flex=0.0, knee_flex=0.0, facing=1,
             shoulder_spread=6.0, torso=TORSO, nose_ahead=18.0, conf=0.9):
    """A side-view person facing image-right (facing=+1) or image-left (-1).

    trunk: degrees of flexion (+) / extension (-). neck: relative to the trunk.
    arm: absolute upper-arm angle from straight down, + = forward; None = hanging (gravity).
    elbow_flex / knee_flex: 0 = straight.
    """
    f = facing
    kp = np.zeros((17, 3))
    hip = np.array([400.0, 400.0])
    up = np.array([0.0, -1.0])
    trunk_dir = rot(up, trunk * f)
    shoulder = hip + trunk_dir * torso
    head_dir = rot(trunk_dir, neck * f)
    ear = shoulder + head_dir * 25
    nose = ear + np.array([nose_ahead * f, 4.0])
    down = np.array([0.0, 1.0])
    arm_dir = rot(down, -(arm or 0.0) * f)  # forward raise turns 'down' toward the facing side
    forearm_dir = rot(arm_dir, -elbow_flex * f)
    shin_dir = rot(down, knee_flex * f)

    def put(i, p):
        kp[i] = (p[0], p[1], conf)

    put(0, nose)
    put(3, ear)
    put(4, ear)
    # Each limb hangs from its own shoulder/hip (left/right are offset by half the spread).
    for side, offset in ((0, -shoulder_spread / 2), (1, shoulder_spread / 2)):
        dx = np.array([offset, 0.0])
        sh, hp = shoulder + dx, hip + dx
        el = sh + arm_dir * 30
        kn = hp + down * 45
        put(5 + side, sh)
        put(7 + side, el)
        put(9 + side, el + forearm_dir * 28)
        put(11 + side, hp)
        put(13 + side, kn)
        put(15 + side, kn + shin_dir * 45)
    return kp


def score(kp, **kw):
    return assess(compute_angles(kp), **kw)


# --- the published tables ------------------------------------------------------------


def test_lookup_tables_spot_checks_against_worksheet():
    assert TABLE_A[0][0] == [1, 2, 3, 4] and TABLE_A[4][2] == [7, 8, 9, 9] and TABLE_A[2][1][3] == 7
    assert TABLE_B[0] == [[1, 2, 2], [1, 2, 3]] and TABLE_B[5][1] == [8, 9, 9] and TABLE_B[3][1][2] == 7
    assert TABLE_C[0] == [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7]
    assert TABLE_C[5][8] == 10 and TABLE_C[7][4] == 10 and TABLE_C[11] == [12] * 12
    assert all(len(r) == 12 for r in TABLE_C) and len(TABLE_C) == 12


@pytest.mark.parametrize("score_value, level, name", [
    (1, 1, "negligible"), (2, 2, "low"), (3, 2, "low"), (4, 3, "medium"), (7, 3, "medium"),
    (8, 4, "high"), (10, 4, "high"), (11, 5, "very_high"), (15, 5, "very_high"),
])
def test_risk_levels(score_value, level, name):
    assert risk_level(score_value) == (level, name)


@pytest.mark.parametrize("angle, expected", [(0, 1), (4, 1), (15, 2), (-15, 2), (30, 3), (-25, 3), (61, 4)])
def test_trunk_sub_score(angle, expected):
    assert trunk_score(angle) == expected


def test_neck_upper_and_lower_arm_sub_scores():
    assert [neck_score(a) for a in (0, 15, 25, -10)] == [1, 1, 2, 2]
    assert [upper_arm_score(a) for a in (0, -25, 30, 60, 120)] == [1, 2, 2, 3, 4]
    assert [lower_arm_score(f) for f in (0, 59, 60, 90, 100, 130)] == [2, 2, 1, 1, 1, 2]


# --- postures ---------------------------------------------------------------------------


def test_upright_standing():
    r = score(skeleton())
    assert (r.trunk, r.neck, r.legs, r.upper_arm, r.lower_arm) == (1, 1, 1, 1, 2)
    assert (r.table_a, r.table_b, r.score, r.level_name) == (1, 1, 1, "negligible")


def test_30_degree_bend_with_arms_hanging():
    r = score(skeleton(trunk=30))
    a = compute_angles(skeleton(trunk=30))
    assert a.trunk == pytest.approx(30, abs=0.5)
    # Arms hang vertically, so relative to the bent trunk they are 30 degrees forward.
    assert a.upper_arm_left == pytest.approx(30, abs=0.5)
    assert (r.trunk, r.upper_arm, r.lower_arm) == (3, 2, 2)
    assert (r.table_a, r.table_b, r.score) == (2, 2, 2)


def test_70_degree_bend():
    r = score(skeleton(trunk=70))
    assert (r.trunk, r.upper_arm) == (4, 3)
    assert (r.table_a, r.table_b, r.score, r.level_name) == (3, 4, 3, "low")
    assert r.dominant == "trunk"


def test_arms_overhead():
    a = compute_angles(skeleton(arm=175))
    assert a.upper_arm_left == pytest.approx(175, abs=1)
    r = assess(a)
    assert (r.upper_arm, r.lower_arm, r.table_b, r.score) == (4, 2, 5, 3)


def test_deep_squat():
    r = score(skeleton(trunk=30, knee_flex=100))
    assert r.legs == 3  # base 1 + 2 for knees flexed > 60
    assert (r.table_a, r.score, r.level_name) == (5, 4, "medium")


def test_load_and_static_raise_the_score():
    base = score(skeleton(trunk=70))
    heavy = score(skeleton(trunk=70), load=2, static=True)
    assert heavy.load == 2 and heavy.activity == 1
    assert heavy.score == TABLE_C[base.table_a + 2 - 1][base.table_b - 1] + 1


# --- lower arm: keypoint elbow angle -> flexion (change 3) ----------------------------------


def test_lower_arm_straight_vs_90_degree_bend():
    straight = compute_angles(skeleton(elbow_flex=0))
    bent = compute_angles(skeleton(elbow_flex=90))
    assert straight.lower_arm_left == pytest.approx(0, abs=0.5)    # interior 180 -> flexion 0
    assert bent.lower_arm_left == pytest.approx(90, abs=0.5)       # interior 90 -> flexion 90
    assert lower_arm_score(straight.lower_arm_left) == 2           # outside 60-100
    assert lower_arm_score(bent.lower_arm_left) == 1


# --- flexion vs extension from facing direction (change 2) --------------------------------


def test_facing_direction_signs_trunk_extension():
    a = compute_angles(skeleton(trunk=-25))
    assert a.facing == 1 and a.trunk == pytest.approx(-25, abs=0.5)
    assert trunk_score(a.trunk) == 3  # > 20 degrees extension


def test_mirrored_person_scores_the_same():
    for kw in ({"trunk": 45}, {"trunk": -15}, {"arm": 60, "trunk": 10}, {"neck": 30}):
        right = assess(compute_angles(skeleton(facing=1, **kw)))
        left = assess(compute_angles(skeleton(facing=-1, **kw)))
        assert compute_angles(skeleton(facing=-1, **kw)).facing == -1
        parts = lambda r: (r.trunk, r.neck, r.upper_arm, r.score)  # noqa: E731
        assert parts(left) == parts(right)


def test_neck_extension_is_signed():
    assert compute_angles(skeleton(neck=-30)).neck == pytest.approx(-30, abs=1)
    assert score(skeleton(neck=-30)).neck == 2


def test_upper_arm_behind_the_body_is_extension():
    a = compute_angles(skeleton(arm=-40))
    assert a.upper_arm_left == pytest.approx(-40, abs=1)
    assert upper_arm_score(a.upper_arm_left) == 2


def test_unclear_facing_lowers_confidence():
    clear = compute_angles(skeleton())
    unclear = compute_angles(skeleton(nose_ahead=0.0))
    assert unclear.facing is None
    assert unclear.confidence == pytest.approx(clear.confidence * 0.5)
    assert any("facing" in n for n in unclear.notes)


# --- view confidence (change 1) ----------------------------------------------------------


def test_side_view_is_confident():
    a = compute_angles(skeleton(shoulder_spread=6))
    assert a.shoulder_torso_ratio < 0.35 and a.confidence == pytest.approx(1.0)


def test_facing_the_camera_is_not_confident():
    # Shoulders spread wide relative to torso length: a frontal view.
    a = compute_angles(skeleton(shoulder_spread=80, nose_ahead=0.0))
    assert a.shoulder_torso_ratio >= 0.75 and a.confidence == 0.0


def test_bending_toward_the_camera_is_not_confident():
    # Seen from the front, bending forward shortens the apparent torso, raising the ratio:
    # the trunk angle can't be read from this view.
    upright_front = compute_angles(skeleton(shoulder_spread=55, torso=TORSO, nose_ahead=0.0))
    bent_front = compute_angles(skeleton(shoulder_spread=55, torso=45, nose_ahead=0.0))
    assert bent_front.shoulder_torso_ratio > upright_front.shoulder_torso_ratio
    assert bent_front.confidence == 0.0


def test_side_view_bend_stays_confident():
    a = compute_angles(skeleton(trunk=70, shoulder_spread=6))
    assert a.confidence == pytest.approx(1.0)


def test_missing_keypoints_are_not_guessed():
    kp = skeleton()
    kp[[7, 8, 9, 10], 2] = 0.05  # arms not visible
    a = compute_angles(kp)
    assert a.upper_arm_left is None and a.lower_arm_left is None
    r = assess(a)
    assert "upper_arm" in r.estimated_parts and a.confidence < 1.0
    kp[[11, 12], 2] = 0.0  # no hips -> no trunk -> no score
    assert assess(compute_angles(kp)) is None


# --- tracker: smoothing, alerts, time at risk -----------------------------------------------


def high_risk_pose():
    # 70 degree bend, deep knee bend, arms reaching forward overhead: REBA high.
    return skeleton(trunk=70, knee_flex=100, arm=150)


def run(tracker, kp, seconds, start=1_790_000_000.0, fps=10, track_id=1, **kw):
    out, alerts, t = None, [], start
    for _ in range(int(seconds * fps)):
        out, alert = tracker.update(track_id, kp, t, **kw)
        if alert:
            alerts.append(alert)
        t += 1 / fps
    return out, alerts, t


def test_high_risk_pose_is_high():
    r = score(high_risk_pose(), load=2)
    assert r.level >= 4, r


def test_sustained_high_risk_raises_one_event_with_details():
    cfg = ErgonomicsConfig(alert_after_s=3.0, alert_cooldown_s=60.0)
    tr = ErgoTracker(cfg)
    view, alerts, _ = run(tr, high_risk_pose(), 10, load=2, zone_ids=["dock"])
    assert view.reliable and view.level >= 4
    assert len(alerts) == 1  # cooldown: one event, not one per frame
    a = alerts[0]
    assert a.duration_s >= 3.0 and a.peak_score >= 8 and a.dominant and a.zone_ids == ["dock"]


def test_brief_high_risk_does_not_alert():
    tr = ErgoTracker(ErgonomicsConfig(alert_after_s=3.0))
    _, alerts, t = run(tr, high_risk_pose(), 2, load=2)
    _, more, _ = run(tr, skeleton(), 2, start=t)
    assert alerts == [] and more == []


def test_low_confidence_never_alerts():
    tr = ErgoTracker()
    frontal = high_risk_pose()
    frontal[5, 0] -= 50
    frontal[6, 0] += 50  # shoulders spread: facing the camera
    view, alerts, _ = run(tr, frontal, 10, load=2)
    assert not view.reliable and alerts == []


def test_time_at_risk_per_zone_and_hour_never_per_track():
    tr = ErgoTracker()
    run(tr, skeleton(), 5, track_id=7, zone_ids=["line_1"])
    rows = tr.drain_time()
    assert {r[1] for r in rows} == {"zone", "hour"}  # no per-person time (privacy)
    by_kind = {k: sum(r[4] for r in rows if r[1] == k) for k in ("zone", "hour")}
    assert by_kind["zone"] == pytest.approx(4.9, abs=0.15)  # 50 frames -> 49 gaps of 0.1 s
    assert by_kind["zone"] == pytest.approx(by_kind["hour"])
    assert {r[2] for r in rows if r[1] == "zone"} == {"line_1"}
    assert tr.drain_time() == []  # drained


def test_static_posture_adds_activity_point():
    tr = ErgoTracker(ErgonomicsConfig(static_after_s=60.0))
    view, _, _ = run(tr, skeleton(trunk=30), 65, fps=2)
    assert view.raw.activity == 1


def test_smoothing_ignores_a_single_glitch_frame():
    tr = ErgoTracker()
    t = 1_790_000_000.0
    for i in range(10):
        kp = high_risk_pose() if i == 5 else skeleton()
        view, _ = tr.update(1, kp, t + i * 0.1, load=2)
    # One high-risk frame among upright ones: the 1 s median keeps the upright score.
    assert view.score == score(skeleton(), load=2).score
    assert view.score < score(high_risk_pose(), load=2).score
