"""Balance prediction: centre of mass vs base of support (synthetic front-view poses)."""

from types import SimpleNamespace

import numpy as np
import pytest

from activity.balance import BalanceTracker, centre_of_mass

C = 0.9


def stance(lean_px=0.0, feet=(380.0, 420.0), conf=C, ankles_conf=C):
    """A front-view person, feet at ``feet`` (x), upper body shifted sideways by ``lean_px``."""
    k = np.zeros((17, 3), np.float32)
    mid = sum(feet) / 2
    k[0] = (mid + lean_px * 1.4, 100, conf)                         # head leans furthest
    k[5], k[6] = (mid - 30 + lean_px, 150, conf), (mid + 30 + lean_px, 150, conf)
    k[7], k[8] = (mid - 40 + lean_px, 220, conf), (mid + 40 + lean_px, 220, conf)
    k[9], k[10] = (mid - 42 + lean_px, 280, conf), (mid + 42 + lean_px, 280, conf)
    k[11], k[12] = (mid - 20 + lean_px * 0.5, 300, conf), (mid + 20 + lean_px * 0.5, 300, conf)
    k[13], k[14] = (feet[0] + lean_px * 0.2, 400, conf), (feet[1] + lean_px * 0.2, 400, conf)
    k[15], k[16] = (feet[0], 500, ankles_conf), (feet[1], 500, ankles_conf)
    return SimpleNamespace(keypoints=k, bbox=(feet[0] - 60, 80, feet[1] + 60, 510), body_height=400.0)


def run(tracker, make, seconds, t=0.0, fps=10):
    alerts = []
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        alerts += tracker.update({1: make()}, t)
    return t, alerts


def test_centre_of_mass_sits_over_the_feet_when_upright_and_shifts_with_a_lean():
    upright = centre_of_mass(stance().keypoints)
    assert upright[0] == pytest.approx(400, abs=1)
    assert centre_of_mass(stance(lean_px=60).keypoints)[0] > upright[0] + 30
    no_trunk = stance().keypoints
    no_trunk[5, 2] = 0
    assert centre_of_mass(no_trunk) is None


def test_steady_stance_is_low_risk_and_never_warns():
    b = BalanceTracker()
    _, alerts = run(b, stance, 3)
    view = b.current[1]
    assert alerts == [] and view["risk"] < 0.3 and view["base"][0] < view["com"][0] < view["base"][1]


def test_leaning_past_the_feet_raises_the_risk_and_warns_losing_balance():
    b = BalanceTracker()
    t, _ = run(b, stance, 1)
    t, alerts = run(b, lambda: stance(lean_px=110), 0.3, t)
    assert alerts == [] and b.current[1]["risk"] > 0.85  # must hold (and the sudden shift reads as motion)
    t, alerts = run(b, lambda: stance(lean_px=110), 1.2, t)
    assert len(alerts) == 1 and alerts[0].alert_type == "losing_balance"
    assert alerts[0].message.startswith("Losing balance: centre of mass") and "base of support" in alerts[0].message
    _, again = run(b, lambda: stance(lean_px=110), 3, t)
    assert again == []  # 10 s cooldown


def test_risk_rises_as_the_margin_shrinks():
    b = BalanceTracker()
    risks = []
    for lean in (0, 40, 80, 120):
        b2 = BalanceTracker()
        run(b2, lambda lean=lean: stance(lean_px=lean), 1)
        risks.append(b2.current[1]["risk"])
    assert risks == sorted(risks) and risks[-1] == 1.0
    assert b.measure(stance().keypoints, 400, (0, 0, 1, 1))["margin_px"] > 0


def test_no_estimate_without_both_ankles():
    b = BalanceTracker()
    run(b, lambda: stance(ankles_conf=0.2), 1)
    assert b.current == {}


def test_no_warning_mid_stride_while_walking():
    b = BalanceTracker()
    t = 0.0
    alerts = []
    for i in range(40):  # walking sideways across the view, the body ahead of the feet each step
        t += 0.1
        x = 200 + 30 * i
        alerts += b.update({1: stance(lean_px=110, feet=(x, x + 40))}, t)
    assert alerts == [] and b.current[1]["moving"] and b.current[1]["risk"] > 0.85  # the bar still shows it
