import json

import numpy as np
import pytest

from anomaly.zone_monitor import Zone, ZoneMonitor

W, H = 1000, 500
BOX = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
INSIDE = np.array([500.0, 250.0])
OUTSIDE = np.array([100.0, 100.0])


@pytest.fixture
def monitor(tmp_path):
    m = ZoneMonitor(zones_file=str(tmp_path / "zones.json"), frame_width=W, frame_height=H,
                    alert_cooldown=30.0)
    return m


def use(monitor, **zone):
    monitor.set_zones([Zone.from_dict({"id": "z", "name": "Z", "polygon": BOX, **zone})])


def run(monitor, positions, t0=0.0, dt=0.1, track_id=1):
    out, t = [], t0
    for p in positions:
        out += monitor.check(track_id, np.asarray(p, dtype=float), t)
        t += dt
    return out, t


def test_missing_zone_file_creates_defaults(tmp_path):
    path = tmp_path / "sub" / "zones.json"
    m = ZoneMonitor(zones_file=str(path))
    assert path.is_file()
    assert {z.zone_type for z in m.zones} == {"restricted", "time_limited", "one_way"}


def test_from_dict_ignores_unknown_keys_and_normalises_polygon():
    z = Zone.from_dict({"id": "a", "name": "A", "polygon": [[0, 0], [1, 0], [1, 1]],
                        "zone_type": "restricted", "color": "#f00"})
    assert z.polygon == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_restricted_alerts_on_entry_then_respects_cooldown(monitor):
    use(monitor, zone_type="restricted")
    alerts, t = run(monitor, [OUTSIDE] + [INSIDE] * 100)          # 10 s inside
    assert [a.violation_type for a in alerts] == ["intrusion"]
    # Boundary jitter (in/out/in) must not re-alert inside the cooldown.
    alerts, t = run(monitor, [OUTSIDE, INSIDE, OUTSIDE, INSIDE], t0=t)
    assert alerts == []
    alerts, _ = run(monitor, [INSIDE], t0=31.0)
    assert len(alerts) == 1


def test_restricted_new_person_alerts_immediately(monitor):
    use(monitor, zone_type="restricted")
    run(monitor, [INSIDE], track_id=1)
    alerts, _ = run(monitor, [INSIDE], t0=1.0, track_id=2)
    assert [a.track_id for a in alerts] == [2]


def test_time_limited_alerts_after_limit_only(monitor):
    use(monitor, zone_type="time_limited", time_limit=5.0)
    alerts, _ = run(monitor, [INSIDE] * 45)                        # 4.5 s
    assert alerts == []
    alerts, _ = run(monitor, [INSIDE] * 20, t0=4.5)                # up to 6.5 s
    assert [a.violation_type for a in alerts] == ["time_exceeded"]
    assert alerts[0].duration > 5.0


def test_time_limited_timer_resets_on_exit(monitor):
    use(monitor, zone_type="time_limited", time_limit=5.0)
    alerts, _t = run(monitor, [INSIDE] * 40 + [OUTSIDE] + [INSIDE] * 40)
    assert alerts == []


@pytest.mark.parametrize("direction,step,wrong", [
    ("up", (0, 10), True), ("up", (0, -10), False),
    ("down", (0, -10), True), ("left", (10, 0), True), ("right", (10, 0), False),
])
def test_one_way(monitor, direction, step, wrong):
    use(monitor, zone_type="one_way", direction=direction)
    path = [INSIDE + np.array(step) * i for i in range(5)]
    alerts, _ = run(monitor, path)
    assert (len(alerts) == 1) is wrong
    if wrong:
        assert alerts[0].violation_type == "wrong_direction"


def test_frame_size_controls_normalisation(monitor):
    use(monitor, zone_type="restricted")
    monitor.set_frame_size(2000, 1000)          # INSIDE is now at (0.25, 0.25): outside
    alerts, _ = run(monitor, [INSIDE])
    assert alerts == []


def test_inactive_zone_is_ignored(monitor):
    use(monitor, zone_type="restricted", active=False)
    assert run(monitor, [INSIDE])[0] == []


def test_prune_drops_state(monitor):
    use(monitor, zone_type="restricted")
    run(monitor, [INSIDE], track_id=7)
    monitor.prune([])
    assert not monitor.last_alert and not monitor.track_inside and not monitor.track_last_pos


def test_overlay_denormalises(monitor):
    use(monitor, zone_type="time_limited", time_limit=3)
    (z,) = monitor.get_zones_for_overlay()
    assert z["polygon"][0] == (400, 200)
    assert z["polygon_normalized"][0] == [0.4, 0.4]
    assert z["time_limit"] == 3


@pytest.mark.parametrize("name", ["corridor_demo.json", "hallway_demo.json"])
def test_bundled_demo_zone_files_are_valid(name):
    from core.samples import DEMO_CONFIG_DIR

    data = json.loads((DEMO_CONFIG_DIR / name).read_text(encoding="utf-8"))
    zones = [Zone.from_dict(z) for z in data["zones"]]
    assert zones
    for z in zones:
        assert z.zone_type in {"restricted", "time_limited", "one_way"}
        assert len(z.polygon) >= 3
        assert all(0.0 <= c <= 1.0 for p in z.polygon for c in p)
        if z.zone_type == "one_way":
            assert z.direction in {"up", "down", "left", "right"}
        if z.zone_type == "time_limited":
            assert z.time_limit > 0
