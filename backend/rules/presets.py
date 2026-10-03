"""Built-in rules and presets.

Built-ins reproduce the hard-coded fall and restricted-zone alerts as rules. They are used
only with RULES_BUILTINS=true, which also switches the hard-coded versions off, so nothing
alerts twice. Their events keep the old types ("fall", "zone_intrusion") so history, search
and analytics don't change.

Presets are sets of ordinary rules. Warehouse safety is zone-free apart from one dwell rule per
time-limited zone, so it works on any camera.
"""

from __future__ import annotations

import re

from rules.dsl import (
    CountGreaterThan,
    Fallen,
    InZone,
    PostureRiskAtLeast,
    Rule,
    StationaryFor,
    TimeWindow,
)


def _safe(zone_id: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(zone_id).lower()).strip("-") or "zone"


BUILTIN_EVENT_TYPES = {"fall": "fall", "zone": "zone_intrusion"}
PRESETS = {
    "warehouse": {"label": "Warehouse safety", "available": True},
    "exam": {"label": "Exam hall", "available": False, "note": "comes with the exam hall work (Phase 3d)"},
    "posture": {"label": "Desk posture coach", "available": False,
                "note": "the posture coach is its own mode: switch to it in the mode menu"},
}


def builtin_rules(zones: list[dict], fall_cooldown_s: float = 30.0, zone_cooldown_s: float = 30.0) -> list[Rule]:
    """``zones``: [{"id", "name", "zone_type", "active"}] from the zone monitor."""
    out = [Rule(id="builtin-fall", name="Fall", builtin=True, conditions=[Fallen(type="fallen")],
                severity="critical", cooldown_s=fall_cooldown_s, actions=["alert", "record_clip", "notify"],
                source_text="Built in: alert when a fall is confirmed")]
    for z in zones:
        if z.get("zone_type") == "restricted" and z.get("active", True):
            out.append(Rule(id=f"builtin-zone-{_safe(z['id'])}"[:48], name=f"Restricted zone: {z['name']}",
                            builtin=True, conditions=[InZone(type="in_zone", zone=z["id"])], severity="high",
                            cooldown_s=zone_cooldown_s, actions=["alert", "record_clip", "notify"],
                            source_text=f"Built in: alert when someone enters {z['name']}"))
    return out


def builtin_event_type(rule: Rule) -> str | None:
    if not rule.builtin:
        return None
    return BUILTIN_EVENT_TYPES["fall"] if rule.id == "builtin-fall" else BUILTIN_EVENT_TYPES["zone"]


def preset_rules(name: str, zones: list[dict]) -> list[Rule]:
    if name != "warehouse":
        raise ValueError(PRESETS.get(name, {}).get("note") or f"unknown preset {name!r}")
    p = "warehouse"
    rules = [
        Rule(id="wh-posture-risk", name="High-risk posture held", preset=p,
             conditions=[PostureRiskAtLeast(type="posture_risk_at_least", level=4)], duration_s=30,
             severity="medium", cooldown_s=300, source_text="Alert if someone works in a high-risk posture for 30 s"),
        Rule(id="wh-motionless", name="Person not moving for 2 minutes", preset=p,
             conditions=[StationaryFor(type="stationary_for", seconds=120, radius_body_heights=0.15)],
             severity="medium", cooldown_s=600,
             source_text="Alert if someone hasn't moved at all for 2 minutes (possible person down)"),
        Rule(id="wh-after-hours", name="Someone on site after hours", preset=p,
             conditions=[TimeWindow(type="time_window", start="22:00", end="06:00"),
                         CountGreaterThan(type="count_greater_than", n=0)],
             duration_s=10, severity="high", cooldown_s=1800, actions=["alert", "record_clip", "notify"],
             source_text="Alert if anyone is here between 22:00 and 06:00"),
    ]
    for z in zones:
        if z.get("zone_type") == "time_limited" and z.get("time_limit") and z.get("active", True):
            rules.append(Rule(id=f"wh-dwell-{_safe(z['id'])}"[:48], name=f"Too long in {z['name']}", preset=p,
                              conditions=[InZone(type="in_zone", zone=z["id"])], duration_s=float(z["time_limit"]),
                              severity="medium", cooldown_s=120,
                              source_text=f"Alert if someone stays in {z['name']} for more than {z['time_limit']:g} s"))
    return rules
