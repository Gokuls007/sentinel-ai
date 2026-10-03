"""Plain-words preview of a rule, written by code (not the LLM), plus warnings worth checking
before confirming."""

from __future__ import annotations

import re

from rules.dsl import REBA_LEVELS, RuleBody

DAY_NAMES = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}


def _secs(s: float) -> str:
    s = float(s)
    if s >= 3600 and s % 3600 == 0:
        return f"{int(s // 3600)} h"
    if s >= 60 and s % 60 == 0:
        return f"{int(s // 60)} min"
    return f"{s:g} s"


def condition_text(c, zones: dict[str, str]) -> str:
    zn = lambda z: zones.get(z, z)  # noqa: E731
    t = c.type
    if t == "in_zone":
        return f"in {zn(c.zone)}"
    if t == "not_in_zone":
        return f"not in {zn(c.zone)}"
    if t == "fallen":
        return "has fallen"
    if t == "stationary_for":
        return f"hasn't moved for {_secs(c.seconds)}"
    if t == "posture_risk_at_least":
        return f"posture risk {REBA_LEVELS[c.level]} or worse"
    if t == "count_greater_than":
        return f"more than {c.n} {'person' if c.n == 1 else 'people'} in view"
    if t == "count_in_zone_greater_than":
        return f"more than {c.n} {'person' if c.n == 1 else 'people'} in {zn(c.zone)}"
    if t == "time_window":
        days = f" on {', '.join(DAY_NAMES[d] for d in c.days)}" if c.days else ""
        if c.start == "00:00" and c.end == "23:59":
            return f"all day{days}" if days else "at any time"
        return f"between {c.start} and {c.end}{days}"
    if t == "holding_object":
        return f"holding a {'phone' if c.object == 'cell phone' else c.object}"
    if t == "head_turned":
        side = "left or right" if c.direction == "either" else c.direction
        return f"head turned {side} ≥{c.min_angle:g}°"
    if t == "looking_down":
        return "looking down"
    return t


def describe(body: RuleBody, zones: dict[str, str]) -> str:
    """e.g. 'Person · in Loading Dock · for 30 s → medium alert, clip'."""
    parts = ["Person" if body.per_person else "Scene"]
    parts += [condition_text(c, zones) for c in body.conditions]
    if body.duration_s:
        parts.append(f"for {_secs(body.duration_s)}")
    acts = [a for a in body.actions if a != "alert"]
    names = {"record_clip": "clip", "notify": "notify"}
    tail = f"{body.severity} alert" + (", " + ", ".join(names[a] for a in acts) if acts else "")
    return " · ".join(parts) + f" → {tail}"


# Words that show a sentence asked for a condition. A condition with none of them in the sentence
# was probably added by the model on its own ("reading a book" -> looking_down), so the preview
# says so. Zones are checked separately (by name).
CUES = {
    "fallen": ("fall", "fell", "down", "collapse", "lying", "floor"),
    "stationary_for": ("still", "move", "moving", "motionless", "idle", "stationary", "frozen", "stuck"),
    "posture_risk_at_least": ("posture", "ergonom", "lift", "bend", "awkward", "strain", "reba", "risk"),
    "count_greater_than": ("people", "persons", "more than", "crowd", "anyone", "anybody", "someone", "nobody",
                           "number", "or more"),
    "count_in_zone_greater_than": ("people", "persons", "more than", "crowd", "or more", "number"),
    "time_window": ("am", "pm", ":", "night", "morning", "evening", "midnight", "noon", "weekend", "weekday",
                    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "hour",
                    "o'clock", "after", "before", "between", "day"),
    "holding_object": ("phone", "laptop", "book", "holding", "carrying", "using", "reading", "texting", "mobile"),
    "head_turned": ("turn", "head", "look", "glanc", "side", "left", "right", "neighbour", "neighbor"),
    "looking_down": ("down", "look", "lap", "floor", "desk"),
}


def unrequested(body: RuleBody, source_text: str) -> list[str]:
    """Condition types with no cue at all in the sentence (probably added by the model)."""
    text = source_text.lower()
    out = []
    for c in body.conditions:
        cues = CUES.get(c.type)
        if cues and not any(re.search((r"\b" if w[0].isalpha() else "") + re.escape(w), text)
                            for w in cues):
            out.append(c.type)
    return out


def warnings(body: RuleBody, zones: dict[str, str], source_text: str) -> list[str]:
    """Things to check: zone names matched loosely, instant rules that may be noisy, etc."""
    out = []
    text = source_text.lower()
    for ctype in unrequested(body, source_text):
        out.append(f"Your sentence doesn't seem to ask for '{ctype.replace('_', ' ')}'. Remove it by rephrasing "
                   "if it wasn't intended.")
    for z in sorted(body.zones()):
        name = zones.get(z, z)
        words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2]
        if name.lower() not in text and not all(w in text for w in words):
            out.append(f"Matched to zone '{name}'. Check it's the zone you meant.")
    types = {c.type for c in body.conditions}
    if not body.duration_s and types & {"head_turned", "looking_down", "holding_object", "not_in_zone"}:
        out.append("Fires the moment this is seen. A duration (e.g. 5 s) avoids alerts on brief glances.")
    if "head_turned" in types or "looking_down" in types:
        out.append("Head direction is a 2D estimate from the face keypoints; it needs a clear view of the face.")
    if "holding_object" in types:
        out.append("Small objects like phones are detected less reliably than people, especially far away.")
    if "posture_risk_at_least" in types:
        out.append("Posture risk (2D REBA) is most reliable from a side view.")
    if body.cooldown_s < 10:
        out.append(f"Cooldown is only {_secs(body.cooldown_s)}: this rule could alert often.")
    return out
