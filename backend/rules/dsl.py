"""The rule language: what a plain-English rule compiles into.

A rule is "a person" plus conditions that must all hold, for at least ``duration_s``, then
actions. Conditions are a closed set (a discriminated union): the compiler can only pick from
these, and anything else is rejected with a clear message, so an LLM can never invent a
check that doesn't exist. Rules are checked every frame by deterministic code
(``rules/engine.py``); the LLM is used once, to compile the sentence.

Per-person conditions are judged for each tracked person; scene conditions (counts, time of
day) once per frame. A rule with only scene conditions fires once for the scene.

Not in 3a (the compiler refuses them and says why): near_object / missing_object (helmets,
vests, forklifts: open-vocabulary detection, 3b), looking_at_seat (exam seats, 3d) and
posture_deviation (the desk posture coach is its own mode).
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SEVERITIES = ("low", "medium", "high", "critical")
ACTIONS = ("alert", "record_clip", "notify")
OBJECT_CLASSES = ("cell phone", "laptop", "book")  # COCO classes the detector already knows
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
REBA_LEVELS = {1: "negligible", 2: "low", 3: "medium", 4: "high", 5: "very high"}
UNAVAILABLE = {
    "near_object": "distance to objects like forklifts needs open-vocabulary detection (Phase 3b)",
    "missing_object": "helmet and vest checks need open-vocabulary detection (Phase 3b)",
    "looking_at_seat": "seat zones for exam halls come with the exam preset (Phase 3d)",
    "posture_deviation": "desk posture is handled by the Desk Posture Coach mode, not rules",
}


class _Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InZone(_Condition):
    """The person's hip point is inside the zone."""
    type: Literal["in_zone"]
    zone: str = Field(..., min_length=1, max_length=64, description="zone id")


class NotInZone(_Condition):
    """The person is visible and outside the zone."""
    type: Literal["not_in_zone"]
    zone: str = Field(..., min_length=1, max_length=64, description="zone id")


class Fallen(_Condition):
    """The fall detector says the person is down (a confirmed fall)."""
    type: Literal["fallen"]


class StationaryFor(_Condition):
    """The person has stayed within ``radius_body_heights`` of one spot for ``seconds``."""
    type: Literal["stationary_for"]
    seconds: float = Field(..., gt=0, le=3600)
    radius_body_heights: float = Field(0.3, gt=0, le=2)


class PostureRiskAtLeast(_Condition):
    """Smoothed 2D REBA risk level (1 negligible .. 5 very high), reliable frames only."""
    type: Literal["posture_risk_at_least"]
    level: int = Field(..., ge=2, le=5)


class CountGreaterThan(_Condition):
    """More than ``n`` people in view."""
    type: Literal["count_greater_than"]
    n: int = Field(..., ge=0, le=100)


class CountInZoneGreaterThan(_Condition):
    """More than ``n`` people inside the zone."""
    type: Literal["count_in_zone_greater_than"]
    zone: str = Field(..., min_length=1, max_length=64)
    n: int = Field(..., ge=0, le=100)


class TimeWindow(_Condition):
    """Local time of day between ``start`` and ``end`` (HH:MM, may wrap past midnight), on
    ``days`` (all days if empty)."""
    type: Literal["time_window"]
    start: str
    end: str
    days: list[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]] = Field(default_factory=list)

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("time must be HH:MM (24-hour), e.g. 22:00")
        return v

    @model_validator(mode="after")
    def _not_empty(self):
        if self.start == self.end:
            raise ValueError("start and end are the same time: the window would be empty")
        return self


class HoldingObject(_Condition):
    """A detected object (COCO class) is at one of the person's hands."""
    type: Literal["holding_object"]
    object: Literal["cell phone", "laptop", "book"]


class HeadTurned(_Condition):
    """Head turned to the person's own left or right by at least ``min_angle`` degrees
    (2D estimate from the nose between the ears; low-confidence frames never count)."""
    type: Literal["head_turned"]
    direction: Literal["left", "right", "either"] = "either"
    min_angle: float = Field(30, ge=15, le=90)


class LookingDown(_Condition):
    """Head pitched down: the nose well below the ear line (2D estimate)."""
    type: Literal["looking_down"]


Condition = Annotated[
    InZone | NotInZone | Fallen | StationaryFor | PostureRiskAtLeast | CountGreaterThan
    | CountInZoneGreaterThan | TimeWindow | HoldingObject | HeadTurned | LookingDown,
    Field(discriminator="type"),
]
CONDITION_TYPES = ("in_zone", "not_in_zone", "fallen", "stationary_for", "posture_risk_at_least",
                   "count_greater_than", "count_in_zone_greater_than", "time_window", "holding_object",
                   "head_turned", "looking_down")
SCENE_TYPES = ("count_greater_than", "count_in_zone_greater_than", "time_window")


class RuleBody(BaseModel):
    """What the compiler produces (the parts of a rule that come from the sentence)."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=80)
    subject: Literal["person"] = "person"
    conditions: list[Condition] = Field(..., min_length=1, max_length=6)
    duration_s: float = Field(0, ge=0, le=3600, description="conditions must hold this long")
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    cooldown_s: float = Field(60, ge=0, le=86400, description="minimum time between alerts per person")
    actions: list[Literal["alert", "record_clip", "notify"]] = Field(
        default_factory=lambda: ["alert", "record_clip"])

    @model_validator(mode="after")
    def _consistent(self):
        keys = [c.model_dump_json() for c in self.conditions]
        if len(set(keys)) != len(keys):
            raise ValueError("the same condition appears twice")
        zones_in = {c.zone for c in self.conditions if c.type == "in_zone"}
        zones_out = {c.zone for c in self.conditions if c.type == "not_in_zone"}
        if zones_in & zones_out:
            raise ValueError(f"in and not in the same zone at once can never hold: {sorted(zones_in & zones_out)}")
        if sum(c.type == "time_window" for c in self.conditions) > 1:
            raise ValueError("use one time_window per rule")
        if "alert" not in self.actions:
            raise ValueError("every rule needs the 'alert' action (it is what records the event)")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("an action is listed twice")
        return self

    @property
    def per_person(self) -> bool:
        return any(c.type not in SCENE_TYPES for c in self.conditions)

    def zones(self) -> set[str]:
        return {c.zone for c in self.conditions if hasattr(c, "zone")}

    def objects(self) -> set[str]:
        return {c.object for c in self.conditions if c.type == "holding_object"}


class Rule(RuleBody):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")
    enabled: bool = True
    preset: str | None = None
    camera_ids: list[str] = Field(default_factory=list, description="empty = every camera")
    source_text: str = Field("", max_length=500)
    builtin: bool = False
    version: int = 1


def check_references(body: RuleBody, zones: dict[str, str]) -> list[str]:
    """Problems with things the rule refers to: zones must exist (by id). ``zones`` maps
    id -> name for the cameras the rule runs on."""
    problems = []
    for z in sorted(body.zones()):
        if z not in zones:
            known = ", ".join(f"{i} ({n})" for i, n in zones.items()) or "none are drawn"
            problems.append(f"zone {z!r} doesn't exist; known zones: {known}")
    return problems


def format_errors(err: ValidationError) -> list[str]:
    """Readable validation messages, e.g. 'conditions.0.seconds: Input should be greater than 0'."""
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"] if not str(p).startswith("function-"))
        out.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return out


def slugify(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "rule"
    rid, n = base, 2
    while rid in taken:
        rid, n = f"{base}-{n}", n + 1
    return rid
