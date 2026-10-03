"""Checks the confirmed rules every frame (deterministic code; no LLM here).

For each rule and each person (or once for the scene, for rules with only scene
conditions) a timer runs while every condition holds. When it has held for ``duration_s`` the
rule fires: **once per episode** (the conditions must stop holding before it can fire again
for that person) and no sooner than ``cooldown_s`` after its last firing. A condition that
drops out for under ``grace_s`` (a missed keypoint, one bad frame) doesn't restart the timer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

from rules.dsl import DAYS, SCENE_TYPES, Rule

SCENE = "scene"


@dataclass
class PersonState:
    track_id: int
    point: tuple[float, float]          # hip midpoint, px (what zones are tested with)
    body_height: float                  # px
    zones: set[str] = field(default_factory=set)
    fallen: bool = False                # confirmed fall
    reba_level: int | None = None       # reliable smoothed REBA level only
    holding: set[str] = field(default_factory=set)
    head: tuple[str, float, float] | None = None  # (left|right, degrees, confidence)
    looking_down: bool | None = None


@dataclass
class SceneState:
    ts: float
    camera_id: str
    persons: dict[int, PersonState]
    local_time: datetime | None = None  # defaults to the local time of ts

    def zone_count(self, zone: str) -> int:
        return sum(zone in p.zones for p in self.persons.values())


@dataclass
class Firing:
    rule: Rule
    track_id: int | None
    ts: float
    held_s: float
    zone_id: str | None


@dataclass
class _Timer:
    since: float
    last_true: float
    fired: bool = False


class RuleEngine:
    def __init__(self, rules: list[Rule] | None = None, grace_s: float = 1.0, head_min_conf: float = 0.5):
        self.grace_s = grace_s
        self.head_min_conf = head_min_conf
        self._lock = threading.Lock()
        self._rules: list[Rule] = []
        self._timers: dict[tuple[str, object], _Timer] = {}
        self._last_fire: dict[tuple[str, object], float] = {}
        self._anchors: dict[tuple[int, float], tuple[tuple[float, float], float]] = {}  # (tid, radius)
        self.fire_counts: dict[str, int] = {}
        self.last_fired: dict[str, float] = {}
        self.set_rules(rules or [])

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    def set_rules(self, rules: list[Rule]) -> None:
        """Swap the rule set (thread-safe). Timers of rules that changed or went away are dropped."""
        with self._lock:
            keep = {(r.id, r.version) for r in rules}
            old = {r.id: r.version for r in self._rules}
            changed = {rid for rid, v in old.items() if (rid, v) not in keep}
            self._rules = [r for r in rules if r.enabled]
            for d in (self._timers, self._last_fire):
                for k in [k for k in d if k[0] in changed]:
                    del d[k]

    # --- conditions ----------------------------------------------------------------------------

    def _stationary(self, p: PersonState, radius_bh: float, ts: float) -> float:
        """Seconds the person has stayed within ``radius_bh`` body heights of one spot."""
        key = (p.track_id, radius_bh)
        anchor = self._anchors.get(key)
        limit = radius_bh * max(p.body_height, 1.0)
        if anchor is None or ((p.point[0] - anchor[0][0]) ** 2 + (p.point[1] - anchor[0][1]) ** 2) ** 0.5 > limit:
            self._anchors[key] = (p.point, ts)
            return 0.0
        return ts - anchor[1]

    def _person_ok(self, c, p: PersonState, ts: float) -> bool:
        t = c.type
        if t == "in_zone":
            return c.zone in p.zones
        if t == "not_in_zone":
            return c.zone not in p.zones
        if t == "fallen":
            return p.fallen
        if t == "stationary_for":
            return self._stationary(p, c.radius_body_heights, ts) >= c.seconds
        if t == "posture_risk_at_least":
            return p.reba_level is not None and p.reba_level >= c.level
        if t == "holding_object":
            return c.object in p.holding
        if t == "head_turned":
            if p.head is None or p.head[2] < self.head_min_conf or p.head[1] < c.min_angle:
                return False
            return c.direction == "either" or c.direction == p.head[0]
        if t == "looking_down":
            return bool(p.looking_down)
        raise ValueError(f"not a person condition: {t}")

    @staticmethod
    def _scene_ok(c, scene: SceneState, now: datetime) -> bool:
        t = c.type
        if t == "count_greater_than":
            return len(scene.persons) > c.n
        if t == "count_in_zone_greater_than":
            return scene.zone_count(c.zone) > c.n
        if t == "time_window":
            hm = now.strftime("%H:%M")
            inside = c.start <= hm < c.end if c.start < c.end else (hm >= c.start or hm < c.end)
            if not inside:
                return False
            if not c.days:
                return True
            # A window that wraps past midnight belongs to the day it started.
            day = now.weekday() if (c.start < c.end or hm >= c.start) else (now.weekday() - 1) % 7
            return DAYS[day] in c.days
        raise ValueError(f"not a scene condition: {t}")

    # --- per frame ------------------------------------------------------------------------------

    def evaluate(self, scene: SceneState) -> list[Firing]:
        ts = scene.ts
        now = scene.local_time or datetime.fromtimestamp(ts)
        with self._lock:
            rules = [r for r in self._rules if not r.camera_ids or scene.camera_id in r.camera_ids]
        firings: list[Firing] = []
        seen_keys = set()
        for rule in rules:
            person_conds = [c for c in rule.conditions if c.type not in SCENE_TYPES]
            scene_conds = [c for c in rule.conditions if c.type in SCENE_TYPES]
            scene_ok = all(self._scene_ok(c, scene, now) for c in scene_conds)
            if person_conds:
                for tid, p in scene.persons.items():
                    # Evaluate every condition (stationary anchors must update each frame).
                    results = [self._person_ok(c, p, ts) for c in person_conds]
                    key = (rule.id, tid)
                    seen_keys.add(key)
                    f = self._step(rule, key, scene_ok and all(results), ts, tid, self._zone_of(rule, p))
                    if f:
                        firings.append(f)
            else:
                key = (rule.id, SCENE)
                seen_keys.add(key)
                zone = next((c.zone for c in scene_conds if hasattr(c, "zone")), None)
                f = self._step(rule, key, scene_ok, ts, None, zone)
                if f:
                    firings.append(f)
        # People who left: drop their timers after the grace period, and their anchors.
        for key in [k for k, t in self._timers.items() if k not in seen_keys and ts - t.last_true > self.grace_s]:
            del self._timers[key]
        present = set(scene.persons)
        for key in [k for k in self._anchors if k[0] not in present]:
            del self._anchors[key]
        return firings

    @staticmethod
    def _zone_of(rule: Rule, p: PersonState) -> str | None:
        return next((c.zone for c in rule.conditions if c.type == "in_zone" and c.zone in p.zones), None)

    def _step(self, rule: Rule, key, ok: bool, ts: float, tid, zone) -> Firing | None:
        timer = self._timers.get(key)
        if ok:
            if timer is None:
                timer = self._timers[key] = _Timer(since=ts, last_true=ts)
            timer.last_true = ts
        elif timer is not None:
            if ts - timer.last_true > self.grace_s:
                del self._timers[key]  # the episode is over
            return None
        else:
            return None
        held = ts - timer.since
        if timer.fired or held < rule.duration_s:
            return None
        last = self._last_fire.get(key)
        if last is not None and ts - last < rule.cooldown_s:
            return None
        timer.fired = True
        self._last_fire[key] = ts
        self.fire_counts[rule.id] = self.fire_counts.get(rule.id, 0) + 1
        self.last_fired[rule.id] = ts
        return Firing(rule=rule, track_id=tid, ts=ts, held_s=round(held, 2), zone_id=zone)
