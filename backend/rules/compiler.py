"""Plain English -> a validated rule, once, with the provider-agnostic LLM client.

The model sees the rule format, this site's real zones and the object classes, and must call
one of two tools: ``submit_rule`` (a rule body) or ``refuse`` (what's missing). The rule is
validated by Pydantic and against the real zones; on errors the model gets them back once to
fix (one repair turn). It never sees video, frames or events. Nothing runs until a person
confirms the preview, which is written by code (rules/describe.py), not by the model.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from llm.base import LLMClient, LLMError, Message, ToolSpec, Usage
from rules.describe import describe, warnings
from rules.dsl import OBJECT_CLASSES, UNAVAILABLE, RuleBody, check_references, format_errors

MAX_TEXT = 300

SYSTEM_PROMPT = """You turn one plain-English safety rule into a structured rule for a camera system.
Call exactly one tool: submit_rule with the rule, or refuse when it can't be expressed.

What the system can check (conditions; all of a rule's conditions must hold at once):
- in_zone(zone) / not_in_zone(zone): the person is inside / outside a drawn zone. Use zone ids
  from the list below.
- fallen: the person has fallen (confirmed fall).
- stationary_for(seconds, radius_body_heights=0.3): the person hasn't moved from one spot for that long.
- posture_risk_at_least(level): 2D REBA ergonomic risk; 2 low, 3 medium, 4 high, 5 very high
  ("bad posture", "awkward lifting" = 4 unless stated).
- count_greater_than(n): more than n people in view. "3 or more" means n=2.
- count_in_zone_greater_than(zone, n): more than n people in a zone.
- time_window(start "HH:MM", end "HH:MM", days=[mon..sun]): local time; can wrap past midnight.
  Days without hours ("at the weekend", "on Sundays", "on weekdays") mean all day on those days:
  start "00:00", end "23:59" with those days. "After hours" or "at night" with no times given
  need explicit times: refuse.
- holding_object(object): one of {objects} at the person's hand ("phone" = "cell phone").
- head_turned(direction left|right|either, min_angle 15-90, default 30): the person's head turned.
- looking_down: head pitched down.

Use only the conditions the sentence asks for. Do not add conditions that merely seem implied or
likely: "reading a book" is holding_object("book") only, not also looking_down. Every condition
you include must correspond to words in the sentence.

Fields:
- name: short, e.g. "Loading dock dwell".
- duration_s: how long the conditions must hold continuously ("for more than 30 seconds" = 30,
  "for 2 minutes" = 120; 0 = immediately).
- severity: low|medium|high|critical. Medium unless the text implies otherwise ("danger",
  "emergency", falls = high or critical).
- cooldown_s: default 60.
- actions: always include "alert"; add "record_clip" by default; add "notify" only if the text asks
  to notify, send, message or page someone.

Zones at this site (id: name, type):
{zones}

Refuse (do not guess) when the rule needs something not in the list above, for example: protective
gear (helmets, vests), distance to vehicles or objects like forklifts, identifying who someone is,
emotions or intent ("looks tired", "is cheating", "acts suspicious"), sounds, or a zone that isn't
listed. Desk posture at a computer (slouching, sitting up straight) is the separate Desk Posture
Coach mode, not a rule: refuse it. In the refusal, say plainly what is missing. Never invent a zone,
object, or condition.
Do not ask questions; refuse instead if it's ambiguous.
Local time now: {now}."""

REFUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "What is missing or can't be checked, in plain words."},
        "unsupported": {"type": "array", "items": {"type": "string"},
                        "description": "The parts of the request that can't be checked, e.g. ['helmet']."},
    },
    "required": ["reason"],
}


def _clean(schema):
    """Pydantic's JSON schema without titles or discriminator hints (some providers reject them)."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k not in ("title", "discriminator")}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("submit_rule", "Submit the structured rule.", _clean(copy.deepcopy(RuleBody.model_json_schema()))),
        ToolSpec("refuse", "Refuse when the rule can't be expressed with the available conditions.", REFUSE_SCHEMA),
    ]


@dataclass
class CompileResult:
    status: str  # "rule" | "refusal" | "error"
    text: str
    body: RuleBody | None = None
    preview: str | None = None
    warnings: list[str] = field(default_factory=list)
    refusal: str | None = None
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status, "text": self.text,
            "rule": self.body.model_dump(mode="json") if self.body else None,
            "preview": self.preview, "warnings": self.warnings, "refusal": self.refusal,
            "unsupported": self.unsupported, "error": self.error, "attempts": self.attempts,
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
            "provider": self.provider, "model": self.model, "latency_s": self.latency_s,
        }


class RuleCompiler:
    def __init__(self, llm: LLMClient, max_attempts: int = 2, max_tokens: int = 4096):
        self.llm = llm
        self.max_attempts = max_attempts  # the first try plus one repair turn
        self.max_tokens = max_tokens
        self.specs = tool_specs()

    def system_prompt(self, zones: dict[str, tuple[str, str]], now: str) -> str:
        zone_lines = "\n".join(f"- {zid}: {name} ({ztype})" for zid, (name, ztype) in zones.items()) or "- (none drawn)"
        return SYSTEM_PROMPT.format(objects=", ".join(f'"{o}"' for o in OBJECT_CLASSES), zones=zone_lines, now=now)

    def compile(self, text: str, zones: dict[str, tuple[str, str]], now: str = "") -> CompileResult:
        """``zones`` maps zone id -> (name, type) for the cameras the rule will run on."""
        t0 = time.time()
        text = (text or "").strip()[:MAX_TEXT]
        res = CompileResult(status="error", text=text, provider=self.llm.provider, model=self.llm.model)
        names = {zid: name for zid, (name, _t) in zones.items()}

        def done(**kw) -> CompileResult:
            for k, v in kw.items():
                setattr(res, k, v)
            res.latency_s = round(time.time() - t0, 3)
            return res

        if not text:
            return done(error="empty rule text")
        system = self.system_prompt(zones, now)
        messages = [Message(role="user", content=text)]
        problems: list[str] = []
        while res.attempts < self.max_attempts:
            try:
                resp = self.llm.chat(system, messages, self.specs, max_tokens=self.max_tokens)
            except LLMError as e:
                return done(error=str(e))
            res.attempts += 1
            res.usage.add(resp.usage)
            messages.append(resp.message)
            if not resp.tool_calls:
                problems = ["no tool was called"]
                messages.append(Message(role="user", content="Call submit_rule or refuse."))
                continue
            call = resp.tool_calls[0]
            if call.name == "refuse":
                args = call.arguments if isinstance(call.arguments, dict) else {}
                reason = str(args.get("reason") or "This rule can't be expressed with the available checks.")
                unsupported = [str(u) for u in args.get("unsupported") or []][:10]
                return done(status="refusal", refusal=reason[:500], unsupported=unsupported)
            body, problems = self._check(call, names)
            if body is not None:
                return done(status="rule", body=body, preview=describe(body, names),
                            warnings=warnings(body, names, text))
            unavailable = self._unavailable(call)
            if unavailable:
                return done(status="refusal", refusal=f"Not available yet: {UNAVAILABLE[unavailable]}.",
                            unsupported=[unavailable])
            feedback = ("That rule is invalid:\n- " + "\n- ".join(problems) +
                        "\nFix it and call submit_rule again, or call refuse if it can't be expressed.")
            for i, c in enumerate(resp.tool_calls):
                messages.append(Message(role="tool", tool_call_id=c.id,
                                        content=feedback if i == 0 else "Ignored: call one tool at a time."))
        return done(error="couldn't produce a valid rule: " + "; ".join(problems))

    @staticmethod
    def _unavailable(call) -> str | None:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        for c in args.get("conditions") or []:
            if isinstance(c, dict) and c.get("type") in UNAVAILABLE:
                return c["type"]
        return None

    @staticmethod
    def _check(call, names: dict[str, str]) -> tuple[RuleBody | None, list[str]]:
        if call.name != "submit_rule":
            return None, [f"unknown tool {call.name!r}; use submit_rule or refuse"]
        if call.parse_error:
            return None, [f"the arguments weren't valid JSON: {call.parse_error}"]
        args = call.arguments
        if isinstance(args, dict) and isinstance(args.get("rule"), dict) and "conditions" not in args:
            args = args["rule"]  # some models wrap the object
        try:
            body = RuleBody.model_validate(args)
        except ValidationError as e:
            # Name invented zones too, so one repair turn can fix everything at once.
            conds = args.get("conditions") if isinstance(args, dict) else None
            zones = {c.get("zone") for c in conds or [] if isinstance(c, dict) and isinstance(c.get("zone"), str)}
            unknown = sorted(z for z in zones if z not in names)
            extra = [f"zone {z!r} doesn't exist; known zones: {', '.join(names) or 'none'}" for z in unknown]
            return None, format_errors(e) + extra
        problems = check_references(body, names)
        return (None, problems) if problems else (body, [])


def dumps_rule(body: RuleBody) -> str:
    return json.dumps(body.model_dump(mode="json"), sort_keys=True)
