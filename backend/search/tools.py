"""Read-only tools the search agent may call. Every query goes through ``EventStore``, whose
SQL is parameterized and whose column / group / order names come from fixed allow-lists, so
nothing the model sends can change or reach beyond the event log.

Times are exchanged in local time as ISO 8601 (``2026-09-30T14:05:00``), which is how people
ask ("after 6pm yesterday"). Zones may be named by id or by display name.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from events.schema import EVENT_TYPES, SEVERITIES, Event, is_valid_type
from events.store import ORDERS, EventStore
from llm.base import ToolSpec

MAX_FIND_LIMIT = 50
DEFAULT_FIND_LIMIT = 10
MAX_RESULT_CHARS = 12_000
WEEKDAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
GROUPS = ("type", "severity", "zone", "camera", "track", "hour", "day", "weekday")

TYPE_HELP = {
    "fall": "a person fell",
    "zone_intrusion": "someone entered a restricted zone",
    "time_exceeded": "someone stayed in a time-limited zone too long",
    "wrong_direction": "someone moved the wrong way through a one-way zone",
    "loitering": "someone stayed in one spot too long",
    "action": "a notable action",
    "ergo_risk": "high-risk working posture (REBA)",
    "near_miss": "a near miss",
}


class ToolError(ValueError):
    """Bad arguments; the message goes back to the model so it can correct itself."""


_FILTER_PROPS: dict[str, Any] = {
    "types": {"type": "array", "items": {"type": "string", "enum": list(EVENT_TYPES)},
              "description": "Only these event types."},
    "min_severity": {"type": "string", "enum": list(SEVERITIES),
                     "description": "Only events at this severity or higher."},
    "zone": {"type": "string", "description": "Zone id or name (e.g. 'loading dock')."},
    "camera": {"type": "string", "description": "Camera id."},
    "track_id": {"type": "integer", "description": "One tracked person (ids are per camera)."},
    "start": {"type": "string",
              "description": "Local time, ISO 8601 (e.g. 2026-09-30T18:00:00 or 2026-09-30). Events that "
                             "overlap [start, end] match."},
    "end": {"type": "string", "description": "Local time, ISO 8601. A bare date means the end of that day."},
}


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("current_time", "The current local date, time and weekday. Use it to resolve 'today', "
                                 "'yesterday', 'this week', 'last night'.",
                 {"type": "object", "properties": {}}),
        ToolSpec("list_values", "Known values for a field, with event counts: zones (ids and names), "
                                "cameras, event types or severities.",
                 {"type": "object", "properties": {
                     "field": {"type": "string", "enum": ["zone", "camera", "type", "severity"]}},
                  "required": ["field"]}),
        ToolSpec("count_events", "Count events matching the filters, optionally grouped. Use this for "
                                 "'how many' questions instead of listing events. hour = hour of day "
                                 "00-23, weekday = day of week, day = calendar date, track = person.",
                 {"type": "object", "properties": {
                     "group_by": {"type": "string", "enum": list(GROUPS)}, **_FILTER_PROPS}}),
        ToolSpec("find_events", "List events matching the filters (at most 50). Returns the total "
                                "number of matches too, so you know if the list is cut short.",
                 {"type": "object", "properties": {
                     **_FILTER_PROPS,
                     "order": {"type": "string", "enum": list(ORDERS),
                               "description": "newest (default), oldest, or longest duration first."},
                     "limit": {"type": "integer", "minimum": 1, "maximum": MAX_FIND_LIMIT}}}),
        ToolSpec("get_event", "Every stored detail of one event, by id.",
                 {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}),
    ]


class SearchTools:
    def __init__(self, store: EventStore, zone_names: dict[str, str] | None = None,
                 now: Callable[[], float] = time.time):
        self.store = store
        self.zone_names = dict(zone_names or {})  # zone id -> display name
        self.now = now
        self.seen_event_ids: set[int] = set()  # ids returned to the model (citation check)

    # --- dispatch -------------------------------------------------------------------------

    def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "current_time": self.current_time,
            "list_values": self.list_values,
            "count_events": self.count_events,
            "find_events": self.find_events,
            "get_event": self.get_event,
        }.get(name)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        return handler(**arguments) if name != "current_time" else handler()

    @staticmethod
    def to_text(result: dict[str, Any]) -> str:
        text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + '..."(truncated: narrow the filters or lower the limit)"'
        return text

    # --- tools ----------------------------------------------------------------------------

    def current_time(self) -> dict[str, Any]:
        now = datetime.fromtimestamp(self.now())
        return {"now": now.strftime("%Y-%m-%dT%H:%M:%S"), "weekday": now.strftime("%A"),
                "note": "Weeks start on Monday. All times are local."}

    def list_values(self, field: str = "zone", **unknown: Any) -> dict[str, Any]:
        _reject_unknown(unknown)
        if field == "zone":
            counts = self.store.count("zone")
            ids = sorted(set(self.zone_names) | {z for z in counts if z})
            zones = [{"id": z, "name": self.zone_names.get(z, z), "events": counts.get(z, 0)} for z in ids]
            return {"zones": zones, "events_without_zone": counts.get("", 0)}
        if field == "camera":
            return {"cameras": [{"id": k, "events": v} for k, v in self.store.count("camera").items()]}
        if field == "type":
            counts = self.store.count("type")
            return {"types": [{"type": t, "meaning": TYPE_HELP.get(t, ""), "events": counts.get(t, 0)}
                              for t in sorted(set(EVENT_TYPES) | set(counts))]}
        if field == "severity":
            counts = self.store.count("severity")
            return {"severities": [{"severity": s, "events": counts.get(s, 0)} for s in SEVERITIES]}
        raise ToolError("field must be one of zone, camera, type, severity")

    def count_events(self, group_by: str | None = None, **raw: Any) -> dict[str, Any]:
        filters, applied = self._filters(raw)
        total = self.store.total(**filters)
        result: dict[str, Any] = {"filters": applied, "total": total}
        if group_by:
            if group_by not in GROUPS:
                raise ToolError(f"group_by must be one of {', '.join(GROUPS)}")
            counts = self.store.count(group_by, **filters)
            if group_by == "zone":
                result["counts"] = {self._zone_label(k): v for k, v in counts.items()}
            elif group_by == "weekday":
                result["counts"] = {WEEKDAYS[int(k)]: v for k, v in counts.items()}
            elif group_by == "track":
                result["counts"] = {(f"track {k}" if k else "(no track)"): v for k, v in counts.items()}
            else:
                result["counts"] = counts
            result["group_by"] = group_by
        return result

    def find_events(self, order: str = "newest", limit: int = DEFAULT_FIND_LIMIT, **raw: Any) -> dict[str, Any]:
        if order not in ORDERS:
            raise ToolError(f"order must be one of {', '.join(ORDERS)}")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise ToolError("limit must be an integer") from None
        limit = max(1, min(limit, MAX_FIND_LIMIT))
        filters, applied = self._filters(raw)
        total = self.store.total(**filters)
        events = self.store.query(limit=limit, order=order, **filters)
        self.seen_event_ids.update(e.id for e in events if e.id is not None)
        return {"filters": applied, "order": order, "total_matching": total, "returned": len(events),
                "events": [self._compact(e) for e in events]}

    def get_event(self, id: Any = None, **unknown: Any) -> dict[str, Any]:
        _reject_unknown(unknown)
        try:
            event_id = int(id)
        except (TypeError, ValueError):
            raise ToolError("id must be an integer") from None
        event = self.store.get(event_id)
        if event is None:
            return {"found": False, "id": event_id}
        self.seen_event_ids.add(event_id)
        item = self._compact(event)
        item.update({"found": True, "attributes": event.attributes, "confidence": event.confidence,
                     "has_clip": bool(event.clip_path)})
        return item

    # --- helpers --------------------------------------------------------------------------

    def _zone_label(self, zone_id: str) -> str:
        if not zone_id:
            return "(no zone)"
        name = self.zone_names.get(zone_id)
        return f"{name} [{zone_id}]" if name and name != zone_id else zone_id

    def resolve_zone(self, text: str) -> str:
        """Zone id from an id or a (partial, case-insensitive) display name."""
        known = set(self.zone_names) | set(self.store.distinct("zone_id"))
        if text in known:
            return text
        needle = _norm(text)
        exact = [z for z in known if _norm(z) == needle or _norm(self.zone_names.get(z, "")) == needle]
        if len(exact) == 1:
            return exact[0]
        partial = [z for z in known if needle and (needle in _norm(z) or needle in _norm(self.zone_names.get(z, "")))]
        if len(partial) == 1:
            return partial[0]
        options = ", ".join(f"{z} ({self.zone_names.get(z, z)})" for z in sorted(known)) or "none"
        if partial:
            raise ToolError(f"zone {text!r} is ambiguous; matches: {', '.join(sorted(partial))}")
        raise ToolError(f"no zone matches {text!r}; known zones: {options}")

    def _filters(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        allowed = set(_FILTER_PROPS)
        unknown = set(raw) - allowed
        if unknown:
            raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}")
        filters: dict[str, Any] = {}
        applied: dict[str, Any] = {}
        types = raw.get("types")
        if types:
            types = [types] if isinstance(types, str) else list(types)
            bad = [t for t in types if not isinstance(t, str) or not is_valid_type(t)]
            if bad:
                raise ToolError(f"unknown event type(s) {bad}; known: {', '.join(EVENT_TYPES)}")
            filters["types"] = applied["types"] = types
        if raw.get("min_severity"):
            sev = raw["min_severity"]
            if sev not in SEVERITIES:
                raise ToolError(f"min_severity must be one of {', '.join(SEVERITIES)}")
            filters["severity"] = list(SEVERITIES[SEVERITIES.index(sev):])
            applied["min_severity"] = sev
        if raw.get("zone"):
            filters["zone_id"] = self.resolve_zone(str(raw["zone"]))
            applied["zone"] = self._zone_label(filters["zone_id"])
        if raw.get("camera"):
            filters["camera_id"] = applied["camera"] = str(raw["camera"])
        if raw.get("track_id") is not None:
            try:
                filters["track_id"] = applied["track_id"] = int(raw["track_id"])
            except (TypeError, ValueError):
                raise ToolError("track_id must be an integer") from None
        if raw.get("start"):
            filters["start"] = parse_local_time(raw["start"], end_of_day=False)
            applied["start"] = _iso(filters["start"])
        if raw.get("end"):
            filters["end"] = parse_local_time(raw["end"], end_of_day=True)
            applied["end"] = _iso(filters["end"])
        if "start" in filters and "end" in filters and filters["end"] < filters["start"]:
            raise ToolError("end is before start")
        return filters, applied

    def _compact(self, e: Event) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": e.id, "type": e.type, "severity": e.severity, "camera": e.camera_id,
            "start": _iso(e.start_ts), "end": _iso(e.end_ts),
            "weekday": datetime.fromtimestamp(e.start_ts).strftime("%A"),
        }
        if e.end_ts - e.start_ts >= 1:
            item["duration_s"] = round(e.end_ts - e.start_ts)
        if e.zone_id:
            item["zone"] = self._zone_label(e.zone_id)
        if e.track_id is not None:
            item["track_id"] = e.track_id
        if e.message:
            item["message"] = e.message
        if e.verified is not None:
            item["verified"] = bool(e.verified)
        return item


def parse_local_time(value: Any, end_of_day: bool) -> float:
    """ISO 8601 local time (or a bare date) -> unix seconds. A bare date as an end bound means
    the end of that day, so "end: 2026-09-30" includes all of the 30th."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        if len(text) == 10:
            d = date.fromisoformat(text)
            dt = datetime(d.year, d.month, d.day)
            if end_of_day:
                dt += timedelta(days=1) - timedelta(microseconds=1)
            return dt.timestamp()
        return datetime.fromisoformat(text).timestamp()  # naive = local time
    except ValueError:
        raise ToolError(f"can't read time {value!r}; use ISO 8601 like 2026-09-30T18:00:00") from None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _reject_unknown(unknown: dict[str, Any]) -> None:
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}")
