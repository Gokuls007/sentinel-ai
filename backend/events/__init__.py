"""Unified events: schema, SQLite store with migrations, and the in-process bus."""

from events.bus import EventBus
from events.schema import DEFAULT_CAMERA, EVENT_TYPES, SEVERITIES, Event
from events.store import GROUP_BY_KEYS, EventStore

__all__ = [
    "DEFAULT_CAMERA",
    "EVENT_TYPES",
    "GROUP_BY_KEYS",
    "SEVERITIES",
    "Event",
    "EventBus",
    "EventStore",
]
