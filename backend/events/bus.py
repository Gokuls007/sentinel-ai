"""In-process publish/subscribe for events.

The pipeline publishes each new event once; subscribers run in the order they subscribed
(the store first, so later subscribers see the event's id). A failing subscriber is logged
and skipped: it must never stop the video pipeline or the other subscribers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from events.schema import Event

logger = logging.getLogger("sentinel.events.bus")

Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[tuple[str, Subscriber]] = []
        self._lock = threading.Lock()

    def subscribe(self, name: str, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers.append((name, callback))

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subscribers = [(n, cb) for n, cb in self._subscribers if n != name]

    def publish(self, event: Event) -> Event:
        with self._lock:
            subscribers = list(self._subscribers)
        for name, callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception("event subscriber %r failed for %s", name, event.alert_id)
        return event
