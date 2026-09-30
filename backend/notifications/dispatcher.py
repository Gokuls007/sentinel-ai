"""Filters, debounces and delivers event notifications on a background thread.

Subscribed to the ``EventBus``: ``handle(event)`` returns immediately, so the video pipeline
never waits on the network. At most one notification is sent per (event type, track_id)
within ``debounce_s`` seconds, and only for events at or above ``min_severity``.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable

from events.schema import SEVERITIES, Event
from notifications.base import Notification, Notifier, NotifierError, summarize

logger = logging.getLogger("sentinel.notifications")


class NotificationDispatcher:
    def __init__(
        self,
        notifiers: list[Notifier],
        *,
        debounce_s: float = 60.0,
        min_severity: str = "medium",
        max_pending: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ):
        if min_severity not in SEVERITIES:
            raise ValueError(f"min_severity must be one of {SEVERITIES}")
        self.notifiers = notifiers
        self.debounce_s = debounce_s
        self._min_rank = SEVERITIES.index(min_severity)
        self._clock = clock
        self._last_sent: dict[tuple[str, int | None], float] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[Notification | None] = queue.Queue(maxsize=max_pending)
        self.stats = {"queued": 0, "sent": 0, "failed": 0, "debounced": 0, "below_severity": 0,
                      "dropped": 0}
        self._thread = threading.Thread(target=self._worker, name="notifications", daemon=True)
        self._thread.start()

    def should_notify(self, event: Event) -> bool:
        """Severity filter plus debounce; records the send time when it says yes."""
        if SEVERITIES.index(event.severity) < self._min_rank:
            self.stats["below_severity"] += 1
            return False
        key = (event.type, event.track_id)
        now = self._clock()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self.debounce_s:
                self.stats["debounced"] += 1
                return False
            self._last_sent[key] = now
            # Bound memory: forget keys older than the debounce window.
            if len(self._last_sent) > 1000:
                cutoff = now - self.debounce_s
                self._last_sent = {k: t for k, t in self._last_sent.items() if t >= cutoff}
        return True

    def handle(self, event: Event) -> None:
        """EventBus subscriber."""
        if not self.notifiers or not self.should_notify(event):
            return
        try:
            self._queue.put_nowait(summarize(event))
            self.stats["queued"] += 1
        except queue.Full:
            self.stats["dropped"] += 1
            logger.warning("notification queue full; dropping %s", event.alert_id)

    def _worker(self) -> None:
        while True:
            notification = self._queue.get()
            try:
                if notification is None:
                    return
                for notifier in self.notifiers:
                    try:
                        notifier.send(notification)
                        self.stats["sent"] += 1
                    except NotifierError as exc:
                        self.stats["failed"] += 1
                        logger.warning("%s notification failed: %s", notifier.name, exc)
                    except Exception:
                        self.stats["failed"] += 1
                        logger.exception("%s notifier crashed", notifier.name)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 30.0) -> None:
        """Wait until queued notifications are delivered (or ``timeout`` passes)."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=1.0)
        self._thread.join(timeout)
