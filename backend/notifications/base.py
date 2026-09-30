"""Common notification types: what gets sent, and the interface every channel implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from events.schema import Event

_ICONS = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "ℹ️"}


@dataclass
class Notification:
    event: Event
    title: str
    text: str
    image_path: str | None = None  # the event's thumbnail (a JPEG), if it exists


def summarize(event: Event) -> Notification:
    """Short human-readable summary of an event, used by every channel."""
    when = datetime.fromtimestamp(event.end_ts).strftime("%Y-%m-%d %H:%M:%S")
    title = f"{_ICONS.get(event.severity, '')} Sentinel AI: {event.type.replace('_', ' ')} ({event.severity})".strip()
    lines = [event.message or event.type, f"Time: {when}", f"Camera: {event.camera_id}"]
    if event.zone_id:
        lines.append(f"Zone: {event.zone_id}")
    if event.track_id is not None:
        lines.append(f"Person: track #{event.track_id}")
    if event.duration_s >= 1:
        lines.append(f"Duration: {event.duration_s:.0f}s")
    if event.confidence is not None:
        lines.append(f"Confidence: {event.confidence:.0%}")
    if event.alert_id:
        lines.append(f"Event: {event.alert_id}")
    return Notification(event=event, title=title, text="\n".join(lines), image_path=event.thumbnail_path)


class NotifierError(RuntimeError):
    """Delivery failed. The message must never contain credentials."""


class Notifier(ABC):
    name: str

    @abstractmethod
    def send(self, notification: Notification) -> None:
        """Deliver one notification or raise NotifierError."""
