"""The unified event record every detector's output ends up as.

Detectors keep producing ``AnomalyAlert`` objects; the pipeline turns each one into an
``Event`` (adding camera, zone, clip and thumbnail) and publishes it on the ``EventBus``.
Search, rules, reports and the dashboard all read this one shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("low", "medium", "high", "critical")

# Known event types. Zone sub-types (time_exceeded, wrong_direction) predate this schema and
# are kept so existing dashboards and filters keep working; rule events are "rule:<rule_id>".
EVENT_TYPES = (
    "fall",
    "possible_fall",  # reached the ground, not yet confirmed: dashboard only, no notification
    "zone_intrusion",
    "time_exceeded",
    "wrong_direction",
    "loitering",
    "action",
    "ergo_risk",
    "near_miss",
)

DEFAULT_CAMERA = "cam-0"


def is_valid_type(event_type: str) -> bool:
    return event_type in EVENT_TYPES or (event_type.startswith("rule:") and len(event_type) > 5)


@dataclass
class Event:
    type: str
    severity: str
    start_ts: float
    end_ts: float
    camera_id: str = DEFAULT_CAMERA
    track_id: int | None = None
    zone_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    clip_path: str | None = None
    thumbnail_path: str | None = None
    verified: int | None = None  # set asynchronously by VLM verification (Phase 3)
    # Kept from the original schema: clip folders and URLs are named by alert_id.
    alert_id: str | None = None
    message: str = ""
    confidence: float | None = None
    id: int | None = None  # assigned by the store

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")
        if not is_valid_type(self.type):
            raise ValueError(f"unknown event type {self.type!r}")
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must not be before start_ts")

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_alert(
        cls,
        alert: Any,
        *,
        camera_id: str = DEFAULT_CAMERA,
        clip_path: str | None = None,
        thumbnail_path: str | None = None,
    ) -> Event:
        """Build an event from an ``AnomalyAlert``.

        Alerts fire at one instant. For conditions that built up over time (dwell in a
        time-limited zone, loitering) the event starts when the condition began, so
        ``end_ts - start_ts`` is how long it lasted.
        """
        details = dict(alert.details or {})
        zone_id = details.pop("zone_id", None)
        details.pop("clip_path", None)  # a column now
        lasted = details.get("dwell_seconds") or details.get("duration") or 0.0
        try:
            lasted = max(0.0, float(lasted))
        except (TypeError, ValueError):
            lasted = 0.0
        return cls(
            type=alert.alert_type,
            severity=alert.severity,
            start_ts=float(alert.timestamp) - lasted,
            end_ts=float(alert.timestamp),
            camera_id=camera_id,
            track_id=alert.track_id,
            zone_id=zone_id,
            attributes=details,
            clip_path=clip_path or None,
            thumbnail_path=thumbnail_path,
            alert_id=alert.alert_id,
            message=alert.message,
            confidence=alert.confidence,
        )

    def to_alert_dict(self) -> dict[str, Any]:
        """The pre-Phase-0 alert shape, for /api/alerts and the WebSocket (unchanged clients)."""
        details = dict(self.attributes)
        if self.zone_id is not None:
            details["zone_id"] = self.zone_id
        if self.clip_path:
            details["clip_path"] = self.clip_path
        return {
            "alert_id": self.alert_id,
            "alert_type": self.type,
            "track_id": self.track_id,
            "timestamp": self.end_ts,
            "confidence": self.confidence,
            "severity": self.severity,
            "message": self.message,
            "details": details,
            "event_id": self.id,
            "camera_id": self.camera_id,
        }
