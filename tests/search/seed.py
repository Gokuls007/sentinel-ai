"""A fixed, hand-written event log for testing and evaluating search.

"Now" is Wednesday 2026-09-30 15:00 local time. The week starts on Monday the 28th; "last
week" is Mon 21 to Sun 27 September. Events are inserted oldest first, so ids follow the
order of SEED (the first event gets id 1). Each event has a short key that the eval
questions use instead of raw ids.

Times are local wall-clock times, so the answers are the same on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

NOW = datetime(2026, 9, 30, 15, 0, 0)

CAMERAS = {
    "cam-0": "Warehouse floor",
    "cam-1": "Yard",
    "laptop": "Assembly workbench (laptop webcam)",
}
ZONES = {
    "loading_dock": "Loading Dock",
    "forklift_lane": "Forklift Lane",
    "storage": "Chemical Storage",
    "gate": "Yard Gate",
    "workbench": "Workbench",
}
ZONE_CAMERA = {"loading_dock": "cam-0", "forklift_lane": "cam-0", "storage": "cam-0", "gate": "cam-1",
               "workbench": "laptop"}

MESSAGES = {
    "fall": "Fall detected",
    "zone_intrusion": "Person entered a restricted zone",
    "time_exceeded": "Stayed in a time-limited zone too long",
    "wrong_direction": "Moved against the one-way direction",
    "loitering": "Loitering",
    "ergo_risk": "High-risk posture (REBA)",
    "near_miss": "Near miss with a forklift",
}


@dataclass(frozen=True)
class SeedEvent:
    key: str
    when: str  # local "YYYY-MM-DD HH:MM" at which the event ended (the alert time)
    type: str
    severity: str
    zone: str | None
    track: int
    duration_s: float = 0.0
    camera: str | None = None  # default: the zone's camera
    verified: int | None = None  # 1 = confirmed, 0 = marked as a false alarm

    @property
    def camera_id(self) -> str:
        return self.camera or ZONE_CAMERA.get(self.zone or "", "cam-0")

    @property
    def end(self) -> datetime:
        return datetime.strptime(self.when, "%Y-%m-%d %H:%M")


SEED: tuple[SeedEvent, ...] = (
    # Mon 15 - Sun 20 Sep
    SeedEvent("sep15_near_miss", "2026-09-15 13:30", "near_miss", "medium", "forklift_lane", 4),
    SeedEvent("sep16_wrong_dir", "2026-09-16 09:10", "wrong_direction", "medium", "forklift_lane", 8),
    SeedEvent("sep16_storage_night", "2026-09-16 21:00", "zone_intrusion", "high", "storage", 2),
    SeedEvent("sep18_fall_false", "2026-09-18 08:20", "fall", "high", "loading_dock", 5, verified=0),
    SeedEvent("sep18_dock_time", "2026-09-18 14:00", "time_exceeded", "medium", "loading_dock", 1, 300),
    SeedEvent("sep20_gate", "2026-09-20 10:00", "zone_intrusion", "high", "gate", 3),
    # Last week: Mon 21 - Sun 27 Sep
    SeedEvent("sep21_dock_longest", "2026-09-21 07:00", "time_exceeded", "high", "loading_dock", 6, 1200),
    SeedEvent("sep21_storage", "2026-09-21 11:11", "zone_intrusion", "high", "storage", 7),
    SeedEvent("sep21_ergo", "2026-09-21 17:45", "ergo_risk", "high", "workbench", 2, 20),
    SeedEvent("sep22_fall", "2026-09-22 06:30", "fall", "critical", "forklift_lane", 4, verified=1),
    SeedEvent("sep22_wrong_dir", "2026-09-22 10:45", "wrong_direction", "low", "forklift_lane", 5),
    SeedEvent("sep23_storage", "2026-09-23 08:00", "zone_intrusion", "high", "storage", 1),
    SeedEvent("sep23_ergo", "2026-09-23 12:30", "ergo_risk", "medium", "workbench", 3, 5),
    SeedEvent("sep23_gate_loiter", "2026-09-23 20:10", "loitering", "medium", "gate", 9, 200),
    SeedEvent("sep24_wrong_dir", "2026-09-24 09:15", "wrong_direction", "medium", "forklift_lane", 2),
    SeedEvent("sep24_near_miss", "2026-09-24 09:40", "near_miss", "high", "forklift_lane", 2),
    SeedEvent("sep24_dock_time", "2026-09-24 16:00", "time_exceeded", "medium", "loading_dock", 12, 360),
    SeedEvent("sep25_fall", "2026-09-25 07:05", "fall", "high", "loading_dock", 3, verified=1),
    SeedEvent("sep25_dock_time", "2026-09-25 07:50", "time_exceeded", "medium", "loading_dock", 3, 480),
    SeedEvent("sep25_ergo", "2026-09-25 13:00", "ergo_risk", "high", "workbench", 6, 15),
    SeedEvent("sep25_storage_evening", "2026-09-25 18:30", "zone_intrusion", "high", "storage", 8),
    SeedEvent("sep26_gate", "2026-09-26 14:00", "zone_intrusion", "high", "gate", 5),
    SeedEvent("sep27_loiter", "2026-09-27 11:00", "loitering", "low", None, 2, 300, camera="cam-1"),
    # This week: Mon 28 Sep
    SeedEvent("mon_ergo", "2026-09-28 06:10", "ergo_risk", "high", "workbench", 1, 6),
    SeedEvent("mon_dock_time", "2026-09-28 08:45", "time_exceeded", "high", "loading_dock", 3, 900),
    SeedEvent("mon_wrong_dir_low", "2026-09-28 10:00", "wrong_direction", "low", "forklift_lane", 4),
    SeedEvent("mon_wrong_dir", "2026-09-28 10:20", "wrong_direction", "medium", "forklift_lane", 4),
    SeedEvent("mon_near_miss", "2026-09-28 15:35", "near_miss", "critical", "loading_dock", 7),
    SeedEvent("mon_fall", "2026-09-28 16:50", "fall", "critical", "storage", 9, verified=1),
    # Yesterday: Tue 29 Sep
    SeedEvent("tue_storage_early", "2026-09-29 05:55", "zone_intrusion", "high", "storage", 1),
    SeedEvent("tue_fall_false", "2026-09-29 07:30", "fall", "critical", "forklift_lane", 6, verified=0),
    SeedEvent("tue_dock_time", "2026-09-29 09:00", "time_exceeded", "medium", "loading_dock", 8, 610),
    SeedEvent("tue_wrong_dir", "2026-09-29 12:15", "wrong_direction", "medium", "forklift_lane", 10),
    SeedEvent("tue_storage_evening", "2026-09-29 18:40", "zone_intrusion", "high", "storage", 14),
    SeedEvent("tue_gate_loiter", "2026-09-29 19:05", "loitering", "low", "gate", 15, 140),
    SeedEvent("tue_gate_night", "2026-09-29 22:30", "zone_intrusion", "critical", "gate", 16),
    # Today: Wed 30 Sep, before 15:00
    SeedEvent("wed_storage", "2026-09-30 06:40", "zone_intrusion", "high", "storage", 3),
    SeedEvent("wed_dock_time", "2026-09-30 07:15", "time_exceeded", "medium", "loading_dock", 5, 420),
    SeedEvent("wed_fall", "2026-09-30 08:02", "fall", "critical", "loading_dock", 7, verified=1),
    SeedEvent("wed_ergo", "2026-09-30 09:30", "ergo_risk", "high", "workbench", 2, 12),
    SeedEvent("wed_wrong_dir", "2026-09-30 10:10", "wrong_direction", "medium", "forklift_lane", 9),
    SeedEvent("wed_loiter", "2026-09-30 11:45", "loitering", "low", None, 11, 95, camera="cam-1"),
    SeedEvent("wed_gate", "2026-09-30 13:20", "zone_intrusion", "high", "gate", 4),
    SeedEvent("wed_ergo_critical", "2026-09-30 14:05", "ergo_risk", "critical", "workbench", 2, 8),
    SeedEvent("wed_near_miss", "2026-09-30 14:30", "near_miss", "high", "forklift_lane", 9),
)

EVENT_IDS = {e.key: i for i, e in enumerate(SEED, start=1)}


def build_store(db_path: str):
    """Create an EventStore at ``db_path`` holding SEED (ids 1..len(SEED))."""
    from events.schema import Event
    from events.store import EventStore

    store = EventStore(db_path)
    for e in SEED:
        end = e.end.timestamp()
        stored = store.emit(Event(
            type=e.type, severity=e.severity, start_ts=end - e.duration_s, end_ts=end, camera_id=e.camera_id,
            track_id=e.track, zone_id=e.zone, alert_id=f"SEED-{e.key}", message=MESSAGES[e.type],
            confidence=0.9, verified=e.verified,
        ))
        assert stored.id == EVENT_IDS[e.key], "seed ids must follow SEED order (use a fresh database)"
    return store


def now_ts() -> float:
    return NOW.timestamp()
