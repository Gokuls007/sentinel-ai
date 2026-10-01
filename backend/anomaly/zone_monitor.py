import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Zone:
    id: str
    name: str
    polygon: list[tuple[float, float]] # Normalized 0-1
    zone_type: str # "restricted", "time_limited", "one_way"
    time_limit: float = 0.0
    required_ppe: list[str] = None
    direction: str | None = None # "left", "right", "up", "down"
    active: bool = True

    def __post_init__(self):
        if self.required_ppe is None:
            self.required_ppe = []
        self.polygon = [tuple(map(float, p)) for p in self.polygon]

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        """Build a Zone from JSON, ignoring unknown keys (e.g. a UI colour)."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

@dataclass
class ZoneViolation:
    track_id: int
    zone_id: str
    zone_name: str
    violation_type: str # "intrusion", "time_exceeded", "wrong_direction"
    timestamp: float
    confidence: float
    duration: float = 0.0
    details: str = ""

class ZoneMonitor:
    # Movement (as a fraction of frame height) needed to count as travelling a direction.
    DIRECTION_MIN_MOVE = 0.01

    def __init__(self, zones_file="config/zones.json", frame_width=1280, frame_height=720,
                 alert_cooldown: float = 30.0):
        # Resolve path relative to backend root if it's a relative path starting with config/
        if zones_file in ("config/zones.json", "backend/config/zones.json") and not os.path.exists(zones_file):
            zones_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "zones.json")

        self.zones_file = zones_file
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alert_cooldown = alert_cooldown

        self.zones: list[Zone] = []
        self._load_or_create_zones()

        # Track state
        self.track_zone_entry: dict[tuple[int, str], float] = {}
        self.track_inside: dict[tuple[int, str], bool] = {}
        self.last_alert: dict[tuple[int, str], float] = {}
        self.track_last_pos: dict[int, np.ndarray] = {}

    def set_frame_size(self, width: int, height: int):
        """Zones are normalised (0-1); tell the monitor the real frame size."""
        self.frame_width, self.frame_height = int(width), int(height)

    def set_zones(self, zones: list[Zone]):
        """Replace the active zones (e.g. from a demo config) without touching zones.json."""
        self.zones = list(zones)
        self.track_zone_entry.clear()
        self.track_inside.clear()
        self.last_alert.clear()

    def replace_zones(self, zones: list[Zone]) -> None:
        """Replace the zones and save them to this monitor's zones file (zone editor)."""
        self.set_zones(zones)
        self._save_zones()

    def _cooled_down(self, key, timestamp: float) -> bool:
        last = self.last_alert.get(key)
        return last is None or timestamp - last >= self.alert_cooldown

    def prune(self, active_track_ids):
        """Forget state for people who are no longer tracked."""
        active = set(active_track_ids)
        for d in (self.track_zone_entry, self.track_inside, self.last_alert):
            for key in [k for k in d if k[0] not in active]:
                del d[key]
        for tid in [t for t in self.track_last_pos if t not in active]:
            del self.track_last_pos[tid]

    def _load_or_create_zones(self):
        if not os.path.exists(self.zones_file):
            os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
            self.zones = [
                Zone("restricted_1", "Restricted Area",
                     [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)], "restricted"),
                Zone("dock_1", "Loading Dock",
                     [(0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (0.75, 1.0)], "time_limited", time_limit=10.0),
                Zone("exit_1", "One-Way Exit",
                     [(0.4, 0.8), (0.6, 0.8), (0.6, 1.0), (0.4, 1.0)], "one_way", direction="down"),
            ]
            self._save_zones()
        else:
            with open(self.zones_file) as f:
                data = json.load(f)
                self.zones = [Zone.from_dict(z) for z in data.get("zones", [])]

    def _save_zones(self):
        with open(self.zones_file, "w") as f:
            json.dump({"zones": [asdict(z) for z in self.zones]}, f, indent=4)

    def check(self, track_id: int, centroid: np.ndarray, timestamp: float) -> list[ZoneViolation]:
        norm_x = centroid[0] / self.frame_width
        norm_y = centroid[1] / self.frame_height
        point = (norm_x, norm_y)
        
        violations = []

        for zone in self.zones:
            if not zone.active:
                continue

            is_inside = self._point_in_polygon(point, zone.polygon)
            key = (track_id, zone.id)
            just_entered = is_inside and not self.track_inside.get(key, False)
            self.track_inside[key] = is_inside

            if not is_inside:
                self.track_zone_entry.pop(key, None)  # reset timer on exit
                continue
            if just_entered:
                self.track_zone_entry[key] = timestamp

            violation = None
            # 1. Restricted: alert on entry, then at most once per cooldown per person
            #    (re-entries from boundary jitter don't re-alert).
            if zone.zone_type == "restricted":
                if self._cooled_down(key, timestamp):
                    violation = ZoneViolation(
                        track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                        violation_type="intrusion", timestamp=timestamp, confidence=0.9,
                        duration=timestamp - self.track_zone_entry[key],
                        details="Unauthorized entry into restricted area")

            # 2. Time limited: alert once the limit is passed, then once per cooldown.
            elif zone.zone_type == "time_limited":
                duration = timestamp - self.track_zone_entry[key]
                if duration > zone.time_limit and self._cooled_down(key, timestamp):
                    violation = ZoneViolation(
                        track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                        violation_type="time_exceeded", timestamp=timestamp, confidence=0.8,
                        duration=duration,
                        details=f"Time limit exceeded ({duration:.1f}s > {zone.time_limit}s)")

            # 3. One way: sustained movement against the allowed direction.
            elif zone.zone_type == "one_way" and track_id in self.track_last_pos:
                last_pos = self.track_last_pos[track_id]
                dx = (centroid[0] - last_pos[0]) / self.frame_height
                dy = (centroid[1] - last_pos[1]) / self.frame_height
                m = self.DIRECTION_MIN_MOVE
                wrong = {
                    "left": dx > m, "right": dx < -m, "up": dy > m, "down": dy < -m,
                }.get(zone.direction, False)
                if wrong and self._cooled_down(key, timestamp):
                    violation = ZoneViolation(
                        track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                        violation_type="wrong_direction", timestamp=timestamp, confidence=0.7,
                        details=f"Walking wrong way in one-way zone (allowed: {zone.direction})")

            if violation:
                self.last_alert[key] = timestamp
                violations.append(violation)

        self.track_last_pos[track_id] = np.array(centroid, dtype=float).copy()
        return violations

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def add_zone(self, zone: Zone):
        self.zones.append(zone)
        self._save_zones()

    def remove_zone(self, zone_id: str):
        self.zones = [z for z in self.zones if z.id != zone_id]
        self._save_zones()

    def get_zones_for_overlay(self) -> list[dict]:
        overlay_zones = []
        for zone in self.zones:
            if not zone.active:
                continue
            # Denormalize polygon for frontend/HUD
            poly = []
            for px, py in zone.polygon:
                poly.append((int(px * self.frame_width), int(py * self.frame_height)))
            
            overlay_zones.append({
                "id": zone.id,
                "name": zone.name,
                "type": zone.zone_type,
                "polygon": poly,
                "polygon_normalized": [list(p) for p in zone.polygon],
                "time_limit": zone.time_limit,
                "direction": zone.direction,
            })
        return overlay_zones
