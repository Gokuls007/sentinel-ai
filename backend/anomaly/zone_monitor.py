import time
import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class Zone:
    id: str
    name: str
    polygon: List[Tuple[float, float]] # Normalized 0-1
    zone_type: str # "restricted", "time_limited", "one_way"
    time_limit: float = 0.0
    required_ppe: List[str] = None
    direction: Optional[str] = None # "left", "right", "up", "down"
    active: bool = True

    def __post_init__(self):
        if self.required_ppe is None:
            self.required_ppe = []

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
    def __init__(self, zones_file="config/zones.json", frame_width=1280, frame_height=720):
        # Resolve path relative to backend root if it's a relative path starting with config/
        if zones_file == "config/zones.json":
            zones_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "zones.json")
            
        self.zones_file = zones_file
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.zones: List[Zone] = []
        self._load_or_create_zones()
        
        # Track state
        self.track_zone_entry: Dict[Tuple[int, str], float] = {}
        self.track_last_pos: Dict[int, np.ndarray] = {}

    def _load_or_create_zones(self):
        if not os.path.exists(self.zones_file):
            os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
            self.zones = [
                Zone("restricted_1", "Restricted Area", [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)], "restricted"),
                Zone("dock_1", "Loading Dock", [(0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (0.75, 1.0)], "time_limited", time_limit=10.0),
                Zone("exit_1", "One-Way Exit", [(0.4, 0.8), (0.6, 0.8), (0.6, 1.0), (0.4, 1.0)], "one_way", direction="down")
            ]
            self._save_zones()
        else:
            with open(self.zones_file, "r") as f:
                data = json.load(f)
                self.zones = [Zone(**z) for z in data.get("zones", [])]

    def _save_zones(self):
        with open(self.zones_file, "w") as f:
            json.dump({"zones": [asdict(z) for z in self.zones]}, f, indent=4)

    def check(self, track_id: int, centroid: np.ndarray, timestamp: float) -> List[ZoneViolation]:
        norm_x = centroid[0] / self.frame_width
        norm_y = centroid[1] / self.frame_height
        point = (norm_x, norm_y)
        
        violations = []
        
        for zone in self.zones:
            if not zone.active: continue
            
            is_inside = self._point_in_polygon(point, zone.polygon)
            key = (track_id, zone.id)
            
            if is_inside:
                # 1. Restricted Entry
                if zone.zone_type == "restricted":
                    violations.append(ZoneViolation(
                        track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                        violation_type="intrusion", timestamp=timestamp, confidence=0.9,
                        details="Unauthorized entry into restricted area"
                    ))
                
                # 2. Time Limited
                elif zone.zone_type == "time_limited":
                    if key not in self.track_zone_entry:
                        self.track_zone_entry[key] = timestamp
                    
                    duration = timestamp - self.track_zone_entry[key]
                    if duration > zone.time_limit:
                        violations.append(ZoneViolation(
                            track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                            violation_type="time_exceeded", timestamp=timestamp, confidence=0.8,
                            duration=duration, details=f"Time limit exceeded ({duration:.1f}s > {zone.time_limit}s)"
                        ))
                
                # 3. One Way
                elif zone.zone_type == "one_way" and track_id in self.track_last_pos:
                    last_pos = self.track_last_pos[track_id]
                    dx = centroid[0] - last_pos[0]
                    dy = centroid[1] - last_pos[1]
                    
                    is_wrong = False
                    if zone.direction == "left" and dx > 2: is_wrong = True # Moving right
                    elif zone.direction == "right" and dx < -2: is_wrong = True # Moving left
                    elif zone.direction == "up" and dy > 2: is_wrong = True # Moving down
                    elif zone.direction == "down" and dy < -2: is_wrong = True # Moving up
                    
                    if is_wrong:
                        violations.append(ZoneViolation(
                            track_id=track_id, zone_id=zone.id, zone_name=zone.name,
                            violation_type="wrong_direction", timestamp=timestamp, confidence=0.7,
                            details=f"Walking wrong way in one-way zone (dir: {zone.direction})"
                        ))
            else:
                # Reset entry time if they exit
                if key in self.track_zone_entry:
                    del self.track_zone_entry[key]
                    
        self.track_last_pos[track_id] = centroid.copy()
        return violations

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
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

    def get_zones_for_overlay(self) -> List[Dict]:
        overlay_zones = []
        for zone in self.zones:
            if not zone.active: continue
            # Denormalize polygon for frontend/HUD
            poly = []
            for px, py in zone.polygon:
                poly.append((int(px * self.frame_width), int(py * self.frame_height)))
            
            overlay_zones.append({
                "id": zone.id,
                "name": zone.name,
                "type": zone.zone_type,
                "polygon": poly
            })
        return overlay_zones
