"""Seat map: rectangles (normalised 0-1 image coordinates), reading-order labels, neighbours,
and which person sits where.

From the invigilator's view (the camera is at the front, facing the students):
- rows run front to back: row A is the one nearest the camera, i.e. lowest in the image;
- seats run left to right as seen in the image: seat 1 is leftmost.
All exam logic is keyed by seat label, not track id (track ids change; seats don't).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

import numpy as np

NOSE, L_SH, R_SH = 0, 5, 6
HEAD = (0, 1, 2, 3, 4)
MIN_CONF = 0.3


@dataclass
class Seat:
    label: str
    rect: tuple[float, float, float, float]  # x1, y1, x2, y2, normalised
    row: int = 0
    col: int = 0
    neighbours: dict[str, str | None] = field(default_factory=dict)  # left/right/front/back -> label
    id: int | None = None

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.rect
        return (x1 + x2) / 2, (y1 + y2) / 2

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "rect": [round(v, 4) for v in self.rect],
                "row": self.row, "col": self.col, "neighbours": self.neighbours}


def shoulders(kp: np.ndarray) -> tuple[np.ndarray, float] | None:
    """(shoulder midpoint in pixels, shoulder width in pixels), or None."""
    kp = np.asarray(kp, float)
    if kp[L_SH, 2] < MIN_CONF or kp[R_SH, 2] < MIN_CONF:
        return None
    mid = (kp[L_SH, :2] + kp[R_SH, :2]) / 2
    return mid, float(abs(kp[L_SH, 0] - kp[R_SH, 0]))


def seat_rect_for(kp: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    """A seat region around a seated person's head, shoulders and desk area (normalised)."""
    s = shoulders(kp)
    if s is None:
        return None
    (mx, my), sw = s
    sw = max(sw, 10.0)
    head = [kp[i, 1] for i in HEAD if kp[i, 2] >= MIN_CONF]
    top = (min(head) if head else my - sw) - 0.5 * sw
    x1, x2 = mx - 1.1 * sw, mx + 1.1 * sw
    y1, y2 = top, my + 1.6 * sw  # down to the desk in front of them
    w, h = max(width, 1), max(height, 1)
    return (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))


def row_letters(n: int) -> list[str]:
    letters = list(string.ascii_uppercase)
    return [letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26] for i in range(n)]


def label_seats(rects: list[tuple[float, float, float, float]]) -> list[Seat]:
    """Reading-order seats: rows by centre height (front = lowest in the image first), then
    left to right. A new row starts when a centre is more than half a seat height above the
    current row's average."""
    if not rects:
        return []
    items = sorted(rects, key=lambda r: -((r[1] + r[3]) / 2))  # front (bottom) first
    heights = [r[3] - r[1] for r in rects]
    gap = 0.5 * float(np.median(heights))
    rows: list[list] = []
    for r in items:
        cy = (r[1] + r[3]) / 2
        if rows and abs(np.mean([(q[1] + q[3]) / 2 for q in rows[-1]]) - cy) <= gap:
            rows[-1].append(r)
        else:
            rows.append([r])
    seats = []
    for ri, (letter, row) in enumerate(zip(row_letters(len(rows)), rows, strict=True)):
        for ci, r in enumerate(sorted(row, key=lambda q: (q[0] + q[2]) / 2)):
            seats.append(Seat(label=f"{letter}{ci + 1}", rect=tuple(float(v) for v in r), row=ri, col=ci))
    return neighbour_graph(seats)


def neighbour_graph(seats: list[Seat]) -> list[Seat]:
    """Left/right: the adjacent seat in the same row. Front/back: the seat in the adjacent row
    whose centre is horizontally nearest, if it's within one seat width."""
    by_row: dict[int, list[Seat]] = {}
    for s in seats:
        by_row.setdefault(s.row, []).append(s)
    for row in by_row.values():
        row.sort(key=lambda s: s.centre[0])
    for s in seats:
        row = by_row[s.row]
        i = row.index(s)
        n = {"left": row[i - 1].label if i > 0 else None,
             "right": row[i + 1].label if i + 1 < len(row) else None}
        width = s.rect[2] - s.rect[0]
        for key, other in (("front", s.row - 1), ("back", s.row + 1)):
            cands = by_row.get(other, [])
            best = min(cands, key=lambda o: abs(o.centre[0] - s.centre[0]), default=None)
            n[key] = best.label if best is not None and abs(best.centre[0] - s.centre[0]) <= width else None
        s.neighbours = n
    return seats


def relabel(seats: list[Seat]) -> list[Seat]:
    """Recompute rows, columns and neighbours after edits, keeping custom labels."""
    fresh = label_seats([s.rect for s in seats])
    old_by_rect = {tuple(float(v) for v in s.rect): s for s in seats}
    for f in fresh:
        old = old_by_rect.get(tuple(f.rect))
        if old is not None and old.label:
            f.id, f.label = old.id, old.label
    return neighbour_graph(fresh)  # neighbours named by the kept labels


def assign(people: dict[int, np.ndarray], seats: list[Seat], width: int, height: int) -> dict[str, int]:
    """{seat label: track id}: a person sits in the seat containing their shoulder midpoint; if
    two do, the one nearer the seat centre. People in no seat aren't returned (staff, others)."""
    w, h = max(width, 1), max(height, 1)
    best: dict[str, tuple[float, int]] = {}
    for tid, kp in people.items():
        s = shoulders(kp)
        if s is None:
            continue
        x, y = s[0][0] / w, s[0][1] / h
        for seat in seats:
            if seat.contains(x, y):
                cx, cy = seat.centre
                d = (x - cx) ** 2 + (y - cy) ** 2
                if seat.label not in best or d < best[seat.label][0]:
                    best[seat.label] = (d, tid)
    return {label: tid for label, (_d, tid) in best.items()}
