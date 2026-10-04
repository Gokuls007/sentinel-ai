"""Zone polygon checks: a zone whose edges cross itself has no clear inside, so it is rejected,
with a suggested fix."""

from __future__ import annotations

import math

Point = tuple[float, float]


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and \
        min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12


def segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Proper or touching intersection of segments p1-p2 and q1-q2."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    eps = 1e-12
    return ((abs(d1) < eps and _on_segment(q1, q2, p1)) or (abs(d2) < eps and _on_segment(q1, q2, p2))
            or (abs(d3) < eps and _on_segment(p1, p2, q1)) or (abs(d4) < eps and _on_segment(p1, p2, q2)))


def self_intersections(points: list[Point]) -> list[tuple[int, int]]:
    """Pairs of edge indexes (edge i runs from point i to point i+1) that cross. Neighbouring
    edges share a corner and don't count."""
    pts = [tuple(map(float, p)) for p in points]
    n = len(pts)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges
            if segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                out.append((i, j))
    return out


def convex_hull(points: list[Point]) -> list[Point]:
    pts = sorted(set(tuple(map(float, p)) for p in points))
    if len(pts) <= 2:
        return pts

    def half(seq):
        h: list[Point] = []
        for p in seq:
            while len(h) >= 2 and _orient(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def fix_polygon(points: list[Point]) -> tuple[list[Point], str]:
    """A simple (non-crossing) version of ``points`` and how it was made: "reordered" (the same
    points, sorted around their centre: keeps the shape when that is possible) or "hull" (the
    convex hull: the outline around all the points)."""
    pts = [tuple(map(float, p)) for p in points]
    if not self_intersections(pts):
        return pts, "unchanged"
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    if not self_intersections(ordered):
        return ordered, "reordered"
    return convex_hull(pts), "hull"
