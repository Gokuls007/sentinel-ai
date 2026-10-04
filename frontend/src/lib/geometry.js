// Zone polygon checks (same rules as backend/anomaly/geometry.py): edges that cross leave a
// zone with no clear inside, so the editor refuses them and offers a fix.

const orient = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
const onSegment = (a, b, c) => Math.min(a[0], b[0]) - 1e-12 <= c[0] && c[0] <= Math.max(a[0], b[0]) + 1e-12
  && Math.min(a[1], b[1]) - 1e-12 <= c[1] && c[1] <= Math.max(a[1], b[1]) + 1e-12;

function segmentsCross(p1, p2, q1, q2) {
  const d1 = orient(q1, q2, p1); const d2 = orient(q1, q2, p2);
  const d3 = orient(p1, p2, q1); const d4 = orient(p1, p2, q2);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  const eps = 1e-12;
  return (Math.abs(d1) < eps && onSegment(q1, q2, p1)) || (Math.abs(d2) < eps && onSegment(q1, q2, p2))
    || (Math.abs(d3) < eps && onSegment(p1, p2, q1)) || (Math.abs(d4) < eps && onSegment(p1, p2, q2));
}

/** Pairs of crossing edges [i, j]; edge i runs from point i to point i+1. */
export function selfIntersections(points) {
  const n = points.length;
  const out = [];
  if (n < 4) return out;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (j === i + 1 || (i === 0 && j === n - 1)) continue;
      if (segmentsCross(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n])) out.push([i, j]);
    }
  }
  return out;
}

function convexHull(points) {
  const pts = [...new Map(points.map((p) => [`${p[0]},${p[1]}`, p])).values()]
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const half = (seq) => {
    const h = [];
    for (const p of seq) {
      while (h.length >= 2 && orient(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    return h;
  };
  const lower = half(pts);
  const upper = half([...pts].reverse());
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** { points, method }: "reordered" (same corners, sorted around the centre) or "hull". */
export function fixPolygon(points) {
  if (!selfIntersections(points).length) return { points, method: 'unchanged' };
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
  const ordered = [...points].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  if (!selfIntersections(ordered).length) return { points: ordered, method: 'reordered' };
  return { points: convexHull(points), method: 'hull' };
}

export const FIX_LABELS = {
  reordered: 'Same corners, connected in order around the centre',
  hull: 'The outline around all the corners (some inner corners are dropped)',
};
