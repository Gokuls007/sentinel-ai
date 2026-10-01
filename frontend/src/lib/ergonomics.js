// REBA (ergonomics) display helpers shared by the live panels, Analytics and Events.

export const ERGO_NOTE = '2D approximation of REBA, not a certified assessment.';

/** Backend level names, lowest to highest, plus "unknown" (view too unreliable to score). */
export const ERGO_LEVELS = ['negligible', 'low', 'medium', 'high', 'very_high'];
export const ERGO_TIME_LEVELS = ['unknown', ...ERGO_LEVELS];

export const LEVEL_LABEL = {
  unknown: 'Unknown',
  negligible: 'Negligible',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  very_high: 'Very high',
};

// Chart fills: green for negligible/low (two shades so stacks stay readable),
// yellow medium, orange high, red very high, grey unknown.
export const LEVEL_HEX = {
  unknown: '#6b7280',
  negligible: '#15803d',
  low: '#4ade80',
  medium: '#facc15',
  high: '#f97316',
  very_high: '#ef4444',
};

// Static class names so Tailwind can see (and generate) them.
const BADGE = {
  negligible: 'text-emerald-300 border-emerald-500/60 bg-emerald-950/40',
  low: 'text-emerald-300 border-emerald-500/60 bg-emerald-950/40',
  medium: 'text-yellow-300 border-yellow-500/60 bg-yellow-950/40',
  high: 'text-orange-300 border-orange-500/60 bg-orange-950/40',
  very_high: 'text-red-300 border-red-500/70 bg-red-950/50',
};
const BADGE_UNRELIABLE = 'text-white/40 border-white/15 bg-white/5';

/** Badge classes for a level; grey when the score is unreliable or missing. */
export function levelBadge(levelName, reliable = true) {
  if (!reliable || !levelName) return BADGE_UNRELIABLE;
  return BADGE[levelName] || BADGE_UNRELIABLE;
}

export const PART_LABEL = {
  trunk: 'Trunk',
  neck: 'Neck',
  legs: 'Legs',
  upper_arm: 'Upper arm',
  lower_arm: 'Lower arm',
  unknown: 'Unknown',
};

export const partLabel = (p) => (p ? PART_LABEL[p] || String(p).replace(/_/g, ' ') : '--');
export const levelLabel = (l) => (l ? LEVEL_LABEL[l] || String(l).replace(/_/g, ' ') : '--');

/** Track IDs are tracker identities, not people: a re-identified person gets a new ID. */
export const trackLabel = (id) => `Track #${id}`;

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * Signed joints (trunk, neck, upper arm): + flexion, - extension. When the person's facing
 * direction is unknown (`facing` null: seen from the front or back) the backend reports
 * magnitudes only, so the direction is shown as unknown rather than guessed as "flex".
 */
export function formatSignedAngle(v, facing) {
  if (!isNum(v)) return '--';
  const deg = Math.round(Math.abs(v));
  if (deg === 0) return '0° neutral';
  if (facing == null) return `${deg}° (dir. ?)`;
  return `${deg}° ${v > 0 ? 'flex' : 'ext'}`;
}

/** Flexion-only joints (lower arm, knee): 0 = straight. */
export function formatFlexion(v) {
  if (!isNum(v)) return '--';
  return `${Math.round(v)}° flex`;
}

/** Rows for the joint-angle table: [label, left, right] (single joints fill `left` only). */
export function angleRows(angles) {
  const a = angles || {};
  return [
    ['Trunk', formatSignedAngle(a.trunk, a.facing), null],
    ['Neck', formatSignedAngle(a.neck, a.facing), null],
    ['Upper arm', formatSignedAngle(a.upper_arm_left, a.facing), formatSignedAngle(a.upper_arm_right, a.facing)],
    ['Lower arm', formatFlexion(a.lower_arm_left), formatFlexion(a.lower_arm_right)],
    ['Knee', formatFlexion(a.knee_left), formatFlexion(a.knee_right)],
  ];
}

/** Seconds -> compact "45s" / "12m 05s" / "3h 20m". */
export function formatSeconds(totalSeconds) {
  if (!isNum(totalSeconds)) return '--';
  const s = Math.max(0, Math.round(totalSeconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
}

export const formatPercent = (v) => (isNum(v) ? `${Math.round(v * 100)}%` : '--');

// --- Day ranges for /api/ergonomics/time (local calendar days) ---------------------------

export const DAY_PRESETS = [
  ['today', 'Today'],
  ['yesterday', 'Yesterday'],
  ['7d', 'Last 7 days'],
];

/** Date -> "YYYY-MM-DD" in local time. */
export function isoDay(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function shiftDays(nowMs, n) {
  const d = new Date(nowMs);
  d.setDate(d.getDate() + n);
  return d;
}

/** { day_from, day_to } for a preset (or the custom days when preset is "custom"). */
export function dayRange(value, nowMs) {
  if (value.preset === 'custom') {
    const from = value.from || isoDay(new Date(nowMs));
    const to = value.to && value.to >= from ? value.to : from;
    return { day_from: from, day_to: to };
  }
  if (value.preset === 'yesterday') {
    const y = isoDay(shiftDays(nowMs, -1));
    return { day_from: y, day_to: y };
  }
  if (value.preset === '7d') return { day_from: isoDay(shiftDays(nowMs, -6)), day_to: isoDay(new Date(nowMs)) };
  const t = isoDay(new Date(nowMs));
  return { day_from: t, day_to: t };
}

/** Unix-second bounds of whole local days [day_from 00:00, day_to + 1 day 00:00). */
export function dayBounds(dayFrom, dayTo) {
  const parse = (s) => {
    const [y, m, d] = s.split('-').map(Number);
    return new Date(y, m - 1, d);
  };
  const start = parse(dayFrom);
  const end = parse(dayTo);
  end.setDate(end.getDate() + 1);
  return { start: Math.floor(start.getTime() / 1000), end: Math.floor(end.getTime() / 1000) };
}

/** Standard REBA action levels for a final score (1 / 2-3 / 4-7 / 8-10 / 11+). */
export function levelForScore(score) {
  if (typeof score !== 'number' || score <= 0) return null;
  if (score <= 1) return 'negligible';
  if (score <= 3) return 'low';
  if (score <= 7) return 'medium';
  if (score <= 10) return 'high';
  return 'very_high';
}
