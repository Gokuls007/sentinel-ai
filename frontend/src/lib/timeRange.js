export const PRESETS = [
  ['1h', '1h', 3600],
  ['24h', '24h', 86400],
  ['7d', '7d', 7 * 86400],
  ['all', 'All', null],
];

export const DEFAULT_RANGE = { preset: '24h', from: '', to: '' };

/**
 * { start, end } in unix seconds for the API (either may be null).
 * Relative presets are anchored at `nowMs` and left open-ended so events
 * still in progress are included.
 */
export function rangeToParams(range, nowMs) {
  if (range.preset === 'custom') {
    const from = range.from ? new Date(range.from).getTime() : NaN;
    const to = range.to ? new Date(range.to).getTime() : NaN;
    return {
      start: Number.isFinite(from) ? Math.floor(from / 1000) : null,
      end: Number.isFinite(to) ? Math.floor(to / 1000) : null,
    };
  }
  const preset = PRESETS.find(([k]) => k === range.preset);
  const seconds = preset ? preset[2] : null;
  return { start: seconds ? Math.floor(nowMs / 1000 - seconds) : null, end: null };
}

/** Axis bounds (ms) for hour charts: the selected window, ending now for open ranges. */
export function rangeBounds(start, end, nowMs) {
  if (start == null) return {};
  return { startMs: start * 1000, endMs: end != null ? end * 1000 : nowMs };
}
