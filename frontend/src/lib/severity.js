export const SEVERITIES = ['low', 'medium', 'high', 'critical'];

// Static class names so Tailwind can see (and generate) them.
export const SEVERITY_BADGE = {
  critical: 'text-red-400 border-red-500/60 bg-red-950/30',
  high: 'text-orange-400 border-orange-500/50 bg-orange-950/20',
  medium: 'text-yellow-400 border-yellow-500/50 bg-yellow-950/20',
  low: 'text-cyan-400 border-cyan-500/40 bg-cyan-950/20',
};

// Chart fills matching the CSS alert palette in index.css.
export const SEVERITY_HEX = {
  critical: '#FF1744',
  high: '#FF6D00',
  medium: '#FFD600',
  low: '#00E5FF',
};

export function severityBadge(sev) {
  return SEVERITY_BADGE[sev] || SEVERITY_BADGE.low;
}
