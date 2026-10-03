// Human labels for backend event types; unknown types fall back to "snake case" -> words.
const TYPE_LABELS = {
  ergo_risk: 'Ergonomic risk',
  fall: 'Confirmed fall',
  possible_fall: 'Possible fall',
};

export function eventTypeLabel(type) {
  if (!type) return '--';
  // Plain-English rules: "rule:loading-dock-dwell" -> "Rule: loading dock dwell" (the event message has its name).
  if (String(type).startsWith('rule:')) return `Rule: ${String(type).slice(5).replace(/[-_]/g, ' ')}`;
  return TYPE_LABELS[type] || String(type).replace(/_/g, ' ');
}
