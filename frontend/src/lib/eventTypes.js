// Human labels for backend event types; unknown types fall back to "snake case" -> words.
const TYPE_LABELS = {
  ergo_risk: 'Ergonomic risk',
};

export function eventTypeLabel(type) {
  if (!type) return '--';
  return TYPE_LABELS[type] || String(type).replace(/_/g, ' ');
}
