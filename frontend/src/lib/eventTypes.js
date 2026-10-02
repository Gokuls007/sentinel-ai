// Human labels for backend event types; unknown types fall back to "snake case" -> words.
const TYPE_LABELS = {
  ergo_risk: 'Ergonomic risk',
  fall: 'Confirmed fall',
  possible_fall: 'Possible fall',
};

export function eventTypeLabel(type) {
  if (!type) return '--';
  return TYPE_LABELS[type] || String(type).replace(/_/g, ' ');
}
