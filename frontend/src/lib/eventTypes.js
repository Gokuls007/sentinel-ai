// Human labels for backend event types; unknown types fall back to "snake case" -> words.
const TYPE_LABELS = {
  ergo_risk: 'Ergonomic risk',
  unsafe_lift: 'Unsafe lift',
  standing_on_chair: 'Standing on chair',
  hazard_contact: 'Hand on hazard',
  losing_balance: 'Losing balance',
  fall: 'Confirmed fall',
  possible_fall: 'Possible fall',
};

export function eventTypeLabel(type) {
  if (!type) return '--';
  // Plain-English rules: "rule:loading-dock-dwell" -> "Rule: loading dock dwell" (the event message has its name).
  if (String(type).startsWith('rule:')) return `Rule: ${String(type).slice(5).replace(/[-_]/g, ' ')}`;
  return TYPE_LABELS[type] || String(type).replace(/_/g, ' ');
}

/** An alert from "Test with demo footage" (the `test` camera): shown, labelled TEST, never counted as new. */
export function isTestAlert(alert) {
  return Boolean(alert?.test) || alert?.camera_id === 'test';
}
