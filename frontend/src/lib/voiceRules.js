import { isTestAlert } from './eventTypes.js';

// Spoken warnings (browser Web Speech API): new live alerts of these types are read out, at most
// once per type every COOLDOWN_MS. History, test footage and other alert types stay silent.
export const COOLDOWN_MS = 10000;
export const SPOKEN_TYPES = new Set([
  'losing_balance', 'hazard_contact', 'unsafe_lift', 'standing_on_chair', 'fall', 'possible_fall', 'trip_hazard',
]);

/** "Losing balance: centre of mass..." -> "Losing balance"; "Hand on tv (hazard)" -> "Hand on tv". */
export function spokenText(alert) {
  const message = String(alert?.message || '');
  const head = message.split(':')[0].replace(/\s*\(hazard\)\s*$/i, '').trim();
  return head || String(alert?.alert_type || '').replace(/_/g, ' ');
}

/** Decide what to say for the newest alerts: returns texts, updating `last` (type -> time). */
export function pickAnnouncements(alerts, since, last, now, cooldownMs = COOLDOWN_MS) {
  const out = [];
  for (const a of alerts) {
    if ((a.timestamp || 0) * 1000 < since) break; // newest first: the rest are older
    if (!SPOKEN_TYPES.has(a.alert_type) || isTestAlert(a)) continue;
    if (now - (last[a.alert_type] || -Infinity) < cooldownMs) continue;
    last[a.alert_type] = now;
    out.push(spokenText(a));
  }
  return out;
}
