import { apiUrl } from './api';

/**
 * POST /api/search and call onStep(step) for every Server-Sent Event as it arrives.
 * Resolves when the stream ends. HTTP errors (403, 429, 503...) reject with `.status`
 * and the backend's `.detail`.
 */
export async function streamSearch(question, onStep, { signal } = {}) {
  let res;
  try {
    res = await fetch(apiUrl('/api/search'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (err) {
    if (err.name !== 'AbortError') err.status = null; // backend offline
    throw err;
  }
  if (!res.ok) {
    let detail = null;
    try {
      const data = await res.json();
      detail = typeof data?.detail === 'string' ? data.detail : null;
    } catch {
      // non-JSON error body
    }
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          onStep(JSON.parse(line.slice(6)));
        } catch {
          // a malformed line: skip it rather than abort the answer
        }
      }
    }
  }
}

/**
 * Split an answer into pieces for rendering: plain text, **bold** and [#id] citations.
 * Returns [{ kind: 'text'|'bold'|'cite', text, id? }].
 */
export function tokenizeAnswer(text) {
  const out = [];
  const re = /\[#(\d+)\]|\*\*([^*]+)\*\*/g;
  let last = 0;
  let m;
  while ((m = re.exec(text || '')) !== null) {
    if (m.index > last) out.push({ kind: 'text', text: text.slice(last, m.index) });
    if (m[1]) out.push({ kind: 'cite', text: `#${m[1]}`, id: Number(m[1]) });
    else out.push({ kind: 'bold', text: m[2] });
    last = re.lastIndex;
  }
  if (last < (text || '').length) out.push({ kind: 'text', text: text.slice(last) });
  return out;
}

/** "count_events" + {types:["fall"], zone:"dock"} -> "count events · types=fall · zone=dock" */
export function describeToolCall(name, args) {
  const label = String(name || '').replace(/_/g, ' ');
  const parts = Object.entries(args || {})
    .filter(([, v]) => v != null && v !== '' && !(Array.isArray(v) && v.length === 0))
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(',') : v}`);
  return [label, ...parts].join(' · ');
}

export const EXAMPLE_QUESTIONS = [
  'How many falls were there today?',
  'Who went into a restricted zone after 6pm yesterday?',
  'Break down this week\'s events by zone',
  'What was the longest time-limit overstay this week?',
  'Which falls were marked as false alarms?',
];
