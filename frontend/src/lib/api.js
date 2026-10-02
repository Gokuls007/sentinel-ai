import { useEffect, useState } from 'react';

// Base URL for REST calls. Empty = same origin (the backend serves the built
// dashboard, and `npm run dev` proxies /api to it).
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export const BACKEND_START_HINT = 'python backend/main.py --demo';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

/** Feed socket URL; `camera` selects which camera's frames to receive (default: primary). */
export function wsUrl(camera) {
  const q = camera ? `?camera=${encodeURIComponent(camera)}` : '';
  if (import.meta.env.VITE_WS_URL) return `${import.meta.env.VITE_WS_URL}${q}`;
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/feed${q}`;
}

export async function fetchJson(path, { signal } = {}) {
  const res = await fetch(apiUrl(path), { signal });
  if (!res.ok) {
    const err = new Error(`${path}: HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/** POST a JSON body; errors carry `.status` and the backend's `.detail`. */
export function postJson(path, body = {}) {
  return sendJson('POST', path, body);
}

/** PUT a JSON body (same error shape as postJson). */
export function putJson(path, body = {}) {
  return sendJson('PUT', path, body);
}

/** DELETE with a JSON body (same error shape as postJson). */
export function deleteJson(path, body = {}) {
  return sendJson('DELETE', path, body);
}

async function sendJson(method, path, body) {
  let res;
  try {
    res = await fetch(apiUrl(path), {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    err.status = null; // network failure: backend offline
    throw err;
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    // empty or non-JSON body
  }
  if (!res.ok) {
    // FastAPI sends a string for HTTPException and a list of issues for validation errors.
    const detail = typeof data?.detail === 'string'
      ? data.detail
      : Array.isArray(data?.detail)
        ? data.detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
        : null;
    const err = new Error(detail || `${path}: HTTP ${res.status}`);
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return data;
}

/** True when an error means the backend is unreachable or the pipeline is still starting. */
export function isOfflineError(err) {
  if (!err) return false;
  const status = typeof err === 'object' ? err.status : null;
  return status == null || status === 503 || status === 502 || status === 504;
}

/** Human text for a fetch error: offline/starting vs a real HTTP error. */
export function describeError(err) {
  if (!err) return null;
  if (isOfflineError(err)) return 'Backend offline or starting';
  if (err.status === 404) return 'Not found';
  return err.message || String(err);
}

/**
 * One-shot GET that refetches whenever `path` or `reloadKey` changes.
 * `path` null = skip. Returns { data, error, loading }; error is the Error
 * object (use describeError). Data is cleared on failure: no stale values.
 */
export function useFetch(path, reloadKey = 0) {
  const [state, setState] = useState({ data: null, error: null, key: null });
  const key = path == null ? null : `${path}#${reloadKey}`;

  useEffect(() => {
    if (path == null) return undefined;
    const controller = new AbortController();
    fetchJson(path, { signal: controller.signal })
      .then((data) => setState({ data, error: null, key }))
      .catch((err) => {
        if (err.name !== 'AbortError') setState({ data: null, error: err, key });
      });
    return () => controller.abort();
  }, [path, key]);

  if (key == null) return { data: null, error: null, loading: false };
  // While a new request is in flight keep showing the previous result (no flicker).
  return { data: state.data, error: state.error, loading: state.key !== key };
}

/** localStorage JSON read/write that never throws (private mode, blocked storage). */
export function loadStored(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw == null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function saveStored(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage unavailable: the setting just won't persist.
  }
}

/** Build a query string from an object, skipping null/undefined/''/[] values. */
export function queryString(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v)) {
      if (v.length) q.set(k, v.join(','));
    } else {
      q.set(k, String(v));
    }
  }
  const s = q.toString();
  return s ? `?${s}` : '';
}

/**
 * Poll a GET endpoint every `intervalMs` while `enabled`.
 * Returns { data, error }; data is null until the first success and is reset
 * to null on failure so stale values are never shown as current.
 */
export function usePoll(path, intervalMs, enabled = true) {
  const [state, setState] = useState({ data: null, error: null });

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    let timer = null;
    let controller = null;

    const tick = async () => {
      controller = new AbortController();
      try {
        const data = await fetchJson(path, { signal: controller.signal });
        if (!cancelled) setState({ data, error: null });
      } catch (err) {
        if (!cancelled && err.name !== 'AbortError') setState({ data: null, error: err.message });
      }
      if (!cancelled) timer = setTimeout(tick, intervalMs);
    };
    tick();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (controller) controller.abort();
    };
  }, [path, intervalMs, enabled]);

  return enabled ? state : { data: null, error: null };
}

/** A clock that re-renders every `intervalMs` (for relative-time UI). */
export function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

export function formatDuration(totalSeconds) {
  if (totalSeconds == null || !Number.isFinite(totalSeconds)) return '--';
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = String(Math.floor(s / 3600)).padStart(2, '0');
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const sec = String(s % 60).padStart(2, '0');
  return `${h}:${m}:${sec}`;
}

export function basename(path) {
  if (path == null) return '';
  return String(path).split(/[\\/]/).pop();
}

export function formatTs(unixSeconds) {
  if (unixSeconds == null || !Number.isFinite(Number(unixSeconds))) return '--';
  return new Date(Number(unixSeconds) * 1000).toLocaleString([], {
    month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** Parse a backend hour_bucket key ("2026-09-30 14:00", local time) to ms. */
export function parseHourBucket(key) {
  const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):/.exec(key || '');
  if (!m) return null;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4])).getTime();
}

/**
 * Turn hour_bucket counts into a gap-filled series [{ key, label, count }].
 * Gaps are only filled when the span is reasonable (<= maxBuckets hours).
 * bounds { startMs, endMs } (optional) extends the axis to the selected range.
 */
export function hourSeries(counts, maxBuckets = 24 * 14, bounds = {}) {
  const entries = Object.entries(counts || {})
    .map(([k, n]) => ({ ms: parseHourBucket(k), key: k, count: n }))
    .filter((e) => e.ms != null)
    .sort((a, b) => a.ms - b.ms);
  if (entries.length === 0) return [];
  const label = (ms) => {
    const d = new Date(ms);
    return `${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:00`;
  };
  const floorHour = (ms) => {
    const d = new Date(ms);
    d.setMinutes(0, 0, 0);
    return d.getTime();
  };
  let first = entries[0].ms;
  let last = entries[entries.length - 1].ms;
  if (bounds.startMs != null) first = Math.min(first, floorHour(bounds.startMs));
  if (bounds.endMs != null) last = Math.max(last, floorHour(bounds.endMs));
  const span = Math.round((last - first) / 3600000) + 1;
  if (span > maxBuckets) return entries.map((e) => ({ key: e.key, label: label(e.ms), count: e.count }));
  const byMs = new Map(entries.map((e) => [e.ms, e.count]));
  const out = [];
  // Step by calendar hour (not +3600000) so DST changes don't skew labels.
  for (let d = new Date(first); d.getTime() <= last; d.setHours(d.getHours() + 1)) {
    const ms = d.getTime();
    out.push({ key: String(ms), label: label(ms), count: byMs.get(ms) || 0 });
  }
  return out;
}
