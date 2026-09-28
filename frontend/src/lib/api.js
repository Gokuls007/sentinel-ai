import { useEffect, useState } from 'react';

// Base URL for REST calls. Empty = same origin (the backend serves the built
// dashboard, and `npm run dev` proxies /api to it).
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export const BACKEND_START_HINT = 'python backend/main.py --demo';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export function wsUrl() {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/feed`;
}

export async function fetchJson(path, { signal } = {}) {
  const res = await fetch(apiUrl(path), { signal });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
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
