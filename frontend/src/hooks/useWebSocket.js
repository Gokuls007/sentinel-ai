import { useEffect, useState } from 'react';
import { fetchJson, wsUrl } from '../lib/api';

const MAX_ALERTS = 200;
const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 10000;

/** Merge alerts by alert_id (later fields win, has_clip is sticky), newest first, capped. */
export function mergeAlerts(prev, incoming) {
  if (!incoming || incoming.length === 0) return prev;
  const byId = new Map(prev.map((a) => [a.alert_id, a]));
  for (const a of incoming) {
    if (!a || !a.alert_id) continue;
    const old = byId.get(a.alert_id);
    byId.set(a.alert_id, old ? { ...old, ...a, has_clip: Boolean(old.has_clip || a.has_clip) } : a);
  }
  return [...byId.values()]
    .sort((x, y) => (y.timestamp || 0) - (x.timestamp || 0))
    .slice(0, MAX_ALERTS);
}

/**
 * Live connection to the Sentinel backend feed (`/ws/feed`).
 *
 * status: 'connecting' | 'live' | 'disconnected'
 * Nothing here is simulated: when the backend is unreachable the frame/stats
 * are cleared and status is 'disconnected'.
 */
const useWebSocket = (url = wsUrl()) => {
  const [status, setStatus] = useState('connecting');
  const [frame, setFrame] = useState(null);
  const [frameData, setFrameData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [lastError, setLastError] = useState(null);

  useEffect(() => {
    // All connection state is local to this effect run, so StrictMode's
    // mount/unmount/mount cannot leak sockets or timers between runs.
    let socket = null;
    let retryTimer = null;
    let backoff = BACKOFF_START_MS;
    let disposed = false;

    const loadPersistedAlerts = async () => {
      try {
        const history = await fetchJson('/api/alerts?limit=100');
        if (!disposed && Array.isArray(history)) setAlerts((prev) => mergeAlerts(prev, history));
      } catch {
        // REST history is best-effort; the socket still delivers live alerts.
      }
    };

    const scheduleReconnect = () => {
      if (disposed) return;
      clearTimeout(retryTimer);
      retryTimer = setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, BACKOFF_MAX_MS);
    };

    const handleMessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        console.warn('[WS] Ignoring non-JSON message');
        return;
      }
      if (!msg || typeof msg !== 'object') return;

      switch (msg.type) {
        case 'history':
          if (Array.isArray(msg.alerts)) setAlerts((prev) => mergeAlerts(prev, msg.alerts));
          break;
        case 'alert':
          if (msg.alert) setAlerts((prev) => mergeAlerts(prev, [msg.alert]));
          break;
        case 'frame':
          if (msg.image) setFrame(msg.image);
          setFrameData(msg.data || null);
          break;
        default:
          break;
      }
    };

    function connect() {
      if (disposed) return;
      setStatus('connecting');
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        setLastError(String(err?.message || err));
        setStatus('disconnected');
        scheduleReconnect();
        return;
      }
      socket = ws;

      ws.onopen = () => {
        if (disposed) return;
        backoff = BACKOFF_START_MS;
        setLastError(null);
        setStatus('live');
        loadPersistedAlerts();
      };
      ws.onmessage = handleMessage;
      ws.onerror = () => {
        if (!disposed) setLastError(`Cannot reach ${url}`);
      };
      ws.onclose = () => {
        if (socket === ws) socket = null;
        if (disposed) return;
        setStatus('disconnected');
        setFrame(null);
        setFrameData(null);
        scheduleReconnect();
      };
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(retryTimer);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [url]);

  const stats = frameData?.stats || null;
  return { status, connected: status === 'live', frame, frameData, alerts, stats, lastError, url };
};

export default useWebSocket;
