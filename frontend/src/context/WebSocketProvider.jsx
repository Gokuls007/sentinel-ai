import { useEffect, useMemo, useState } from 'react';
import { fetchJson, wsUrl } from '../lib/api';
import { FeedContext, FrameContext, mergeAlerts } from './liveFeed';

const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 10000;

/**
 * One live connection to the Sentinel backend feed (`/ws/feed`), shared by every page.
 *
 * status: 'connecting' | 'live' | 'disconnected'
 * Nothing here is simulated: when the backend is unreachable the frame is
 * cleared and status is 'disconnected'.
 */
export default function WebSocketProvider({ url = wsUrl(), children }) {
  const [status, setStatus] = useState('connecting');
  const [frameState, setFrameState] = useState({ frame: null, frameData: null });
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
          // One state update per frame (image + data together).
          setFrameState((prev) => ({ frame: msg.image || prev.frame, frameData: msg.data || null }));
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
        setFrameState({ frame: null, frameData: null });
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

  const feed = useMemo(
    () => ({ status, connected: status === 'live', alerts, lastError, url }),
    [status, alerts, lastError, url],
  );

  // `children` is created by the parent, so a frame update re-renders only
  // FrameContext consumers, not the page tree.
  return (
    <FeedContext.Provider value={feed}>
      <FrameContext.Provider value={frameState}>{children}</FrameContext.Provider>
    </FeedContext.Provider>
  );
}
