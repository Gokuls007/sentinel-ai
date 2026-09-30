import { useEffect, useMemo, useState } from 'react';
import { fetchJson, wsUrl } from '../lib/api';
import { connectFeed } from '../lib/feedSocket';
import { FeedContext, FrameContext, mergeAlerts } from './liveFeed';

/**
 * One live connection to the Sentinel backend feed (`/ws/feed`, primary camera frames +
 * alerts from every camera), shared by every page.
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
    let disposed = false;

    const loadPersistedAlerts = async () => {
      try {
        const history = await fetchJson('/api/alerts?limit=100');
        if (!disposed && Array.isArray(history)) setAlerts((prev) => mergeAlerts(prev, history));
      } catch {
        // REST history is best-effort; the socket still delivers live alerts.
      }
    };

    const dispose = connectFeed(url, {
      onStatus: setStatus,
      onError: setLastError,
      onOpen: loadPersistedAlerts,
      onClose: () => setFrameState({ frame: null, frameData: null }),
      onMessage: (msg) => {
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
      },
    });

    return () => {
      disposed = true;
      dispose();
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
