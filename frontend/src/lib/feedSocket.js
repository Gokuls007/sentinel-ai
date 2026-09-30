const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 10000;

/**
 * Open a `/ws/feed` socket with reconnect + exponential backoff.
 *
 * handlers: { onStatus(status), onError(message|null), onOpen(), onMessage(msg), onClose() }
 * status: 'connecting' | 'live' | 'disconnected'
 * Returns dispose(); all state lives in this closure, so StrictMode's double
 * mount cannot leak sockets or timers.
 */
export function connectFeed(url, handlers) {
  const h = handlers;
  let socket = null;
  let retryTimer = null;
  let backoff = BACKOFF_START_MS;
  let disposed = false;

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
    if (msg && typeof msg === 'object') h.onMessage?.(msg);
  };

  function connect() {
    if (disposed) return;
    h.onStatus?.('connecting');
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      h.onError?.(String(err?.message || err));
      h.onStatus?.('disconnected');
      scheduleReconnect();
      return;
    }
    socket = ws;

    ws.onopen = () => {
      if (disposed) return;
      backoff = BACKOFF_START_MS;
      h.onError?.(null);
      h.onStatus?.('live');
      h.onOpen?.();
    };
    ws.onmessage = handleMessage;
    ws.onerror = () => {
      if (!disposed) h.onError?.(`Cannot reach ${url}`);
    };
    ws.onclose = () => {
      if (socket === ws) socket = null;
      if (disposed) return;
      h.onStatus?.('disconnected');
      h.onClose?.();
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
}
