import { createContext, useContext } from 'react';

export const MAX_ALERTS = 200;

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
 * Two contexts so the ~15 fps frame stream only re-renders its readers:
 * - FeedContext: { status, connected, alerts, lastError, url } (changes rarely)
 * - FrameContext: { frame, frameData } (changes every frame)
 */
export const FeedContext = createContext(null);
export const FrameContext = createContext({ frame: null, frameData: null });

export function useFeed() {
  const ctx = useContext(FeedContext);
  if (!ctx) throw new Error('useFeed must be used inside <WebSocketProvider>');
  return ctx;
}

/** Latest annotated frame + per-frame data. Only components that render the video/stats should call this. */
export function useFrame() {
  return useContext(FrameContext);
}
