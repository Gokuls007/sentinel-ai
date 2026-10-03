import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchJson, loadStored, postJson, saveStored, wsUrl } from './api';
import { connectFeed } from './feedSocket';

export const LAPTOP = 'laptop';
export const INDEX_KEY = 'sentinel.camera.index';
export const AUTOSTART_KEY = 'sentinel.camera.autostart';
const TRANSITIONAL = new Set(['starting', 'stopping']);

/**
 * Poll /api/cameras for the laptop entry: every 1 s while starting/stopping,
 * every 3 s otherwise. `apply(info)` shows a POST result immediately and
 * restarts the loop (aborting any in-flight, now stale, request).
 */
export function useLaptopCamera() {
  const [state, setState] = useState({ cam: null, error: null, loaded: false });
  const [restart, setRestart] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    let controller = null;
    const tick = async () => {
      controller = new AbortController();
      let delay = 3000;
      try {
        const cams = await fetchJson('/api/cameras', { signal: controller.signal });
        const cam = Array.isArray(cams) ? cams.find((c) => c.id === LAPTOP) || null : null;
        if (cancelled) return;
        setState({ cam, error: null, loaded: true });
        if (cam && TRANSITIONAL.has(cam.status)) delay = 1000;
      } catch (err) {
        if (cancelled || err.name === 'AbortError') return;
        setState({ cam: null, error: err, loaded: true });
      }
      if (!cancelled) timer = setTimeout(tick, delay);
    };
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [restart]);

  const apply = useCallback((info) => {
    if (info && info.id === LAPTOP) setState((prev) => ({ ...prev, cam: { ...prev.cam, ...info }, error: null }));
    setRestart((n) => n + 1);
  }, []);
  return { ...state, apply };
}

/** Frames from `/ws/feed?camera=laptop`, local to the component that renders them.
 * `onFrame(frameData)` (optional) is called for every frame, outside React rendering. */
export function useLaptopFrames(onFrame) {
  const [state, setState] = useState({ status: 'connecting', frame: null, frameData: null });
  const onFrameRef = useRef(onFrame);
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);
  useEffect(
    () =>
      connectFeed(wsUrl(LAPTOP), {
        onStatus: (status) => setState((prev) => ({ ...prev, status })),
        onClose: () => setState((prev) => ({ ...prev, frame: null, frameData: null })),
        onMessage: (msg) => {
          // Alerts arrive through the shared provider already; only frames matter here.
          if (msg.type === 'frame') {
            setState((prev) => ({ ...prev, frame: msg.image || prev.frame, frameData: msg.data || null }));
            onFrameRef.current?.(msg.data || null);
          }
        },
      }),
    [],
  );
  return state;
}

/** Stop the webcam from anywhere (the header button) and turn auto-start off. */
export async function stopLaptopCamera() {
  saveStored(AUTOSTART_KEY, false);
  return postJson(`/api/cameras/${LAPTOP}/stop`, {});
}

export function storedCameraIndex() {
  const v = Number(loadStored(INDEX_KEY, 0));
  return [0, 1, 2, 3].includes(v) ? v : 0;
}

/**
 * Start / stop the laptop webcam. After the first manual start the choice is remembered
 * (`AUTOSTART_KEY`), and pages that use the webcam start it on load. The camera never starts
 * before that first click.
 */
export function useCameraControl(camState) {
  const { cam, loaded, apply } = camState;
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const tried = useRef(false);

  const run = useCallback(async (kind, index = storedCameraIndex()) => {
    setBusy(kind);
    setError(null);
    try {
      const info = kind === 'start'
        ? await postJson(`/api/cameras/${LAPTOP}/start`, { index })
        : await postJson(`/api/cameras/${LAPTOP}/stop`, {});
      if (kind === 'start') saveStored(AUTOSTART_KEY, true);
      if (kind === 'stop') saveStored(AUTOSTART_KEY, false);
      apply(info);
    } catch (err) {
      setError(err);
      apply(null);
    } finally {
      setBusy(null);
    }
  }, [apply]);

  useEffect(() => {
    if (!loaded || tried.current) return;
    tried.current = true;
    const status = cam?.status || 'stopped';
    if (status === 'stopped' && loadStored(AUTOSTART_KEY, false)) run('start');
  }, [loaded, cam, run]);

  return { busy, error, start: (index) => run('start', index), stop: () => run('stop') };
}
