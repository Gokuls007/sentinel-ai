import { useEffect, useId, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Webcam, Users, Crosshair, Gauge } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import VideoFeed from '../components/VideoFeed';
import AlertPanel from '../components/AlertPanel';
import ZoneEditor from '../components/ZoneEditor';
import ErgonomicsPanel from '../components/ErgonomicsPanel';
import { useFeed } from '../context/liveFeed';
import { BACKEND_START_HINT, describeError, fetchJson, isOfflineError, loadStored, postJson, saveStored, wsUrl } from '../lib/api';
import { connectFeed } from '../lib/feedSocket';
import { inputClass, labelClass } from '../lib/ui';

const LAPTOP = 'laptop';
const INDEX_KEY = 'sentinel.camera.index';
const INDEX_OPTIONS = [0, 1, 2, 3];
const TRANSITIONAL = new Set(['starting', 'stopping']);

/**
 * Poll /api/cameras for the laptop entry: every 1 s while starting/stopping,
 * every 3 s otherwise. `apply(info)` shows a POST result immediately and
 * restarts the loop (aborting any in-flight, now stale, request).
 */
function useLaptopCamera() {
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

  const apply = (info) => {
    if (info && info.id === LAPTOP) setState((prev) => ({ ...prev, cam: { ...prev.cam, ...info }, error: null }));
    setRestart((n) => n + 1);
  };
  return { ...state, apply };
}

/** Frames from `/ws/feed?camera=laptop`, local to the component that renders them. */
function useLaptopFrames() {
  const [state, setState] = useState({ status: 'connecting', frame: null, frameData: null });
  useEffect(
    () =>
      connectFeed(wsUrl(LAPTOP), {
        onStatus: (status) => setState((prev) => ({ ...prev, status })),
        onClose: () => setState((prev) => ({ ...prev, frame: null, frameData: null })),
        onMessage: (msg) => {
          // Alerts arrive through the shared provider already; only frames matter here.
          if (msg.type === 'frame') {
            setState((prev) => ({ ...prev, frame: msg.image || prev.frame, frameData: msg.data || null }));
          }
        },
      }),
    [],
  );
  return state;
}

// --- Pieces ------------------------------------------------------------------------------

const Spinner = ({ children }) => (
  <div className="flex flex-col items-center justify-center py-16 space-y-4" role="status">
    <div className="relative">
      <div className="w-14 h-14 border-2 rounded-full border-cyan-500/20" />
      <div className="absolute top-0 w-14 h-14 border-2 rounded-full animate-spin border-t-cyan-400" />
    </div>
    <p className="mono text-[11px] tracking-[0.3em] uppercase text-cyan-400 animate-pulse text-center">{children}</p>
  </div>
);

const buttonBase =
  'px-4 py-2 text-[11px] mono uppercase font-bold tracking-widest border cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';

const Message = ({ tone = 'red', children }) => (
  <div
    role="alert"
    className={`p-3 border-l-2 text-[11px] outfit ${
      tone === 'red' ? 'border-red-500 bg-red-950/20 text-red-200' : 'border-amber-500 bg-amber-950/20 text-amber-100'
    }`}
  >
    {children}
  </div>
);

const DeviceHints = () => (
  <ul className="mt-2 list-disc pl-5 space-y-0.5 text-[10px] text-white/70">
    <li>Wrong device index: 0 is usually the built-in camera; try another index for a USB camera.</li>
    <li>The camera is in use by another app (Teams, Zoom, the Windows Camera app): close it and try again.</li>
    <li>Windows camera privacy is off: Settings, Privacy &amp; security, Camera, allow camera access for desktop apps.</li>
  </ul>
);

const StatTile = ({ icon, label, value }) => {
  const Icon = icon;
  return (
    <div className="px-3 py-2 bg-cyan-950/20 border-l border-cyan-500/20">
      <div className="flex items-center gap-1.5 text-[9px] uppercase font-bold text-cyan-500/70 tracking-widest">
        <Icon className="w-3 h-3" aria-hidden="true" />
        {label}
      </div>
      <div className="text-xl font-bold mono text-white leading-tight">{value}</div>
    </div>
  );
};

const show = (v, fmt = (x) => x) => (v == null ? '--' : fmt(v));

/** Live webcam panel: owns the frame socket, so frames only re-render this subtree. */
const CameraFeed = ({ cam }) => {
  const { status, frame, frameData } = useLaptopFrames();
  const [drawing, setDrawing] = useState(false); // drawing a zone: the paused frame replaces the feed
  const stats = frameData?.stats || null;
  const sourceStats = cam?.hardware_error
    ? { hardware_error: true, error_message: cam.source_error || 'Camera could not be opened' }
    : null;
  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="grid grid-cols-3 gap-px bg-cyan-500/10 border border-cyan-500/10">
        <StatTile icon={Gauge} label="FPS" value={show(stats?.fps ?? cam?.fps, (v) => Number(v).toFixed(1))} />
        <StatTile icon={Users} label="Persons" value={show(stats?.person_count)} />
        <StatTile icon={Crosshair} label="Tracks" value={show(stats?.active_tracks)} />
      </div>
      <div className={drawing ? 'hidden' : 'h-[min(62vh,640px)] min-h-[320px]'}>
        <VideoFeed
          title="Laptop Camera"
          frame={frame}
          status={status}
          stats={stats}
          sourceStats={sourceStats}
          source={cam?.source != null ? `Device ${cam.source}` : null}
          offlineHint={
            <p className="text-[10px] mono text-red-400/70 mt-3 max-w-md">Connecting to the laptop camera feed...</p>
          }
        />
      </div>
      {!drawing && (
        // Frames (and their `ergonomics`) come from this camera's own socket.
        <ErgonomicsPanel connected={status === 'live'} ergonomics={frameData ? frameData.ergonomics : null} />
      )}
      <ZoneEditor camera={LAPTOP} frame={frame} onDrawingChange={setDrawing} />
    </div>
  );
};

// --- Page --------------------------------------------------------------------------------

const CameraPage = () => {
  const id = useId();
  const { alerts, connected } = useFeed();
  const { cam, error: pollError, loaded, apply } = useLaptopCamera();
  const [index, setIndex] = useState(() => {
    const v = Number(loadStored(INDEX_KEY, 0));
    return INDEX_OPTIONS.includes(v) ? v : 0;
  });
  const [busy, setBusy] = useState(null); // 'start' | 'stop' | null
  const [actionError, setActionError] = useState(null);

  const laptopAlerts = useMemo(() => alerts.filter((a) => a.camera_id === LAPTOP), [alerts]);
  const status = cam?.status || 'stopped';

  const chooseIndex = (v) => {
    setIndex(v);
    saveStored(INDEX_KEY, v);
  };

  const run = async (kind) => {
    setBusy(kind);
    setActionError(null);
    try {
      const info = kind === 'start'
        ? await postJson(`/api/cameras/${LAPTOP}/start`, { index })
        : await postJson(`/api/cameras/${LAPTOP}/stop`, {});
      apply(info);
    } catch (err) {
      setActionError(err);
      apply(null);
    } finally {
      setBusy(null);
    }
  };

  const actionMessage = (() => {
    if (!actionError) return null;
    if (actionError.status === 403) return 'Camera control only works from this computer.';
    if (isOfflineError(actionError)) return 'Backend offline or starting. Try again in a moment.';
    return actionError.detail || actionError.message;
  })();

  // --- Offline / loading ---
  if (pollError && isOfflineError(pollError)) {
    return (
      <DashboardPanel title="My Camera" severity="red">
        <div className="py-10 text-center space-y-3">
          <p className="mono tracking-[0.4em] text-sm font-bold text-red-500">BACKEND_OFFLINE</p>
          <p className="text-[10px] mono text-red-400/70">Backend offline or starting. Start it with:</p>
          <p className="text-[11px] mono text-cyan-300 px-2 py-1 bg-black/60 border border-cyan-500/20 inline-block">{BACKEND_START_HINT}</p>
        </div>
      </DashboardPanel>
    );
  }
  if (!loaded) return <Spinner>Checking camera status...</Spinner>;

  const running = status === 'running';
  const starting = status === 'starting' || busy === 'start';
  const stopping = status === 'stopping' || busy === 'stop';

  // --- Header row (shown in every state) ---
  const header = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        {running && !stopping && cam?.hardware_error ? (
          <span className="flex items-center gap-2 px-3 py-1 border border-amber-500/60 bg-amber-950/30 text-amber-300 text-[11px] mono font-bold uppercase tracking-widest" role="status">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" aria-hidden="true" />
            Camera error: retrying
          </span>
        ) : running && !stopping ? (
          <span className="flex items-center gap-2 px-3 py-1 border border-red-500/60 bg-red-950/40 text-red-400 text-[11px] mono font-bold uppercase tracking-widest" role="status">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" aria-hidden="true" />
            LIVE: LAPTOP CAMERA
          </span>
        ) : (
          <span className="flex items-center gap-2 px-3 py-1 border border-white/10 text-white/40 text-[11px] mono font-bold uppercase tracking-widest">
            <Webcam className="w-3.5 h-3.5" aria-hidden="true" />
            Camera {stopping ? 'stopping' : starting ? 'starting' : status === 'error' ? 'error' : 'off'}
          </span>
        )}
        <Link to={`/events?camera_id=${LAPTOP}`} className="text-[10px] mono uppercase text-cyan-400/80 hover:text-cyan-300 underline">
          Laptop camera events
        </Link>
      </div>
      {(running || starting) && (
        <button
          type="button"
          onClick={() => run('stop')}
          disabled={busy != null || status === 'stopping'}
          className={`${buttonBase} border-red-500/60 text-red-300 bg-red-950/30 hover:bg-red-900/40`}
        >
          Stop camera
        </button>
      )}
    </div>
  );

  let body;
  if (stopping) {
    body = <DashboardPanel title="My Camera"><Spinner>Stopping camera...</Spinner></DashboardPanel>;
  } else if (starting) {
    body = <DashboardPanel title="My Camera"><Spinner>Starting camera and loading models...</Spinner></DashboardPanel>;
  } else if (running) {
    body = (
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-8 space-y-3">
          {cam?.hardware_error && (
            <Message>
              <strong className="mono uppercase">Camera {cam.source} could not be opened</strong>
              {cam.source_error ? `: ${cam.source_error}` : ''}. Retrying every 5 s.
              <DeviceHints />
            </Message>
          )}
          <CameraFeed cam={cam} />
        </div>
        <div className="col-span-12 xl:col-span-4 min-h-[360px]">
          <AlertPanel alerts={laptopAlerts} connected={connected} />
        </div>
      </div>
    );
  } else {
    // stopped or error: the start panel. The camera only ever starts from this button.
    body = (
      <DashboardPanel title="Start your laptop camera" className="max-w-2xl">
        <div className="space-y-4">
          <p className="text-sm outfit text-white/80 leading-relaxed">
            Start your laptop camera: detection (people, poses, falls, loitering) runs on this computer;
            video is not uploaded anywhere.
          </p>
          {status === 'error' && (
            <Message>
              <strong className="mono uppercase">The camera stopped with an error</strong>
              {cam?.error ? `: ${cam.error}` : ''}
              <DeviceHints />
            </Message>
          )}
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label htmlFor={`${id}-index`} className={labelClass}>Camera device</label>
              <select
                id={`${id}-index`}
                value={index}
                onChange={(e) => chooseIndex(Number(e.target.value))}
                className={inputClass}
              >
                {INDEX_OPTIONS.map((i) => (
                  <option key={i} value={i}>{i === 0 ? '0 (built-in camera)' : `${i}`}</option>
                ))}
              </select>
            </div>
            <button
              type="button"
              onClick={() => run('start')}
              disabled={busy != null}
              className={`${buttonBase} border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 flex items-center gap-2`}
            >
              <Webcam className="w-4 h-4" aria-hidden="true" />
              Start camera
            </button>
          </div>
        </div>
      </DashboardPanel>
    );
  }

  return (
    <div className="space-y-4">
      {header}
      {actionMessage && <Message tone={actionError?.status === 403 ? 'amber' : 'red'}>{actionMessage}</Message>}
      {pollError && !isOfflineError(pollError) && <Message>{describeError(pollError)}</Message>}
      {body}
    </div>
  );
};

export default CameraPage;
