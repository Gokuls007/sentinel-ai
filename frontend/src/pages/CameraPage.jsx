import { useId, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Webcam, Users, Crosshair, Gauge } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import VideoFeed from '../components/VideoFeed';
import AlertPanel from '../components/AlertPanel';
import ZoneEditor from '../components/ZoneEditor';
import ErgonomicsPanel from '../components/ErgonomicsPanel';
import { ActivityPanel, ViewBanner } from '../components/ActivityPanel';
import DemoFootagePanel from '../components/DemoFootagePanel';
import { useFeed } from '../context/liveFeed';
import {
  BACKEND_START_HINT, describeError, formatDuration, isOfflineError, postJson, saveStored, useNow,
} from '../lib/api';
import {
  INDEX_KEY, LAPTOP, storedCameraIndex, useCameraControl, useLaptopCamera, useLaptopFrames,
} from '../lib/laptopCamera';
import { inputClass, labelClass } from '../lib/ui';

const INDEX_OPTIONS = [0, 1, 2, 3];

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

/** Record the raw webcam video (no overlays) to data/recordings, for test clips. */
const RecordControl = ({ cam, onChange }) => {
  const rec = cam?.recording || {};
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [startedAt, setStartedAt] = useState(null);
  const now = useNow(500);
  const recording = Boolean(rec.recording);
  // Elapsed time: the server's count at the last poll, advanced locally between polls.
  const [base, setBase] = useState({ seconds: 0, at: 0 });
  const [seenSeconds, setSeenSeconds] = useState(null);
  if (rec.seconds !== seenSeconds) {
    setSeenSeconds(rec.seconds);
    setBase({ seconds: rec.seconds || 0, at: Date.now() });
  }
  const elapsed = recording ? Math.max(base.seconds + (now - base.at) / 1000, startedAt ? (now - startedAt) / 1000 : 0) : 0;

  const toggle = async () => {
    setBusy(true);
    setError(null);
    try {
      const info = await postJson(`/api/cameras/${LAPTOP}/record/${recording ? 'stop' : 'start'}`, {});
      setStartedAt(recording ? null : Date.now());
      onChange({ id: LAPTOP, recording: info });
    } catch (err) {
      setError(err.detail || err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      {!recording && rec.last_path && (
        <span className="text-[10px] mono text-white/50" title={rec.folder}>
          Saved {rec.last_path} ({Number(rec.last_seconds).toFixed(0)} s) in data/recordings
        </span>
      )}
      {error && <span className="text-[10px] mono text-red-300">{error}</span>}
      <button
        type="button"
        onClick={toggle}
        disabled={busy}
        aria-pressed={recording}
        title="Saves the raw camera video (no overlays) to data/recordings, e.g. for fall or ergonomics test clips"
        className={`${buttonBase} flex items-center gap-2 ${
          recording ? 'border-red-500 text-white bg-red-600 hover:bg-red-500' : 'border-white/30 text-white/80 hover:border-red-400 hover:text-red-300'
        }`}
      >
        <span className={`w-2.5 h-2.5 rounded-full ${recording ? 'bg-white animate-pulse' : 'bg-red-500'}`} aria-hidden="true" />
        {recording ? `Stop recording ${formatDuration(elapsed).slice(3)}` : 'Record clip'}
      </button>
    </div>
  );
};

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
      {!drawing && <ViewBanner view={frameData?.view} />}
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
        <div className="grid md:grid-cols-2 gap-3">
          <ActivityPanel connected={status === 'live'} activity={frameData?.activity} ergonomics={frameData?.ergonomics} />
          <ErgonomicsPanel connected={status === 'live'} ergonomics={frameData ? frameData.ergonomics : null} />
        </div>
      )}
      <details className="border border-white/10 bg-black/30 open:pb-2" open={drawing || undefined}>
        <summary className="cursor-pointer px-3 py-2 text-[11px] mono uppercase text-white/60 hover:text-white/80">
          Optional: zones (restricted areas, time limits, one-way lanes)
        </summary>
        <div className="px-2 pt-2">
          <ZoneEditor camera={LAPTOP} frame={frame} onDrawingChange={setDrawing} />
        </div>
      </details>
    </div>
  );
};

// --- Page --------------------------------------------------------------------------------

const CameraPage = () => {
  const id = useId();
  const { alerts, connected } = useFeed();
  const camState = useLaptopCamera();
  const { cam, error: pollError, loaded, apply } = camState;
  const [index, setIndex] = useState(storedCameraIndex);
  const control = useCameraControl(camState); // also auto-starts after the first manual start
  const { busy, error: actionError } = control;

  const laptopAlerts = useMemo(() => alerts.filter((a) => a.camera_id === LAPTOP), [alerts]);
  const status = cam?.status || 'stopped';

  const chooseIndex = (v) => {
    setIndex(v);
    saveStored(INDEX_KEY, v);
  };

  const run = (kind) => (kind === 'start' ? control.start(index) : control.stop());

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
      <div className="flex items-center gap-3">
      {running && !stopping && <RecordControl cam={cam} onChange={apply} />}
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
      <DemoFootagePanel />
    </div>
  );
};

export default CameraPage;
