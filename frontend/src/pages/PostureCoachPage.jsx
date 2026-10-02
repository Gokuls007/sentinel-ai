import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, Bell, BellOff, Crosshair, RotateCcw, Volume2, VolumeX, Webcam } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import VideoFeed from '../components/VideoFeed';
import { useAppMode } from '../context/appMode';
import { describeError, fetchJson, formatDuration, loadStored, postJson, saveStored } from '../lib/api';
import { useCameraControl, useLaptopCamera, useLaptopFrames } from '../lib/laptopCamera';
import { chipClass } from '../lib/ui';

const BAD = new Set(['slouching', 'leaning', 'too_close', 'slumped']);
const SETTINGS_KEY = 'sentinel.posture.reminders';
const INTERVALS = [
  [30, '30 s (demo)'],
  [60, '1 min'],
  [300, '5 min'],
  [600, '10 min'],
];

// Static class names so Tailwind generates them.
const STATUS_STYLE = {
  good: { text: 'text-emerald-300', ring: 'border-emerald-400/60 bg-emerald-950/30', bar: 'bg-emerald-400' },
  slouching: { text: 'text-amber-300', ring: 'border-amber-400/60 bg-amber-950/30', bar: 'bg-amber-400' },
  leaning: { text: 'text-orange-300', ring: 'border-orange-400/60 bg-orange-950/30', bar: 'bg-orange-400' },
  too_close: { text: 'text-red-300', ring: 'border-red-400/60 bg-red-950/30', bar: 'bg-red-400' },
  away: { text: 'text-white/50', ring: 'border-white/15 bg-white/5', bar: 'bg-white/20' },
  calibrating: { text: 'text-cyan-300', ring: 'border-cyan-400/60 bg-cyan-950/30', bar: 'bg-cyan-400' },
  no_baseline: { text: 'text-white/70', ring: 'border-white/20 bg-white/5', bar: 'bg-white/20' },
  moved: { text: 'text-sky-300', ring: 'border-sky-400/50 bg-sky-950/30', bar: 'bg-sky-500/60' },
  unclear: { text: 'text-violet-300', ring: 'border-violet-400/50 bg-violet-950/30', bar: 'bg-violet-500/60' },
  slumped: { text: 'text-rose-300', ring: 'border-rose-400/60 bg-rose-950/30', bar: 'bg-rose-500' },
};
const style = (s) => STATUS_STYLE[s] || STATUS_STYLE.away;
const STATUS_LABELS = {
  good: 'Good', slouching: 'Slouching', leaning: 'Leaning', too_close: 'Too close', away: 'Away', moved: 'Moved',
  unclear: 'Unclear', slumped: 'Slumped',
};
const SUMMARY_KEYS = ['good', 'slouching', 'leaning', 'slumped', 'too_close', 'moved', 'unclear', 'away'];
const DEBUG_KEY = 'sentinel.posture.debug';

const fmtNum = (v, digits = 2) => (v == null || Number.isNaN(v) ? '--' : Number(v).toFixed(digits));

/** The raw values behind each status: now vs baseline, the change, and the limit. */
const DebugPanel = ({ rows }) => (
  <div className="border border-white/10 bg-black/40 p-2 overflow-x-auto">
    {!rows ? (
      <p className="text-[10px] mono text-white/40">No measurements yet (set a baseline and sit in view).</p>
    ) : (
      <table className="w-full text-[10px] mono">
        <caption className="sr-only">Posture measurements</caption>
        <thead>
          <tr className="text-white/40 text-left">
            <th scope="col" className="font-normal pr-2">Measure</th>
            <th scope="col" className="font-normal pr-2 text-right">Now</th>
            <th scope="col" className="font-normal pr-2 text-right">Baseline</th>
            <th scope="col" className="font-normal pr-2 text-right">Change</th>
            <th scope="col" className="font-normal text-right">Limit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className={r.triggered ? style(r.status).text : 'text-white/70'}>
              <td className="pr-2 py-0.5">{r.label}{r.triggered ? ' ●' : ''}</td>
              <td className="pr-2 text-right">{fmtNum(r.now, r.key === 'tilt_deg' || r.key === 'face_size' ? 1 : 2)}</td>
              <td className="pr-2 text-right">{fmtNum(r.baseline, r.key === 'tilt_deg' || r.key === 'face_size' ? 1 : 2)}</td>
              <td className="pr-2 text-right">
                {r.key === 'tilt_deg' || r.key === 'lateral'
                  ? `${r.change >= 0 ? '+' : ''}${fmtNum(r.change, r.key === 'tilt_deg' ? 1 : 2)}`
                  : r.change == null ? '--' : `${fmtNum(r.change, 2)}x`}
              </td>
              <td className="text-right text-white/40">
                {r.limit}
                {r.note ? <span className="block text-[9px] text-sky-300/80">{r.note}</span> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    )}
    <p className="mt-1 text-[9px] mono text-white/30">
      All relative to your shoulders and your own baseline (smoothed over 1 s). ● = over its limit.
    </p>
  </div>
);

/** A soft two-note chime with the Web Audio API (no audio file needed). */
function chime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [[660, 0], [880, 0.18]].forEach(([freq, at]) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + at);
      gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + at + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + at + 0.5);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + at);
      osc.stop(ctx.currentTime + at + 0.55);
    });
    setTimeout(() => ctx.close(), 1200);
  } catch {
    // audio unavailable: the on-screen reminder still shows
  }
}

function useReminderSettings() {
  const [settings, setSettings] = useState(() => ({
    notify: true, sound: true, interval: 300, ...loadStored(SETTINGS_KEY, {}),
  }));
  const update = (patch) => setSettings((s) => {
    const next = { ...s, ...patch };
    saveStored(SETTINGS_KEY, next);
    return next;
  });
  return [settings, update];
}

/** A gentle reminder once a poor posture has been held for `interval` seconds, then again every
 * `interval` while it continues. Returns [banner, dismiss, onFrame]; pass onFrame to the frame
 * socket so the check runs on every frame. */
function useReminders(settings) {
  const [banner, setBanner] = useState(null);
  const last = useRef({ episode: null, at: 0 });
  const settingsRef = useRef(settings);
  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  const onFrame = useCallback((frameData) => {
    const posture = frameData?.posture;
    const s = settingsRef.current;
    if (!posture || !BAD.has(posture.status) || posture.held_s < s.interval) return;
    const now = Date.now();
    const sameEpisode = last.current.episode === posture.episode;
    if (sameEpisode && now - last.current.at < s.interval * 1000) return;
    last.current = { episode: posture.episode, at: now };
    const body = posture.reasons?.[0] || 'Take a moment to sit tall.';
    const title = `${posture.label} for ${formatDuration(posture.held_s).slice(3)}`;
    setBanner({ title, body, at: now });
    if (s.sound) chime();
    if (s.notify && 'Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(`Posture: ${title}`, { body, tag: 'sentinel-posture', silent: true });
      } catch {
        // some browsers only allow notifications from a service worker
      }
    }
    postJson('/api/posture/reminder?camera=laptop', {}).catch(() => {});
  }, []);

  return [banner, () => setBanner(null), onFrame];
}

const StartCamera = ({ control, cam }) => (
  <DashboardPanel title="Start your webcam" className="max-w-2xl">
    <div className="space-y-4">
      <p className="text-sm outfit text-white/80 leading-relaxed">
        The posture coach watches you through your laptop webcam. Everything runs on this computer:
        no video or posture data leaves it, and nothing is sent to any AI service.
      </p>
      {cam?.status === 'error' && <p className="text-[11px] text-red-300">The camera stopped with an error: {cam.error}</p>}
      {control.error && <p className="text-[11px] text-red-300">{control.error.detail || describeError(control.error)}</p>}
      <button
        type="button"
        onClick={() => control.start()}
        disabled={control.busy != null || cam?.status === 'starting'}
        className="px-4 py-2 text-[11px] mono uppercase font-bold tracking-widest border border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 flex items-center gap-2 disabled:opacity-40"
      >
        <Webcam className="w-4 h-4" aria-hidden="true" />
        {cam?.status === 'starting' || control.busy === 'start' ? 'Starting camera...' : 'Start camera'}
      </button>
      <p className="text-[10px] mono text-white/40">After this first start, the camera starts automatically when you open the app.</p>
    </div>
  </DashboardPanel>
);

const StatusCard = ({ posture, connected, onBaseline, busy, mode }) => {
  const [debug, setDebug] = useState(() => Boolean(loadStored(DEBUG_KEY, false)));
  const toggleDebug = () => setDebug((d) => {
    saveStored(DEBUG_KEY, !d);
    return !d;
  });
  if (!connected || !posture) {
    return (
      <div className="border border-white/10 bg-white/5 p-6 text-center text-[11px] mono text-white/50 uppercase tracking-widest">
        {mode !== 'posture' ? 'Switch to Desk Posture Coach mode to start coaching' : 'Waiting for the camera...'}
      </div>
    );
  }
  const s = style(posture.status);
  const bad = BAD.has(posture.status);
  return (
    <div className={`border-2 ${s.ring} p-5 space-y-4`} role="status" aria-live="polite">
      <div className="text-[10px] mono uppercase tracking-[0.3em] text-white/40">Posture</div>
      <div className={`text-5xl md:text-6xl font-bold outfit leading-none ${s.text}`}>{posture.label}</div>
      {posture.calibrating && (
        <div className="text-sm text-cyan-200">
          {posture.calibration_phase === 'get_ready'
            ? `Recording starts in ${Math.ceil(posture.calibration_left_s)}...`
            : `Recording... ${posture.calibration_left_s.toFixed(1)} s left`}
        </div>
      )}
      {bad && (
        <div className="text-[11px] mono uppercase text-white/50">for {formatDuration(posture.held_s).slice(3)}</div>
      )}
      {posture.reasons?.length > 0 && (
        <ul className="space-y-1 text-sm text-white/85 outfit">
          {posture.reasons.map((r) => <li key={r}>• {r}</li>)}
        </ul>
      )}
      {posture.status === 'no_baseline' && !posture.calibrating && (
        <div className="space-y-2">
          <p className="text-sm text-white/75 outfit">
            Sit upright, facing the screen with both shoulders in view. The coach compares you with that posture.
          </p>
          <div className="flex items-center justify-center gap-2 text-cyan-300 animate-bounce" aria-hidden="true">
            <ArrowDown className="w-5 h-5" />
            <span className="text-[11px] mono uppercase tracking-widest">Start here</span>
            <ArrowDown className="w-5 h-5" />
          </div>
        </div>
      )}
      {posture.hint && (
        <div className="flex items-center justify-between gap-3 p-2 border border-sky-400/40 bg-sky-950/30 text-[11px] text-sky-200">
          <span>{posture.hint}</span>
          <button type="button" onClick={onBaseline} disabled={busy} className={chipClass(true)}>Reset baseline</button>
        </div>
      )}
      {posture.status === 'unclear' && (
        <ul className="text-[11px] text-violet-200/90 outfit space-y-0.5">
          <li>• Turn on a light in front of you (not behind)</li>
          <li>• A lighter wall or chair behind dark clothes helps</li>
          <li>• Keep both shoulders inside the picture</li>
        </ul>
      )}
      {posture.calibration_error && <p className="text-[11px] text-red-300">{posture.calibration_error}</p>}
      <button
        type="button"
        onClick={onBaseline}
        disabled={busy || posture.calibrating}
        className={`w-full px-4 py-2.5 text-[11px] mono uppercase font-bold tracking-widest border border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 disabled:opacity-40 flex items-center justify-center gap-2 ${
          !posture.has_baseline && !posture.calibrating ? 'ring-4 ring-cyan-300/70 ring-offset-2 ring-offset-black shadow-[0_0_24px_rgba(34,211,238,0.6)] py-3.5 text-sm' : ''
        }`}
      >
        <Crosshair className="w-4 h-4" aria-hidden="true" />
        {posture.has_baseline ? 'Reset baseline' : 'Set baseline'}
      </button>
      {posture.has_baseline && posture.baseline_recorded_at && (
        <p className="text-[9px] mono text-white/30 uppercase">
          Baseline from {new Date(posture.baseline_recorded_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}
        </p>
      )}
      <button type="button" onClick={toggleDebug} aria-pressed={debug} className={chipClass(debug)}>
        {debug ? 'Hide measurements' : 'Show measurements'}
      </button>
      {debug && <DebugPanel rows={posture.debug} />}
    </div>
  );
};

const ReminderSettings = ({ settings, update }) => {
  const [permission, setPermission] = useState(() => ('Notification' in window ? Notification.permission : 'unsupported'));
  const toggleNotify = async () => {
    const on = !settings.notify;
    if (on && permission === 'default') {
      try {
        setPermission(await Notification.requestPermission());
      } catch {
        setPermission('denied');
      }
    }
    update({ notify: on });
  };
  return (
    <DashboardPanel title="Reminders">
      <div className="space-y-3">
        <p className="text-[11px] text-white/60 outfit">
          A gentle reminder when a poor posture has lasted this long:
        </p>
        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Remind after">
          {INTERVALS.map(([s, label]) => (
            <button key={s} type="button" role="radio" aria-checked={settings.interval === s}
              onClick={() => update({ interval: s })} className={chipClass(settings.interval === s)}>
              {label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={toggleNotify} aria-pressed={settings.notify}
            className={`${chipClass(settings.notify)} flex items-center gap-1.5`}>
            {settings.notify ? <Bell className="w-3 h-3" aria-hidden="true" /> : <BellOff className="w-3 h-3" aria-hidden="true" />}
            Notification {settings.notify ? 'on' : 'off'}
          </button>
          <button type="button" onClick={() => update({ sound: !settings.sound })} aria-pressed={settings.sound}
            className={`${chipClass(settings.sound)} flex items-center gap-1.5`}>
            {settings.sound ? <Volume2 className="w-3 h-3" aria-hidden="true" /> : <VolumeX className="w-3 h-3" aria-hidden="true" />}
            Sound {settings.sound ? 'on' : 'off'}
          </button>
          <button type="button" onClick={chime} className={chipClass(false)}>Test sound</button>
        </div>
        {settings.notify && permission === 'denied' && (
          <p className="text-[10px] text-amber-300">
            Browser notifications are blocked for this site, so reminders show here on the page instead.
          </p>
        )}
      </div>
    </DashboardPanel>
  );
};

const SessionSummary = ({ posture, timeline, onReset }) => {
  const s = posture?.session;
  const total = s ? Object.values(s.seconds).reduce((a, b) => a + b, 0) : 0;
  const pct = s?.good_fraction != null ? Math.round(s.good_fraction * 100) : null;
  return (
    <DashboardPanel title="This session" headerAction={s ? `${formatDuration(total)} TRACKED` : null}>
      {!s ? (
        <p className="text-[11px] mono text-white/40">No data yet.</p>
      ) : (
        <div className="space-y-4">
          <div className="flex items-end gap-6 flex-wrap">
            <div>
              <div className="text-4xl font-bold outfit text-emerald-300 leading-none">{pct != null ? `${pct}%` : '--'}</div>
              <div className="text-[10px] mono uppercase text-white/40 mt-1">good posture</div>
            </div>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] mono">
              {SUMMARY_KEYS.map((k) => (
                <div key={k} className="contents">
                  <dt className={`${style(k).text} uppercase`}>{STATUS_LABELS[k]}</dt>
                  <dd className="text-white/70 text-right">{formatDuration(s.seconds[k] || 0)}</dd>
                </div>
              ))}
            </dl>
            <div className="text-[11px] mono text-white/50">{s.reminders} reminder{s.reminders === 1 ? '' : 's'}</div>
          </div>
          <div>
            <div className="text-[9px] mono uppercase text-white/40 mb-1">Timeline (10 s per block)</div>
            {timeline.length ? (
              <div className="flex h-6 w-full overflow-hidden border border-white/10" aria-label="Posture over the session">
                {timeline.map((b) => (
                  <div key={b.t} className={`flex-1 min-w-[2px] ${style(b.status).bar}`}
                    title={`${formatDuration(b.t)}: ${STATUS_LABELS[b.status] || b.status}`} />
                ))}
              </div>
            ) : (
              <p className="text-[10px] mono text-white/30">The timeline fills in every 10 seconds.</p>
            )}
            <div className="flex flex-wrap gap-3 mt-2">
              {SUMMARY_KEYS.map((k) => (
                <span key={k} className="flex items-center gap-1 text-[9px] mono uppercase text-white/50">
                  <span className={`w-2 h-2 ${style(k).bar}`} aria-hidden="true" /> {STATUS_LABELS[k]}
                </span>
              ))}
            </div>
          </div>
          <button type="button" onClick={onReset} className={`${chipClass(false)} flex items-center gap-1.5`}>
            <RotateCcw className="w-3 h-3" aria-hidden="true" /> Reset session
          </button>
        </div>
      )}
    </DashboardPanel>
  );
};

const PostureCoachPage = () => {
  const { mode, setMode } = useAppMode();
  const camState = useLaptopCamera();
  const control = useCameraControl(camState);
  const { cam, loaded } = camState;
  const running = cam?.status === 'running';
  const [settings, update] = useReminderSettings();
  const [banner, dismiss, onFrame] = useReminders(settings);
  const { status: feedStatus, frame, frameData } = useLaptopFrames(onFrame);
  const posture = frameData?.posture || null;
  const [timeline, setTimeline] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!running) return undefined;
    let cancelled = false;
    const load = () => fetchJson('/api/posture?camera=laptop')
      .then((d) => !cancelled && setTimeline(d.timeline || []))
      .catch(() => {});
    load();
    const id = setInterval(load, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, [running]);

  const call = async (path, after) => {
    setBusy(true);
    setError(null);
    try {
      const data = await postJson(path, {});
      after?.(data);
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  if (!loaded) return <p className="mono text-[11px] text-white/40 uppercase">Checking camera...</p>;

  return (
    <div className="space-y-4">
      {mode !== 'posture' && (
        <div className="p-3 border-l-2 border-amber-500 bg-amber-950/20 text-amber-100 text-[11px] flex items-center gap-3">
          The camera is in {mode} mode, so posture isn&apos;t being analysed.
          <button type="button" className={chipClass(true)} onClick={() => setMode('posture')}>Switch to posture</button>
        </div>
      )}
      {banner && (
        <div className="p-4 border-2 border-amber-400/70 bg-amber-950/40 flex items-start justify-between gap-4" role="alert">
          <div>
            <div className="text-lg font-bold text-amber-200 outfit">{banner.title}</div>
            <div className="text-sm text-amber-100/90 outfit">{banner.body}</div>
          </div>
          <button type="button" onClick={dismiss} className={chipClass(true)}>Got it</button>
        </div>
      )}
      {error && <p className="text-[11px] text-red-300">{error}</p>}
      {!running ? (
        <StartCamera control={control} cam={cam} />
      ) : (
        <>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 lg:col-span-7 h-[min(60vh,560px)] min-h-[300px]">
              <VideoFeed title="You" frame={frame} status={feedStatus} stats={frameData?.stats || null}
                source="Laptop webcam" />
            </div>
            <div className="col-span-12 lg:col-span-5 space-y-4">
              <StatusCard posture={posture} connected={feedStatus === 'live'} busy={busy} mode={mode}
                onBaseline={() => call('/api/posture/baseline?camera=laptop')} />
              <ReminderSettings settings={settings} update={update} />
            </div>
          </div>
          <SessionSummary posture={posture} timeline={timeline}
            onReset={() => call('/api/posture/session/reset?camera=laptop', () => setTimeline([]))} />
          <p className="text-[10px] mono text-white/30">
            Runs entirely on this computer. Front-facing webcam estimate compared with your own baseline;
            not a medical or ergonomic assessment.
          </p>
        </>
      )}
    </div>
  );
};

export default PostureCoachPage;
