import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, ChevronDown, ChevronUp, Crosshair, RotateCcw, Webcam } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import VideoFeed from '../components/VideoFeed';
import { useAppMode } from '../context/appMode';
import { deleteJson, describeError, fetchJson, formatDuration, loadStored, postJson, putJson, saveStored } from '../lib/api';
import { useCameraControl, useLaptopCamera, useLaptopFrames } from '../lib/laptopCamera';
import { chipClass } from '../lib/ui';
import { CalibrationPanel, ProbabilityBars, RecordingPrompt } from './PostureCalibration';
import { BreakBanner, BreakPanel, TipCard } from './PostureCoaching';
import { MovementCard, MovementSettings, TodayPanel } from './MovementCoach';

const ALERTS_KEY = 'sentinel.posture.reminders';
const ADVANCED_KEY = 'sentinel.posture.advanced';

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
  looking_down: { text: 'text-teal-300', ring: 'border-teal-400/50 bg-teal-950/30', bar: 'bg-teal-500/70' },
  checking: { text: 'text-cyan-200/80', ring: 'border-cyan-400/30 bg-cyan-950/20', bar: 'bg-cyan-700/50' },
  looking_away: { text: 'text-sky-200', ring: 'border-sky-300/40 bg-sky-950/20', bar: 'bg-sky-300/50' },
  not_sure: { text: 'text-fuchsia-300', ring: 'border-fuchsia-400/40 bg-fuchsia-950/20', bar: 'bg-fuchsia-500/50' },
};
const style = (s) => STATUS_STYLE[s] || STATUS_STYLE.away;
const STATUS_LABELS = {
  good: 'Good', slouching: 'Slouching', leaning: 'Leaning', too_close: 'Too close', away: 'Away', moved: 'Moved',
  unclear: 'Unclear', slumped: 'Slumped', looking_down: 'Looking down', checking: 'Checking',
  looking_away: 'Looking away', not_sure: 'Not sure',
};
const SUMMARY_KEYS = [
  'good', 'slouching', 'leaning', 'slumped', 'too_close', 'looking_down', 'looking_away', 'not_sure', 'moved', 'unclear',
  'away',
];
const DEBUG_KEY = 'sentinel.posture.debug';

const fmtNum = (v, digits = 2) => (v == null || Number.isNaN(v) ? '--' : Number(v).toFixed(digits));

const FEATURE_LABELS = [
  ['head_ratio', 'Head height (/ shoulder width)', 2], ['face_ratio', 'Face / shoulder width', 2],
  ['tilt_deg', 'Shoulder tilt (deg)', 1], ['lateral', 'Head offset (/ shoulder width)', 2],
  ['pitch', 'Head pitch (nose vs ears, eye spans)', 2], ['yaw', 'Head yaw (nose vs eyes, eye spans)', 2],
  ['roll_deg', 'Head roll (deg)', 1], ['eye_span', 'Eye span (px)', 1], ['shoulder_width', 'Shoulder width (px)', 0],
];

/** Classifier mode: the model's probabilities plus the inputs it saw on the last usable frame. */
const ModelPanel = ({ posture }) => {
  const { probabilities: probs, features, yaw, ood, turn_reason: turn } = posture;
  const unfamiliar = ood?.distance != null && ood.distance > ood.threshold;
  const turned = turn != null;
  return (
  <div className="border border-white/10 bg-black/40 p-2 space-y-2">
    <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px] mono">
      <dt className={turned ? 'text-sky-300' : 'text-white/50'}>Head yaw (vs your Good, limit ±0.50){turned ? ' ●' : ''}</dt>
      <dd className={`text-right ${turned ? 'text-sky-300' : 'text-white/75'}`}>
        {yaw == null ? '--' : `${yaw >= 0 ? '+' : ''}${fmtNum(yaw, 2)}`}{turned ? ` · ${turn}` : ''}
      </dd>
      <dt className={unfamiliar ? 'text-fuchsia-300' : 'text-white/50'}>
        Distance to calibration (limit {fmtNum(ood?.threshold, 2)}){unfamiliar ? ' ●' : ''}
      </dt>
      <dd className={`text-right ${unfamiliar ? 'text-fuchsia-300' : 'text-white/75'}`}>{fmtNum(ood?.distance, 2)}</dd>
    </dl>
    <ProbabilityBars probs={probs} />
    {features && (
      <dl className="grid grid-cols-2 gap-x-4 text-[10px] mono">
        {FEATURE_LABELS.map(([k, label, d]) => (
          <div key={k} className="contents">
            <dt className="text-white/50">{label}</dt>
            <dd className="text-right text-white/75">{fmtNum(features[k], d)}</dd>
          </div>
        ))}
      </dl>
    )}
    <p className="text-[9px] mono text-white/30">
      Checked before the model: a turned head is Looking away; a pose further from your calibration than
      99% of held-out calibration frames is Not sure (after 3 s). ● = over its limit.
    </p>
  </div>
  );
};

/** The raw values behind each status: now vs baseline, the change, and the limit. */
const DebugPanel = ({ rows, yaw, turn }) => (
  <div className="border border-white/10 bg-black/40 p-2 overflow-x-auto">
    <p className={`text-[10px] mono mb-1 ${turn ? 'text-sky-300' : 'text-white/60'}`}>
      Head yaw (vs baseline, limit ±0.50): {yaw == null ? '--' : `${yaw >= 0 ? '+' : ''}${fmtNum(yaw, 2)}`}
      {turn ? ` ● ${turn}: looking away, posture not judged` : ''}
    </p>
    {!rows ? (
      <p className="text-[10px] mono text-white/40">
        {turn ? 'Face the screen to see the posture measurements.' : 'No measurements yet (set a baseline and sit in view).'}
      </p>
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

function useAlertSettings() {
  const [settings, setSettings] = useState(() => ({ notify: true, sound: true, ...loadStored(ALERTS_KEY, {}) }));
  const update = (patch) => setSettings((s) => {
    const next = { ...s, ...patch };
    saveStored(ALERTS_KEY, next);
    return next;
  });
  return [settings, update];
}

/** A chime and a browser notification once per movement reminder and per long-hold warning
 * (the coach decides when; the page shows them). Returns onFrame for the frame socket. */
function useCoachAlerts(alerts) {
  const seen = useRef({ reminder: null, holds: new Set() });
  const alertsRef = useRef(alerts);
  useEffect(() => {
    alertsRef.current = alerts;
  }, [alerts]);

  return useCallback((frameData) => {
    const posture = frameData?.posture;
    if (!posture?.movement) return;
    const s = alertsRef.current;
    const ping = (title, body) => {
      if (s.sound) chime();
      if (s.notify && 'Notification' in window && Notification.permission === 'granted') {
        try {
          new Notification(title, { body, tag: 'sentinel-movement', silent: true });
        } catch {
          // some browsers only allow notifications from a service worker
        }
      }
      postJson('/api/posture/reminder?camera=laptop', {}).catch(() => {});
    };
    const r = posture.movement.reminder;
    if (r.offered && seen.current.reminder !== r.id) {
      seen.current.reminder = r.id;
      ping('Time to move', `You've been still for ${Math.round(posture.movement.still_s / 60)} min. Stand up or take a stretch break.`);
    }
    for (const w of posture.holds?.active || []) {
      if (!seen.current.holds.has(w.id)) {
        seen.current.holds.add(w.id);
        ping(w.label, w.instruction);
      }
    }
  }, []);
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

/** The main column: calibration prompts and breaks take over; otherwise the movement card. */
const CoachCard = ({ posture, connected, onCancel, onBreak, busy, mode }) => {
  if (!connected || !posture) {
    return (
      <div className="border border-white/10 bg-white/5 p-6 text-center text-[11px] mono text-white/50 uppercase tracking-widest">
        {mode !== 'posture' ? 'Switch to Desk Posture Coach mode to start coaching' : 'Waiting for the camera...'}
      </div>
    );
  }
  if (posture.recording) return <RecordingPrompt rec={posture.recording} onCancel={onCancel} />;
  if (posture.break?.routine) {
    return (
      <BreakPanel routine={posture.break.routine} note={posture.break.note} busy={busy}
        onStood={() => onBreak('stood')} onCancel={() => onBreak('cancel')} />
    );
  }
  return <MovementCard posture={posture} onBreak={onBreak} busy={busy} />;
};

/** "Show details": the moment-to-moment posture view (never flagged; for curiosity and tuning). */
const PostureDetails = ({ posture, onBaseline, busy }) => {
  const [debug, setDebug] = useState(() => Boolean(loadStored(DEBUG_KEY, false)));
  const toggleDebug = () => setDebug((d) => {
    saveStored(DEBUG_KEY, !d);
    return !d;
  });
  if (!posture) return null;
  const s = style(posture.status);
  const classifier = posture.method === 'classifier';
  return (
    <div className={`border ${s.ring} p-4 space-y-3`}>
      <div className="text-[10px] mono uppercase tracking-[0.3em] text-white/40">Posture right now (not flagged)</div>
      <div className={`text-3xl font-bold outfit leading-none ${s.text}`}>{posture.label}</div>
      {posture.calibrating && (
        <div className="text-sm text-cyan-200">
          {posture.calibration_phase === 'get_ready'
            ? `Recording starts in ${Math.ceil(posture.calibration_left_s)}...`
            : `Recording... ${posture.calibration_left_s.toFixed(1)} s left`}
        </div>
      )}
      {posture.reasons?.length > 0 && (
        <ul className="space-y-1 text-sm text-white/85 outfit">
          {posture.reasons.map((r) => <li key={r}>• {r}</li>)}
        </ul>
      )}
      {posture.status === 'no_baseline' && !posture.calibrating && (
        <div className="space-y-2">
          <p className="text-sm text-white/75 outfit">
            A baseline (or Personal calibration) turns on the long-hold warnings: sit upright, facing the
            screen with both shoulders in view. Movement tracking works without it.
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
      {classifier ? (
        <p className="text-[10px] mono text-white/40 uppercase">Using your personal model ({posture.model?.name})</p>
      ) : (
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
      )}
      {!classifier && posture.has_baseline && posture.baseline_recorded_at && (
        <p className="text-[9px] mono text-white/30 uppercase">
          Baseline from {new Date(posture.baseline_recorded_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}
        </p>
      )}
      <button type="button" onClick={toggleDebug} aria-pressed={debug} className={chipClass(debug)}>
        {debug ? 'Hide measurements' : 'Show measurements'}
      </button>
      {debug && (classifier
        ? <ModelPanel posture={posture} />
        : <DebugPanel rows={posture.debug} yaw={posture.yaw} turn={posture.turn_reason} />)}
    </div>
  );
};

const SessionSummary = ({ posture, timeline, onReset }) => {
  const s = posture?.session;
  const total = s ? Object.values(s.seconds).reduce((a, b) => a + b, 0) : 0;
  const pct = s?.good_fraction != null ? Math.round(s.good_fraction * 100) : null;
  return (
    <DashboardPanel title="Posture this session (details)" headerAction={s ? `${formatDuration(total)} TRACKED` : null}>
      {!s ? (
        <p className="text-[11px] mono text-white/40">No data yet.</p>
      ) : (
        <div className="space-y-4">
          <div className="flex items-end gap-6 flex-wrap">
            <div>
              <div className="text-2xl font-bold outfit text-white/70 leading-none">{pct != null ? `${pct}%` : '--'}</div>
              <div className="text-[10px] mono uppercase text-white/40 mt-1">upright (for reference)</div>
            </div>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] mono">
              {SUMMARY_KEYS.map((k) => (
                <div key={k} className="contents">
                  <dt className={`${style(k).text} uppercase`}>{STATUS_LABELS[k]}</dt>
                  <dd className="text-white/70 text-right">{formatDuration(s.seconds[k] || 0)}</dd>
                </div>
              ))}
            </dl>
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
  const [alerts, updateAlerts] = useAlertSettings();
  const onFrame = useCoachAlerts(alerts);
  const { status: feedStatus, frame, frameData } = useLaptopFrames(onFrame);
  const posture = frameData?.posture || null;
  const [timeline, setTimeline] = useState([]);
  const [moveTimeline, setMoveTimeline] = useState([]);
  // Calibration, the posture details and table, and the timeline: collapsed by default.
  const [advanced, setAdvanced] = useState(() => Boolean(loadStored(ADVANCED_KEY, false)));
  const toggleAdvanced = () => setAdvanced((d) => {
    saveStored(ADVANCED_KEY, !d);
    return !d;
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!running) return undefined;
    let cancelled = false;
    const load = () => fetchJson('/api/posture?camera=laptop')
      .then((d) => {
        if (cancelled) return;
        setTimeline(d.timeline || []);
        setMoveTimeline(d.movement_timeline || []);
      })
      .catch(() => {});
    load();
    const id = setInterval(load, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, [running]);

  const call = async (path, after, body = {}, send = postJson) => {
    setBusy(true);
    setError(null);
    try {
      const data = await send(path, body);
      after?.(data);
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const breakAction = (action) => call(`/api/posture/break/${action}?camera=laptop`);

  if (!loaded) return <p className="mono text-[11px] text-white/40 uppercase">Checking camera...</p>;

  return (
    <div className="space-y-4">
      {mode !== 'posture' && (
        <div className="p-3 border-l-2 border-amber-500 bg-amber-950/20 text-amber-100 text-[11px] flex items-center gap-3">
          The camera is in {mode} mode, so posture isn&apos;t being analysed.
          <button type="button" className={chipClass(true)} onClick={() => setMode('posture')}>Switch to posture</button>
        </div>
      )}
      {error && <p className="text-[11px] text-red-300">{error}</p>}
      {running && <BreakBanner brk={posture?.break} />}
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
              <CoachCard posture={posture} connected={feedStatus === 'live'} busy={busy} mode={mode}
                onCancel={() => call('/api/posture/recording/cancel?camera=laptop')}
                onBreak={breakAction} />
              <TipCard enabled={running} />
              <MovementSettings settings={posture?.settings} alerts={alerts} updateAlerts={updateAlerts} busy={busy}
                onChange={(patch) => call('/api/posture/settings?camera=laptop', null, patch, putJson)} />
            </div>
          </div>
          <button type="button" onClick={toggleAdvanced} aria-expanded={advanced}
            className={`${chipClass(advanced)} flex items-center gap-1.5`}>
            {advanced ? <ChevronUp className="w-3 h-3" aria-hidden="true" /> : <ChevronDown className="w-3 h-3" aria-hidden="true" />}
            Advanced: calibration, posture details, timeline
          </button>
          {advanced && (
            <div className="space-y-4">
              <div className="grid grid-cols-12 gap-4">
                <div className="col-span-12 lg:col-span-6">
                  <PostureDetails posture={posture} busy={busy}
                    onBaseline={() => call('/api/posture/baseline?camera=laptop')} />
                </div>
                <div className="col-span-12 lg:col-span-6">
                  <CalibrationPanel posture={posture} busy={busy}
                    call={(path, body) => call(path, null, body)}
                    remove={(path) => call(path, null, {}, deleteJson)} />
                </div>
              </div>
              <TodayPanel timeline={moveTimeline} />
              <SessionSummary posture={posture} timeline={timeline}
                onReset={() => call('/api/posture/session/reset?camera=laptop', () => setTimeline([]))} />
            </div>
          )}
          <p className="text-[10px] mono text-white/30">
            Runs entirely on this computer. A movement coach: it tracks how long you stay still, not every
            posture change. Front-facing webcam estimates; not a medical or ergonomic assessment.
          </p>
        </>
      )}
    </div>
  );
};

export default PostureCoachPage;
