import { useState } from 'react';
import { Activity, Bell, BellOff, Coffee, Footprints, Volume2, VolumeX } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import { chipClass } from '../lib/ui';
import { BackToGood, FixGuidance } from './PostureCoaching';

const REMINDER_MINUTES = [15, 20, 30, 45, 60];
const HOLD_MINUTES = [10, 20, 30];

// Static class names so Tailwind generates them.
const STATE_STYLE = {
  moving: { text: 'text-emerald-300', ring: 'border-emerald-400/50 bg-emerald-950/20', bar: 'bg-emerald-400' },
  still: { text: 'text-sky-200', ring: 'border-sky-400/40 bg-sky-950/20', bar: 'bg-sky-500/60' },
  long_still: { text: 'text-amber-300', ring: 'border-amber-400/60 bg-amber-950/30', bar: 'bg-amber-400' },
  away: { text: 'text-white/50', ring: 'border-white/15 bg-white/5', bar: 'bg-white/15' },
  on_break: { text: 'text-teal-300', ring: 'border-teal-400/50 bg-teal-950/30', bar: 'bg-teal-400' },
};
const STATE_LABELS = { moving: 'Moving', still: 'Still', long_still: 'Still 10+ min', away: 'Away', on_break: 'Break' };
const stateStyle = (s) => STATE_STYLE[s] || STATE_STYLE.away;

/** "23 min" / "1 h 05 min" / "40 s" */
function span(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
}

const headline = (mv) => {
  if (mv.state === 'away') return { big: 'Away', small: 'Welcome back when you sit down again.' };
  if (mv.state === 'on_break') return { big: 'On a break', small: 'Nice. The timer starts fresh afterwards.' };
  if (mv.state === 'moving') return { big: 'Just moved', small: mv.last_change?.why ? `You ${mv.last_change.why}.` : '' };
  return { big: `Still for ${span(mv.still_s)}`, small: 'Time since you last moved' };
};

/** The main card: time since you last moved, the movement reminder, and long-hold warnings. */
export const MovementCard = ({ posture, onBreak, busy }) => {
  const mv = posture.movement;
  const holds = posture.holds;
  const st = stateStyle(mv.state);
  const h = headline(mv);
  const reminder = mv.reminder;
  return (
    <div className={`border-2 ${st.ring} p-5 space-y-4`} role="status" aria-live="polite">
      <div className="flex items-center justify-between">
        <div className="text-[10px] mono uppercase tracking-[0.3em] text-white/40">Movement</div>
        {mv.demo_timings && (
          <span className="text-[9px] mono uppercase font-bold text-black bg-amber-300 px-1.5 py-0.5">Demo timings</span>
        )}
      </div>
      <div>
        <div className={`text-5xl md:text-6xl font-bold outfit leading-none ${st.text}`}>{h.big}</div>
        {h.small && <div className="text-sm text-white/60 outfit mt-2">{h.small}</div>}
      </div>
      {reminder.offered && (
        <div className="p-3 border-2 border-teal-400/70 bg-teal-950/40 space-y-2" role="alert">
          <div className="text-lg font-semibold outfit text-teal-100 flex items-center gap-2">
            <Footprints className="w-5 h-5" aria-hidden="true" />
            {span(mv.still_s)} without moving: time to get up?
          </div>
          <p className="text-sm text-teal-100/80 outfit">
            Stand up for a minute, or take the guided stretch break (neck tilts, shoulder shrugs, a stand-up).
          </p>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy} className={chipClass(true)} onClick={() => onBreak('start')}>
              Start stretch break
            </button>
            <button type="button" disabled={busy} className={chipClass(false)} onClick={() => onBreak('snooze')}>
              Snooze {Math.round(reminder.snooze_s / 60)} min
            </button>
            <button type="button" disabled={busy} className={chipClass(false)} onClick={() => onBreak('skip')}>Skip</button>
          </div>
        </div>
      )}
      <BackToGood correction={posture.back_to_good} />
      {holds.active.map((w) => (
        <div key={w.id} className="p-3 border-2 border-amber-400/70 bg-amber-950/30 space-y-2" role="alert">
          <div className="text-base font-semibold outfit text-amber-100">
            {w.label} ({span(w.held_s)})
          </div>
          {posture.guidance?.active && <FixGuidance guidance={posture.guidance} />}
        </div>
      ))}
      <dl className="grid grid-cols-3 gap-3 text-center">
        <div className="border border-white/10 p-2">
          <dt className="text-[9px] mono uppercase text-white/40">Breaks today</dt>
          <dd className="text-2xl font-bold outfit text-white">{mv.breaks_today}</dd>
        </div>
        <div className="border border-white/10 p-2">
          <dt className="text-[9px] mono uppercase text-white/40">Static time today</dt>
          <dd className="text-2xl font-bold outfit text-white">{span(mv.static_today_s)}</dd>
        </div>
        <div className="border border-white/10 p-2">
          <dt className="text-[9px] mono uppercase text-white/40">Longest still</dt>
          <dd className="text-2xl font-bold outfit text-white">{span(mv.longest_still_s)}</dd>
        </div>
      </dl>
      <p className="text-[10px] text-white/40 outfit">
        Shifting around is normal; only staying still too long is flagged. Static time counts still stretches over{' '}
        {Math.round(mv.static_min_s / 60)} min. Reminder after {span(reminder.after_s)} still
        {reminder.snoozed ? ' (snoozed)' : ''}{reminder.skipped ? ' (skipped until you next move)' : ''}.
      </p>
      {!holds.available && <p className="text-[10px] text-white/40 outfit">{holds.reason}.</p>}
      <button type="button" disabled={busy || mv.state === 'on_break'} onClick={() => onBreak('start')}
        className={`${chipClass(false)} flex items-center gap-1.5`}>
        <Coffee className="w-3 h-3" aria-hidden="true" /> Stretch break now
      </button>
    </div>
  );
};

/** Today: the movement timeline (10 s per block). */
export const TodayPanel = ({ timeline }) => (
  <DashboardPanel title="Today" headerAction="10 S PER BLOCK">
    {timeline.length ? (
      <div className="space-y-2">
        <div className="flex h-6 w-full overflow-hidden border border-white/10" aria-label="Movement today">
          {timeline.map((b) => (
            <div key={b.t} className={`flex-1 min-w-[1px] ${stateStyle(b.state).bar}`}
              title={`${new Date(b.t * 1000).toLocaleTimeString([], { timeStyle: 'short' })}: ${STATE_LABELS[b.state] || b.state}`} />
          ))}
        </div>
        <div className="flex flex-wrap gap-3">
          {Object.keys(STATE_LABELS).map((k) => (
            <span key={k} className="flex items-center gap-1 text-[9px] mono uppercase text-white/50">
              <span className={`w-2 h-2 ${stateStyle(k).bar}`} aria-hidden="true" /> {STATE_LABELS[k]}
            </span>
          ))}
        </div>
      </div>
    ) : (
      <p className="text-[10px] mono text-white/30">The timeline fills in every 10 seconds.</p>
    )}
  </DashboardPanel>
);

const Chips = ({ values, value, field, label, disabled, onChange }) => (
  <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={label}>
    {values.map((m) => (
      <button key={m} type="button" role="radio" aria-checked={value === m} disabled={disabled}
        onClick={() => onChange({ [field]: m })} className={chipClass(value === m)}>{m} min</button>
    ))}
  </div>
);

/** Movement reminder, long-hold warnings, and how alerts reach you. */
export const MovementSettings = ({ settings, alerts, updateAlerts, onChange, busy }) => {
  const [permission, setPermission] = useState(() => ('Notification' in window ? Notification.permission : 'unsupported'));
  const toggleNotify = async () => {
    const on = !alerts.notify;
    if (on && permission === 'default') {
      try {
        setPermission(await Notification.requestPermission());
      } catch {
        setPermission('denied');
      }
    }
    updateAlerts({ notify: on });
  };
  if (!settings) return null;
  const demo = settings.demo_timings;
  return (
    <DashboardPanel title="Reminders">
      <div className="space-y-3">
        {demo && (
          <p className="text-[11px] text-amber-300 outfit">
            Demo timings are on (Settings): reminder after 1 min, long-hold warnings after 2 min.
          </p>
        )}
        <p className="text-[11px] text-white/60 outfit">Remind me to move after being still for:</p>
        <Chips values={REMINDER_MINUTES} value={settings.reminder_min} field="reminder_min" label="Movement reminder"
          disabled={busy || demo} onChange={onChange} />
        <p className="text-[11px] text-white/60 outfit">
          Warn only about extremes held a long time (short-term posture is never flagged):
        </p>
        {[['head_down', 'Head far down'], ['lean', 'Strong lean to one side']].map(([k, label]) => (
          <div key={k} className="space-y-1.5">
            <button type="button" aria-pressed={settings[`${k}_enabled`]} disabled={busy}
              onClick={() => onChange({ [`${k}_enabled`]: !settings[`${k}_enabled`] })}
              className={`${chipClass(settings[`${k}_enabled`])} flex items-center gap-1.5`}>
              <Activity className="w-3 h-3" aria-hidden="true" /> {label}: {settings[`${k}_enabled`] ? 'on' : 'off'}
            </button>
            {settings[`${k}_enabled`] && (
              <Chips values={HOLD_MINUTES} value={settings[`${k}_min`]} field={`${k}_min`} label={`${label} after`}
                disabled={busy || demo} onChange={onChange} />
            )}
          </div>
        ))}
        <div className="flex flex-wrap gap-2 pt-1">
          <button type="button" onClick={toggleNotify} aria-pressed={alerts.notify}
            className={`${chipClass(alerts.notify)} flex items-center gap-1.5`}>
            {alerts.notify ? <Bell className="w-3 h-3" aria-hidden="true" /> : <BellOff className="w-3 h-3" aria-hidden="true" />}
            Notification {alerts.notify ? 'on' : 'off'}
          </button>
          <button type="button" onClick={() => updateAlerts({ sound: !alerts.sound })} aria-pressed={alerts.sound}
            className={`${chipClass(alerts.sound)} flex items-center gap-1.5`}>
            {alerts.sound ? <Volume2 className="w-3 h-3" aria-hidden="true" /> : <VolumeX className="w-3 h-3" aria-hidden="true" />}
            Sound {alerts.sound ? 'on' : 'off'}
          </button>
        </div>
        {alerts.notify && permission === 'denied' && (
          <p className="text-[10px] text-amber-300">
            Browser notifications are blocked for this site, so reminders show here on the page instead.
          </p>
        )}
      </div>
    </DashboardPanel>
  );
};
