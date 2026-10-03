import { useEffect, useState } from 'react';
import { CheckCircle2, Lightbulb, MoveUp, X } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import { fetchJson, postJson } from '../lib/api';
import { chipClass } from '../lib/ui';

/** While a poor posture is shown: what to do, and whether the ghost is on the video. */
export const FixGuidance = ({ guidance }) => {
  if (!guidance?.instruction) return null;
  return (
    <div className="p-3 border border-white/20 bg-white/5 space-y-1">
      <div className="flex items-center gap-2 text-lg font-semibold outfit text-white">
        <MoveUp className="w-5 h-5 text-cyan-300" aria-hidden="true" /> {guidance.instruction}
      </div>
      <p className="text-[11px] text-white/55 outfit">
        {guidance.ghost_available
          ? 'Line yourself up with the faint white outline of your good posture on the video.'
          : 'Record Good (Personal calibration) or reset your baseline once to see an outline of your good posture on the video.'}
      </p>
    </div>
  );
};

/** "Back to good" right after the quick check sees you corrected. */
export const BackToGood = ({ correction }) => {
  if (!correction) return null;
  return (
    <div className="p-3 border-2 border-emerald-400/70 bg-emerald-950/40 flex items-center gap-2 text-emerald-200" role="status">
      <CheckCircle2 className="w-5 h-5" aria-hidden="true" />
      <span className="text-lg font-semibold outfit">Nice, back to good</span>
      <span className="text-[11px] mono text-emerald-200/70">corrected in {correction.seconds.toFixed(0)} s</span>
    </div>
  );
};

/** The guided break, step by step, with live rep counts. */
export const BreakPanel = ({ routine, note, onStood, onCancel, busy }) => (
  <div className="border-2 border-teal-400/60 bg-teal-950/30 p-5 space-y-3" role="status" aria-live="polite">
    <div className="text-[10px] mono uppercase tracking-[0.3em] text-white/40">
      Stretch break{routine.phase === 'step' ? ` · ${routine.step} of ${routine.steps}` : ''}
    </div>
    <div className="text-5xl font-bold outfit leading-none text-teal-200">{routine.label}</div>
    <p className="text-sm text-white/85 outfit">{routine.instruction}</p>
    {routine.phase === 'get_ready' ? (
      <div className="text-sm text-teal-200">Starting in {Math.ceil(routine.left_s)}...</div>
    ) : (
      <div className="space-y-1">
        {Object.entries(routine.counts).map(([k, [n, target]]) => (
          <div key={k} className="flex items-center gap-2 text-[11px] mono">
            <span className="w-20 text-white/60">
              {{ side_a: 'One side', side_b: 'Other side', reps: 'Reps', stood: 'Standing' }[k] || k}
            </span>
            <div className="flex gap-1">
              {Array.from({ length: target }, (_, i) => (
                <span key={i} className={`w-4 h-4 border ${i < n ? 'bg-teal-400 border-teal-300' : 'border-white/25'}`} />
              ))}
            </div>
            <span className="text-white/60">{Math.min(n, target)}/{target}</span>
          </div>
        ))}
        {routine.waiting_for_reference && (
          <p className="text-[11px] text-amber-300">Sit facing the camera so it can see your face and shoulders.</p>
        )}
        <p className="text-[10px] mono text-white/40">{Math.ceil(routine.left_s)} s left for this step</p>
      </div>
    )}
    <p className="text-[10px] text-white/50 outfit">{note}</p>
    <div className="flex gap-2">
      {routine.key === 'stand_up' && (
        <button type="button" onClick={onStood} disabled={busy} className={chipClass(true)}>I stood up</button>
      )}
      <button type="button" onClick={onCancel} disabled={busy} className={chipClass(false)}>End break</button>
    </div>
  </div>
);

/** The result after a break (the offer comes with the movement reminder). */
export const BreakBanner = ({ brk }) => {
  if (!brk?.result) return null;
  const done = brk.result.outcome === 'completed';
  return (
    <div className="p-3 border-l-2 border-teal-400 bg-teal-950/30 text-teal-100 text-sm outfit flex items-center gap-2">
      <CheckCircle2 className="w-4 h-4" aria-hidden="true" />
      {done ? 'Break done. Nice work: all three movements confirmed.'
        : brk.result.outcome === 'cancelled' ? 'Break ended early.' : 'Break finished (some steps not confirmed).'}
    </div>
  );
};

/** One setup tip at a time, from the last 7 days, with the evidence behind it. */
export const TipCard = ({ enabled }) => {
  const [tip, setTip] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    const load = () => fetchJson('/api/posture/tips?camera=laptop')
      .then((d) => !cancelled && setTip(d.tip))
      .catch(() => {});
    load();
    const id = setInterval(load, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled]);

  if (!tip) return null;
  const dismiss = async () => {
    setBusy(true);
    try {
      const d = await postJson(`/api/posture/tips/${tip.rule}/dismiss?camera=laptop`, {});
      setTip(d.tip);
    } catch {
      // keep showing it; try again later
    } finally {
      setBusy(false);
    }
  };
  return (
    <DashboardPanel title="Setup tip" headerAction="LAST 7 DAYS">
      <div className="space-y-2">
        <div className="flex items-start gap-2">
          <Lightbulb className="w-5 h-5 text-amber-300 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="space-y-1">
            <div className="text-base font-semibold outfit text-white">{tip.title}</div>
            <p className="text-sm text-white/80 outfit">{tip.advice}</p>
            <p className="text-[11px] mono text-white/45">Why: {tip.evidence}</p>
          </div>
        </div>
        <button type="button" onClick={dismiss} disabled={busy} className={`${chipClass(false)} flex items-center gap-1.5`}>
          <X className="w-3 h-3" aria-hidden="true" /> Hide for a week
        </button>
      </div>
    </DashboardPanel>
  );
};
