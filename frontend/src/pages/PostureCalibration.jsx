import { useEffect, useState } from 'react';
import { CircleDot, FlaskConical, GraduationCap, Trash2 } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import { fetchJson } from '../lib/api';
import { chipClass } from '../lib/ui';

const POSTURES = ['good', 'slouching', 'leaning_left', 'leaning_right', 'too_close', 'looking_down'];
const POSTURE_LABELS = {
  good: 'Good', slouching: 'Slouching', leaning_left: 'Leaning left', leaning_right: 'Leaning right',
  too_close: 'Too close', looking_down: 'Looking down',
};
const MIN_FRAMES = 30;

// What to do during each recording. Move naturally: a frozen pose teaches the model a pose you
// never actually hold.
const INSTRUCTIONS = {
  good: 'Sit the way you do when you are sitting well. Keep working: type, read, shift slightly. Don\'t freeze.',
  slouching: 'Slouch the way you really do when you\'re tired or absorbed: sink down, shoulders rounded, head '
    + 'forward. Keep typing or reading while you do.',
  leaning_left: 'Lean to your left the way you really do (onto an armrest or an elbow). Keep moving naturally '
    + 'within that lean.',
  leaning_right: 'Lean to your right the way you really do (onto an armrest or an elbow). Keep moving naturally '
    + 'within that lean.',
  too_close: 'Lean in toward the screen, the way you do when reading small text. Keep reading and moving a little.',
  looking_down: 'Look down at your keyboard or phone the way you normally do: type, scroll, glance back up '
    + 'now and then.',
};

const pct = (v) => (v == null ? '--' : `${Math.round(v * 100)}%`);

/** The big prompt shown while a calibration step or the test is recording. */
export const RecordingPrompt = ({ rec, onCancel }) => {
  const ready = rec.phase === 'get_ready';
  const progress = ready ? 0 : 1 - rec.left_s / rec.record_s;
  return (
    <div className="border-2 border-cyan-400/60 bg-cyan-950/30 p-5 space-y-3" role="status" aria-live="polite">
      <div className="text-[10px] mono uppercase tracking-[0.3em] text-white/40">
        {rec.kind === 'test' ? `Test my calibration · ${rec.step} of ${rec.steps}` : 'Calibration'}
      </div>
      <div className="text-5xl md:text-6xl font-bold outfit leading-none text-cyan-200">{rec.label}</div>
      <p className="text-sm text-white/85 outfit">{INSTRUCTIONS[rec.posture]}</p>
      <div className="text-sm text-cyan-200">
        {ready ? `Get into position... recording starts in ${Math.ceil(rec.left_s)}`
          : `Recording... ${rec.left_s.toFixed(0)} s left · ${rec.frames} frames`}
      </div>
      <div className="h-1.5 w-full bg-white/10" aria-hidden="true">
        <div className="h-full bg-cyan-400 transition-[width]" style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      {rec.skipped > 10 && !ready && (
        <p className="text-[11px] text-amber-300">
          Some frames are unusable: keep your face and both shoulders in view.
        </p>
      )}
      <button type="button" onClick={onCancel} className={chipClass(false)}>Cancel</button>
    </div>
  );
};

/** Per-posture probabilities (averaged over 10 s, as the status uses them). */
export const ProbabilityBars = ({ probs }) => (
  <div className="space-y-1">
    {POSTURES.map((p) => {
      const v = probs?.[p] ?? 0;
      return (
        <div key={p} className="flex items-center gap-2 text-[10px] mono">
          <span className="w-24 text-white/60 shrink-0">{POSTURE_LABELS[p]}</span>
          <div className="flex-1 h-2 bg-white/10 relative">
            <div className={`h-full ${v >= 0.7 ? 'bg-cyan-400' : 'bg-white/40'}`} style={{ width: `${Math.round(v * 100)}%` }} />
            <div className="absolute top-0 bottom-0 left-[70%] w-px bg-white/40" aria-hidden="true" />
          </div>
          <span className="w-9 text-right text-white/70">{pct(v)}</span>
        </div>
      );
    })}
    <p className="text-[9px] mono text-white/30">
      Averaged over 10 s. The status changes only when a posture stays above the line (70%) for 5 s.
    </p>
  </div>
);

const ConfusionMatrix = ({ confusion }) => {
  const { classes, matrix } = confusion;
  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] mono">
        <caption className="sr-only">Confusion matrix: rows are what you did, columns what the model said</caption>
        <thead>
          <tr>
            <th scope="col" className="font-normal text-white/40 text-left pr-2">did ↓ / said →</th>
            {classes.map((c) => (
              <th key={c} scope="col" className="font-normal text-white/50 px-1 text-center">{POSTURE_LABELS[c]?.replace('Leaning ', 'Lean ') || c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const n = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={classes[i]}>
                <th scope="row" className="font-normal text-white/60 text-left pr-2">{POSTURE_LABELS[classes[i]]}</th>
                {row.map((v, j) => {
                  const share = n ? v / n : 0;
                  const cls = i === j ? 'bg-emerald-500' : 'bg-red-500';
                  return (
                    <td key={classes[j]} className="px-1 text-center relative">
                      <span className={`absolute inset-0.5 ${cls}`} style={{ opacity: share * 0.7 }} aria-hidden="true" />
                      <span className="relative text-white/85">{v || '·'}</span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

const Report = ({ title, note, report }) => (
  <div className="space-y-2 border border-white/10 p-3">
    <div className="flex items-baseline justify-between gap-3 flex-wrap">
      <div className="text-[11px] mono uppercase text-white/70">{title}</div>
      <div className="text-[11px] mono text-white/50">{report.frames} frames</div>
    </div>
    <div className="flex gap-6">
      <div>
        <div className="text-3xl font-bold outfit text-white leading-none">{pct(report.accuracy)}</div>
        <div className="text-[9px] mono uppercase text-white/40 mt-1">accuracy</div>
      </div>
      <div>
        <div className="text-3xl font-bold outfit text-white/80 leading-none">{pct(report.balanced_accuracy)}</div>
        <div className="text-[9px] mono uppercase text-white/40 mt-1">balanced (avg per posture)</div>
      </div>
    </div>
    <ConfusionMatrix confusion={report.confusion} />
    <p className="text-[10px] text-white/45 outfit">{note}</p>
  </div>
);

/** Guided calibration: record each posture, train, read the hold-out report, then test on new data. */
export const CalibrationPanel = ({ posture, busy, call, remove }) => {
  const counts = posture?.calibration_counts || {};
  const model = posture?.model;
  const recording = Boolean(posture?.recording);
  const ready = POSTURES.every((p) => (counts[p] || 0) >= MIN_FRAMES);
  const [details, setDetails] = useState(null);
  const reportKey = model ? `${model.trained_at}:${model.test_accuracy}` : null;

  useEffect(() => {
    if (!reportKey) return undefined;
    let cancelled = false;
    fetchJson('/api/posture/model?camera=laptop')
      .then((d) => !cancelled && setDetails(d))
      .catch(() => {});
    return () => { cancelled = true; };
  }, [reportKey]);
  const shown = model ? details : null;

  return (
    <DashboardPanel title="Personal calibration"
      headerAction={model ? 'USING YOUR MODEL' : 'USING BASELINE THRESHOLDS'}>
      <div className="space-y-4">
        <p className="text-[11px] text-white/65 outfit leading-relaxed">
          Record each posture for about 20 seconds (a 3 s countdown first). <b>Move naturally</b> during each one:
          type, read, shift slightly. Don&apos;t freeze. For Slouching, slouch the way you really do. A small model
          then learns your postures; everything stays on this computer.
        </p>
        <ul className="space-y-1.5">
          {POSTURES.map((p) => {
            const n = counts[p] || 0;
            const ok = n >= MIN_FRAMES;
            return (
              <li key={p} className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-[11px] outfit text-white/85">
                  <CircleDot className={`w-3 h-3 ${ok ? 'text-emerald-400' : 'text-white/25'}`} aria-hidden="true" />
                  {POSTURE_LABELS[p]}
                  <span className="mono text-[10px] text-white/40">{n ? `${n} frames` : 'not recorded'}</span>
                  {p === 'looking_down' && <span className="mono text-[9px] text-teal-300/80">neutral</span>}
                </span>
                <button type="button" disabled={busy || recording} className={chipClass(!ok)}
                  onClick={() => call('/api/posture/calibration/record?camera=laptop', { posture: p })}>
                  {n ? 'Redo' : 'Record'}
                </button>
              </li>
            );
          })}
        </ul>
        {posture?.recording_error && <p className="text-[11px] text-red-300">{posture.recording_error}</p>}
        <div className="flex flex-wrap gap-2">
          <button type="button" disabled={busy || recording || !ready}
            onClick={() => call('/api/posture/calibration/train?camera=laptop')}
            className="px-3 py-2 text-[11px] mono uppercase font-bold tracking-widest border border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 disabled:opacity-40 flex items-center gap-2">
            <GraduationCap className="w-4 h-4" aria-hidden="true" /> {model ? 'Retrain' : 'Train my model'}
          </button>
          <button type="button" disabled={busy || recording || !model}
            onClick={() => call('/api/posture/test?camera=laptop')}
            className={`${chipClass(Boolean(model))} flex items-center gap-1.5`}>
            <FlaskConical className="w-3 h-3" aria-hidden="true" /> Test my calibration (~2 min)
          </button>
          {model && (
            <button type="button" disabled={busy || recording}
              onClick={() => remove('/api/posture/model?camera=laptop')}
              className={`${chipClass(false)} flex items-center gap-1.5`}>
              <Trash2 className="w-3 h-3" aria-hidden="true" /> Use thresholds instead
            </button>
          )}
        </div>
        {!ready && <p className="text-[10px] mono text-white/35">Record all six postures to train.</p>}
        {shown?.report && (
          <div className="space-y-3">
            {shown.test_report ? (
              <Report title="Test my calibration (realistic)" report={shown.test_report}
                note={`Scored on new recordings made after training, in random order (${new Date(shown.test_report.tested_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}). This is the number to trust.${
                  shown.test_report.unfamiliar_fraction != null
                    ? ` ${pct(shown.test_report.unfamiliar_fraction)} of these frames would have shown Not sure.` : ''}`} />
            ) : (
              <p className="text-[11px] text-amber-200/90 outfit">
                Run &quot;Test my calibration&quot; for the realistic accuracy: the number below is measured on the
                same sessions the model learned from, so it is optimistic.
              </p>
            )}
            <Report title={`Calibration hold-out (${shown.report.model})`} report={shown.report}
              note={`Trained on the first 75% of each recording, scored on the last 25% (held out by time). Compared: ${
                Object.entries(shown.report.compared).map(([n, r]) => `${n} ${pct(r.balanced_accuracy)}`).join(', ')
              } balanced accuracy; the better one was kept and refitted on everything.${
                shown.report.ood ? ` Not sure beyond distance ${shown.report.ood.threshold.toFixed(2)} (99th percentile of held-out frames).` : ''}`} />
          </div>
        )}
      </div>
    </DashboardPanel>
  );
};
