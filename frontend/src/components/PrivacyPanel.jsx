import { useEffect, useRef, useState } from 'react';
import DashboardPanel from './DashboardPanel';
import { describeError, postJson, putJson, useFetch } from '../lib/api';
import { chipClass, inputClass } from '../lib/ui';

const RETENTION_OPTIONS = [7, 14, 30, 90, 365, 0];

const formatBytes = (n) => {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
};

const daysLabel = (d) => (d === 0 ? 'Keep forever' : `${d} days`);

const lastRunText = (run) => {
  if (!run) return 'Not run yet (it runs at startup, then every hour).';
  const when = new Date(run.at * 1000).toLocaleString();
  if (run.kept_forever) return `${when}: keeping everything (retention off).`;
  if (run.dry_run) return `${when}: dry run, would have deleted ${run.count} item(s). Nothing was deleted.`;
  return `${when}: deleted ${run.count} item(s)${run.manual ? ' (Delete now)' : ''}.`;
};

/** Retention: what's stored, what would be deleted, and the setting. Opening this panel ends the
 * dry-run period (retention starts deleting for real). */
const PrivacyPanel = () => {
  const [reload, setReload] = useState(0);
  const { data, error } = useFetch('/api/privacy', reload);
  const [wasDryRun, setWasDryRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const marked = useRef(false);

  // The first response tells whether retention was still dry-running; then mark the panel seen.
  useEffect(() => {
    if (!data || marked.current) return;
    marked.current = true;
    setWasDryRun(data.dry_run);
    if (data.dry_run) {
      postJson('/api/privacy/seen', {})
        .then(() => setReload((n) => n + 1))
        .catch((err) => setSaveError(describeError(err)));
    }
  }, [data]);

  const save = async (body) => {
    setBusy(true);
    setSaveError(null);
    setMessage(null);
    setConfirming(false);
    try {
      await putJson('/api/privacy', body);
      setReload((n) => n + 1);
    } catch (err) {
      setSaveError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const deleteNow = async () => {
    setBusy(true);
    setSaveError(null);
    try {
      const r = await postJson('/api/privacy/delete-now', { retention_days: data.retention_days });
      setMessage(`Deleted ${r.count} item(s), ${formatBytes(r.bytes)}.`);
      setConfirming(false);
      setReload((n) => n + 1);
    } catch (err) {
      setSaveError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const would = data?.would_delete;
  const wouldBy = Object.fromEntries((would?.items || []).map((i) => [i.category, i]));
  return (
    <DashboardPanel title="Privacy" headerAction={data ? daysLabel(data.retention_days).toUpperCase() : null}>
      {!data ? (
        <p className="text-[11px] text-white/50 outfit">{error ? describeError(error) : 'Loading...'}</p>
      ) : (
        <div className="space-y-3">
          {wasDryRun && (
            <p className="text-[11px] outfit text-amber-300">
              Until now retention only did dry runs (nothing was deleted). From now on it deletes history older than the
              setting below, at startup and every hour.
            </p>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <label htmlFor="retention-days" className="text-[12px] font-bold text-white outfit">
              Keep history for
            </label>
            <select id="retention-days" className={inputClass} disabled={busy} value={data.retention_days}
              onChange={(e) => save({ retention_days: Number(e.target.value) })}>
              {RETENTION_OPTIONS.map((d) => <option key={d} value={d}>{daysLabel(d)}</option>)}
            </select>
          </div>
          <p className="text-[10px] text-white/50 outfit">
            Covers events with their clips and snapshots, camera recordings, time-at-risk totals, posture history,
            the movement-coach log and database backups. Settings (zones, rules, calibration) are kept.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <caption className="sr-only">What is stored, and what the retention setting would delete now</caption>
              <thead>
                <tr className="text-[9px] uppercase text-white/40 tracking-widest small-caps border-b border-white/10">
                  <th scope="col" className="py-1.5 pr-3 font-bold">Stored</th>
                  <th scope="col" className="py-1.5 pr-3 font-bold text-right">Items</th>
                  <th scope="col" className="py-1.5 pr-3 font-bold text-right">Size</th>
                  <th scope="col" className="py-1.5 font-bold text-right">Older than {daysLabel(data.retention_days)}</th>
                </tr>
              </thead>
              <tbody>
                {data.stored.items.map((i) => (
                  <tr key={i.category} className="text-[10px] mono border-b border-white/5">
                    <th scope="row" className="py-1.5 pr-3 font-normal text-white/70 outfit">{i.label}</th>
                    <td className="py-1.5 pr-3 text-right tabular-nums text-white/70">{i.count}</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums text-white/50">{i.bytes ? formatBytes(i.bytes) : '--'}</td>
                    <td className={`py-1.5 text-right tabular-nums ${wouldBy[i.category]?.count ? 'text-amber-300' : 'text-white/30'}`}>
                      {would ? wouldBy[i.category]?.count ?? 0 : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] mono text-white/40">Last automatic run: {lastRunText(data.last_run)}</p>
          <div className="flex items-start justify-between gap-4 border-t border-white/10 pt-3">
            <div>
              <div className="text-[12px] font-bold text-white outfit">Skeleton-only mode</div>
              <p className="text-[10px] text-white/50 outfit max-w-md">
                Never store video or images: no incident clips, no snapshots, no camera recordings. Each event keeps
                the keypoints from 10 s before to 5 s after instead, played back as a stick figure, and notifications
                are sent as text only. The live view still shows the camera.
              </p>
            </div>
            <button type="button" role="switch" aria-checked={data.skeleton_only} disabled={busy}
              onClick={() => save({ skeleton_only: !data.skeleton_only })}
              className={`relative flex-none w-11 h-6 border transition-colors disabled:opacity-40 ${data.skeleton_only ? 'bg-cyan-400 border-cyan-200' : 'bg-black/60 border-white/20'}`}>
              <span className="sr-only">Skeleton-only mode</span>
              <span className={`absolute top-0.5 bg-white transition-all ${data.skeleton_only ? 'left-[22px]' : 'left-0.5'}`}
                style={{ width: 18, height: 18 }} aria-hidden="true" />
            </button>
          </div>
          {would && would.count > 0 && !confirming && (
            <button type="button" disabled={busy} onClick={() => setConfirming(true)} className={chipClass(false)}>
              Delete now...
            </button>
          )}
          {confirming && (
            <div className="border border-red-500/40 bg-red-950/30 p-2 space-y-2">
              <p className="text-[11px] outfit text-red-200">
                Permanently delete {would.count} item(s) ({formatBytes(would.bytes)}) older than{' '}
                {daysLabel(data.retention_days)}? This can&apos;t be undone.
              </p>
              <div className="flex gap-2">
                <button type="button" disabled={busy} onClick={deleteNow}
                  className="text-[10px] mono uppercase font-bold px-2 py-1 bg-red-500 text-black disabled:opacity-40">
                  Delete {would.count} item(s)
                </button>
                <button type="button" disabled={busy} onClick={() => setConfirming(false)} className={chipClass(false)}>
                  Cancel
                </button>
              </div>
            </div>
          )}
          {message && <p className="text-[11px] text-emerald-300 outfit">{message}</p>}
          {saveError && <p className="text-[11px] text-red-300">{saveError}</p>}
        </div>
      )}
    </DashboardPanel>
  );
};

export default PrivacyPanel;
