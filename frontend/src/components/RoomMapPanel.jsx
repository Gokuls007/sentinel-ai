import { useState } from 'react';
import { Map as MapIcon } from 'lucide-react';
import { describeError, postJson, putJson, usePoll } from '../lib/api';

const buttonBase =
  'px-3 py-1.5 text-[10px] mono uppercase font-bold tracking-widest border cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';

const Progress = ({ label, value, max }) => {
  const pct = Math.max(0, Math.min(100, Math.round((100 * (value || 0)) / (max || 1))));
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-[11px]">
        <span className="outfit text-white/50">{label}</span>
        <span className="mono text-white/60">{pct}%</span>
      </div>
      <div className="h-1.5 bg-white/10" role="progressbar" aria-label={label} aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={`h-full ${pct >= 100 ? 'bg-emerald-400' : 'bg-cyan-400'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

const Item = ({ label, value, ok }) => (
  <div className="flex items-baseline justify-between gap-3 text-[11px]">
    <span className="outfit text-white/50">{label}</span>
    <span className={`mono ${ok ? 'text-emerald-300' : 'text-white/60'}`}>{value}</span>
  </div>
);

/**
 * What the camera has learned about the room from the people it watches (backend
 * scene/room_model.py): the floor, furniture that hides people's legs (a couch back, a counter),
 * and standing height by position. Hidden legs are not judged for falls.
 */
const RoomMapPanel = ({ camera, enabled = true }) => {
  const [reload, setReload] = useState(0);
  const { data } = usePoll(`/api/room?camera=${camera}&r=${reload}`, 3000, enabled);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const act = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      setReload((n) => n + 1);
    } catch (err) {
      setError(err.status === 403 ? 'Only works from this computer.' : err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const show = Boolean(data?.show);
  return (
    <div className="border border-white/10 bg-black/30 p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-[11px] mono uppercase tracking-widest text-cyan-300">
          <MapIcon size={14} aria-hidden="true" /> Room map
        </h3>
        <div className="flex gap-2">
          <button
            type="button"
            className={`${buttonBase} ${show ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10' : 'border-white/20 text-white/70'}`}
            disabled={busy || !data}
            aria-pressed={show}
            onClick={() => act(() => putJson(`/api/room?camera=${camera}`, { show: !show }))}
          >
            {show ? 'Hide on video' : 'Show on video'}
          </button>
          <button
            type="button"
            className={`${buttonBase} border-white/20 text-white/70 hover:border-red-400/60`}
            disabled={busy || !data}
            onClick={() => act(() => postJson(`/api/room/reset?camera=${camera}`, {}))}
          >
            Relearn
          </button>
        </div>
      </div>
      <p className="text-[11px] outfit text-white/50">
        Learned from people moving around: green is floor, red is furniture that hides legs (people behind it are not
        judged by their legs). Relearn after moving the camera or furniture.
      </p>
      {data && !data.height_ready && (
        <div className="border border-amber-400/30 bg-amber-500/5 p-2 space-y-2">
          <p className="text-[11px] outfit text-amber-100/90">
            <span className="font-bold">Teach the room (about 30 s):</span> walk slowly around the room, standing upright with
            your head and feet in view, once close to the camera and once far away. Until this is done, &quot;on the
            floor&quot; only works when the person&apos;s legs can be seen.
          </p>
          <Progress label="Standing seen head to feet" value={data.height_samples} max={data.height_samples_needed} />
          <Progress label="Near and far" value={data.depth_spread} max={data.depth_spread_needed} />
        </div>
      )}
      {data && (
        <div className="space-y-1">
          <Item label="Floor areas" value={data.floor_cells} ok={data.floor_cells > 0} />
          <Item label="Furniture hiding legs" value={data.furniture_cells} ok={data.furniture_cells > 0} />
          <Item
            label="Standing height by position"
            value={data.height_ready ? `ready (${Math.round(data.height_error * 100)}% error)` : 'learning'}
            ok={data.height_ready}
          />
        </div>
      )}
      {error && <p role="alert" className="text-[11px] outfit text-red-300">{error}</p>}
    </div>
  );
};

export default RoomMapPanel;
