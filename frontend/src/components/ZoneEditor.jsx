import { useEffect, useId, useRef, useState } from 'react';
import { Plus, Trash2, Undo2, X, Save } from 'lucide-react';
import DashboardPanel from './DashboardPanel';
import { describeError, fetchJson, putJson } from '../lib/api';
import { inputClass, labelClass } from '../lib/ui';

const TYPES = {
  restricted: { label: 'Restricted: alert as soon as someone enters', color: '#ef4444' },
  time_limited: { label: 'Time limit: alert if someone stays too long', color: '#f97316' },
  one_way: { label: 'One-way: alert on walking the wrong way', color: '#d946ef' },
};
const DIRECTIONS = ['up', 'down', 'left', 'right'];

const button =
  'px-3 py-1.5 text-[10px] mono uppercase font-bold tracking-widest border cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5';

const slug = (name) =>
  (name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 24) || 'zone') +
  '_' + Math.random().toString(36).slice(2, 6);

/** A zone from /api/zones (overlay shape) -> the editor/PUT shape. */
const fromOverlay = (z) => ({
  id: z.id,
  name: z.name,
  zone_type: z.type,
  polygon: z.polygon_normalized,
  time_limit: z.time_limit || 0,
  direction: z.direction || null,
});

const Polygon = ({ points, color, dashed = false, label }) => {
  if (!points.length) return null;
  const d = points.map(([x, y]) => `${x * 100},${y * 100}`).join(' ');
  const [lx, ly] = points[0];
  return (
    <g>
      {points.length >= 3 ? (
        <polygon points={d} fill={color} fillOpacity="0.22" stroke={color} strokeWidth="0.4"
          strokeDasharray={dashed ? '1.2 0.8' : undefined} vectorEffect="non-scaling-stroke" />
      ) : (
        <polyline points={d} fill="none" stroke={color} strokeWidth="0.4" vectorEffect="non-scaling-stroke" />
      )}
      {dashed && points.map(([x, y], i) => <circle key={i} cx={x * 100} cy={y * 100} r="0.8" fill={color} />)}
      {label && (
        <text x={lx * 100 + 0.8} y={ly * 100 - 1} fill={color} fontSize="2.6" fontFamily="monospace" fontWeight="bold">
          {label}
        </text>
      )}
    </g>
  );
};

/**
 * Draw the laptop camera's danger zones on a frozen frame and save them.
 * Zones are normalised (0-1) image coordinates, so they fit any resolution.
 */
const ZoneEditor = ({ camera, frame }) => {
  const id = useId();
  const imgRef = useRef(null);
  const [zones, setZones] = useState(null); // saved state from the backend (editor shape)
  const [draft, setDraft] = useState(null); // {points, name, zone_type, time_limit, direction}
  const [still, setStill] = useState(null); // the frame being drawn on
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchJson(`/api/zones?camera=${camera}`)
      .then((data) => !cancelled && setZones((data || []).map(fromOverlay)))
      .catch((err) => !cancelled && setError(describeError(err)));
    return () => { cancelled = true; };
  }, [camera]);

  const startDraft = () => {
    setStill(frame);
    setSaved(false);
    setError(null);
    setDraft({ points: [], name: `Zone ${(zones?.length || 0) + 1}`, zone_type: 'restricted', time_limit: 10, direction: 'up' });
  };

  const addPoint = (e) => {
    if (!draft || !imgRef.current) return;
    const r = imgRef.current.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    setDraft((d) => ({ ...d, points: [...d.points, [Number(x.toFixed(4)), Number(y.toFixed(4))]] }));
  };

  const finishDraft = () => {
    const zone = {
      id: slug(draft.name),
      name: draft.name.trim() || 'Zone',
      zone_type: draft.zone_type,
      polygon: draft.points,
      time_limit: draft.zone_type === 'time_limited' ? Number(draft.time_limit) : 0,
      direction: draft.zone_type === 'one_way' ? draft.direction : null,
    };
    setZones((z) => [...(z || []), zone]);
    setDraft(null);
    setDirty(true);
  };

  const remove = (zoneId) => {
    setZones((z) => z.filter((x) => x.id !== zoneId));
    setDirty(true);
    setSaved(false);
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const overlay = await putJson(`/api/zones?camera=${camera}`, { zones });
      setZones((overlay || []).map(fromOverlay));
      setDirty(false);
      setSaved(true);
      setStill(null);
    } catch (err) {
      setError(err.status === 403 ? 'Zones can only be edited from this computer.' : err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const drawing = draft != null;
  const image = drawing ? still : null;
  const canFinish = drawing && draft.points.length >= 3 && (draft.zone_type !== 'time_limited' || Number(draft.time_limit) > 0);

  return (
    <DashboardPanel title="Danger zones" headerAction={zones ? `${zones.length} zone${zones.length === 1 ? '' : 's'}` : null}>
      <div className="space-y-3">
        {!drawing && (
          <p className="text-[11px] outfit text-white/70">
            Draw areas on your camera view. <strong className="text-white">Restricted</strong> zones alert as soon as a
            person enters; <strong className="text-white">time-limit</strong> zones alert if someone stays longer than
            the limit. Zones are checked on every frame once saved.
          </p>
        )}

        {drawing && (
          <div className="space-y-2">
            <p className="text-[11px] mono text-cyan-300">
              Click on the image to place corners ({draft.points.length} so far, at least 3). The frame is paused
              while you draw.
            </p>
            <div className="relative inline-block w-full select-none">
              {image ? (
                <img
                  ref={imgRef}
                  src={`data:image/jpeg;base64,${image}`}
                  alt="Paused camera frame: click to place zone corners"
                  className="w-full h-auto block cursor-crosshair border border-cyan-500/30"
                  onClick={addPoint}
                  draggable={false}
                />
              ) : (
                <div className="p-6 text-[11px] mono text-amber-300 border border-amber-500/30">
                  No frame yet: wait until the camera image appears, then press Add zone again.
                </div>
              )}
              {image && (
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden="true">
                  {(zones || []).map((z) => (
                    <Polygon key={z.id} points={z.polygon} color={TYPES[z.zone_type]?.color || '#22d3ee'} label={z.name} />
                  ))}
                  <Polygon points={draft.points} color={TYPES[draft.zone_type].color} dashed label={draft.name} />
                </svg>
              )}
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label htmlFor={`${id}-name`} className={labelClass}>Name</label>
                <input id={`${id}-name`} className={inputClass} value={draft.name} maxLength={60}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
              </div>
              <div>
                <label htmlFor={`${id}-type`} className={labelClass}>Type</label>
                <select id={`${id}-type`} className={inputClass} value={draft.zone_type}
                  onChange={(e) => setDraft({ ...draft, zone_type: e.target.value })}>
                  {Object.entries(TYPES).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                </select>
              </div>
              {draft.zone_type === 'time_limited' && (
                <div>
                  <label htmlFor={`${id}-limit`} className={labelClass}>Limit (seconds)</label>
                  <input id={`${id}-limit`} type="number" min="1" className={`${inputClass} w-24`} value={draft.time_limit}
                    onChange={(e) => setDraft({ ...draft, time_limit: e.target.value })} />
                </div>
              )}
              {draft.zone_type === 'one_way' && (
                <div>
                  <label htmlFor={`${id}-dir`} className={labelClass}>Allowed direction</label>
                  <select id={`${id}-dir`} className={inputClass} value={draft.direction}
                    onChange={(e) => setDraft({ ...draft, direction: e.target.value })}>
                    {DIRECTIONS.map((d) => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
              )}
              <button type="button" className={`${button} border-white/20 text-white/70`} disabled={!draft.points.length}
                onClick={() => setDraft({ ...draft, points: draft.points.slice(0, -1) })}>
                <Undo2 className="w-3.5 h-3.5" aria-hidden="true" /> Undo point
              </button>
              <button type="button" className={`${button} border-white/20 text-white/70`} onClick={() => setDraft(null)}>
                <X className="w-3.5 h-3.5" aria-hidden="true" /> Cancel
              </button>
              <button type="button" className={`${button} border-cyan-400 bg-cyan-400 text-black`} disabled={!canFinish} onClick={finishDraft}>
                <Plus className="w-3.5 h-3.5" aria-hidden="true" /> Add this zone
              </button>
            </div>
          </div>
        )}

        {zones && zones.length > 0 && (
          <ul className="divide-y divide-white/5 border border-white/10">
            {zones.map((z) => (
              <li key={z.id} className="flex items-center justify-between px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="w-3 h-3 flex-none" style={{ background: TYPES[z.zone_type]?.color }} aria-hidden="true" />
                  <span className="text-[12px] outfit font-bold text-white truncate">{z.name}</span>
                  <span className="text-[10px] mono uppercase text-white/50">
                    {z.zone_type === 'time_limited' ? `time limit ${z.time_limit}s` : z.zone_type === 'one_way' ? `one-way (${z.direction})` : 'restricted'}
                  </span>
                </div>
                <button type="button" className={`${button} border-red-500/40 text-red-300`} onClick={() => remove(z.id)}
                  aria-label={`Delete zone ${z.name}`}>
                  <Trash2 className="w-3.5 h-3.5" aria-hidden="true" /> Delete
                </button>
              </li>
            ))}
          </ul>
        )}

        {error && <p role="alert" className="text-[11px] text-red-300 border-l-2 border-red-500 pl-2">{error}</p>}
        {saved && !dirty && <p role="status" className="text-[11px] text-emerald-300">Saved. Zones are active now.</p>}

        {!drawing && (
          <div className="flex flex-wrap gap-3">
            <button type="button" className={`${button} border-cyan-400/70 text-cyan-300`} onClick={startDraft} disabled={zones == null}>
              <Plus className="w-3.5 h-3.5" aria-hidden="true" /> Add zone
            </button>
            <button type="button" className={`${button} border-emerald-400 bg-emerald-400 text-black`} onClick={save}
              disabled={!dirty || busy}>
              <Save className="w-3.5 h-3.5" aria-hidden="true" /> {busy ? 'Saving...' : dirty ? 'Save zones' : 'Saved'}
            </button>
          </div>
        )}
      </div>
    </DashboardPanel>
  );
};

export default ZoneEditor;
