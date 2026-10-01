import { useId, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import DashboardPanel from './DashboardPanel';
import EmptyState from './EmptyState';
import HudStackedBarChart, { StackLegend } from './HudStackedBarChart';
import { describeError, queryString, useFetch, useNow } from '../lib/api';
import {
  DAY_PRESETS, ERGO_LEVELS, ERGO_NOTE, LEVEL_HEX, LEVEL_LABEL, dayBounds, dayRange, formatSeconds,
  levelBadge, levelForScore, partLabel, trackLabel,
} from '../lib/ergonomics';
import { chipClass, inputClass, labelClass } from '../lib/ui';

export const ERGO_EMPTY = 'No ergonomic data yet: it accumulates while a camera runs.';

// Lowest to highest risk, then unknown (view too unreliable to score) in grey.
const ORDER = [...ERGO_LEVELS, 'unknown'];
const SERIES = ORDER.map((k) => ({ key: k, label: LEVEL_LABEL[k], color: LEVEL_HEX[k] }));
const RISKY = ['high', 'very_high'];

/** rows {key: {level: seconds}} -> [{ key, label, <level>: s, total, risky }]. */
function toStacks(rows, labelFor) {
  return Object.entries(rows || {}).map(([key, levels]) => {
    const row = { key, label: labelFor(key) };
    let total = 0;
    for (const lvl of ORDER) {
      const s = Number(levels?.[lvl]) || 0;
      row[lvl] = s;
      total += s;
    }
    row.total = total;
    row.risky = RISKY.reduce((n, l) => n + row[l], 0);
    return row;
  });
}

const byRisk = (a, b) => b.risky - a.risky || b.total - a.total || String(a.label).localeCompare(String(b.label));

/** Minutes on the value axis once the largest bar passes 10 minutes. */
function axisFormat(rows) {
  const max = Math.max(0, ...rows.map((r) => r.total));
  if (max >= 600) return (v) => `${Math.round(v / 60)}m`;
  return (v) => `${Math.round(v)}s`;
}

const totalOf = (rows) => rows.reduce((n, r) => n + r.total, 0);

/** Loading / error / empty, else children. */
function stateMessage(result, empty) {
  if (result.error) return describeError(result.error);
  if (!result.data) return 'Loading...';
  if (empty) return ERGO_EMPTY;
  return null;
}

// --- Controls ----------------------------------------------------------------------------

const DayControl = ({ value, onChange }) => {
  const id = useId();
  const custom = value.preset === 'custom';
  return (
    <fieldset className="min-w-0">
      <legend className={labelClass}>Days</legend>
      <div className="flex flex-wrap items-end gap-1.5">
        {DAY_PRESETS.map(([key, label]) => (
          <button key={key} type="button" aria-pressed={value.preset === key} onClick={() => onChange({ ...value, preset: key })} className={chipClass(value.preset === key)}>
            {label}
          </button>
        ))}
        <button type="button" aria-pressed={custom} onClick={() => onChange({ ...value, preset: 'custom' })} className={chipClass(custom)}>
          Pick days
        </button>
        {custom && (
          <>
            <div>
              <label htmlFor={`${id}-from`} className={labelClass}>From</label>
              <input id={`${id}-from`} type="date" value={value.from} onChange={(e) => onChange({ ...value, from: e.target.value })} className={inputClass} />
            </div>
            <div>
              <label htmlFor={`${id}-to`} className={labelClass}>To</label>
              <input id={`${id}-to`} type="date" value={value.to} min={value.from || undefined} onChange={(e) => onChange({ ...value, to: e.target.value })} className={inputClass} />
            </div>
          </>
        )}
      </div>
    </fieldset>
  );
};

// --- Postures + tracks -------------------------------------------------------------------

const PosturesTable = ({ postures }) => {
  const rows = Object.entries(postures || {}).map(([part, p]) => ({ part, ...p }));
  const maxEvents = Math.max(1, ...rows.map((r) => r.events || 0));
  return (
    <table className="w-full text-left border-collapse">
      <caption className="sr-only">Ergonomic risk events by main body part</caption>
      <thead>
        <tr className="text-[9px] uppercase text-white/40 tracking-widest small-caps border-b border-white/10">
          <th scope="col" className="py-1.5 pr-2 font-bold">Main factor</th>
          <th scope="col" className="py-1.5 pr-2 font-bold">Events</th>
          <th scope="col" className="py-1.5 pr-2 font-bold text-right">Peak REBA</th>
          <th scope="col" className="py-1.5 font-bold text-right">Time</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const lvl = levelForScore(r.peak_reba);
          return (
            <tr key={r.part} className="text-[10px] mono border-b border-white/5">
              <th scope="row" className="py-1.5 pr-2 font-normal text-white/80">{partLabel(r.part)}</th>
              <td className="py-1.5 pr-2">
                <div className="flex items-center gap-2">
                  <div className="flex-1 min-w-[40px] h-1.5 bg-white/5">
                    <div className="h-full" style={{ width: `${((r.events || 0) / maxEvents) * 100}%`, background: lvl ? LEVEL_HEX[lvl] : LEVEL_HEX.unknown }} />
                  </div>
                  <span className="w-6 text-right tabular-nums text-cyan-100">{r.events}</span>
                </div>
              </td>
              <td className="py-1.5 pr-2 text-right">
                <span className={`inline-block px-1.5 py-px border text-[9px] font-bold ${levelBadge(lvl)}`}>{r.peak_reba || '--'}</span>
              </td>
              <td className="py-1.5 text-right tabular-nums text-white/60">{formatSeconds(r.total_duration_s)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
};

const TRACK_LIMIT = 50;

const TrackTable = ({ rows }) => {
  const shown = rows.slice(0, TRACK_LIMIT);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <caption className="sr-only">Time at each risk level per track ID, most high-risk time first</caption>
        <thead>
          <tr className="text-[9px] uppercase text-white/40 tracking-widest small-caps border-b border-white/10">
            <th scope="col" className="py-1.5 pr-3 font-bold">Track</th>
            <th scope="col" className="py-1.5 pr-3 font-bold text-right">High + very high</th>
            {ORDER.map((l) => (
              <th key={l} scope="col" className="py-1.5 pr-3 font-bold text-right whitespace-nowrap">
                <span className="inline-block w-2 h-2 mr-1 align-middle" style={{ background: LEVEL_HEX[l] }} aria-hidden="true" />
                {LEVEL_LABEL[l]}
              </th>
            ))}
            <th scope="col" className="py-1.5 font-bold text-right">Total</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={r.key} className="text-[10px] mono border-b border-white/5">
              <th scope="row" className="py-1.5 pr-3 font-bold text-cyan-400 whitespace-nowrap">{r.label}</th>
              <td className={`py-1.5 pr-3 text-right tabular-nums font-bold ${r.risky > 0 ? 'text-orange-300' : 'text-white/30'}`}>{formatSeconds(r.risky)}</td>
              {ORDER.map((l) => (
                <td key={l} className={`py-1.5 pr-3 text-right tabular-nums ${r[l] > 0 ? 'text-white/70' : 'text-white/20'}`}>{formatSeconds(r[l])}</td>
              ))}
              <td className="py-1.5 text-right tabular-nums text-white/70">{formatSeconds(r.total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > TRACK_LIMIT && (
        <p className="mt-2 text-[9px] mono uppercase text-white/40">Showing the {TRACK_LIMIT} track IDs with the most high-risk time of {rows.length}.</p>
      )}
    </div>
  );
};

// --- Section -----------------------------------------------------------------------------

/**
 * Time at each REBA risk level: per zone (main view), per hour of day, top risky postures
 * and per track ID. `status` (the feed status) refetches on reconnect.
 */
const ErgonomicsAnalytics = ({ status, meta }) => {
  const id = useId();
  const [days, setDays] = useState({ preset: 'today', from: '', to: '' });
  const [camera, setCamera] = useState('');
  const [refresh, setRefresh] = useState(0);
  const now = useNow(60000); // re-anchors "today" and refreshes once a minute
  const { day_from, day_to } = dayRange(days, now);
  const reloadKey = `${status}|${refresh}|${now}`;

  const common = { day_from, day_to, camera_id: camera };
  const zoneRes = useFetch(`/api/ergonomics/time${queryString({ group_by: 'zone', ...common })}`, reloadKey);
  const hourRes = useFetch(`/api/ergonomics/time${queryString({ group_by: 'hour', ...common })}`, reloadKey);
  const trackRes = useFetch(`/api/ergonomics/time${queryString({ group_by: 'track', ...common })}`, reloadKey);
  const { start, end } = dayBounds(day_from, day_to);
  const postureRes = useFetch(`/api/ergonomics/postures${queryString({ start, end, camera_id: camera })}`, reloadKey);

  // Camera choices: cameras with stored events (meta) plus every configured camera.
  const { data: cams } = useFetch('/api/cameras', status);
  const camIds = Array.isArray(cams) ? cams.map((c) => c.id) : [];
  const cameraOptions = [...new Set([...(meta?.cameras || []), ...camIds])].filter(Boolean).sort();

  // Zone names: /api/meta lists the primary camera's zones; the laptop's come from its own
  // zone list, which is only available while that camera runs.
  const laptopRunning = Array.isArray(cams) && cams.some((c) => c.id === 'laptop' && c.status === 'running');
  const { data: laptopZones } = useFetch(laptopRunning ? '/api/zones?camera=laptop' : null, status);
  const zoneNames = new Map([
    ...(Array.isArray(laptopZones) ? laptopZones : []).map((z) => [z.id, z.name || z.id]),
    ...(meta?.zones || []).map((z) => [z.id, z.name || z.id]),
  ]);

  const zoneRows = toStacks(zoneRes.data?.rows, (k) => (k === '' ? 'No zone' : zoneNames.get(k) || k)).sort(byRisk);
  const hourRaw = toStacks(hourRes.data?.rows, (k) => k);
  const hourRows = hourRaw.length === 0
    ? []
    : Array.from({ length: 24 }, (_, h) => {
      const r = hourRaw.find((x) => Number(x.key) === h);
      return { ...(r || Object.fromEntries(ORDER.map((l) => [l, 0]))), key: String(h), label: String(h).padStart(2, '0') };
    });
  const trackRows = toStacks(trackRes.data?.rows, trackLabel).sort(byRisk);
  const postures = postureRes.data?.postures || {};

  const zoneEmpty = zoneRows.every((r) => r.total === 0);
  const zoneMsg = stateMessage(zoneRes, zoneEmpty);
  const hourMsg = stateMessage(hourRes, hourRows.every((r) => r.total === 0));
  const trackMsg = stateMessage(trackRes, trackRows.length === 0);
  let postureMsg = null;
  if (postureRes.error) postureMsg = describeError(postureRes.error);
  else if (!postureRes.data) postureMsg = 'Loading...';
  else if (Object.keys(postures).length === 0) postureMsg = 'No ergonomic risk events in these days';

  const fmtValue = (v) => formatSeconds(v);
  const rangeLabel = day_from === day_to ? day_from : `${day_from} - ${day_to}`;

  return (
    <section aria-label="Ergonomics" className="space-y-4">
      <DashboardPanel
        title="Time at risk per zone"
        headerAction={zoneRes.data && !zoneEmpty ? `${formatSeconds(totalOf(zoneRows))} OBSERVED // ${rangeLabel}` : rangeLabel}
      >
        <div className="space-y-4">
          <div className="flex flex-wrap gap-x-6 gap-y-3 items-end">
            <DayControl value={days} onChange={setDays} />
            <div>
              <label htmlFor={`${id}-camera`} className={labelClass}>Camera</label>
              <select id={`${id}-camera`} value={camera} onChange={(e) => setCamera(e.target.value)} className={inputClass}>
                <option value="">All cameras</option>
                {cameraOptions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <button
              type="button"
              onClick={() => setRefresh((n) => n + 1)}
              className={`${chipClass(false)} flex items-center gap-1`}
              title="Fetch the latest accumulated time"
            >
              <RefreshCw className={`w-3 h-3 ${zoneRes.loading ? 'animate-spin' : ''}`} aria-hidden="true" /> Refresh
            </button>
          </div>

          <StackLegend series={SERIES} />

          <div className="min-h-[220px] flex flex-col justify-center">
            {zoneMsg ? (
              <EmptyState>{zoneMsg}</EmptyState>
            ) : (
              <HudStackedBarChart
                data={zoneRows}
                series={SERIES}
                horizontal
                height={Math.max(200, zoneRows.length * 44 + 30)}
                formatValue={fmtValue}
                formatTick={axisFormat(zoneRows)}
              />
            )}
          </div>
          <p className="text-[9px] outfit text-white/40">
            Time each tracked person spent at each REBA risk level, by the zone they were in. Unknown is time the
            view was too unreliable to score (people are best measured from the side). {ERGO_NOTE}
          </p>
        </div>
      </DashboardPanel>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <DashboardPanel title="Time at risk per hour" headerAction="HOUR OF DAY" className="xl:col-span-2">
          <div className="min-h-[220px] flex flex-col justify-center">
            {hourMsg ? (
              <EmptyState>{hourMsg}</EmptyState>
            ) : (
              <HudStackedBarChart data={hourRows} series={SERIES} height={220} formatValue={fmtValue} formatTick={axisFormat(hourRows)} />
            )}
          </div>
        </DashboardPanel>
        <DashboardPanel title="Top risky postures" headerAction="FROM RISK EVENTS">
          <div className="min-h-[220px] flex flex-col justify-center">
            {postureMsg ? <EmptyState>{postureMsg}</EmptyState> : <PosturesTable postures={postures} />}
          </div>
        </DashboardPanel>
      </div>

      <DashboardPanel title="Time per track ID" headerAction={trackRes.data && trackRows.length ? `${trackRows.length} TRACK IDS` : null}>
        {trackMsg ? (
          <EmptyState className="py-6">{trackMsg}</EmptyState>
        ) : (
          <>
            <TrackTable rows={trackRows} />
            <p className="mt-2 text-[9px] outfit text-white/40">
              Track IDs are tracker identities, not people: someone who leaves and is picked up again gets a new ID.
            </p>
          </>
        )}
      </DashboardPanel>
    </section>
  );
};

export default ErgonomicsAnalytics;
