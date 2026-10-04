import { useEffect, useId, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ChevronLeft, ChevronRight, ImageOff, X, Film } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import EmptyState from '../components/EmptyState';
import HudBarChart from '../components/HudBarChart';
import AngleTable from '../components/AngleTable';
import SkeletonPlayer from '../components/SkeletonPlayer';
import TimeRangeControl from '../components/TimeRangeControl';
import { useFeed } from '../context/liveFeed';
import { apiUrl, describeError, formatTs, hourSeries, queryString, useFetch, useNow } from '../lib/api';
import { DEFAULT_RANGE, rangeBounds, rangeToParams } from '../lib/timeRange';
import { SEVERITIES, severityBadge } from '../lib/severity';
import { eventTypeLabel } from '../lib/eventTypes';
import {
  ERGO_NOTE, formatPercent, formatSeconds, levelBadge, levelLabel, partLabel,
} from '../lib/ergonomics';
import { chipClass, inputClass, labelClass } from '../lib/ui';

const PAGE_SIZE = 50;
const DEFAULT_FILTERS = { types: [], severity: [], camera_id: '', zone_id: '', range: DEFAULT_RANGE };

const toggleIn = (list, value) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

const SeverityTag = ({ severity }) => (
  <span className={`inline-block px-1.5 py-px border text-[8px] mono uppercase font-bold ${severityBadge(severity)}`}>
    {severity}
  </span>
);

const Thumb = ({ url, className = 'w-16 h-10' }) => {
  const [failed, setFailed] = useState(false);
  if (!url || failed) {
    return (
      <div className={`${className} flex items-center justify-center border border-white/10 bg-black/60`}>
        <ImageOff className="w-3.5 h-3.5 text-white/20" aria-label="No thumbnail" />
      </div>
    );
  }
  return (
    <img
      src={apiUrl(url)}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className={`${className} object-cover border border-white/10 bg-black`}
    />
  );
};

// --- Filters -----------------------------------------------------------------------------

const ChipGroup = ({ legend, options, selected, onToggle, render = (o) => o }) => (
  <fieldset className="min-w-0">
    <legend className={labelClass}>{legend}</legend>
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => (
        <button key={o} type="button" aria-pressed={selected.includes(o)} onClick={() => onToggle(o)} className={chipClass(selected.includes(o))}>
          {render(o)}
        </button>
      ))}
    </div>
  </fieldset>
);

const Filters = ({ meta, filters, onChange }) => {
  const id = useId();
  const types = meta?.event_types || [];
  const severities = meta?.severities || SEVERITIES;
  // Keep a deep-linked camera selectable even before it has any events.
  const known = meta?.cameras || [];
  const cameras = filters.camera_id && !known.includes(filters.camera_id) ? [...known, filters.camera_id] : known;
  const zones = meta?.zones || [];
  const set = (patch) => onChange({ ...filters, ...patch });
  const dirty = JSON.stringify(filters) !== JSON.stringify(DEFAULT_FILTERS);

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-3 items-start">
      <ChipGroup
        legend="Type"
        options={types}
        selected={filters.types}
        onToggle={(t) => set({ types: toggleIn(filters.types, t) })}
        render={eventTypeLabel}
      />
      <ChipGroup
        legend="Severity"
        options={severities}
        selected={filters.severity}
        onToggle={(s) => set({ severity: toggleIn(filters.severity, s) })}
      />
      <div>
        <label htmlFor={`${id}-camera`} className={labelClass}>Camera</label>
        <select id={`${id}-camera`} value={filters.camera_id} onChange={(e) => set({ camera_id: e.target.value })} className={inputClass}>
          <option value="">All cameras</option>
          {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <div>
        <label htmlFor={`${id}-zone`} className={labelClass}>Zone</label>
        <select id={`${id}-zone`} value={filters.zone_id} onChange={(e) => set({ zone_id: e.target.value })} className={inputClass}>
          <option value="">All zones</option>
          {zones.map((z) => <option key={z.id} value={z.id}>{z.name || z.id}</option>)}
        </select>
      </div>
      <TimeRangeControl value={filters.range} onChange={(range) => set({ range })} />
      {dirty && (
        <button type="button" onClick={() => onChange(DEFAULT_FILTERS)} className={`${chipClass(false)} self-end`}>
          Reset filters
        </button>
      )}
    </div>
  );
};

// --- Detail panel ------------------------------------------------------------------------

const Field = ({ label, children }) => (
  <div className="flex justify-between gap-4 py-1 border-b border-white/5 text-[10px] mono">
    <dt className="text-white/40 uppercase">{label}</dt>
    <dd className="text-cyan-100 text-right break-all">{children}</dd>
  </div>
);

const formatAttr = (v) => (v != null && typeof v === 'object' ? JSON.stringify(v) : String(v));

// ergo_risk attributes shown in their own section (not repeated in the raw list).
const ERGO_ATTRS = new Set(['reba_score', 'risk_level', 'dominant', 'duration', 'view_confidence', 'angles']);

const ErgoDetail = ({ attrs }) => {
  const score = attrs.reba_score;
  const angles = attrs.angles && typeof attrs.angles === 'object' ? attrs.angles : null;
  return (
    <div>
      <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400 small-caps mb-1">Ergonomic risk (REBA)</h3>
      <dl>
        <Field label="REBA score">
          <span className={`inline-block px-1.5 py-px border text-[9px] mono uppercase font-bold ${levelBadge(attrs.risk_level)}`}>
            {score ?? '--'}
          </span>
        </Field>
        <Field label="Risk level">{levelLabel(attrs.risk_level)}</Field>
        <Field label="Main factor">{partLabel(attrs.dominant)}</Field>
        <Field label="Duration">{typeof attrs.duration === 'number' ? formatSeconds(attrs.duration) : '--'}</Field>
        <Field label="View confidence">{formatPercent(attrs.view_confidence)}</Field>
      </dl>
      {angles && <AngleTable angles={angles} caption="Joint angles at the peak" className="mt-2 text-[10px]" />}
      <p className="mt-2 text-[9px] outfit text-white/40">{ERGO_NOTE}</p>
    </div>
  );
};

const EventDetail = ({ eventId, onClose }) => {
  const [reload, setReload] = useState(0);
  const [clipFailed, setClipFailed] = useState(false);
  const { data: ev, error, loading } = useFetch(`/api/events/${encodeURIComponent(eventId)}`, reload);
  const closeRef = useRef(null);

  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  // Focus the close button when opened; Escape closes.
  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e) => {
      if (e.key === 'Escape') onCloseRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const retry = () => {
    setClipFailed(false);
    setReload((n) => n + 1);
  };

  let body;
  if (error) body = <EmptyState>{error.status === 404 ? `Event ${eventId} not found` : describeError(error)}</EmptyState>;
  else if (!ev) body = <EmptyState>{loading ? 'Loading event...' : 'No data'}</EmptyState>;
  else {
    const isErgo = ev.type === 'ergo_risk';
    const attrs = Object.entries(ev.attributes || {}).filter(([k]) => !(isErgo && ERGO_ATTRS.has(k)));
    body = (
      <div className="space-y-4">
        {!ev.clip_url && ev.skeleton_url ? (
          <SkeletonPlayer url={ev.skeleton_url} />
        ) : ev.clip_url && !clipFailed ? (
          <video
            key={`${ev.clip_url}#${reload}`}
            src={apiUrl(ev.clip_url) + (reload ? `?try=${reload}` : '')}
            controls
            autoPlay
            muted
            playsInline
            onError={() => setClipFailed(true)}
            className="w-full max-h-72 bg-black border border-white/10"
          />
        ) : (
          <div className="space-y-2">
            <Thumb url={ev.thumbnail_url} className="w-full h-48" />
            <div className="flex items-center gap-2 text-[9px] mono uppercase text-white/50">
              <Film className="w-3 h-3" aria-hidden="true" />
              {clipFailed ? 'Clip could not be loaded' : 'No clip yet (it may still be recording)'}
              <button type="button" onClick={retry} className={chipClass(false)}>Retry</button>
            </div>
          </div>
        )}
        <p className="text-sm outfit text-white/90">{ev.message || '(no message)'}</p>
        <dl>
          <Field label="Event id">{ev.id}</Field>
          <Field label="Type">{eventTypeLabel(ev.type)}</Field>
          <Field label="Severity"><SeverityTag severity={ev.severity} /></Field>
          <Field label="Camera">{ev.camera_id || '--'}</Field>
          <Field label="Zone">{ev.zone_id || '--'}</Field>
          <Field label="Track">{ev.track_id ?? '--'}</Field>
          <Field label="Start">{formatTs(ev.start_ts)}</Field>
          <Field label="End">{ev.end_ts != null ? formatTs(ev.end_ts) : 'ongoing'}</Field>
          <Field label="Confidence">{typeof ev.confidence === 'number' ? `${(ev.confidence * 100).toFixed(0)}%` : '--'}</Field>
          <Field label="Verified">{ev.verified ? 'yes' : 'no'}</Field>
          <Field label="Alert id">{ev.alert_id || '--'}</Field>
        </dl>
        {isErgo && <ErgoDetail attrs={ev.attributes || {}} />}
        {attrs.length > 0 && (
          <div>
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400 small-caps mb-1">Attributes</h3>
            <dl>
              {attrs.map(([k, v]) => <Field key={k} label={k}>{formatAttr(v)}</Field>)}
            </dl>
          </div>
        )}
        {ev.clip_url && (
          <a href={apiUrl(ev.clip_url)} target="_blank" rel="noreferrer" className="inline-block text-[9px] mono uppercase text-cyan-400/70 hover:text-cyan-300 underline">
            Open clip in new tab
          </a>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/50" onClick={onClose} aria-hidden="true" />
      <aside
        role="dialog"
        aria-label={`Event ${eventId}`}
        className="fixed inset-y-0 right-0 z-50 w-full sm:w-[480px] bg-[#070b10] border-l border-cyan-500/20 flex flex-col"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/10">
          <h2 className="text-[11px] font-bold uppercase tracking-[0.2em] text-cyan-400 small-caps">Event #{eventId}</h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close event details"
            className="p-1 border border-white/10 text-white/60 hover:text-white cursor-pointer outline-none focus-visible:ring-1 focus-visible:ring-cyan-400"
          >
            <X className="w-4 h-4" aria-hidden="true" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">{body}</div>
      </aside>
    </>
  );
};

// --- Page --------------------------------------------------------------------------------

const EventsPage = () => {
  const { status, alerts } = useFeed();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedId = searchParams.get('event');
  // `?camera_id=` deep link (e.g. from My Camera); kept in sync with the camera filter.
  const urlCamera = searchParams.get('camera_id') || '';
  const [filters, setFilters] = useState(() => ({ ...DEFAULT_FILTERS, camera_id: urlCamera }));
  const [seenUrlCamera, setSeenUrlCamera] = useState(urlCamera);
  if (urlCamera !== seenUrlCamera) {
    // URL changed underneath us (back/forward, a link): follow it.
    setSeenUrlCamera(urlCamera);
    setFilters((f) => ({ ...f, camera_id: urlCamera }));
  }
  const [offset, setOffset] = useState(0);

  // Relative time presets re-anchor every 30 s; a new live alert refreshes page 1.
  const now = useNow(30000);
  const newestAlert = alerts[0]?.alert_id || '';
  const reloadKey = `${status}|${offset === 0 ? newestAlert : ''}`;

  const { data: meta } = useFetch('/api/meta', status);
  const { start, end } = rangeToParams(filters.range, now);
  const common = {
    types: filters.types,
    severity: filters.severity,
    camera_id: filters.camera_id,
    zone_id: filters.zone_id,
    start,
    end,
  };
  const events = useFetch(`/api/events${queryString({ ...common, limit: PAGE_SIZE, offset })}`, reloadKey);
  const timeline = useFetch(`/api/events/stats${queryString({ ...common, group_by: 'hour_bucket' })}`, reloadKey);

  const updateFilters = (next) => {
    setFilters(next);
    setOffset(0);
    if (next.camera_id !== urlCamera) {
      setSeenUrlCamera(next.camera_id);
      setSearchParams(
        (p) => {
          const q = new URLSearchParams(p);
          if (next.camera_id) q.set('camera_id', next.camera_id);
          else q.delete('camera_id');
          return q;
        },
        { replace: true },
      );
    }
  };
  const openEvent = (id) =>
    setSearchParams((p) => {
      const next = new URLSearchParams(p);
      next.set('event', String(id));
      return next;
    });
  const closeEvent = () =>
    setSearchParams((p) => {
      const next = new URLSearchParams(p);
      next.delete('event');
      return next;
    });

  const total = events.data?.total ?? 0;
  const rows = events.data?.events || [];
  const series = hourSeries(timeline.data?.counts, 24 * 14, rangeBounds(start, end, now));

  let empty = null;
  if (events.error) empty = describeError(events.error);
  else if (!events.data) empty = 'Loading events...';
  else if (rows.length === 0) empty = 'No events match these filters';

  return (
    <div className="space-y-4">
      <DashboardPanel title="Filters">
        <Filters meta={meta} filters={filters} onChange={updateFilters} />
      </DashboardPanel>

      <DashboardPanel title="Events per hour" headerAction={timeline.data ? `${series.reduce((n, d) => n + d.count, 0)} IN RANGE` : null}>
        {timeline.error ? (
          <EmptyState>{describeError(timeline.error)}</EmptyState>
        ) : series.length === 0 ? (
          <EmptyState>{timeline.data ? 'No events in range' : 'Loading...'}</EmptyState>
        ) : (
          <HudBarChart data={series} height={120} />
        )}
      </DashboardPanel>

      <DashboardPanel
        title="Event log"
        headerAction={events.data ? `${total} TOTAL${events.loading ? ' // UPDATING' : ''}` : null}
      >
        {empty ? (
          <EmptyState className="py-8">{empty}</EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <caption className="sr-only">Events, newest first</caption>
              <thead>
                <tr className="text-[9px] uppercase text-white/40 tracking-widest small-caps border-b border-white/10">
                  <th scope="col" className="py-2 pr-3 font-bold">Preview</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Time</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Type</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Severity</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Camera</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Zone</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Track</th>
                  <th scope="col" className="py-2 pr-3 font-bold">Message</th>
                  <th scope="col" className="py-2 font-bold"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((ev) => {
                  const selected = String(ev.id) === selectedId;
                  return (
                    <tr
                      key={ev.id}
                      onClick={() => openEvent(ev.id)}
                      className={`text-[10px] mono border-b border-white/5 cursor-pointer transition-colors ${
                        selected ? 'bg-cyan-500/15' : 'hover:bg-cyan-500/5'
                      }`}
                    >
                      <td className="py-1.5 pr-3"><Thumb url={ev.thumbnail_url} /></td>
                      <td className="py-1.5 pr-3 whitespace-nowrap text-white/70">{formatTs(ev.start_ts)}</td>
                      <td className="py-1.5 pr-3 whitespace-nowrap text-cyan-300 uppercase">{eventTypeLabel(ev.type)}</td>
                      <td className="py-1.5 pr-3"><SeverityTag severity={ev.severity} /></td>
                      <td className="py-1.5 pr-3 text-white/60">{ev.camera_id || '--'}</td>
                      <td className="py-1.5 pr-3 text-white/60">{ev.zone_id || '--'}</td>
                      <td className="py-1.5 pr-3 text-white/60">{ev.track_id ?? '--'}</td>
                      <td className="py-1.5 pr-3 outfit text-white/80 max-w-md truncate" title={ev.message || ''}>{ev.message}</td>
                      <td className="py-1.5 whitespace-nowrap">
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); openEvent(ev.id); }}
                          aria-label={`Open event ${ev.id}`}
                          className={chipClass(selected)}
                        >
                          {ev.clip_url || ev.skeleton_url ? 'Play' : 'View'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="flex items-center justify-between mt-3 text-[9px] mono uppercase text-white/50">
          <span>
            {total > 0 ? `${offset + 1}-${Math.min(offset + PAGE_SIZE, total)} of ${total}` : '0 events'}
          </span>
          <div className="flex gap-1.5">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              className={`${chipClass(false)} flex items-center gap-1 disabled:opacity-30 disabled:cursor-not-allowed`}
            >
              <ChevronLeft className="w-3 h-3" aria-hidden="true" /> Prev
            </button>
            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => setOffset((o) => o + PAGE_SIZE)}
              className={`${chipClass(false)} flex items-center gap-1 disabled:opacity-30 disabled:cursor-not-allowed`}
            >
              Next <ChevronRight className="w-3 h-3" aria-hidden="true" />
            </button>
          </div>
        </div>
      </DashboardPanel>

      {selectedId && <EventDetail key={selectedId} eventId={selectedId} onClose={closeEvent} />}
    </div>
  );
};

export default EventsPage;
