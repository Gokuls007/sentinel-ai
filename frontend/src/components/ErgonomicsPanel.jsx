import { useState } from 'react';
import DashboardPanel from './DashboardPanel';
import EmptyState from './EmptyState';
import AngleTable from './AngleTable';
import { loadStored, saveStored } from '../lib/api';
import {
  ERGO_NOTE, formatPercent, levelBadge, levelLabel, partLabel, trackLabel,
} from '../lib/ergonomics';
import { chipClass } from '../lib/ui';

const SHOW_ANGLES_KEY = 'sentinel.ergonomics.showAngles';

const RiskBadge = ({ t }) => {
  const reliable = Boolean(t.reliable) && t.score != null;
  if (!reliable) {
    return (
      <span
        className={`inline-flex items-center gap-1 px-1.5 py-px border text-[8px] mono uppercase ${levelBadge(null, false)}`}
        title="Not scored: REBA needs the whole body, ideally seen from the side"
      >
        <span className="italic">REBA {t.score ?? '?'}</span>
        <span>// {t.reason || 'not scored'}</span>
      </span>
    );
  }
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-px border text-[9px] mono uppercase font-bold ${levelBadge(t.level_name)}`}>
      REBA {t.score} <span className="font-normal opacity-80">{levelLabel(t.level_name)}</span>
    </span>
  );
};

const ConfidenceBar = ({ value, reliable }) => {
  const pct = typeof value === 'number' ? Math.max(0, Math.min(1, value)) * 100 : 0;
  return (
    <div className="flex items-center gap-1.5" title="View confidence: how well the pose can be measured from this angle">
      <span className="text-[8px] mono uppercase text-white/30">View</span>
      <div className="w-16 h-1 bg-white/10" role="meter" aria-label="View confidence" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
        <div className={`h-full ${reliable ? 'bg-cyan-400/70' : 'bg-white/25'}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-7 text-right text-[8px] mono tabular-nums text-white/50">{formatPercent(value)}</span>
    </div>
  );
};

const PersonRow = ({ t, showAngles }) => {
  const reliable = Boolean(t.reliable) && t.score != null;
  return (
    <li className={`p-2 border-l-2 ${reliable ? 'bg-white/5 border-cyan-500/30' : 'bg-white/[0.02] border-white/10'}`}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-[11px] font-bold mono ${reliable ? 'text-cyan-400' : 'text-cyan-400/50'}`}>{trackLabel(t.track_id)}</span>
          <RiskBadge t={t} />
        </div>
        <ConfidenceBar value={t.confidence} reliable={reliable} />
      </div>
      <div className={`mt-1 text-[9px] mono uppercase ${reliable ? 'text-white/50' : 'text-white/30'}`}>
        Main factor: <span className={reliable ? 'text-white/80' : ''}>{partLabel(t.dominant)}</span>
      </div>
      {showAngles && <AngleTable angles={t.angles} estimated={t.estimated_parts} className="mt-1.5 text-[9px]" />}
    </li>
  );
};

/**
 * Per-person REBA risk from frame data `ergonomics` ({track_id: {...}}).
 * `ergonomics` undefined/null = no frame (or no ergonomics in it) yet.
 */
const ErgonomicsPanel = ({ ergonomics, connected, className = '' }) => {
  const [showAngles, setShowAngles] = useState(() => loadStored(SHOW_ANGLES_KEY, false) === true);
  const toggleAngles = () =>
    setShowAngles((prev) => {
      saveStored(SHOW_ANGLES_KEY, !prev);
      return !prev;
    });

  const rows = ergonomics && typeof ergonomics === 'object'
    ? Object.values(ergonomics).filter(Boolean).sort((a, b) => Number(a.track_id) - Number(b.track_id))
    : [];
  const reliableCount = rows.filter((t) => t.reliable && t.score != null).length;

  let empty = null;
  if (!connected) empty = 'Backend offline';
  else if (ergonomics == null) empty = 'Waiting for ergonomics data...';
  else if (rows.length === 0) empty = 'No people tracked';

  return (
    <DashboardPanel
      title="Ergonomics (REBA)"
      headerAction={connected && ergonomics != null ? `${rows.length} TRACKED // ${reliableCount} SCORED` : null}
      className={className}
    >
      <div className="flex flex-col gap-2 h-full">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[9px] outfit text-white/40">{ERGO_NOTE}</p>
          <button type="button" aria-pressed={showAngles} onClick={toggleAngles} className={`${chipClass(showAngles)} flex-none`}>
            Show angles
          </button>
        </div>
        {empty ? (
          <EmptyState>{empty}</EmptyState>
        ) : (
          <ul className="space-y-1.5 overflow-y-auto pr-1 max-h-80" aria-label="Ergonomic risk per tracked person">
            {rows.map((t) => <PersonRow key={t.track_id} t={t} showAngles={showAngles} />)}
          </ul>
        )}
      </div>
    </DashboardPanel>
  );
};

export default ErgonomicsPanel;
