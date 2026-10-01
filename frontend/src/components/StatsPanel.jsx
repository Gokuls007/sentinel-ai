import { Users, Crosshair, Zap, AlertTriangle } from 'lucide-react';
import DashboardPanel from './DashboardPanel';

// Static class names so Tailwind can see (and generate) them.
const COLOR_CLASSES = {
  cyan: { icon: 'text-cyan-400', label: 'text-cyan-500/70' },
  red: { icon: 'text-red-400', label: 'text-red-500/70' },
};

// Pipeline layers reported in frame data `timings_ms`, in pipeline order.
const LAYERS = [
  ['detect_track', 'Detect+Track'],
  ['pose', 'Pose'],
  ['analytics', 'Analytics'],
  ['ergonomics', 'Ergonomics'], // part of "analytics", shown on its own
  ['annotate', 'Annotate'],
  ['events_and_clips', 'Events+Clips'],
  ['stream', 'Stream'],
];

const StatCard = ({ icon, value, label, color = 'cyan' }) => {
  const Icon = icon;
  const c = COLOR_CLASSES[color] || COLOR_CLASSES.cyan;
  return (
    <div className="p-4 bg-cyan-950/20 border-l border-cyan-500/20">
      <div className="flex items-center space-x-2 mb-1">
        <Icon className={`w-3 h-3 ${c.icon} opacity-60`} />
        <span className={`text-[9px] uppercase font-bold ${c.label} tracking-widest outfit`}>
          {label}
        </span>
      </div>
      <div className="flex items-baseline">
        <span className="text-2xl font-bold mono text-white leading-none">{value}</span>
      </div>
    </div>
  );
};

const show = (v, fmt = (x) => x) => (v == null ? '--' : fmt(v));

/** Small per-layer latency breakdown (bars scaled to the slowest layer). */
const Timings = ({ timings }) => {
  const rows = LAYERS.filter(([k]) => typeof timings?.[k] === 'number').map(([k, label]) => [label, timings[k]]);
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map(([, v]) => v), 1);
  return (
    <div className="mt-3 space-y-1" aria-label="Per-layer processing time">
      {rows.map(([label, v]) => (
        <div key={label} className="flex items-center gap-2 text-[8px] mono uppercase">
          <span className="w-20 flex-none text-white/40 truncate">{label}</span>
          <div className="flex-1 h-1 bg-cyan-500/10">
            <div className="h-full bg-cyan-400/60" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="w-12 flex-none text-right text-cyan-500/80 tabular-nums">{v.toFixed(1)}ms</span>
        </div>
      ))}
    </div>
  );
};

const StatsPanel = ({ stats, timings }) => (
  <DashboardPanel title="Nexus Statistics" headerAction={stats ? 'LIVE' : 'NO_DATA'}>
    <div className="grid grid-cols-2 gap-px bg-cyan-500/10 border border-cyan-500/10">
      <StatCard icon={Users} value={show(stats?.person_count)} label="Persons" />
      <StatCard icon={Crosshair} value={show(stats?.active_tracks)} label="Tracks" />
      <StatCard
        icon={Zap}
        value={show(stats?.processing_time_ms, (v) => `${Math.round(v)}ms`)}
        label="Latency"
      />
      <StatCard icon={AlertTriangle} value={show(stats?.alert_count)} label="Alerts" color="red" />
    </div>
    <Timings timings={timings} />
  </DashboardPanel>
);

export default StatsPanel;
