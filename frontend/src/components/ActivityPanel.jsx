import DashboardPanel from './DashboardPanel';
import EmptyState from './EmptyState';
import { levelBadge, levelLabel, partLabel, trackLabel } from '../lib/ergonomics';

// Static class names so Tailwind generates them.
const LABEL_COLOR = {
  Standing: 'bg-cyan-500/60', Walking: 'bg-emerald-500/70', Sitting: 'bg-sky-500/60', Bending: 'bg-amber-400',
  Lifting: 'bg-orange-500', Carrying: 'bg-violet-500/70', 'Reaching overhead': 'bg-yellow-300',
  'Lying down': 'bg-rose-400', Fallen: 'bg-red-600', 'Upper body only': 'bg-white/20',
};
const HISTORY_S = 120;

/** The last two minutes of one person's labels, as a strip (newest at the right). */
const History = ({ history, now }) => {
  if (!history?.length) return null;
  const start = now - HISTORY_S;
  return (
    <div className="relative h-2 w-full bg-white/5" aria-hidden="true">
      {history.filter((h) => h.end >= start).map((h) => {
        const left = ((Math.max(h.start, start) - start) / HISTORY_S) * 100;
        const width = Math.max(0.5, ((h.end - Math.max(h.start, start)) / HISTORY_S) * 100);
        return <span key={`${h.label}-${h.start}`} title={h.label}
          className={`absolute top-0 h-full ${LABEL_COLOR[h.label] || 'bg-white/30'}`}
          style={{ left: `${left}%`, width: `${width}%` }} />;
      })}
    </div>
  );
};

/** What each person is doing right now, with REBA when it can be trusted (live only). */
export const ActivityPanel = ({ connected, activity, ergonomics }) => {
  const ids = Object.keys(activity || {});
  const now = Math.max(0, ...ids.flatMap((id) => (activity[id].history || []).map((h) => h.end)));
  return (
    <DashboardPanel title="Activity" headerAction={connected ? `${ids.length} PEOPLE` : null}>
      {!connected || !ids.length ? (
        <EmptyState>{connected ? 'Nobody in view' : 'Waiting for video'}</EmptyState>
      ) : (
        <ul className="space-y-2">
          {ids.map((id) => {
            const a = activity[id];
            const e = ergonomics?.[id];
            const reba = e && e.score != null && e.reliable;
            return (
              <li key={id} className="space-y-1">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="flex items-center gap-2">
                    <span className="text-[10px] mono text-white/40">{trackLabel(id)}</span>
                    <span className="text-sm font-semibold outfit text-white">
                      {a.label === 'Carrying' && a.detail ? `Carrying ${a.detail}`
                        : ['Sitting', 'Lying down'].includes(a.label) && a.detail ? `${a.label.split(' ')[0]} on ${a.detail}` : a.label}
                    </span>
                    {a.confidence === 'low' && <span className="text-[9px] mono uppercase text-white/40">unsure</span>}
                  </span>
                  {reba ? (
                    <span className={`px-1.5 py-px border text-[9px] mono uppercase font-bold ${levelBadge(e.level_name)}`}>
                      REBA {e.score} {levelLabel(e.level_name)}{e.dominant ? ` · ${partLabel(e.dominant)}` : ''}
                    </span>
                  ) : e?.reason ? (
                    <span className="text-[9px] mono text-white/40">REBA not scored: {e.reason}</span>
                  ) : null}
                </div>
                <History history={a.history} now={now} />
              </li>
            );
          })}
        </ul>
      )}
      <p className="text-[9px] mono text-white/30 mt-2">
        Live only: the last 2 minutes per person, nothing stored. Pose rules; front-on views hide some activities.
      </p>
    </DashboardPanel>
  );
};

/** "Only upper body visible..." when nobody in view has hips and knees. */
export const ViewBanner = ({ view }) => {
  if (!view?.upper_body_only) return null;
  return (
    <div className="p-3 border-l-2 border-amber-400 bg-amber-950/30 text-amber-100 text-[12px] outfit" role="status">
      {view.message}
    </div>
  );
};
