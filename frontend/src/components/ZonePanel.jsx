import DashboardPanel from './DashboardPanel';
import { usePoll } from '../lib/api';

const ZonePanel = ({ connected }) => {
  const { data: zones } = usePoll('/api/zones', 10000, connected);
  const list = Array.isArray(zones) ? zones : [];

  let empty = null;
  if (!connected) empty = 'Backend offline';
  else if (!zones) empty = 'Loading zones...';
  else if (list.length === 0) empty = 'No zones configured';

  return (
    <DashboardPanel title="Zones" headerAction={connected && zones ? `${list.length} DEFINED` : null} className="h-full">
      <div className="h-full overflow-y-auto space-y-1.5 pr-1">
        {empty ? (
          <div className="flex items-center justify-center h-full opacity-30">
            <p className="text-[10px] mono uppercase tracking-widest text-center">{empty}</p>
          </div>
        ) : (
          list.map((z, i) => (
            <div key={z.id ?? i} className="p-2 bg-white/5 border-l-2 border-cyan-500/30">
              <div className="flex justify-between items-baseline">
                <span className="text-[10px] font-bold uppercase outfit text-cyan-300 truncate">
                  {z.name || z.id}
                </span>
                <span className="text-[8px] mono uppercase text-white/40">{z.type}</span>
              </div>
              {((z.type === 'time_limited' && z.time_limit) || z.direction || z.load_score > 0) && (
                <div className="text-[8px] mono text-white/40 mt-0.5">
                  {z.type === 'time_limited' && z.time_limit ? <span>LIMIT::{z.time_limit}s </span> : null}
                  {z.direction && <span>DIR::{Array.isArray(z.direction) ? z.direction.join(',') : String(z.direction)} </span>}
                  {z.load_score > 0 && <span title="REBA load/force score">LOAD::{z.load_score}</span>}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </DashboardPanel>
  );
};

export default ZonePanel;
