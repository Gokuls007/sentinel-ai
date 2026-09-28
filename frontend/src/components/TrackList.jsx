import DashboardPanel from './DashboardPanel';
import { usePoll } from '../lib/api';

const FALL_STATE_CLASSES = {
  upright: 'text-cyan-600 border-cyan-500/20',
  falling: 'text-amber-400 border-amber-400/40',
  fallen: 'text-orange-500 border-orange-500/50',
  confirmed: 'text-red-500 border-red-500/60 animate-pulse',
};

const TrackList = ({ connected, frameData }) => {
  const { data: tracks } = usePoll('/api/tracks', 1000, connected);

  // Confidence per track id from the latest frame (detections may have track_id: null).
  const confidenceById = new Map();
  for (const det of frameData?.detections?.detections || []) {
    if (det.track_id != null) confidenceById.set(det.track_id, det.confidence);
  }

  const rows = Array.isArray(tracks)
    ? [...tracks].sort((a, b) => a.track_id - b.track_id)
    : [];

  let empty = null;
  if (!connected) empty = 'Backend offline';
  else if (!tracks) empty = 'Loading tracks...';
  else if (rows.length === 0) empty = 'No active tracks';

  return (
    <DashboardPanel title="Target Registry" headerAction={connected ? `${rows.length} TRACKED` : null} className="h-full">
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-2 h-64">
        {empty ? (
          <div className="flex flex-col items-center justify-center h-full opacity-30">
            <p className="text-[10px] mono uppercase tracking-widest text-center">{empty}</p>
          </div>
        ) : (
          rows.map((t) => {
            const conf = confidenceById.get(t.track_id);
            const fallState = t.fall_state || 'upright';
            return (
              <div
                key={t.track_id}
                className="group flex items-center justify-between p-2 bg-white/5 hover:bg-cyan-500/10 border-l-2 border-transparent hover:border-cyan-400 transition-all cursor-default"
              >
                <div className="flex items-center space-x-3">
                  <span className="text-[11px] font-bold mono text-cyan-400">
                    ID::{String(t.track_id).padStart(3, '0')}
                  </span>
                  <span className={`text-[8px] uppercase font-bold mono px-1.5 py-px border ${FALL_STATE_CLASSES[fallState] || FALL_STATE_CLASSES.upright}`}>
                    {fallState}
                  </span>
                </div>
                <div className="flex flex-col items-end">
                  <div className="text-[9px] mono text-cyan-600 font-bold">
                    {conf != null ? `${(conf * 100).toFixed(1)}%` : 'not in frame'}
                  </div>
                  <div className="text-[8px] mono text-gray-600">
                    S::{Number(t.speed ?? 0).toFixed(1)} // T::{Number(t.time_tracked ?? 0).toFixed(0)}s
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </DashboardPanel>
  );
};

export default TrackList;
