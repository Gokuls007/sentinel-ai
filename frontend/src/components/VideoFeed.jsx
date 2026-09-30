import DashboardPanel from './DashboardPanel';
import { BACKEND_START_HINT, basename } from '../lib/api';

const Placeholder = ({ tone, title, children }) => {
  const red = tone === 'red';
  return (
    <div className="flex flex-col items-center justify-center h-full space-y-4">
      <div className="relative">
        <div className={`w-16 h-16 border-2 rounded-full ${red ? 'border-red-500/40' : 'border-cyan-500/20'}`} />
        <div className={`absolute top-0 w-16 h-16 border-2 rounded-full animate-spin ${red ? 'border-t-red-500' : 'border-t-cyan-400'}`} />
      </div>
      <div className="text-center px-6">
        <p className={`mono tracking-[0.4em] animate-pulse text-sm font-bold ${red ? 'text-red-500' : 'text-cyan-400'}`}>
          {title}
        </p>
        {children}
      </div>
    </div>
  );
};

const Tag = ({ className = 'bg-cyan-950/80 border-cyan-400/30 text-cyan-400', children }) => (
  <div className={`px-2 py-0.5 border text-[9px] mono font-bold uppercase tracking-widest ${className}`}>
    {children}
  </div>
);

const VideoFeed = ({
  frame, status, stats, sourceStats, source, showOverlay = true,
  title = 'Live Observation Terminal', offlineHint = null,
}) => {
  const connected = status === 'live';
  const hwError = connected && sourceStats?.hardware_error;
  const sourceName = source != null ? basename(source) : null;

  let placeholder = null;
  if (!connected) {
    placeholder = (
      <Placeholder tone="red" title={status === 'connecting' ? 'CONNECTING...' : 'BACKEND_OFFLINE'}>
        {offlineHint || (
          <>
            <p className="text-[10px] mono text-red-400/70 mt-3 max-w-md">
              No live feed. Start the backend (runs on bundled sample videos, no camera needed):
            </p>
            <p className="text-[11px] mono text-cyan-300 mt-2 px-2 py-1 bg-black/60 border border-cyan-500/20 inline-block">
              {BACKEND_START_HINT}
            </p>
          </>
        )}
      </Placeholder>
    );
  } else if (hwError) {
    placeholder = (
      <Placeholder tone="red" title="SOURCE_ERROR">
        <p className="text-[10px] mono text-red-400/60 uppercase mt-2 max-w-xs">
          {sourceStats?.error_message || 'Video source could not be opened.'}
        </p>
      </Placeholder>
    );
  } else if (sourceStats?.finished) {
    placeholder = <Placeholder title="SOURCE_FINISHED" />;
  } else if (!frame) {
    placeholder = <Placeholder title="WAITING_FOR_FRAMES..." />;
  }

  return (
    <DashboardPanel
      className="h-full overflow-hidden"
      title={title}
      headerAction={connected && sourceName ? sourceName : null}
    >
      <div className="relative w-full h-full bg-black/60 rounded-sm overflow-hidden group">
        {placeholder || (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            className="w-full h-full object-contain"
            alt="Annotated live frame from the Sentinel pipeline"
          />
        )}

        {/* Cinematic Overlays */}
        <div className="absolute inset-0 scan-line scan-animate opacity-30 pointer-events-none" />
        <div className="absolute inset-0 border-[20px] border-black/20 pointer-events-none" />

        {/* Real source info from /api/stats + /api/health */}
        {connected && showOverlay && (
          <div className="absolute bottom-4 left-4 flex flex-wrap gap-2">
            {sourceName && <Tag>Source: {sourceName}</Tag>}
            {sourceStats?.is_file && sourceStats?.loop && (
              <Tag>Looping // {sourceStats.loops_completed ?? 0} loops</Tag>
            )}
            {sourceStats?.source_fps > 0 && <Tag>Src {sourceStats.source_fps} fps</Tag>}
            {sourceStats?.frames_dropped > 0 && (
              <Tag className="bg-amber-950/80 border-amber-400/30 text-amber-400">
                Dropped {sourceStats.frames_dropped}
              </Tag>
            )}
          </div>
        )}

        {connected && showOverlay && stats?.fps != null && (
          <div className="absolute top-4 right-4 text-right">
            <div className="text-[10px] mono text-cyan-500/70 font-bold uppercase">Pipeline FPS</div>
            <div className="text-xl mono font-bold text-white tracking-tighter">
              {Number(stats.fps).toFixed(1)}
            </div>
          </div>
        )}
      </div>
    </DashboardPanel>
  );
};

export default VideoFeed;
