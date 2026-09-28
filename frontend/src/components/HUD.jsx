import useWebSocket from '../hooks/useWebSocket';
import { BACKEND_START_HINT, basename, formatDuration, usePoll } from '../lib/api';
import VideoFeed from './VideoFeed';
import StatsPanel from './StatsPanel';
import TrackList from './TrackList';
import AlertPanel from './AlertPanel';
import AlertTimeline from './AlertTimeline';
import ZonePanel from './ZonePanel';
import { Shield, Clock, Terminal } from 'lucide-react';

const STATUS_STYLES = {
  live: {
    box: 'border-green-500/30 bg-green-500/5',
    dot: 'bg-green-400 animate-pulse',
    text: 'text-green-400',
    label: 'LIVE',
  },
  connecting: {
    box: 'border-amber-500/30 bg-amber-500/5',
    dot: 'bg-amber-400 animate-pulse',
    text: 'text-amber-400',
    label: 'CONNECTING',
  },
  disconnected: {
    box: 'border-red-500/30 bg-red-500/5',
    dot: 'bg-red-500',
    text: 'text-red-500',
    label: 'BACKEND_OFFLINE',
  },
};

const HUD = () => {
  const { status, connected, frame, frameData, alerts, stats, lastError, url } = useWebSocket();
  const { data: backendStats } = usePoll('/api/stats', 2000, connected);
  const { data: health } = usePoll('/api/health', 5000, connected);
  const style = STATUS_STYLES[status] || STATUS_STYLES.disconnected;

  return (
    <div className="flex flex-col h-screen p-4 bg-[#0A0A0F] text-white selection:bg-cyan-500/30 overflow-hidden font-outfit">

      {/* --- SITE HEADER --- */}
      <header className="flex items-center justify-between mb-4 pb-4 border-b border-cyan-500/10">
        <div className="flex items-center space-x-6">
          <div className="relative group">
            <div className="absolute inset-0 bg-cyan-400/20 blur-lg group-hover:bg-cyan-400/40 transition-all" />
            <div className="relative p-2 border border-cyan-400/40 bg-black">
              <Shield className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-[0.4em] text-cyan-400 uppercase leading-none small-caps">
              Sentinel AI
            </h1>
            <p className="text-[9px] text-cyan-700/80 font-bold tracking-[0.6em] uppercase mt-1.5 mono">
              Video Anomaly Detection
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-12">
          {/* Connection badge (with hover tooltip) */}
          <div className="flex items-center space-x-4">
            <span className="text-[10px] uppercase font-bold text-white/30 tracking-widest small-caps">Link Status:</span>
            <div className={`relative group flex items-center space-x-2 px-3 py-1 rounded-sm border ${style.box}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
              <span className={`text-[10px] font-bold uppercase mono tracking-tighter ${style.text}`}>
                {style.label}
              </span>
              <div className="absolute top-full right-0 mt-2 w-72 p-2 bg-black/95 border border-cyan-500/20 rounded-sm opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                <p className="text-[9px] text-cyan-300/80 mono leading-relaxed break-all">
                  Feed: {url}
                </p>
                {status !== 'live' && (
                  <p className="text-[9px] text-amber-300/80 mono leading-relaxed mt-1">
                    {lastError ? `${lastError}. ` : ''}Start the backend with `{BACKEND_START_HINT}`.
                  </p>
                )}
              </div>
            </div>
          </div>

          {backendStats && (
            <div className="hidden lg:flex items-center space-x-4">
              <Clock className="w-3.5 h-3.5 text-cyan-800" />
              <span className="text-[11px] mono text-cyan-500/60 font-bold tabular-nums tracking-widest">
                UPTIME::{formatDuration(backendStats.uptime_seconds)}
              </span>
            </div>
          )}
        </div>
      </header>

      {/* --- DASHBOARD CORE --- */}
      <main className="flex-1 grid grid-cols-12 gap-4 min-h-0">

        {/* CENTER COLUMN: FEED & CHART */}
        <div className="col-span-12 lg:col-span-8 flex flex-col space-y-4 min-h-0">
          <div className="flex-[3] min-h-0">
            <VideoFeed
              frame={frame}
              status={status}
              stats={stats}
              sourceStats={backendStats?.source_stats}
              source={health?.source}
            />
          </div>
          <div className="flex-[1] min-h-[180px] grid grid-cols-3 gap-4">
            <div className="col-span-2 min-h-0">
              <AlertTimeline alerts={alerts} />
            </div>
            <div className="col-span-1 min-h-0">
              <ZonePanel connected={connected} />
            </div>
          </div>
        </div>

        {/* SIDE COLUMN: TELEMETRY */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-4 min-h-0">
          <div className="flex-none">
            <StatsPanel stats={stats} />
          </div>
          <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
            <TrackList connected={connected} frameData={frameData} />
          </div>
          <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
            <AlertPanel alerts={alerts} connected={connected} />
          </div>
        </div>

      </main>

      {/* --- SITE FOOTER --- */}
      <footer className="mt-4 pt-2 border-t border-white/5 flex justify-between items-center opacity-30 select-none">
        <div className="flex items-center space-x-6">
          <div className="text-[9px] uppercase font-bold tracking-[0.3em] text-cyan-600 mono">
            Source // {health?.source != null ? basename(health.source) : 'unknown'}
          </div>
          {backendStats && (
            <>
              <div className="h-3 w-px bg-white/10" />
              <div className="text-[9px] uppercase font-medium text-white/40 tracking-widest mono">
                Frames: {backendStats.frames_processed} // Avg FPS: {backendStats.avg_fps}
              </div>
            </>
          )}
        </div>

        <div className="flex items-center space-x-6">
          {frameData?.processing_time_ms != null && (
            <div className="text-[9px] mono text-cyan-600">
              ENGINE_MS: {Number(frameData.processing_time_ms).toFixed(2)}
            </div>
          )}
          <div className="flex items-center space-x-2">
            <Terminal className="w-3 h-3 text-cyan-700" />
            <span className="text-[9px] mono text-cyan-700 tracking-tighter font-bold uppercase">
              {health ? `Backend ${health.status}` : status}
            </span>
          </div>
        </div>
      </footer>

    </div>
  );
};

export default HUD;
