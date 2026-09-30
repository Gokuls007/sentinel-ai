import { useState } from 'react';
import { Clock, Terminal } from 'lucide-react';
import { basename, formatDuration, loadStored, saveStored, usePoll } from '../lib/api';
import { useFeed, useFrame } from '../context/liveFeed';
import VideoFeed from '../components/VideoFeed';
import StatsPanel from '../components/StatsPanel';
import TrackList from '../components/TrackList';
import AlertPanel from '../components/AlertPanel';
import AlertTimeline from '../components/AlertTimeline';
import ZonePanel from '../components/ZonePanel';

const PANELS_KEY = 'sentinel.live.panels';
const PANEL_OPTIONS = [
  ['overlay', 'Video overlay'],
  ['stats', 'Stats'],
  ['tracks', 'Tracks'],
  ['alerts', 'Alerts'],
  ['timeline', 'Timeline'],
  ['zones', 'Zones'],
];
const DEFAULT_PANELS = Object.fromEntries(PANEL_OPTIONS.map(([k]) => [k, true]));

function usePanelToggles() {
  const [panels, setPanels] = useState(() => {
    const stored = loadStored(PANELS_KEY, null);
    return stored && typeof stored === 'object' ? { ...DEFAULT_PANELS, ...stored } : DEFAULT_PANELS;
  });
  const toggle = (key) =>
    setPanels((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      saveStored(PANELS_KEY, next);
      return next;
    });
  return [panels, toggle];
}

// --- Frame readers: only these re-render at the frame rate. ---------------------------

const LiveVideo = ({ status, sourceStats, source, showOverlay }) => {
  const { frame, frameData } = useFrame();
  return (
    <VideoFeed
      frame={frame}
      status={status}
      stats={frameData?.stats || null}
      sourceStats={sourceStats}
      source={source}
      showOverlay={showOverlay}
    />
  );
};

const LiveStats = () => {
  const { frameData } = useFrame();
  return <StatsPanel stats={frameData?.stats || null} timings={frameData?.timings_ms} />;
};

const LiveTracks = ({ connected }) => {
  const { frameData } = useFrame();
  return <TrackList connected={connected} frameData={frameData} />;
};

const EngineMs = () => {
  const { frameData } = useFrame();
  if (frameData?.processing_time_ms == null) return null;
  return <div className="text-[9px] mono text-cyan-600">ENGINE_MS: {Number(frameData.processing_time_ms).toFixed(2)}</div>;
};

// --------------------------------------------------------------------------------------

const PanelToggles = ({ panels, onToggle }) => (
  <div role="group" aria-label="Show or hide panels" className="flex flex-wrap items-center gap-1.5">
    <span className="text-[9px] uppercase font-bold text-white/30 tracking-widest small-caps mr-1">Panels:</span>
    {PANEL_OPTIONS.map(([key, label]) => (
      <button
        key={key}
        type="button"
        aria-pressed={panels[key]}
        onClick={() => onToggle(key)}
        className={`px-2 py-0.5 text-[9px] mono uppercase font-bold border cursor-pointer transition-colors outline-none focus-visible:ring-1 focus-visible:ring-cyan-400 ${
          panels[key]
            ? 'border-cyan-400/50 text-cyan-300 bg-cyan-500/10'
            : 'border-white/10 text-white/30 hover:text-white/60'
        }`}
      >
        {label}
      </button>
    ))}
  </div>
);

const LivePage = () => {
  const { status, connected, alerts } = useFeed();
  const { data: backendStats } = usePoll('/api/stats', 2000, connected);
  const { data: health } = usePoll('/api/health', 5000, connected);
  const [panels, toggle] = usePanelToggles();

  const sideVisible = panels.stats || panels.tracks || panels.alerts;
  const bottomVisible = panels.timeline || panels.zones;

  return (
    <div className="flex flex-col lg:h-full gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <PanelToggles panels={panels} onToggle={toggle} />
        {backendStats && (
          <div className="flex items-center space-x-2">
            <Clock className="w-3.5 h-3.5 text-cyan-800" aria-hidden="true" />
            <span className="text-[11px] mono text-cyan-500/60 font-bold tabular-nums tracking-widest">
              UPTIME::{formatDuration(backendStats.uptime_seconds)}
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        {/* CENTER COLUMN: FEED & CHART */}
        <div className={`col-span-12 ${sideVisible ? 'lg:col-span-8' : ''} flex flex-col gap-4 min-h-0`}>
          <div className="flex-[3] min-h-[320px] lg:min-h-0">
            <LiveVideo
              status={status}
              sourceStats={backendStats?.source_stats}
              source={health?.source}
              showOverlay={panels.overlay}
            />
          </div>
          {bottomVisible && (
            <div className="flex-[1] min-h-[180px] grid grid-cols-3 gap-4">
              {panels.timeline && (
                <div className={`${panels.zones ? 'col-span-3 md:col-span-2' : 'col-span-3'} min-h-[180px]`}>
                  <AlertTimeline alerts={alerts} />
                </div>
              )}
              {panels.zones && (
                <div className={`${panels.timeline ? 'col-span-3 md:col-span-1' : 'col-span-3'} min-h-[180px]`}>
                  <ZonePanel connected={connected} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* SIDE COLUMN: TELEMETRY */}
        {sideVisible && (
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-4 min-h-0">
            {panels.stats && (
              <div className="flex-none">
                <LiveStats />
              </div>
            )}
            {panels.tracks && (
              <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
                <LiveTracks connected={connected} />
              </div>
            )}
            {panels.alerts && (
              <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
                <AlertPanel alerts={alerts} connected={connected} />
              </div>
            )}
          </div>
        )}
      </div>

      {/* FOOTER */}
      <footer className="pt-2 border-t border-white/5 flex flex-wrap gap-2 justify-between items-center opacity-30 select-none">
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
          <EngineMs />
          <div className="flex items-center space-x-2">
            <Terminal className="w-3 h-3 text-cyan-700" aria-hidden="true" />
            <span className="text-[9px] mono text-cyan-700 tracking-tighter font-bold uppercase">
              {health ? `Backend ${health.status}` : status}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LivePage;
