import React from 'react';
import DashboardPanel from './DashboardPanel';

const VideoFeed = ({ frame, connected, stats, demoMode }) => {
  return (
    <DashboardPanel className="h-full overflow-hidden" title="Live Observation Terminal" headerAction="SEC-01 // ACTIVE">
      <div className="relative w-full h-full bg-black/60 rounded-sm overflow-hidden group">
        {frame ? (
          <img 
            src={`data:image/jpeg;base64,${frame}`} 
            className="w-full h-full object-contain"
            alt="Sentinel Stream"
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full space-y-4">
            <div className="relative">
              <div className={`w-16 h-16 border-2 rounded-full ${stats?.source_stats?.hardware_error ? 'border-red-500/40' : 'border-cyan-500/20'}`} />
              <div className={`absolute top-0 w-16 h-16 border-2 rounded-full animate-spin ${stats?.source_stats?.hardware_error ? 'border-t-red-500' : 'border-t-cyan-400'}`} />
            </div>
            <div className="text-center px-6">
              <p className={`mono tracking-[0.4em] animate-pulse text-sm font-bold ${stats?.source_stats?.hardware_error ? 'text-red-500' : 'text-cyan-400'}`}>
                {!connected ? "LINK_SEVERED" : stats?.source_stats?.hardware_error ? "HARDWARE_FAILURE" : "SYNCING_SIGNAL..."}
              </p>
              {stats?.source_stats?.hardware_error && (
                <p className="text-[10px] mono text-red-400/60 uppercase mt-2 max-w-xs">
                  {stats?.source_stats?.error_message || "Critical Error: Camera source inaccessible or locked by another process."}
                </p>
              )}
            </div>
          </div>
        )}

        {/* Cinematic Overlays */}
        <div className="absolute inset-0 scan-line scan-animate opacity-30 pointer-events-none" />
        
        {/* Reticle Focus Effect */}
        <div className="absolute inset-0 border-[20px] border-black/20 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-cyan-500/10 rounded-full pointer-events-none" />
        
        {/* HUD Data Overlays */}
        <div className="absolute bottom-4 left-4 flex space-x-2">
          <div className={`px-2 py-0.5 border text-[9px] mono font-bold uppercase tracking-widest ${demoMode ? 'bg-amber-950/80 border-amber-400/30 text-amber-400' : 'bg-cyan-950/80 border-cyan-400/30 text-cyan-400'}`}>
            Source: {demoMode ? 'Demo_Simulation' : 'Neural_Cam_v2'}
          </div>
          <div className="px-2 py-0.5 bg-red-950/80 border border-red-500/30 text-[9px] text-red-400 mono font-bold uppercase tracking-widest animate-pulse">
            Recording
          </div>
        </div>

        {/* Track Count Overlay */}
        <div className="absolute top-4 right-4 text-right">
           <div className="text-[10px] mono text-cyan-500/70 font-bold uppercase">Signal Density</div>
           <div className="text-xl mono font-bold text-white tracking-tighter">100%</div>
        </div>
      </div>
    </DashboardPanel>
  );
};

export default VideoFeed;
