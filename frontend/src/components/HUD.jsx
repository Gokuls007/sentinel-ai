import React, { useState, useEffect } from 'react';
import useWebSocket from '../hooks/useWebSocket';
import VideoFeed from './VideoFeed';
import StatsPanel from './StatsPanel';
import TrackList from './TrackList';
import AlertPanel from './AlertPanel';
import AlertTimeline from './AlertTimeline';
import { Shield, Wifi, WifiOff, Clock, Terminal } from 'lucide-react';

const HUD = () => {
  const { frame, frameData, alerts, connected, stats } = useWebSocket();
  const [uptime, setUptime] = useState("00:00:00");

  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      const diff = Math.floor((Date.now() - start) / 1000);
      const h = String(Math.floor(diff / 3600)).padStart(2, '0');
      const m = String(Math.floor((diff % 3600) / 60)).padStart(2, '0');
      const s = String(diff % 60).padStart(2, '0');
      setUptime(`${h}:${m}:${s}`);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

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
              Sector Intelligence Engine // v1.2.4
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-12">
          <div className="flex items-center space-x-4">
            <span className="text-[10px] uppercase font-bold text-white/30 tracking-widest small-caps">Link Status:</span>
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-sm border ${connected ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
              <span className={`text-[10px] font-bold uppercase mono tracking-tighter ${connected ? 'text-green-400' : 'text-red-500'}`}>
                {connected ? "LINK_ESTABLISHED" : "LINK_FAILURE"}
              </span>
            </div>
          </div>
          
          <div className="hidden lg:flex items-center space-x-4">
            <Clock className="w-3.5 h-3.5 text-cyan-800" />
            <span className="text-[11px] mono text-cyan-500/60 font-bold tabular-nums tracking-widest">
              UPTIME::{uptime}
            </span>
          </div>
        </div>
      </header>

      {/* --- DASHBOARD CORE --- */}
      <main className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        
        {/* CENTER COLUMN: FEED & CHART (65%) */}
        <div className="col-span-12 lg:col-span-8 flex flex-col space-y-4 min-h-0">
          <div className="flex-[3] min-h-0">
            <VideoFeed frame={frame} connected={connected} stats={stats} />
          </div>
          <div className="flex-[1] min-h-0 min-h-[180px]">
             <AlertTimeline alerts={alerts} />
          </div>
        </div>

        {/* SIDE COLUMN: TELEMETRY (35%) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-4 min-h-0">
          <div className="flex-none">
            <StatsPanel stats={stats} />
          </div>
          <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
            <TrackList frameData={frameData} />
          </div>
          <div className="flex-1 min-h-0 border-t border-cyan-500/5 pt-2">
            <AlertPanel alerts={alerts} />
          </div>
        </div>

      </main>

      {/* --- SITE FOOTER --- */}
      <footer className="mt-4 pt-2 border-t border-white/5 flex justify-between items-center opacity-30 select-none">
        <div className="flex items-center space-x-6">
          <div className="text-[9px] uppercase font-bold tracking-[0.3em] text-cyan-600 mono">
            Neural Architecture // Sector 7 Access
          </div>
          <div className="h-3 w-px bg-white/10" />
          <div className="text-[9px] uppercase font-medium text-white/40 tracking-widest">
            Authorization: Restricted
          </div>
        </div>
        
        <div className="flex items-center space-x-6">
           {frameData?.processing_time_ms && (
             <div className="text-[9px] mono text-cyan-600">
               ENGINE_MS: {frameData.processing_time_ms.toFixed(2)}
             </div>
           )}
           <div className="flex items-center space-x-2">
             <Terminal className="w-3 h-3 text-cyan-700" />
             <span className="text-[9px] mono text-cyan-700 tracking-tighter font-bold uppercase">Ready</span>
           </div>
        </div>
      </footer>

    </div>
  );
};

export default HUD;
