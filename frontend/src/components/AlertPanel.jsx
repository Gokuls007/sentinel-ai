import React, { useEffect, useRef } from 'react';
import DashboardPanel from './DashboardPanel';

const SEVERITY_COLORS = {
  critical: "text-red-500 bg-red-950/20 border-red-500/50 glow-red",
  high: "text-orange-500 bg-orange-950/20 border-orange-500/50",
  medium: "text-yellow-500 bg-yellow-950/20 border-yellow-500/50",
  low: "text-cyan-500 bg-cyan-950/20 border-cyan-500/50"
};

const AlertPanel = ({ alerts }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) containerRef.current.scrollTop = 0;
  }, [alerts]);

  return (
    <DashboardPanel title="Anomaly Feed" headerAction="LIVE_LOG" severity="red" className="h-full">
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-red-900/40 h-64"
      >
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full opacity-30">
            <p className="text-[10px] mono uppercase tracking-widest text-center">Threat assessment stable...</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <div 
              key={alert.alert_id} 
              className={`p-2 border-l-2 ${SEVERITY_COLORS[alert.severity] || SEVERITY_COLORS.low} transition-all`}
            >
              <div className="flex justify-between items-baseline mb-0.5">
                <span className="text-[9px] font-bold mono uppercase opacity-50 outfit">
                  {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                </span>
                <span className="text-[8px] font-extrabold tracking-widest uppercase small-caps">
                  LEVEL::{alert.severity}
                </span>
              </div>
              <div className="text-[10px] uppercase font-bold tracking-tight outfit">
                {alert.message}
              </div>
              <div className="flex justify-between items-center mt-1.5 opacity-40">
                <span className="text-[8px] mono">TID::{alert.track_id}</span>
                <span className="text-[8px] mono uppercase">{alert.alert_id}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </DashboardPanel>
  );
};

export default AlertPanel;
