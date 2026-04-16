import React from 'react';
import DashboardPanel from './DashboardPanel';

const TrackList = ({ frameData }) => {
  const detections = frameData?.detections?.detections || [];

  return (
    <DashboardPanel title="Target Registry" headerAction="LIVE_IDENT" className="h-full">
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-cyan-900/40 h-64">
        {detections.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full opacity-30">
            <p className="text-[10px] mono uppercase tracking-widest text-center">Scanning sectors...</p>
          </div>
        ) : (
          detections.map((det) => (
            <div 
              key={det.track_id} 
              className="group flex items-center justify-between p-2 bg-white/5 hover:bg-cyan-500/10 border-l-2 border-transparent hover:border-cyan-400 transition-all cursor-default"
            >
              <div className="flex items-center space-x-3">
                <span className="text-[11px] font-bold mono text-cyan-400">
                  ID::{String(det.track_id).padStart(3, '0')}
                </span>
                <span className="text-[9px] uppercase font-bold text-gray-500 tracking-tighter outfit">
                  {det.class_name}
                </span>
              </div>
              <div className="flex flex-col items-end">
                <div className="text-[9px] mono text-cyan-600 font-bold">
                  {(det.confidence * 100).toFixed(1)}%
                </div>
                {det.speed !== undefined && (
                   <div className="text-[8px] mono text-gray-600">
                     S::{det.speed.toFixed(1)}
                   </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </DashboardPanel>
  );
};

export default TrackList;
