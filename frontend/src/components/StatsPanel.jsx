import React from 'react';
import { Users, Crosshair, Zap, AlertTriangle } from 'lucide-react';
import DashboardPanel from './DashboardPanel';

const StatCard = ({ icon: Icon, value, label, color = "cyan" }) => (
  <div className="p-4 bg-cyan-950/20 border-l border-cyan-500/20">
    <div className="flex items-center space-x-2 mb-1">
      <Icon className={`w-3 h-3 text-${color}-400 opacity-60`} />
      <span className={`text-[9px] uppercase font-bold text-${color}-500/70 tracking-widest outfit`}>
        {label}
      </span>
    </div>
    <div className="flex items-baseline">
      <span className="text-2xl font-bold mono text-white leading-none">
        {value}
      </span>
    </div>
  </div>
);

const StatsPanel = ({ stats }) => {
  return (
    <DashboardPanel title="Nexus Statistics" headerAction="SYS_DENSITY">
      <div className="grid grid-cols-2 gap-px bg-cyan-500/10 border border-cyan-500/10">
        <StatCard 
          icon={Users} 
          value={stats.person_count || 0} 
          label="Persons" 
        />
        <StatCard 
          icon={Crosshair} 
          value={stats.active_tracks || 0} 
          label="Tracks" 
        />
        <StatCard 
          icon={Zap} 
          value={`${Math.round(stats.processing_time_ms || 0)}ms`} 
          label="Latency" 
        />
        <StatCard 
          icon={AlertTriangle} 
          value={stats.alert_count || 0} 
          label="Total Vol" 
          color="red"
        />
      </div>
    </DashboardPanel>
  );
};

export default StatsPanel;
