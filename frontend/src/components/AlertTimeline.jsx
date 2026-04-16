import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import DashboardPanel from './DashboardPanel';

const AlertTimeline = ({ alerts }) => {
  const chartData = useMemo(() => {
    const now = Math.floor(Date.now() / 1000);
    const buckets = {};
    for (let i = 0; i < 30; i++) {
      const time = now - (29 - i) * 60;
      const key = new Date(time * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      buckets[key] = 0;
    }
    alerts.forEach(alert => {
      const key = new Date(alert.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      if (buckets.hasOwnProperty(key)) buckets[key]++;
    });
    return Object.entries(buckets).map(([time, count]) => ({ time, count }));
  }, [alerts]);

  return (
    <DashboardPanel title="Temporal Threat Density" headerAction="T_DOM_ANALYTICS" className="h-full">
      <div className="w-full h-full min-h-[140px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#00E5FF" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#ffffff20" 
              fontSize={7} 
              tickLine={false}
              axisLine={false}
              interval={5}
              className="mono"
            />
            <YAxis 
              stroke="#ffffff20" 
              fontSize={7} 
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              className="mono"
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#05111b', borderColor: '#00E5FF33', fontSize: '9px', fontFamily: 'JetBrains Mono' }}
              itemStyle={{ color: '#00E5FF', fontWeight: 'bold' }}
              cursor={{ stroke: '#00E5FF33', strokeWidth: 1 }}
            />
            <Area 
              type="monotone" 
              dataKey="count" 
              stroke="#00E5FF" 
              fillOpacity={1} 
              fill="url(#colorCount)" 
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </DashboardPanel>
  );
};

export default AlertTimeline;
