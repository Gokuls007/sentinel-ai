import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import DashboardPanel from './DashboardPanel';
import { useNow } from '../lib/api';

const MINUTES = 30;

const AlertTimeline = ({ alerts }) => {
  // Ticking clock so buckets roll forward even when no new alerts arrive.
  const now = useNow(10000);

  const chartData = useMemo(() => {
    const currentMinute = Math.floor(now / 60000);
    const counts = new Array(MINUTES).fill(0);
    for (const alert of alerts) {
      if (typeof alert.timestamp !== 'number') continue;
      const age = currentMinute - Math.floor(alert.timestamp / 60);
      if (age >= 0 && age < MINUTES) counts[MINUTES - 1 - age] += 1;
    }
    return counts.map((count, i) => ({
      time: new Date((currentMinute - (MINUTES - 1 - i)) * 60000)
        .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      count,
    }));
  }, [alerts, now]);

  return (
    <DashboardPanel title="Temporal Threat Density" headerAction={`LAST ${MINUTES} MIN`} className="h-full">
      <div className="w-full h-full min-h-[140px]">
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 1, height: 1 }}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00E5FF" stopOpacity={0} />
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
              name="alerts"
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
