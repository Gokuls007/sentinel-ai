import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TOOLTIP = {
  contentStyle: { backgroundColor: '#05111b', borderColor: '#00E5FF33', fontSize: '10px', fontFamily: 'JetBrains Mono' },
  labelStyle: { color: '#ffffffaa' },
  itemStyle: { color: '#00E5FF', fontWeight: 'bold' },
  cursor: { fill: '#00E5FF10' },
};

const AXIS = { stroke: '#ffffff40', fontSize: 9, tickLine: false, axisLine: false, className: 'mono' };

/**
 * Single-series bar chart in the HUD style.
 * data: [{ label, count, fill? }]. `horizontal` puts categories on the y-axis
 * (better for long category names).
 */
const HudBarChart = ({ data, horizontal = false, height = 220, name = 'events' }) => {
  const radius = horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0];
  const longest = Math.max(0, ...data.map((d) => String(d.label).length));
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 1, height: 1 }}>
        <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={horizontal} horizontal={!horizontal} />
          {horizontal ? (
            <>
              <XAxis type="number" allowDecimals={false} {...AXIS} />
              <YAxis type="category" dataKey="label" width={Math.min(140, 12 + longest * 6)} interval={0} {...AXIS} />
            </>
          ) : (
            <>
              <XAxis dataKey="label" minTickGap={16} {...AXIS} />
              <YAxis allowDecimals={false} width={32} {...AXIS} />
            </>
          )}
          <Tooltip {...TOOLTIP} />
          <Bar dataKey="count" name={name} fill="#00E5FF" fillOpacity={0.75} radius={radius} isAnimationActive={false} maxBarSize={36}>
            {data.map((d) => (
              <Cell key={d.label} fill={d.fill || '#00E5FF'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default HudBarChart;
