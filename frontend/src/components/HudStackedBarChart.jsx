import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TOOLTIP = {
  contentStyle: { backgroundColor: '#05111b', borderColor: '#00E5FF33', fontSize: '10px', fontFamily: 'JetBrains Mono' },
  labelStyle: { color: '#ffffffaa' },
  cursor: { fill: '#00E5FF10' },
};

const AXIS = { stroke: '#ffffff40', fontSize: 9, tickLine: false, axisLine: false, className: 'mono' };

/** Swatch legend for the series (HTML, so it wraps on narrow screens). */
export const StackLegend = ({ series }) => (
  <ul className="flex flex-wrap gap-x-3 gap-y-1 text-[9px] mono uppercase text-white/60" aria-label="Legend">
    {series.map((s) => (
      <li key={s.key} className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 flex-none" style={{ background: s.color }} aria-hidden="true" />
        {s.label}
      </li>
    ))}
  </ul>
);

/**
 * Stacked bar chart in the HUD style.
 * data: [{ label, [series.key]: number }]; series: [{ key, label, color }] bottom/left first.
 * `horizontal` puts categories on the y-axis. `formatValue` formats tooltip values,
 * `formatTick` the value axis.
 */
const HudStackedBarChart = ({
  data, series, horizontal = false, height = 220, formatValue = (v) => v, formatTick = (v) => v,
}) => {
  const longest = Math.max(0, ...data.map((d) => String(d.label).length));
  const last = series.length - 1;
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 1, height: 1 }}>
        <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={horizontal} horizontal={!horizontal} />
          {horizontal ? (
            <>
              <XAxis type="number" tickFormatter={formatTick} {...AXIS} />
              <YAxis type="category" dataKey="label" width={Math.min(160, 12 + longest * 6)} interval={0} {...AXIS} />
            </>
          ) : (
            <>
              <XAxis dataKey="label" interval={0} {...AXIS} />
              <YAxis width={40} tickFormatter={formatTick} {...AXIS} />
            </>
          )}
          <Tooltip
            {...TOOLTIP}
            formatter={(v, name) => [formatValue(v), name]}
            itemSorter={() => 0}
          />
          {series.map((s, i) => (
            <Bar
              key={s.key}
              dataKey={s.key}
              name={s.label}
              stackId="stack"
              fill={s.color}
              fillOpacity={0.85}
              isAnimationActive={false}
              maxBarSize={horizontal ? 28 : 24}
              radius={i === last ? (horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]) : 0}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default HudStackedBarChart;
