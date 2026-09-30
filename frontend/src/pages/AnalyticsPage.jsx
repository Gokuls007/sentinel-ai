import { useState } from 'react';
import DashboardPanel from '../components/DashboardPanel';
import EmptyState from '../components/EmptyState';
import HudBarChart from '../components/HudBarChart';
import TimeRangeControl from '../components/TimeRangeControl';
import { useFeed } from '../context/liveFeed';
import { describeError, hourSeries, queryString, useFetch, useNow } from '../lib/api';
import { DEFAULT_RANGE, rangeBounds, rangeToParams } from '../lib/timeRange';
import { SEVERITIES, SEVERITY_HEX } from '../lib/severity';

function useStats(groupBy, start, end, reloadKey) {
  return useFetch(`/api/events/stats${queryString({ group_by: groupBy, start, end })}`, reloadKey);
}

/** counts {key: n} -> [{label, count}] sorted by count desc. */
const byCount = (counts, labelFor = (k) => k) =>
  Object.entries(counts || {})
    .map(([k, n]) => ({ label: labelFor(k), count: n }))
    .sort((a, b) => b.count - a.count);

const ChartCard = ({ title, result, data, className = '', children }) => {
  const total = data.reduce((n, d) => n + d.count, 0);
  let content;
  if (result.error) content = <EmptyState>{describeError(result.error)}</EmptyState>;
  else if (!result.data) content = <EmptyState>Loading...</EmptyState>;
  else if (data.length === 0) content = <EmptyState>No events in range</EmptyState>;
  else content = children;
  return (
    <DashboardPanel title={title} headerAction={result.data ? `${total} EVENTS` : null} className={className}>
      <div className="min-h-[220px] flex flex-col justify-center">{content}</div>
    </DashboardPanel>
  );
};

const AnalyticsPage = () => {
  const { status } = useFeed();
  const [range, setRange] = useState(DEFAULT_RANGE);
  const now = useNow(60000);
  const { start, end } = rangeToParams(range, now);

  const { data: meta } = useFetch('/api/meta', status);
  const zoneNames = new Map((meta?.zones || []).map((z) => [z.id, z.name || z.id]));

  const perHour = useStats('hour_bucket', start, end, status);
  const perZone = useStats('zone', start, end, status);
  const perType = useStats('type', start, end, status);
  const perSeverity = useStats('severity', start, end, status);

  const hourData = hourSeries(perHour.data?.counts, 24 * 31, rangeBounds(start, end, now));
  const zoneData = byCount(perZone.data?.counts, (k) => (k === '' ? 'No zone' : zoneNames.get(k) || k));
  const typeData = byCount(perType.data?.counts, (k) => k.replace(/_/g, ' '));
  const sevCounts = perSeverity.data?.counts || {};
  // Fixed low -> critical order and status colors; unknown severities appended.
  const sevData = [...SEVERITIES, ...Object.keys(sevCounts).filter((k) => !SEVERITIES.includes(k))]
    .filter((k) => sevCounts[k])
    .map((k) => ({ label: k, count: sevCounts[k], fill: SEVERITY_HEX[k] }));

  return (
    <div className="space-y-4">
      <DashboardPanel title="Range" headerAction="APPLIES TO ALL CHARTS">
        <TimeRangeControl value={range} onChange={setRange} />
      </DashboardPanel>

      <ChartCard title="Events per hour" result={perHour} data={hourData}>
        <HudBarChart data={hourData} height={220} />
      </ChartCard>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <ChartCard title="Events per zone" result={perZone} data={zoneData}>
          <HudBarChart data={zoneData} horizontal height={Math.max(160, zoneData.length * 28)} />
        </ChartCard>
        <ChartCard title="Events per type" result={perType} data={typeData}>
          <HudBarChart data={typeData} horizontal height={Math.max(160, typeData.length * 28)} />
        </ChartCard>
        <ChartCard title="Events per severity" result={perSeverity} data={sevData}>
          <HudBarChart data={sevData} height={200} />
        </ChartCard>
      </div>

      <DashboardPanel title="Ergonomic risk trends" headerAction="PHASE 1">
        <div className="min-h-[120px] flex items-center justify-center border border-dashed border-white/10">
          <p className="text-[10px] mono uppercase tracking-widest text-white/40 text-center px-4">
            Placeholder: ergonomic risk trends arrive in Phase 1. No data is shown here yet.
          </p>
        </div>
      </DashboardPanel>
    </div>
  );
};

export default AnalyticsPage;
