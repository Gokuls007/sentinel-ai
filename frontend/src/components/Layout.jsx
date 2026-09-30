import { Suspense, useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Shield, Video, ListVideo, Search, ScrollText, ChartColumn, Settings } from 'lucide-react';
import StatusBadge from './StatusBadge';
import EmptyState from './EmptyState';
import PageErrorBoundary from './PageErrorBoundary';
import { useFeed } from '../context/liveFeed';
import { loadStored, saveStored } from '../lib/api';

const NAV = [
  { to: '/', label: 'Live', title: 'Live Observation', icon: Video, end: true },
  { to: '/events', label: 'Events', title: 'Event Log', icon: ListVideo, badge: true },
  { to: '/search', label: 'Search', title: 'Search', icon: Search },
  { to: '/rules', label: 'Rules', title: 'Safety Rules', icon: ScrollText },
  { to: '/analytics', label: 'Analytics', title: 'Analytics', icon: ChartColumn },
  { to: '/settings', label: 'Settings', title: 'Settings', icon: Settings },
];

const LAST_SEEN_KEY = 'sentinel.events.lastSeen';

/** Count of alerts newer than the last visit to /events (shown as a sidebar badge). */
function useUnseenAlerts(onEvents) {
  const { alerts } = useFeed();
  // First ever visit: start counting from now rather than flagging all history as new.
  const [lastSeen, setLastSeen] = useState(() => loadStored(LAST_SEEN_KEY, Date.now() / 1000));
  const newest = alerts[0]?.timestamp || 0;

  // Adjust state during render (React's recommended alternative to an effect here).
  if (onEvents && newest > lastSeen) setLastSeen(newest);

  useEffect(() => {
    saveStored(LAST_SEEN_KEY, lastSeen);
  }, [lastSeen]);

  if (onEvents) return 0;
  let n = 0;
  for (const a of alerts) {
    if ((a.timestamp || 0) > lastSeen) n += 1;
    else break; // alerts are newest first
  }
  return n;
}

const Sidebar = ({ unseen }) => (
  <nav aria-label="Main" className="flex-none w-14 lg:w-52 flex flex-col border-r border-cyan-500/10 bg-black/40 backdrop-blur-md">
    <div className="flex items-center gap-3 px-3 lg:px-4 py-4 border-b border-cyan-500/10">
      <div className="relative group flex-none">
        <div className="absolute inset-0 bg-cyan-400/20 blur-lg group-hover:bg-cyan-400/40 transition-all" />
        <div className="relative p-1.5 border border-cyan-400/40 bg-black">
          <Shield className="w-5 h-5 text-cyan-400" aria-hidden="true" />
        </div>
      </div>
      <div className="hidden lg:block min-w-0">
        <div className="text-sm font-bold tracking-[0.3em] text-cyan-400 uppercase leading-none small-caps">Sentinel AI</div>
        <div className="text-[7px] text-cyan-700/80 font-bold tracking-[0.35em] uppercase mt-1.5 mono">Video Anomaly Detection</div>
      </div>
    </div>
    <ul className="flex-1 py-3 space-y-1">
      {NAV.map(({ to, label, icon, end, badge }) => {
        const Icon = icon;
        return (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              title={label}
              className={({ isActive }) =>
                `relative flex items-center gap-3 mx-2 px-2.5 py-2 border-l-2 text-[11px] font-bold uppercase tracking-[0.2em] small-caps outline-none transition-all focus-visible:ring-1 focus-visible:ring-cyan-400 ${
                  isActive
                    ? 'border-cyan-400 bg-cyan-500/10 text-cyan-300'
                    : 'border-transparent text-white/40 hover:text-cyan-300 hover:bg-white/5'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-none" aria-hidden="true" />
              <span className="hidden lg:inline">{label}</span>
              {badge && unseen > 0 && (
                <span
                  className="absolute lg:static top-0.5 right-0.5 lg:ml-auto min-w-[18px] px-1 py-px text-center text-[9px] mono font-bold bg-red-600 text-white rounded-sm tracking-normal"
                  aria-label={`${unseen} new alerts`}
                >
                  {unseen > 99 ? '99+' : unseen}
                </span>
              )}
            </NavLink>
          </li>
        );
      })}
    </ul>
  </nav>
);

const Layout = () => {
  const { pathname } = useLocation();
  const onEvents = pathname.startsWith('/events');
  const unseen = useUnseenAlerts(onEvents);
  const current = NAV.find((n) => (n.end ? pathname === n.to : pathname.startsWith(n.to))) || null;

  useEffect(() => {
    document.title = current ? `Sentinel AI - ${current.label}` : 'Sentinel AI';
  }, [current]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0A0A0F] text-white selection:bg-cyan-500/30 font-outfit">
      <Sidebar unseen={unseen} />
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="flex-none flex items-center justify-between px-4 py-3 border-b border-cyan-500/10">
          <h1 className="text-base font-bold tracking-[0.35em] text-cyan-400 uppercase leading-none small-caps">
            {current?.title || 'Not found'}
          </h1>
          <StatusBadge />
        </header>
        <main className="flex-1 min-h-0 overflow-y-auto p-4">
          <PageErrorBoundary key={pathname}>
            <Suspense fallback={<EmptyState>Loading...</EmptyState>}>
              <Outlet />
            </Suspense>
          </PageErrorBoundary>
        </main>
      </div>
    </div>
  );
};

export default Layout;
