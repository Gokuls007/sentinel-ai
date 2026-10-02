import { Suspense, useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Shield, Video, ListVideo, Search, ScrollText, ChartColumn, Settings, PersonStanding, GraduationCap,
} from 'lucide-react';
import { MODES, useAppMode } from '../context/appMode';
import StatusBadge from './StatusBadge';
import EmptyState from './EmptyState';
import PageErrorBoundary from './PageErrorBoundary';
import { useFeed } from '../context/liveFeed';
import { loadStored, saveStored, usePoll } from '../lib/api';

const SETTINGS = { to: '/settings', label: 'Settings', title: 'Settings', icon: Settings };
// Each mode shows only its own pages.
const NAV_BY_MODE = {
  posture: [
    { to: '/coach', label: 'Posture Coach', title: 'Desk Posture Coach', icon: PersonStanding, cameraDot: true },
    SETTINGS,
  ],
  warehouse: [
    { to: '/', label: 'Live', title: 'Live', icon: Video, end: true, cameraDot: true },
    { to: '/events', label: 'Events', title: 'Event Log', icon: ListVideo, badge: true },
    { to: '/search', label: 'Search', title: 'Search', icon: Search },
    { to: '/rules', label: 'Rules', title: 'Safety Rules', icon: ScrollText },
    { to: '/analytics', label: 'Analytics', title: 'Analytics', icon: ChartColumn },
    SETTINGS,
  ],
  exam: [
    { to: '/exam', label: 'Exam Hall', title: 'Exam Hall', icon: GraduationCap },
    SETTINGS,
  ],
};
// Pages reachable by URL but not in the current mode's sidebar still get a title.
const EXTRA_TITLES = [
  { to: '/camera', label: 'My Camera', title: 'My Camera' },
  { to: '/demo', label: 'Demo footage', title: 'Demo footage' },
  ...Object.values(NAV_BY_MODE).flat(),
];

const ModeSwitcher = () => {
  const { mode, setMode } = useAppMode();
  const navigate = useNavigate();
  const choose = (id) => {
    if (id === mode) return;
    setMode(id);
    navigate('/');
  };
  return (
    <div role="radiogroup" aria-label="Mode" className="flex border border-cyan-500/30 bg-black/40">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          role="radio"
          aria-checked={mode === m.id}
          onClick={() => choose(m.id)}
          title={m.label}
          className={`px-2.5 sm:px-3 py-1.5 text-[10px] mono uppercase font-bold tracking-wider outline-none focus-visible:ring-1 focus-visible:ring-cyan-400 transition-colors ${
            mode === m.id ? 'bg-cyan-400 text-black' : 'text-white/50 hover:text-cyan-300'
          }`}
        >
          <span className="hidden md:inline">{m.label}</span>
          <span className="md:hidden">{m.short}</span>
        </button>
      ))}
    </div>
  );
};

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

const Sidebar = ({ nav, unseen, cameraOn }) => (
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
      {nav.map(({ to, label, icon, end, badge, cameraDot }) => {
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
              <span className="relative flex-none">
              <Icon className="w-4 h-4" aria-hidden="true" />
              {cameraDot && cameraOn && (
                <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-red-500 animate-pulse ring-2 ring-black" aria-hidden="true" />
              )}
            </span>
              <span className="hidden lg:inline">{label}</span>
            {cameraDot && cameraOn && <span className="sr-only">(camera on)</span>}
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
  const { connected } = useFeed();
  // "Camera on" dot for My Camera: the webcam keeps running when you leave that page.
  const { data: cams } = usePoll('/api/cameras', 5000, connected);
  const cameraOn = Array.isArray(cams) && cams.some((c) => c.id === 'laptop' && c.status === 'running');
  const { mode, demoFootage } = useAppMode();
  const nav = NAV_BY_MODE[mode] || NAV_BY_MODE.warehouse;
  const match = (n) => (n.end || n.to === '/' ? pathname === n.to : pathname.startsWith(n.to));
  const found = nav.find(match) || EXTRA_TITLES.find(match) || null;
  const current = found && found.to === '/' && mode === 'warehouse'
    ? { ...found, title: demoFootage ? 'Live: demo footage' : 'Live: your webcam' }
    : found;

  const pageLabel = current?.label;
  useEffect(() => {
    document.title = pageLabel ? `Sentinel AI - ${pageLabel}` : 'Sentinel AI';
  }, [pageLabel]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0A0A0F] text-white selection:bg-cyan-500/30 font-outfit">
      <Sidebar nav={nav} unseen={unseen} cameraOn={cameraOn} />
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="flex-none flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-cyan-500/10">
          <h1 className="text-base font-bold tracking-[0.35em] text-cyan-400 uppercase leading-none small-caps">
            {current?.title || 'Not found'}
          </h1>
          <div className="flex items-center gap-3">
            <ModeSwitcher />
            <StatusBadge />
          </div>
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
