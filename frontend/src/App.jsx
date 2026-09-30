import { lazy } from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import WebSocketProvider from './context/WebSocketProvider';
import Layout from './components/Layout';
import LivePage from './pages/LivePage';

// Non-live pages load on demand to keep the initial bundle small.
const CameraPage = lazy(() => import('./pages/CameraPage'));
const EventsPage = lazy(() => import('./pages/EventsPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const SearchPage = lazy(() => import('./pages/SearchPage'));
const RulesPage = lazy(() => import('./pages/RulesPage'));

const NotFound = () => (
  <div className="p-6 mono text-[11px] uppercase tracking-widest text-white/50">
    Page not found. <Link to="/" className="text-cyan-400 underline">Back to Live</Link>
  </div>
);

function App() {
  return (
    <BrowserRouter>
      <WebSocketProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<LivePage />} />
            <Route path="camera" element={<CameraPage />} />
            <Route path="events" element={<EventsPage />} />
            <Route path="search" element={<SearchPage />} />
            <Route path="rules" element={<RulesPage />} />
            <Route path="analytics" element={<AnalyticsPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </WebSocketProvider>
    </BrowserRouter>
  );
}

export default App;
