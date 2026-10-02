import { lazy } from 'react';
import { BrowserRouter, Link, Navigate, Route, Routes } from 'react-router-dom';
import WebSocketProvider from './context/WebSocketProvider';
import AppModeProvider from './context/AppModeProvider';
import { useAppMode } from './context/appMode';
import Layout from './components/Layout';
import LivePage from './pages/LivePage';

// Non-live pages load on demand to keep the initial bundle small.
const CameraPage = lazy(() => import('./pages/CameraPage'));
const EventsPage = lazy(() => import('./pages/EventsPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const SearchPage = lazy(() => import('./pages/SearchPage'));
const RulesPage = lazy(() => import('./pages/RulesPage'));
const PostureCoachPage = lazy(() => import('./pages/PostureCoachPage'));
const ExamPage = lazy(() => import('./pages/ExamPage'));

const NotFound = () => (
  <div className="p-6 mono text-[11px] uppercase tracking-widest text-white/50">
    Page not found. <Link to="/" className="text-cyan-400 underline">Back to the start</Link>
  </div>
);

/** "/" opens each mode's main page. Warehouse shows the live webcam, or the demo footage when
 * that is switched on in Settings. */
const ModeHome = () => {
  const { mode, demoFootage } = useAppMode();
  if (mode === 'posture') return <Navigate to="/coach" replace />;
  if (mode === 'exam') return <Navigate to="/exam" replace />;
  return demoFootage ? <LivePage /> : <CameraPage />;
};

function App() {
  return (
    <BrowserRouter>
      <AppModeProvider>
        <WebSocketProvider>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<ModeHome />} />
              <Route path="coach" element={<PostureCoachPage />} />
              <Route path="exam" element={<ExamPage />} />
              <Route path="demo" element={<LivePage />} />
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
      </AppModeProvider>
    </BrowserRouter>
  );
}

export default App;
