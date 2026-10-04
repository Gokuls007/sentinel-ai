import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import DashboardPanel from './DashboardPanel';
import { apiUrl, useNow } from '../lib/api';
import { eventTypeLabel, isTestAlert } from '../lib/eventTypes';

const SEVERITY_COLORS = {
  critical: 'text-red-500 bg-red-950/20 border-red-500/50 glow-red',
  high: 'text-orange-500 bg-orange-950/20 border-orange-500/50',
  medium: 'text-yellow-500 bg-yellow-950/20 border-yellow-500/50',
  low: 'text-cyan-500 bg-cyan-950/20 border-cyan-500/50',
};

// The backend finishes writing the incident clip ~5 s after the alert fires.
const CLIP_READY_AFTER_S = 6;

const AlertItem = ({ alert, nowMs }) => {
  const [snapshotFailed, setSnapshotFailed] = useState(false);
  const [showClip, setShowClip] = useState(false);
  const [clipFailed, setClipFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);

  const id = encodeURIComponent(alert.alert_id);
  // A new query string on retry, so the browser doesn't reuse a cached 404.
  const clipUrl = apiUrl(`/api/clips/${id}`) + (attempt ? `?try=${attempt}` : '');
  const ageS = nowMs / 1000 - (alert.timestamp || 0);
  const test = isTestAlert(alert); // test footage: nothing is stored, so no snapshot, clip or details
  const clipLikely = !test && (alert.has_clip || ageS >= CLIP_READY_AFTER_S);

  return (
    <div className={`p-2 border-l-2 ${SEVERITY_COLORS[alert.severity] || SEVERITY_COLORS.low} transition-all`}>
      <div className="flex gap-2">
        {!snapshotFailed && !test && (
          <img
            src={apiUrl(`/api/snapshots/${id}`)}
            alt=""
            loading="lazy"
            onError={() => setSnapshotFailed(true)}
            className="w-14 h-10 object-cover flex-none border border-white/10 bg-black"
          />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex justify-between items-baseline mb-0.5">
            <span className="text-[9px] font-bold mono uppercase opacity-50 outfit">
              {new Date(alert.timestamp * 1000).toLocaleTimeString()}
            </span>
            <span className="text-[8px] font-extrabold tracking-widest uppercase small-caps">
              {test && (
                <span className="mr-1.5 px-1 border border-amber-400/60 text-amber-300" title="From test footage: not stored, not notified">
                  TEST
                </span>
              )}
              {eventTypeLabel(alert.alert_type)} // {alert.severity}
            </span>
          </div>
          <div className="text-[10px] uppercase font-bold tracking-tight outfit break-words">
            {alert.message}
          </div>
          <div className="flex justify-between items-center mt-1.5">
            <span className="text-[8px] mono opacity-40">
              TID::{alert.track_id ?? '--'} // {alert.alert_id}
              {typeof alert.confidence === 'number' && ` // ${(alert.confidence * 100).toFixed(0)}%`}
            </span>
            <span className="flex items-center gap-1.5">
            {alert.event_id != null && !test && (
              <Link
                to={`/events?event=${encodeURIComponent(alert.event_id)}`}
                className="text-[8px] mono uppercase font-bold px-1.5 py-px border border-current opacity-70 hover:opacity-100"
              >
                Details
              </Link>
            )}
            {clipLikely && !clipFailed && (
              <button
                type="button"
                onClick={() => setShowClip((v) => !v)}
                className="text-[8px] mono uppercase font-bold px-1.5 py-px border border-current opacity-70 hover:opacity-100 cursor-pointer"
              >
                {showClip ? 'Hide clip' : 'View clip'}
              </button>
            )}
            </span>
          </div>
        </div>
      </div>
      {showClip && !clipFailed && (
        <div className="mt-2">
          <video
            src={clipUrl}
            controls
            autoPlay
            muted
            playsInline
            onError={() => setClipFailed(true)}
            className="w-full max-h-48 bg-black border border-white/10"
          />
          <a href={clipUrl} target="_blank" rel="noreferrer" className="text-[8px] mono uppercase opacity-60 hover:opacity-100 underline">
            Open clip in new tab
          </a>
        </div>
      )}
      {clipFailed && (
        <div className="flex items-center gap-2 text-[8px] mono uppercase mt-1">
          <span className="opacity-50">Clip not ready yet (still recording)</span>
          <button
            type="button"
            onClick={() => { setClipFailed(false); setShowClip(true); setAttempt((n) => n + 1); }}
            className="font-bold px-1.5 py-px border border-current opacity-70 hover:opacity-100 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
};

const AlertPanel = ({ alerts, connected }) => {
  const containerRef = useRef(null);
  const nowMs = useNow(2000);
  const newestId = alerts[0]?.alert_id;

  useEffect(() => {
    if (containerRef.current) containerRef.current.scrollTop = 0;
  }, [newestId]);

  return (
    <DashboardPanel title="Anomaly Feed" headerAction={`${alerts.length} ALERTS`} severity="red" className="h-full">
      <div ref={containerRef} className="flex-1 overflow-y-auto space-y-2 pr-2 h-64">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full opacity-30">
            <p className="text-[10px] mono uppercase tracking-widest text-center">
              {connected ? 'No alerts yet' : 'Backend offline'}
            </p>
          </div>
        ) : (
          alerts.map((alert) => <AlertItem key={alert.alert_id} alert={alert} nowMs={nowMs} />)
        )}
      </div>
    </DashboardPanel>
  );
};

export default AlertPanel;
