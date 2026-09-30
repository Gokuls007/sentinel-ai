import { BACKEND_START_HINT } from '../lib/api';
import { useFeed } from '../context/liveFeed';

const STATUS_STYLES = {
  live: {
    box: 'border-green-500/30 bg-green-500/5',
    dot: 'bg-green-400 animate-pulse',
    text: 'text-green-400',
    label: 'LIVE',
  },
  connecting: {
    box: 'border-amber-500/30 bg-amber-500/5',
    dot: 'bg-amber-400 animate-pulse',
    text: 'text-amber-400',
    label: 'CONNECTING',
  },
  disconnected: {
    box: 'border-red-500/30 bg-red-500/5',
    dot: 'bg-red-500',
    text: 'text-red-500',
    label: 'BACKEND_OFFLINE',
  },
};

/** Connection badge with a hover/focus tooltip (feed URL, start hint when offline). */
const StatusBadge = () => {
  const { status, lastError, url } = useFeed();
  const style = STATUS_STYLES[status] || STATUS_STYLES.disconnected;
  return (
    <div className="flex items-center space-x-4">
      <span className="hidden sm:inline text-[10px] uppercase font-bold text-white/30 tracking-widest small-caps">Link Status:</span>
      <div
        tabIndex={0}
        role="status"
        aria-label={`Backend link: ${style.label}`}
        className={`relative group flex items-center space-x-2 px-3 py-1 rounded-sm border outline-none focus-visible:ring-1 focus-visible:ring-cyan-400 ${style.box}`}
      >
        <div className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
        <span className={`text-[10px] font-bold uppercase mono tracking-tighter ${style.text}`}>
          {style.label}
        </span>
        <div className="absolute top-full right-0 mt-2 w-72 p-2 bg-black/95 border border-cyan-500/20 rounded-sm opacity-0 group-hover:opacity-100 group-focus:opacity-100 transition-opacity z-50 pointer-events-none">
          <p className="text-[9px] text-cyan-300/80 mono leading-relaxed break-all">Feed: {url}</p>
          {status !== 'live' && (
            <p className="text-[9px] text-amber-300/80 mono leading-relaxed mt-1">
              {lastError ? `${lastError}. ` : ''}Start the backend with `{BACKEND_START_HINT}`.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatusBadge;
