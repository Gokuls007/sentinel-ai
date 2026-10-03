import { useState } from 'react';
import { Square } from 'lucide-react';
import { describeError } from '../lib/api';
import { stopLaptopCamera } from '../lib/laptopCamera';

/** Stops the laptop webcam (releases the device, so the camera light goes off) and turns off
 * auto-start: it only starts again when you press Start. `onStop` overrides the request (a page
 * that already has camera controls passes its own). */
const StopCameraButton = ({ onStop, busy: outerBusy = false, className = '' }) => {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const stop = async () => {
    setBusy(true);
    setError(null);
    try {
      await (onStop ? onStop() : stopLaptopCamera());
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <button type="button" onClick={stop} disabled={busy || outerBusy}
        className="px-3 py-1.5 text-[10px] mono uppercase font-bold tracking-widest border border-red-400/70 text-red-200 bg-red-950/40 hover:bg-red-900/50 disabled:opacity-40 flex items-center gap-1.5">
        <Square className="w-3 h-3 fill-current" aria-hidden="true" />
        {busy || outerBusy ? 'Stopping...' : 'Stop camera'}
      </button>
      {error && <span className="text-[10px] text-red-300">{error}</span>}
    </span>
  );
};

export default StopCameraButton;
