import { useEffect, useState } from 'react';
import { Film, Square } from 'lucide-react';
import DashboardPanel from './DashboardPanel';
import ErgonomicsPanel from './ErgonomicsPanel';
import VideoFeed from './VideoFeed';
import { ActivityPanel, ViewBanner } from './ActivityPanel';
import { describeError, fetchJson, postJson } from '../lib/api';
import { useCameraFrames } from '../lib/laptopCamera';
import { chipClass, inputClass } from '../lib/ui';

const TEST = 'test';

const TestFeed = ({ label }) => {
  const { status, frame, frameData } = useCameraFrames(TEST);
  const live = status === 'live';
  return (
    <div className="space-y-3">
      <ViewBanner view={frameData?.view} />
      <div className="h-[min(56vh,560px)] min-h-[300px]">
        <VideoFeed title="Test footage" frame={frame} status={status} stats={frameData?.stats || null} source={label} />
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <ActivityPanel connected={live} activity={frameData?.activity} ergonomics={frameData?.ergonomics} />
        <ErgonomicsPanel connected={live} ergonomics={frameData ? frameData.ergonomics : null} />
      </div>
    </div>
  );
};

/** "Test with demo footage": plays a local clip (CAUCAFall subjects 1-5, or your own recording) as a
 * separate "test" camera in warehouse mode, so activity labels and REBA can be seen working. */
const DemoFootagePanel = () => {
  const [clips, setClips] = useState(null);
  const [choice, setChoice] = useState('');
  const [cam, setCam] = useState(null);
  const [playing, setPlaying] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetchJson('/api/demo/clips')
        .then((d) => {
          if (cancelled) return;
          setClips(d.clips);
          setCam(d.test_camera);
          setChoice((c) => c || d.clips.find((x) => x.source === 'recording')?.id || d.clips[0]?.id || '');
        })
        .catch((err) => !cancelled && setError(describeError(err)));
    load();
    const id = setInterval(load, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const act = async (path, body) => {
    setBusy(true);
    setError(null);
    try {
      const r = await postJson(path, body);
      setPlaying(path.endsWith('play') ? r.label : null);
      setCam((c) => ({ ...c, status: r.status }));
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const status = cam?.status || 'stopped';
  const on = status === 'running' || status === 'starting';
  return (
    <DashboardPanel title="Test with demo footage" headerAction={on ? status.toUpperCase() : null}>
      <div className="space-y-3">
        <p className="text-[11px] text-white/60 outfit">
          Plays a clip as a separate test camera in warehouse mode (no notifications), so you can see activity labels
          and REBA working without moving your camera. Clips: your recordings in{' '}
          <code className="mono text-cyan-300">data/recordings</code> (a side view of lifting and carrying works best)
          and local CAUCAFall clips (subjects 1–5).
        </p>
        {clips && !clips.length && (
          <p className="text-[11px] text-amber-300 outfit">
            No clips found. Record one with the Record button, or put the CAUCAFall dataset in data/datasets.
          </p>
        )}
        {clips && clips.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="demo-clip">
              Clip
            </label>
            <select id="demo-clip" className={inputClass} value={choice} onChange={(e) => setChoice(e.target.value)}>
              {clips.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={busy || !choice}
              onClick={() => act('/api/demo/play', { clip: choice })}
              className={`${chipClass(true)} flex items-center gap-1.5`}
            >
              <Film className="w-3 h-3" aria-hidden="true" /> {on ? 'Play this clip instead' : 'Play'}
            </button>
            {on && (
              <button
                type="button"
                disabled={busy}
                onClick={() => act('/api/demo/stop', {})}
                className={`${chipClass(false)} flex items-center gap-1.5`}
              >
                <Square className="w-3 h-3" aria-hidden="true" /> Stop test footage
              </button>
            )}
          </div>
        )}
        {error && <p className="text-[11px] text-red-300">{error}</p>}
        {status === 'starting' && <p className="text-[11px] mono text-cyan-300">Loading models and the clip...</p>}
        {status === 'running' && <TestFeed label={playing || cam?.source} />}
      </div>
    </DashboardPanel>
  );
};

export default DemoFootagePanel;
