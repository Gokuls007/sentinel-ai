import { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play } from 'lucide-react';
import { apiUrl, describeError, fetchJson } from '../lib/api';
import { chipClass } from '../lib/ui';

// COCO-17 bones (same as the live overlay).
const BONES = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];
const MIN_CONF = 0.3;
const W = 640;
const H = 360;

function draw(ctx, frame, focusId) {
  ctx.fillStyle = '#05080c';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  for (let x = 0; x <= W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y <= H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  for (const person of frame?.people || []) {
    const focus = focusId == null || person.track_id === focusId;
    const kp = person.keypoints.map(([x, y, c]) => [x * W, y * H, c]);
    ctx.lineWidth = focus ? 3 : 2;
    ctx.strokeStyle = focus ? '#22d3ee' : 'rgba(255,255,255,0.35)';
    for (const [a, b] of BONES) {
      if (kp[a][2] < MIN_CONF || kp[b][2] < MIN_CONF) continue;
      ctx.beginPath();
      ctx.moveTo(kp[a][0], kp[a][1]);
      ctx.lineTo(kp[b][0], kp[b][1]);
      ctx.stroke();
    }
    ctx.fillStyle = focus ? '#fbbf24' : 'rgba(255,255,255,0.5)';
    for (const [x, y, c] of kp) {
      if (c < MIN_CONF) continue;
      ctx.beginPath();
      ctx.arc(x, y, focus ? 3 : 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** Skeleton-only mode's incident playback: the stored keypoints as an animated stick figure
 * (10 s before the alert to 5 s after). No video or image is ever stored in that mode. */
const SkeletonPlayer = ({ url }) => {
  const canvasRef = useRef(null);
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchJson(url)
      .then((d) => { if (!cancelled) { setDoc(d); setIndex(0); } })
      .catch((err) => { if (!cancelled) setError(describeError(err)); });
    return () => { cancelled = true; };
  }, [url]);

  const frames = useMemo(() => doc?.frames || [], [doc]);
  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const next = (index + 1) % frames.length;
    const wait = next === 0 ? 1000 : Math.max(10, (frames[next].t - frames[index].t) * 1000);
    const id = setTimeout(() => setIndex(next), wait);
    return () => clearTimeout(id);
  }, [playing, index, frames]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && frames.length) draw(ctx, frames[index], doc.track_id);
  }, [index, frames, doc]);

  if (error) return <p className="text-[10px] text-red-300">{error}</p>;
  if (!doc) return <p className="text-[10px] mono text-white/50">Loading skeleton...</p>;
  if (!frames.length) return <p className="text-[10px] mono text-white/50">No keypoints were captured around this alert.</p>;
  const t = frames[index].t;
  return (
    <div className="space-y-1.5">
      <canvas ref={canvasRef} width={W} height={H} className="w-full bg-black border border-white/10"
        role="img" aria-label="Stick-figure playback of the stored keypoints around the alert" />
      <div className="flex items-center gap-2">
        <button type="button" onClick={() => setPlaying((p) => !p)} className={`${chipClass(false)} flex items-center gap-1`}>
          {playing ? <Pause className="w-3 h-3" aria-hidden="true" /> : <Play className="w-3 h-3" aria-hidden="true" />}
          {playing ? 'Pause' : 'Play'}
        </button>
        <input type="range" min={0} max={frames.length - 1} value={index} aria-label="Position"
          onChange={(e) => { setPlaying(false); setIndex(Number(e.target.value)); }} className="flex-1" />
        <span className={`text-[10px] mono tabular-nums w-16 text-right ${t >= 0 ? 'text-amber-300' : 'text-white/60'}`}>
          {t >= 0 ? '+' : ''}{t.toFixed(1)} s
        </span>
      </div>
      <p className="text-[9px] mono uppercase text-white/40">
        Skeleton only: keypoints, no video. 0 s = the alert; the highlighted figure is the person it was about.{' '}
        <a href={apiUrl(url)} target="_blank" rel="noreferrer" className="underline hover:text-white/70">JSON</a>
      </p>
    </div>
  );
};

export default SkeletonPlayer;
