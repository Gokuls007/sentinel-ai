import { useRef, useState } from 'react';

const MIN = 0.02;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Seat rectangles over the camera frame, in normalised 0-1 coordinates. Drag a seat to move
 * it, drag its corner to resize, click to select. Read-only when `editable` is false. */
const SeatEditor = ({ frame, seats, onChange, selected, onSelect, editable = true, states = {} }) => {
  const svgRef = useRef(null);
  const [drag, setDrag] = useState(null); // { index, mode: 'move'|'resize', start: [x, y], rect }

  const point = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };

  const begin = (e, index, mode) => {
    if (!editable) return;
    e.stopPropagation();
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId); // keep the drag when the pointer leaves the seat
    } catch {
      // no active pointer (e.g. a synthetic event): dragging still works while over the map
    }
    onSelect(index);
    setDrag({ index, mode, start: point(e), rect: seats[index].rect });
  };

  const move = (e) => {
    if (!drag) return;
    const [x, y] = point(e);
    const dx = x - drag.start[0];
    const dy = y - drag.start[1];
    const [x1, y1, x2, y2] = drag.rect;
    let rect;
    if (drag.mode === 'move') {
      const w = x2 - x1;
      const h = y2 - y1;
      const nx = clamp(x1 + dx, 0, 1 - w);
      const ny = clamp(y1 + dy, 0, 1 - h);
      rect = [nx, ny, nx + w, ny + h];
    } else {
      rect = [x1, y1, clamp(x2 + dx, x1 + MIN, 1), clamp(y2 + dy, y1 + MIN, 1)];
    }
    onChange(seats.map((s, i) => (i === drag.index ? { ...s, rect } : s)));
  };

  const colour = (s) => {
    const st = states[s.label];
    if (!st) return '#22d3ee';
    if (!st.occupied) return '#9ca3af';
    return st.calibrated ? '#4ade80' : '#22d3ee';
  };

  return (
    <div className="relative w-full bg-black border border-white/10">
      {frame ? (
        <img src={`data:image/jpeg;base64,${frame}`} alt="Camera view with seats" className="w-full block" draggable={false} />
      ) : (
        <div className="aspect-video flex items-center justify-center text-[11px] mono text-white/40">No video yet</div>
      )}
      <svg ref={svgRef} viewBox="0 0 1 1" preserveAspectRatio="none" className="absolute inset-0 w-full h-full"
        onPointerMove={move} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}
        onPointerDown={() => onSelect(null)} role="group" aria-label="Seat map">
        {seats.map((s, i) => {
          const [x1, y1, x2, y2] = s.rect;
          const c = colour(s);
          const isSel = i === selected;
          return (
            <g key={`${s.label}-${i}`}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={isSel ? 'rgba(34,211,238,0.12)' : 'transparent'}
                stroke={c} strokeWidth={isSel ? 3 : 2} vectorEffect="non-scaling-stroke"
                style={{ cursor: editable ? 'move' : 'default' }}
                onPointerDown={(e) => begin(e, i, 'move')}>
                <title>{`Seat ${s.label}`}</title>
              </rect>
              {editable && (
                <rect x={x2 - 0.012} y={y2 - 0.02} width={0.012} height={0.02} fill={c}
                  style={{ cursor: 'nwse-resize' }} onPointerDown={(e) => begin(e, i, 'resize')} />
              )}
            </g>
          );
        })}
      </svg>
      {/* Labels at the bottom-left corner of each seat (the desk), never over a face. */}
      {seats.map((s, i) => (
        <span key={`l-${s.label}-${i}`} className="absolute text-[11px] mono font-bold px-1 bg-black/70 pointer-events-none"
          style={{ left: `${s.rect[0] * 100}%`, top: `${s.rect[3] * 100}%`, transform: 'translateY(-100%)', color: colour(s) }}>
          {s.label}
        </span>
      ))}
    </div>
  );
};

export default SeatEditor;
