/**
 * Sentinel AI — Demo Mode Hook
 * 
 * Generates a convincing surveillance feed simulation using HTML Canvas
 * and replays pre-recorded detection/alert data at 15fps.
 * 
 * When no backend is connected, this provides a fully functional dashboard
 * experience without requiring Python, PyTorch, or a camera.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

// ──── Skeleton connections (COCO-17 format) ────
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8],
  [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
];

// ──── Relative keypoint offsets for a standing person (normalized to bbox) ────
const STANDING_POSE = [
  [0.50, 0.06],  // 0: nose
  [0.47, 0.04],  // 1: left_eye
  [0.53, 0.04],  // 2: right_eye
  [0.42, 0.07],  // 3: left_ear
  [0.58, 0.07],  // 4: right_ear
  [0.38, 0.28],  // 5: left_shoulder
  [0.62, 0.28],  // 6: right_shoulder
  [0.30, 0.48],  // 7: left_elbow
  [0.70, 0.48],  // 8: right_elbow
  [0.28, 0.62],  // 9: left_wrist
  [0.72, 0.62],  // 10: right_wrist
  [0.40, 0.58],  // 11: left_hip
  [0.60, 0.58],  // 12: right_hip
  [0.38, 0.76],  // 13: left_knee
  [0.62, 0.76],  // 14: right_knee
  [0.36, 0.95],  // 15: left_ankle
  [0.64, 0.95],  // 16: right_ankle
];

const FALLEN_POSE = [
  [0.12, 0.50], [0.10, 0.45], [0.14, 0.45], [0.06, 0.52], [0.18, 0.52],
  [0.25, 0.38], [0.25, 0.62], [0.42, 0.32], [0.42, 0.68], [0.55, 0.30], [0.55, 0.70],
  [0.50, 0.42], [0.50, 0.58], [0.70, 0.38], [0.70, 0.62], [0.88, 0.36], [0.88, 0.64],
];

const WALKING_POSE_A = [
  [0.50, 0.06], [0.47, 0.04], [0.53, 0.04], [0.42, 0.07], [0.58, 0.07],
  [0.36, 0.28], [0.64, 0.28], [0.26, 0.46], [0.74, 0.46], [0.24, 0.60], [0.76, 0.60],
  [0.42, 0.58], [0.58, 0.58], [0.34, 0.78], [0.66, 0.74], [0.30, 0.96], [0.70, 0.92],
];

const WALKING_POSE_B = [
  [0.50, 0.06], [0.47, 0.04], [0.53, 0.04], [0.42, 0.07], [0.58, 0.07],
  [0.36, 0.28], [0.64, 0.28], [0.28, 0.44], [0.72, 0.50], [0.26, 0.58], [0.74, 0.64],
  [0.42, 0.58], [0.58, 0.58], [0.36, 0.74], [0.64, 0.78], [0.32, 0.92], [0.68, 0.96],
];


/**
 * Interpolate between two keyframe person arrays based on normalized time
 */
function interpolatePersons(kfA, kfB, t) {
  const personsA = kfA.persons;
  const personsB = kfB.persons;
  const result = [];
  
  // Build lookup by ID
  const mapA = {};
  const mapB = {};
  personsA.forEach(p => mapA[p.id] = p);
  personsB.forEach(p => mapB[p.id] = p);
  
  // Get all unique IDs across both keyframes
  const allIds = new Set([...Object.keys(mapA), ...Object.keys(mapB)]);
  
  for (const id of allIds) {
    const a = mapA[id];
    const b = mapB[id];
    
    if (a && b) {
      // Interpolate
      result.push({
        id: Number(id),
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        w: a.w + (b.w - a.w) * t,
        h: a.h + (b.h - a.h) * t,
        conf: a.conf + (b.conf - a.conf) * t,
        action: t < 0.5 ? a.action : b.action,
        vx: a.vx + (b.vx - a.vx) * t,
        vy: a.vy + (b.vy - a.vy) * t,
      });
    } else if (a && !b) {
      // Fading out — reduce confidence
      result.push({ ...a, conf: a.conf * (1 - t), id: Number(id) });
    } else if (!a && b) {
      // Fading in — increase confidence
      result.push({ ...b, conf: b.conf * t, id: Number(id) });
    }
  }
  
  return result;
}


/**
 * Get the pose keypoints for a person based on action + animation phase
 */
function getPoseForAction(action, phase) {
  if (action === 'fallen' || action === 'falling') {
    return FALLEN_POSE;
  }
  if (action === 'walking') {
    return phase < 0.5 ? WALKING_POSE_A : WALKING_POSE_B;
  }
  return STANDING_POSE;
}


/**
 * Render a single surveillance frame on a Canvas element
 */
function renderFrame(canvas, persons, zones, alerts, frameNum, totalAlerts, elapsedSec) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  
  // ─── Background: dark surveillance scene ───
  ctx.fillStyle = '#0a0d12';
  ctx.fillRect(0, 0, W, H);
  
  // Subtle noise pattern
  const imgData = ctx.getImageData(0, 0, W, H);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 16) {
    const noise = Math.random() * 6;
    data[i] += noise;
    data[i + 1] += noise;
    data[i + 2] += noise;
  }
  ctx.putImageData(imgData, 0, 0);
  
  // Grid overlay (subtle)
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.04)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 0; y < H; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  
  // Floor line (perspective hint)
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H * 0.82);
  ctx.lineTo(W, H * 0.82);
  ctx.stroke();
  
  // ─── Zone overlays ───
  zones.forEach(zone => {
    const poly = zone.polygon.map(([px, py]) => [px * W, py * H]);
    
    const color = zone.type === 'restricted' 
      ? 'rgba(255, 30, 30, 0.12)'
      : 'rgba(255, 165, 0, 0.10)';
    const borderColor = zone.type === 'restricted'
      ? 'rgba(255, 50, 50, 0.6)'
      : 'rgba(255, 165, 0, 0.5)';
    
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    poly.forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fill();
    
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Zone label
    ctx.fillStyle = borderColor;
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText(zone.name, poly[0][0] + 5, poly[0][1] + 14);
  });
  
  // ─── Render persons ───
  const walkPhase = (frameNum % 30) / 30;
  
  persons.forEach(person => {
    const bx = person.x * W;
    const by = person.y * H;
    const bw = person.w * W;
    const bh = person.h * H;
    
    // Check if person is in an alert state
    const isAlerted = person.action === 'fallen' || person.action === 'falling';
    const boxColor = isAlerted ? '#ff1744' : '#00e5ff';
    const boxAlpha = isAlerted ? 0.9 : 0.7;
    
    // Bounding box
    ctx.strokeStyle = boxColor;
    ctx.globalAlpha = boxAlpha;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx - bw/2, by - bh/2, bw, bh);
    ctx.globalAlpha = 1.0;
    
    // Track ID label
    ctx.fillStyle = boxColor;
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.fillText(`ID:${person.id} ${(person.conf * 100).toFixed(0)}%`, bx - bw/2, by - bh/2 - 5);
    
    // ─── Skeleton ───
    const pose = getPoseForAction(person.action, walkPhase);
    const keypoints = pose.map(([kx, ky]) => [
      bx - bw/2 + kx * bw,
      by - bh/2 + ky * bh
    ]);
    
    // Draw bones
    ctx.strokeStyle = isAlerted ? 'rgba(255, 23, 68, 0.7)' : 'rgba(0, 255, 255, 0.6)';
    ctx.lineWidth = 2;
    SKELETON.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(keypoints[a][0], keypoints[a][1]);
      ctx.lineTo(keypoints[b][0], keypoints[b][1]);
      ctx.stroke();
    });
    
    // Draw joints
    keypoints.forEach(([jx, jy]) => {
      ctx.fillStyle = isAlerted ? '#ff6b6b' : '#ffeb3b';
      ctx.beginPath();
      ctx.arc(jx, jy, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  
  // ─── Alert Banners (top-left, below header) ───
  const activeAlerts = alerts.slice(-3);
  const severityColors = {
    critical: '#ff1744',
    high: '#ff6d00',
    medium: '#ffd600',
    low: '#00e5ff'
  };
  
  activeAlerts.forEach((alert, i) => {
    const ay = 70 + i * 36;
    const color = severityColors[alert.severity] || '#ffffff';
    
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(10, ay, 380, 30);
    ctx.globalAlpha = 1.0;
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px "Outfit", sans-serif';
    ctx.fillText(`⚠ ${alert.message}`, 16, ay + 20);
  });
  
  // ─── Header bar (semi-transparent) ───
  const headerH = 50;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  ctx.fillRect(0, 0, W, headerH);
  
  // Accent line
  ctx.strokeStyle = '#ffe500';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, headerH);
  ctx.lineTo(W, headerH);
  ctx.stroke();
  
  // Title
  ctx.fillStyle = '#ffe500';
  ctx.font = 'bold 16px "Outfit", sans-serif';
  ctx.fillText('SENTINEL AI // HUD_ACTIVE', 18, 32);
  
  // Metrics
  ctx.fillStyle = '#ffffff';
  ctx.font = '11px "JetBrains Mono", monospace';
  const metrics = [
    `SECTOR: DEMO-01`,
    `TRACKS: ${persons.length}`,
    `ALERTS: ${totalAlerts}`,
    `LATENCY: ${Math.floor(18 + Math.random() * 12)}ms`
  ];
  let mx = W * 0.38;
  metrics.forEach(m => {
    ctx.strokeStyle = '#ffe500';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mx - 8, 15);
    ctx.lineTo(mx - 8, 38);
    ctx.stroke();
    ctx.fillText(m, mx, 32);
    mx += W * 0.15;
  });
  
  // ─── Forensic overlay (bottom) ───
  const now = new Date();
  const ts = now.toISOString().replace('T', ' ').substring(0, 23);
  ctx.fillStyle = '#ff0000';
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText(`REC ${ts} // DEMO-01-NORTH`, 18, H - 16);
  
  // Corner reticles
  ctx.strokeStyle = '#ffe500';
  ctx.lineWidth = 1;
  // Top-right
  ctx.beginPath(); ctx.moveTo(W - 28, 58); ctx.lineTo(W - 48, 58); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W - 28, 58); ctx.lineTo(W - 28, 78); ctx.stroke();
  // Bottom-left
  ctx.beginPath(); ctx.moveTo(28, H - 28); ctx.lineTo(48, H - 28); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(28, H - 28); ctx.lineTo(28, H - 48); ctx.stroke();
  
  // Scan line effect
  const scanY = (frameNum * 3) % H;
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.08)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, scanY);
  ctx.lineTo(W, scanY);
  ctx.stroke();
}


/**
 * Main demo mode hook
 */
const useDemoMode = () => {
  const [frame, setFrame] = useState(null);
  const [frameData, setFrameData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({
    person_count: 0,
    active_tracks: 0,
    alert_count: 0,
    processing_time_ms: 0
  });
  
  const canvasRef = useRef(null);
  const demoDataRef = useRef(null);
  const frameIdxRef = useRef(0);
  const totalAlertsRef = useRef(0);
  const firedAlertsRef = useRef(new Set());
  const intervalRef = useRef(null);
  
  // Load demo data
  useEffect(() => {
    const loadData = async () => {
      try {
        const resp = await fetch('/demo_data.json');
        const data = await resp.json();
        demoDataRef.current = data;
        
        // Create offscreen canvas
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 360;
        canvasRef.current = canvas;
        
        console.log('[DEMO] Data loaded. Starting playback...');
        startPlayback();
      } catch (err) {
        console.error('[DEMO] Failed to load demo data:', err);
      }
    };
    
    loadData();
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);
  
  const startPlayback = useCallback(() => {
    const FPS = 15;
    const DURATION = 30; // seconds
    const TOTAL_FRAMES = FPS * DURATION;
    
    intervalRef.current = setInterval(() => {
      const data = demoDataRef.current;
      const canvas = canvasRef.current;
      if (!data || !canvas) return;
      
      const idx = frameIdxRef.current;
      const currentTime = (idx / FPS) % DURATION;
      
      // Find surrounding keyframes
      const keyframes = data.keyframes;
      let kfA = keyframes[0];
      let kfB = keyframes[1] || keyframes[0];
      
      for (let i = 0; i < keyframes.length - 1; i++) {
        if (currentTime >= keyframes[i].time && currentTime < keyframes[i + 1].time) {
          kfA = keyframes[i];
          kfB = keyframes[i + 1];
          break;
        }
      }
      // Handle wrap-around at end
      if (currentTime >= keyframes[keyframes.length - 1].time) {
        kfA = keyframes[keyframes.length - 1];
        kfB = keyframes[0];
      }
      
      // Interpolate persons
      const timeSpan = kfB.time > kfA.time ? kfB.time - kfA.time : 1;
      const t = Math.max(0, Math.min(1, (currentTime - kfA.time) / timeSpan));
      const persons = interpolatePersons(kfA, kfB, t);
      
      // Process scheduled alerts
      const newAlerts = [];
      const loopNum = Math.floor(idx / TOTAL_FRAMES);
      data.alert_schedule.forEach(scheduled => {
        const alertKey = `${scheduled.alert_id}_${loopNum}`;
        if (currentTime >= scheduled.time && currentTime < scheduled.time + 0.2 && !firedAlertsRef.current.has(alertKey)) {
          firedAlertsRef.current.add(alertKey);
          totalAlertsRef.current += 1;
          // Generate unique alert_id for React keys
          const uniqueId = `${scheduled.alert_id}-${loopNum}-${Date.now().toString(36)}`;
          newAlerts.push({
            alert_id: uniqueId,
            alert_type: scheduled.type,
            severity: scheduled.severity,
            track_id: scheduled.track_id,
            message: scheduled.message,
            timestamp: Date.now() / 1000
          });
        }
      });
      
      if (newAlerts.length > 0) {
        setAlerts(prev => [...newAlerts, ...prev].slice(0, 100));
      }
      
      // Get recent alerts for banner display
      const recentAlerts = [];
      data.alert_schedule.forEach(a => {
        if (currentTime >= a.time && currentTime < a.time + 4) {
          recentAlerts.push(a);
        }
      });
      
      // Render frame
      renderFrame(canvas, persons, data.zones, recentAlerts, idx, totalAlertsRef.current, currentTime);
      
      // Convert canvas to base64 JPEG
      const base64 = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];
      setFrame(base64);
      
      // Build frame data (matches WebSocket format)
      const detections = persons.map(p => ({
        track_id: p.id,
        class_name: 'person',
        confidence: p.conf,
        bbox: [
          (p.x - p.w/2) * canvas.width,
          (p.y - p.h/2) * canvas.height,
          (p.x + p.w/2) * canvas.width,
          (p.y + p.h/2) * canvas.height
        ],
        speed: Math.sqrt(p.vx * p.vx + p.vy * p.vy) * 1000
      }));
      
      const fd = {
        timestamp: currentTime,
        frame_number: idx,
        processing_time_ms: 18 + Math.random() * 12,
        detections: {
          person_count: persons.length,
          detections: detections
        },
        stats: {
          person_count: persons.length,
          active_tracks: persons.length,
          alert_count: totalAlertsRef.current,
          processing_time_ms: 18 + Math.random() * 12
        }
      };
      
      setFrameData(fd);
      setStats(fd.stats);
      
      // Loop
      frameIdxRef.current = (idx + 1) % TOTAL_FRAMES;
      if (idx + 1 >= TOTAL_FRAMES) {
        // Reset fired alerts for next loop
        firedAlertsRef.current.clear();
      }
      
    }, 1000 / 15);
  }, []);
  
  return { frame, frameData, alerts, stats };
};

export default useDemoMode;
