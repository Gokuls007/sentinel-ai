import { useState, useEffect, useCallback, useRef } from 'react';
import useDemoMode from './useDemoMode';

const useWebSocket = (url = "ws://localhost:8000/ws/feed") => {
  const [frame, setFrame] = useState(null);
  const [frameData, setFrameData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [stats, setStats] = useState({
    person_count: 0,
    active_tracks: 0,
    alert_count: 0,
    processing_time_ms: 0
  });

  const ws = useRef(null);
  const connectionAttempts = useRef(0);
  const fallbackTimer = useRef(null);
  const isDestroyed = useRef(false);

  // Demo mode data source
  const demo = useDemoMode();

  const connect = useCallback(() => {
    if (isDestroyed.current || demoMode) return;

    console.log(`[WS] Connecting to ${url} (attempt ${connectionAttempts.current + 1})...`);
    
    try {
      ws.current = new WebSocket(url);
    } catch (err) {
      console.warn('[WS] WebSocket constructor failed:', err);
      connectionAttempts.current += 1;
      checkFallback();
      return;
    }

    ws.current.onopen = () => {
      console.log("[WS] HUD LINK ESTABLISHED");
      setConnected(true);
      connectionAttempts.current = 0;
      if (fallbackTimer.current) {
        clearTimeout(fallbackTimer.current);
        fallbackTimer.current = null;
      }
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "frame") {
        setFrame(data.image);
        setFrameData(data.data);
        if (data.data && data.data.stats) {
          setStats((prev) => ({
            ...prev,
            ...data.data.stats
          }));
        }
      } else if (data.type === "alert") {
        setAlerts((prev) => {
          const newAlerts = [data.alert, ...prev];
          return newAlerts.slice(0, 100); // Keep last 100
        });
      }
    };

    ws.current.onclose = () => {
      console.log("[WS] HUD LINK SEVERED.");
      setConnected(false);
      connectionAttempts.current += 1;
      checkFallback();
    };

    ws.current.onerror = (err) => {
      console.error("[WS] HUD LINK ERROR:", err);
      ws.current.close();
    };
  }, [url, demoMode]);

  const checkFallback = useCallback(() => {
    if (isDestroyed.current || demoMode) return;
    
    // After 3 seconds / multiple failures, switch to demo mode
    if (connectionAttempts.current >= 1) {
      if (!fallbackTimer.current) {
        console.log("[WS] Starting 3s fallback timer...");
        fallbackTimer.current = setTimeout(() => {
          if (!isDestroyed.current && connectionAttempts.current >= 1) {
            console.log("[WS] Switching to DEMO MODE — no backend detected");
            setDemoMode(true);
          }
        }, 3000);
      }
    }
    
    // Still retry WebSocket while waiting for fallback
    if (!demoMode && connectionAttempts.current < 3) {
      setTimeout(() => {
        if (!isDestroyed.current && !demoMode) connect();
      }, 2000);
    }
  }, [demoMode, connect]);

  useEffect(() => {
    isDestroyed.current = false;
    connect();
    return () => {
      isDestroyed.current = true;
      if (ws.current) ws.current.close();
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
    };
  }, [connect]);

  // If demo mode is active, return demo data instead
  if (demoMode) {
    return {
      frame: demo.frame,
      frameData: demo.frameData,
      alerts: demo.alerts,
      connected: true, // Report as "connected" so UI doesn't show error state
      stats: demo.stats,
      demoMode: true
    };
  }

  return { frame, frameData, alerts, connected, stats, demoMode: false };
};

export default useWebSocket;
