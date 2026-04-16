import { useState, useEffect, useCallback, useRef } from 'react';

const useWebSocket = (url = "ws://localhost:8000/ws/feed") => {
  const [frame, setFrame] = useState(null);
  const [frameData, setFrameData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState({
    person_count: 0,
    active_tracks: 0,
    alert_count: 0,
    processing_time_ms: 0
  });

  const ws = useRef(null);

  const connect = useCallback(() => {
    console.log(`Establishing HUD Connection to ${url}...`);
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log("HUD LINK ESTABLISHED");
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "frame") {
        setFrame(data.image);
        setFrameData(data.data);
        if (data.data.stats) {
          setStats(data.data.stats);
        }
      } else if (data.type === "alert") {
        setAlerts((prev) => {
          const newAlerts = [data.alert, ...prev];
          return newAlerts.slice(0, 100); // Keep last 100
        });
      }
    };

    ws.current.onclose = () => {
      console.log("HUD LINK SEVERED. Retrying in 3s...");
      setConnected(false);
      setTimeout(connect, 3000);
    };

    ws.current.onerror = (err) => {
      console.error("HUD LINK ERROR:", err);
      ws.current.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, [connect]);

  return { frame, frameData, alerts, connected, stats };
};

export default useWebSocket;
