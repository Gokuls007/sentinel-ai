import logging
import asyncio
import cv2
import json
import base64
import time
import threading
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import SentinelPipeline, FrameResult
from anomaly.engine import AnomalyAlert
from config.settings import SentinelConfig

logger = logging.getLogger("sentinel.api")

# --- Global State ---
pipeline: Optional[SentinelPipeline] = None
connected_clients: List[WebSocket] = []
latest_result: Optional[FrameResult] = None
alert_history: List[dict] = []
history_lock = threading.Lock()

# --- App Lifecycle ---
app = FastAPI(title="Sentinel AI API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def start_pipeline(config: SentinelConfig):
    global pipeline
    pipeline = SentinelPipeline(config)
    
    # Register Callbacks
    @pipeline.on_alert
    def handle_alert(alert: AnomalyAlert):
        with history_lock:
            alert_dict = alert.to_dict()
            alert_history.append(alert_dict)
            if len(alert_history) > 1000:
                alert_history.pop(0)
    
    @pipeline.on_frame
    def handle_frame(result: FrameResult):
        global latest_result
        latest_result = result

    # Run pipeline in a daemon thread
    thread = threading.Thread(target=pipeline.run, daemon=True)
    thread.start()
    logger.info("Sentinel Pipeline started in background thread.")

# --- REST Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/stats")
def get_stats():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.stats

@app.get("/api/alerts")
def get_alerts(limit: int = 50, severity: Optional[str] = None):
    with history_lock:
        data = alert_history[::-1] # Newest first
        if severity:
            data = [a for a in data if a["severity"] == severity]
        return data[:limit]

@app.get("/api/zones")
def get_zones():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.anomaly_engine.zone_overlay_data

@app.get("/api/tracks")
def get_tracks():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    features = pipeline.pose_estimator.get_all_features()
    output = []
    for tid, feat in features.items():
        output.append({
            "track_id": tid,
            "speed": round(feat.speed, 2),
            "displacement": round(feat.displacement, 2),
            "time_tracked": round(feat.time_tracked, 1),
            "last_action": "N/A" # Placeholder for future logic
        })
    return output

# --- WebSockets ---

@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"Client connected to feed. Total: {len(connected_clients)}")
    
    last_frame_number = -1
    
    try:
        while True:
            await asyncio.sleep(1/15) # Cap at ~15fps
            
            if latest_result is not None and latest_result.frame_number > last_frame_number:
                last_frame_number = latest_result.frame_number
                
                # 1. Encode annotated frame
                if latest_result.annotated_frame is not None:
                    _, buffer = cv2.imencode('.jpg', latest_result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # 2. Main frame package
                    await websocket.send_json({
                        "type": "frame",
                        "image": img_base64,
                        "data": latest_result.to_dict()
                    })
                    
                    # 3. Individual alert pushes
                    for alert in latest_result.alerts:
                        await websocket.send_json({
                            "type": "alert",
                            "alert": alert.to_dict()
                        })
                        
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)
