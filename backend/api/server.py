"""FastAPI server: REST API, WebSocket live feed, and the built dashboard.

Threading model: the pipeline runs in a background thread. It publishes each frame
as one pre-serialised JSON message (encoded once, shared by every client) and pushes
alerts into a per-client asyncio queue via ``loop.call_soon_threadsafe``, so alerts are
never lost to frame-rate throttling.
"""

import asyncio
import base64
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from anomaly.engine import AnomalyAlert
from config.settings import SentinelConfig
from core.pipeline import FrameResult, SentinelPipeline
from core.utils import to_serializable

logger = logging.getLogger("sentinel.api")

MAX_FEED_FPS = 15
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# --- Global state (written by the pipeline thread, read by async handlers) ---
pipeline: Optional[SentinelPipeline] = None
config: Optional[SentinelConfig] = None
alert_history: deque = deque(maxlen=1000)
history_lock = threading.Lock()
_latest_message: Optional[str] = None  # pre-serialised "frame" message
_latest_frame_number = -1
_frame_lock = threading.Lock()
_clients: Dict[WebSocket, asyncio.Queue] = {}
_clients_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None

app = FastAPI(title="Sentinel AI API", version="1.3.0")


def _configure_cors(origins):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


# --- Pipeline wiring -----------------------------------------------------------------

def start_pipeline(cfg: SentinelConfig) -> SentinelPipeline:
    """Create the pipeline, hook its callbacks, and run it in a daemon thread."""
    global pipeline, config
    config = cfg
    _configure_cors(cfg.cors_origins)
    pipeline = SentinelPipeline(cfg)

    @pipeline.on_alert
    def handle_alert(alert: AnomalyAlert):
        alert_dict = to_serializable(alert.to_dict())
        with history_lock:
            alert_history.append(alert_dict)
        message = json.dumps({"type": "alert", "alert": alert_dict})
        _broadcast_alert(message)

    @pipeline.on_frame
    def handle_frame(result: FrameResult):
        global _latest_message, _latest_frame_number
        image = None
        if result.annotated_frame is not None:
            ok, buffer = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                image = base64.b64encode(buffer).decode("ascii")
        payload = result.to_dict()
        payload["stats"]["fps"] = pipeline.stats["avg_fps"] if pipeline else 0
        message = json.dumps({"type": "frame", "image": image, "data": payload})
        with _frame_lock:  # serialise once; every client sends the same string
            _latest_message = message
            _latest_frame_number = result.frame_number

    thread = threading.Thread(target=pipeline.run, daemon=True, name="sentinel-pipeline")
    thread.start()
    logger.info("Sentinel Pipeline started in background thread.")
    return pipeline


def _broadcast_alert(message: str):
    if _loop is None:
        return
    with _clients_lock:
        queues = list(_clients.values())
    for q in queues:
        _loop.call_soon_threadsafe(_offer, q, message)


def _offer(q: asyncio.Queue, message: str):
    if q.full():  # a stalled client must not grow memory without bound
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(message)


# --- REST endpoints ------------------------------------------------------------------

def _require_pipeline() -> SentinelPipeline:
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.get("/api/health")
def health_check():
    source = pipeline.video_source.stats if pipeline else {}
    return {
        "status": "ok" if pipeline and not source.get("hardware_error") else "degraded",
        "pipeline": pipeline is not None,
        "source": config.source if config else None,
        "source_error": source.get("error_message", ""),
        "timestamp": time.time(),
    }


@app.get("/api/stats")
def get_stats():
    return to_serializable(_require_pipeline().stats)


@app.get("/api/alerts")
def get_alerts(limit: int = Query(50, ge=1, le=1000), severity: Optional[str] = None,
               alert_type: Optional[str] = None):
    """Recent alerts, newest first, from the SQLite event log (survives restarts)."""
    if pipeline:
        events = pipeline.event_logger.get_events(limit=limit, severity=severity,
                                                  alert_type=alert_type)
        for e in events:
            e["has_clip"] = _clip_path(e["alert_id"]) is not None
        return events
    with history_lock:
        data = list(alert_history)[::-1]
    if severity:
        data = [a for a in data if a["severity"] == severity]
    if alert_type:
        data = [a for a in data if a["alert_type"] == alert_type]
    return data[:limit]


@app.get("/api/zones")
def get_zones():
    return to_serializable(_require_pipeline().anomaly_engine.zone_overlay_data)


@app.get("/api/tracks")
def get_tracks():
    p = _require_pipeline()
    output = []
    for tid, feat in p.pose_estimator.get_all_features().items():
        output.append({
            "track_id": tid,
            "speed": round(float(feat.speed), 2),
            "displacement": round(float(feat.displacement), 2),
            "time_tracked": round(float(feat.time_tracked), 1),
            "fall_state": p.anomaly_engine.fall_detector.state_of(tid),
        })
    return to_serializable(output)


def _incident_file(alert_id: str, prefix: str, ext: str) -> Optional[Path]:
    """Resolve a clip/snapshot path for an alert id, refusing anything outside clips_dir."""
    if not config or not alert_id or not all(c.isalnum() or c in "-_" for c in alert_id):
        return None
    root = Path(config.output.clips_dir).resolve()
    path = (root / alert_id / f"{prefix}_{alert_id}.{ext}").resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path


def _clip_path(alert_id: str) -> Optional[Path]:
    return _incident_file(alert_id, "clip", "mp4")


@app.get("/api/clips/{alert_id}")
def get_clip(alert_id: str):
    path = _clip_path(alert_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Clip not found (it may still be recording)")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/snapshots/{alert_id}")
def get_snapshot(alert_id: str):
    path = _incident_file(alert_id, "snapshot", "jpg")
    if path is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


# --- WebSocket live feed -------------------------------------------------------------

@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    global _loop
    await websocket.accept()
    _loop = asyncio.get_running_loop()
    alerts: asyncio.Queue = asyncio.Queue(maxsize=200)
    with _clients_lock:
        _clients[websocket] = alerts
    logger.info("Client connected to feed. Total: %d", len(_clients))

    try:
        with history_lock:
            history = list(alert_history)[-50:]
        await websocket.send_text(json.dumps({"type": "history", "alerts": history}))

        last_sent = -1
        while True:
            # Alerts first, so none are delayed by frame throttling.
            while not alerts.empty():
                await websocket.send_text(alerts.get_nowait())
            with _frame_lock:
                message, number = _latest_message, _latest_frame_number
            if message is not None and number > last_sent:
                await websocket.send_text(message)
                last_sent = number
            await asyncio.sleep(1 / MAX_FEED_FPS)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away; RuntimeError = send after close
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        with _clients_lock:
            _clients.pop(websocket, None)
        logger.info("Client disconnected. Total: %d", len(_clients))


# --- Dashboard (built React app) -----------------------------------------------------

if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):  # unknown API routes are 404s, not the app
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (FRONTEND_DIST / path).resolve()
        if path and FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "Sentinel AI API. Build the dashboard with `npm run build` in "
                           "frontend/ to serve it here, or run `npm run dev`.",
                "docs": "/docs"}
