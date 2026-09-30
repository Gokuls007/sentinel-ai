"""FastAPI server: REST API, WebSocket live feed, and the built dashboard.

Threading model: the pipeline runs in a background thread. It publishes each frame
as one pre-serialised JSON message (encoded once, shared by every client) and pushes
alerts into a per-client asyncio queue via ``loop.call_soon_threadsafe``, so alerts are
never lost to frame-rate throttling.
"""

import asyncio
import base64
import contextlib
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import SentinelConfig
from core.pipeline import FrameResult, SentinelPipeline
from core.utils import to_serializable
from events import EVENT_TYPES, GROUP_BY_KEYS, SEVERITIES, Event, EventStore

logger = logging.getLogger("sentinel.api")

MAX_FEED_FPS = 15
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# --- Global state (written by the pipeline thread, read by async handlers) ---
pipeline: SentinelPipeline | None = None
config: SentinelConfig | None = None
alert_history: deque = deque(maxlen=1000)
history_lock = threading.Lock()
_latest_message: str | None = None  # pre-serialised "frame" message
_latest_frame_number = -1
_frame_lock = threading.Lock()
_clients: dict[WebSocket, asyncio.Queue] = {}
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

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

    def handle_event(event: Event):
        # Same alert shape the dashboard always received, plus event_id and camera_id.
        alert_dict = to_serializable(event.to_alert_dict())
        with history_lock:
            alert_history.append(alert_dict)
        _broadcast_alert(json.dumps({"type": "alert", "alert": alert_dict}))

    pipeline.event_bus.subscribe("websocket", handle_event)

    @pipeline.on_frame
    def handle_frame(result: FrameResult):
        global _latest_message, _latest_frame_number
        message = encode_frame_message(result, pipeline.stats["avg_fps"] if pipeline else 0)
        with _frame_lock:  # serialise once; every client sends the same string
            _latest_message = message
            _latest_frame_number = result.frame_number

    thread = threading.Thread(target=pipeline.run, daemon=True, name="sentinel-pipeline")
    thread.start()
    logger.info("Sentinel Pipeline started in background thread.")
    return pipeline


def encode_frame_message(result: FrameResult, fps: float, jpeg_quality: int = 70) -> str:
    """The WebSocket "frame" message: annotated JPEG (base64) plus the frame's data."""
    image = None
    if result.annotated_frame is not None:
        ok, buffer = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            image = base64.b64encode(buffer).decode("ascii")
    payload = result.to_dict()
    payload["stats"]["fps"] = fps
    return json.dumps({"type": "frame", "image": image, "data": payload})


def _broadcast_alert(message: str):
    if _loop is None:
        return
    with _clients_lock:
        queues = list(_clients.values())
    for q in queues:
        _loop.call_soon_threadsafe(_offer, q, message)


def _offer(q: asyncio.Queue, message: str):
    if q.full():  # a stalled client must not grow memory without bound
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()
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
def get_alerts(limit: int = Query(50, ge=1, le=1000), severity: str | None = None,
               alert_type: str | None = None):
    """Recent alerts, newest first, from the SQLite event log (survives restarts)."""
    store = getattr(pipeline, "event_store", None) if pipeline else None
    if store is not None:
        events = store.query(limit=limit, severity=severity, types=[alert_type] if alert_type else None)
        out = []
        for e in events:
            item = to_serializable(e.to_alert_dict())
            item["has_clip"] = bool(e.alert_id) and _clip_path(e.alert_id) is not None
            out.append(item)
        return out
    with history_lock:
        data = list(alert_history)[::-1]
    if severity:
        data = [a for a in data if a["severity"] == severity]
    if alert_type:
        data = [a for a in data if a["alert_type"] == alert_type]
    return data[:limit]


# --- Events (unified schema) ---------------------------------------------------------

def _store() -> EventStore:
    store = getattr(pipeline, "event_store", None) if pipeline else None
    if store is None:
        raise HTTPException(status_code=503, detail="Event store not initialized")
    return store


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _event_json(event: Event) -> dict:
    item = to_serializable(event.to_dict())
    has_clip = bool(event.alert_id) and _clip_path(event.alert_id) is not None
    has_thumb = bool(event.alert_id) and _incident_file(event.alert_id, "snapshot", "jpg") is not None
    item["clip_url"] = f"/api/clips/{event.alert_id}" if has_clip else None
    item["thumbnail_url"] = f"/api/snapshots/{event.alert_id}" if has_thumb else None
    return item


def _event_filters(types, severity, camera_id, zone_id, track_id, start, end) -> dict:
    return {
        "types": _csv(types),
        "severity": _csv(severity),
        "camera_id": camera_id,
        "zone_id": zone_id,
        "track_id": track_id,
        "start": start,
        "end": end,
    }


@app.get("/api/events")
def list_events(
    types: str | None = Query(None, description="Comma-separated event types"),
    severity: str | None = Query(None, description="Comma-separated severities"),
    camera_id: str | None = None,
    zone_id: str | None = None,
    track_id: int | None = None,
    start: float | None = Query(None, description="Unix seconds; events overlapping [start, end]"),
    end: float | None = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Events in the unified schema, newest first, with paging and filters."""
    store = _store()
    filters = _event_filters(types, severity, camera_id, zone_id, track_id, start, end)
    return {
        "total": store.total(**filters),
        "events": [_event_json(e) for e in store.query(limit=limit, offset=offset, **filters)],
    }


@app.get("/api/events/stats")
def event_stats(
    group_by: str = Query("type", description=f"One of: {', '.join(GROUP_BY_KEYS)}"),
    types: str | None = None,
    severity: str | None = None,
    camera_id: str | None = None,
    zone_id: str | None = None,
    start: float | None = None,
    end: float | None = None,
):
    """Event counts grouped by type, severity, zone, camera, hour of day, hour bucket or day."""
    if group_by not in GROUP_BY_KEYS:
        raise HTTPException(status_code=422, detail=f"group_by must be one of {list(GROUP_BY_KEYS)}")
    filters = _event_filters(types, severity, camera_id, zone_id, None, start, end)
    return {"group_by": group_by, "counts": _store().count(group_by, **filters)}


@app.get("/api/events/{event_id}")
def get_event(event_id: int):
    event = _store().get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _event_json(event)


@app.get("/api/meta")
def get_meta():
    """Values for filter dropdowns and the Settings page (never includes secrets)."""
    p = _require_pipeline()
    store = _store()
    n = p.config.notifications
    return to_serializable({
        "camera_id": p.config.camera_id,
        "cameras": sorted(set(store.distinct("camera_id")) | {p.config.camera_id}),
        "source": p.config.source,
        "event_types": list(EVENT_TYPES),
        "severities": list(SEVERITIES),
        "zones": [
            {"id": z["id"], "name": z["name"], "type": z["type"], "time_limit": z.get("time_limit"),
             "direction": z.get("direction")}
            for z in p.anomaly_engine.zone_overlay_data
        ],
        "notifications": {
            "channels": [x.name for x in p.notifier.notifiers],
            "telegram": n.telegram_enabled,
            "email": n.email_enabled,
            "webhook": bool(p.config.output.webhook_url),
            "debounce_s": n.debounce_s,
            "min_severity": n.min_severity,
        },
        "thresholds": {
            "detection_confidence": p.config.detector.confidence_threshold,
            "fall": vars(p.config.fall),
            "loiter": vars(p.config.loiter),
            "zone_alert_cooldown_s": p.config.zone.alert_cooldown,
        },
    })


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


def _incident_file(alert_id: str, prefix: str, ext: str) -> Path | None:
    """Resolve a clip/snapshot path for an alert id, refusing anything outside clips_dir."""
    if not config or not alert_id or not all(c.isalnum() or c in "-_" for c in alert_id):
        return None
    root = Path(config.output.clips_dir).resolve()
    path = (root / alert_id / f"{prefix}_{alert_id}.{ext}").resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path


def _clip_path(alert_id: str) -> Path | None:
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
