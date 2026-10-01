"""FastAPI server: REST API, WebSocket live feed, and the built dashboard.

Threading model: the pipeline runs in a background thread. It publishes each frame
as one pre-serialised JSON message (encoded once, shared by every client) and pushes
alerts into a per-client asyncio queue via ``loop.call_soon_threadsafe``, so alerts are
never lost to frame-rate throttling.
"""

import asyncio
import base64
import contextlib
import copy
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

# Extra cameras beside the primary pipeline (the laptop webcam). Each has its own pipeline,
# frame slot and state; events from every camera share the store and the alert stream.
LAPTOP_CAMERA = "laptop"
_cameras: dict[str, SentinelPipeline] = {}
_camera_threads: dict[str, threading.Thread] = {}
_camera_state: dict[str, dict] = {}
_camera_frames: dict[str, tuple[str, int]] = {}
_cameras_lock = threading.Lock()

@contextlib.asynccontextmanager
async def _lifespan(_app):
    yield
    stop_cameras()  # release the webcam when the server shuts down


app = FastAPI(title="Sentinel AI API", version="1.3.0", lifespan=_lifespan)


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
    _attach(pipeline, primary=True)
    thread = threading.Thread(target=pipeline.run, daemon=True, name="sentinel-pipeline")
    thread.start()
    logger.info("Sentinel Pipeline started in background thread.")
    return pipeline


def _attach(p: SentinelPipeline, primary: bool) -> None:
    """Broadcast ``p``'s events to every client and publish its frames to its slot."""

    def handle_event(event: Event):
        # Same alert shape the dashboard always received, plus event_id and camera_id.
        alert_dict = to_serializable(event.to_alert_dict())
        with history_lock:
            alert_history.append(alert_dict)
        _broadcast_alert(json.dumps({"type": "alert", "alert": alert_dict}))

    p.event_bus.subscribe("websocket", handle_event)
    camera_id = p.config.camera_id

    @p.on_frame
    def handle_frame(result: FrameResult):
        global _latest_message, _latest_frame_number
        message = encode_frame_message(result, p.stats["avg_fps"])
        with _frame_lock:  # serialise once; every client of this camera sends the same string
            if primary:
                _latest_message, _latest_frame_number = message, result.frame_number
            else:
                _camera_frames[camera_id] = (message, result.frame_number)


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


def _pipeline_for(camera: str | None) -> SentinelPipeline:
    """The primary pipeline, or the extra camera named ``camera`` (404/503 if it isn't running)."""
    primary = _require_pipeline() if camera is None else pipeline
    if camera is None or (primary is not None and camera == primary.config.camera_id):
        return _require_pipeline()
    with _cameras_lock:
        p = _cameras.get(camera)
    if p is None:
        raise HTTPException(status_code=404, detail=f"camera {camera!r} is not running")
    return p


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
def get_stats(camera: str | None = None):
    return to_serializable(_pipeline_for(camera).stats)


# --- Cameras: the primary source plus the laptop webcam, started on request ----------

ZONE_TYPES = ("restricted", "time_limited", "one_way")


class ZoneIn(BaseModel):
    """One zone from the editor. Points are normalised (0-1) image coordinates."""

    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=60)
    zone_type: str = Field(pattern=r"^(restricted|time_limited|one_way)$")
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=50)
    time_limit: float = Field(0.0, ge=0, le=86400)
    direction: str | None = Field(None, pattern=r"^(up|down|left|right)$")
    active: bool = True

    def checked(self) -> "ZoneIn":
        if any(not (0.0 <= v <= 1.0) for point in self.polygon for v in point):
            raise HTTPException(status_code=422, detail=f"zone {self.id!r}: points must be within 0-1")
        if self.zone_type == "time_limited" and self.time_limit <= 0:
            raise HTTPException(status_code=422, detail=f"zone {self.id!r}: time_limited needs time_limit > 0")
        if self.zone_type == "one_way" and not self.direction:
            raise HTTPException(status_code=422, detail=f"zone {self.id!r}: one_way needs a direction")
        return self


class ZonesIn(BaseModel):
    zones: list[ZoneIn] = Field(max_length=20)


class CameraStart(BaseModel):
    index: int = Field(0, ge=0, le=9, description="Webcam device index (0 = built-in camera)")


def _check_camera_control(request: Request) -> None:
    """Turning a webcam on is sensitive, so:
    - only this machine may do it (the server listens on the LAN by default), unless
      ALLOW_REMOTE_CAMERA_CONTROL=true;
    - the body must be JSON. A cross-site page can't send a JSON POST without a CORS
      preflight, and CORS here allows GET only, so another site can't switch the camera on.
    """
    _check_local_json(request, bool(config and config.allow_remote_camera_control), "camera control")


def _check_local_json(request: Request, allow_remote: bool, action: str) -> None:
    """This machine only (unless ``allow_remote``), and a JSON body (see above)."""
    host = request.client.host if request.client else ""
    local = host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")
    if not local and not allow_remote:
        raise HTTPException(status_code=403, detail=f"{action} is only allowed from this computer")
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=415, detail="send a JSON body")


def _camera_info(camera_id: str) -> dict:
    with _cameras_lock:
        p = _cameras.get(camera_id)
        state = dict(_camera_state.get(camera_id, {"status": "stopped"}))
    if p is not None:
        src = p.video_source.stats
        state.update(fps=p.stats["avg_fps"], frames=p.frame_count,
                     source_error=src.get("error_message") or "",
                     hardware_error=bool(src.get("hardware_error")))
    return {"id": camera_id, "kind": "webcam", **state}


@app.get("/api/cameras")
def list_cameras():
    cams = []
    if pipeline:
        cams.append({"id": pipeline.config.camera_id, "kind": "primary", "status": "running",
                     "source": pipeline.config.source, "fps": pipeline.stats["avg_fps"]})
    cams.append(_camera_info(LAPTOP_CAMERA))
    return to_serializable(cams)


def _laptop_config(index: int) -> SentinelConfig:
    cfg = copy.deepcopy(config)
    # LAPTOP_CAMERA_SOURCE (set by the operator in .env, never by the API) swaps the webcam
    # for an IP camera URL or a video file, which loops like the demo.
    override = config.laptop_camera_source
    cfg.source = override or str(index)
    cfg.loop = bool(override) and not override.isdigit() and "://" not in override
    cfg.camera_id = LAPTOP_CAMERA
    cfg.frame_width, cfg.frame_height = 1280, 720
    # No zones by default: the primary camera's zones are drawn for its own view.
    zones = Path(cfg.output.db_path).resolve().parent / "zones_laptop.json"
    zones.parent.mkdir(parents=True, exist_ok=True)
    if not zones.exists():
        zones.write_text(json.dumps({"zones": []}), encoding="utf-8")
    cfg.zone.zones_file = str(zones)
    return cfg


def _run_camera(camera_id: str, cfg: SentinelConfig) -> None:
    try:
        p = SentinelPipeline(cfg)  # loads the models: a few seconds
        _attach(p, primary=False)
        with _cameras_lock:
            stopped_meanwhile = _camera_state.get(camera_id, {}).get("status") != "starting"
            if not stopped_meanwhile:
                _cameras[camera_id] = p
                _camera_state[camera_id] = {"status": "running", "source": cfg.source,
                                            "started_at": time.time()}
        if stopped_meanwhile:  # Stop was pressed while the models were loading
            p.stop()
            return
        p.run()  # returns when stopped (or the camera can't be opened and is stopped)
    except Exception as e:
        logger.exception("camera %s failed", camera_id)
        with _cameras_lock:
            _camera_state[camera_id] = {"status": "error", "error": str(e), "source": cfg.source}
    finally:
        with _cameras_lock:
            _cameras.pop(camera_id, None)
            _camera_frames.pop(camera_id, None)
            if _camera_state.get(camera_id, {}).get("status") in ("running", "stopping"):
                _camera_state[camera_id] = {"status": "stopped"}


@app.post("/api/cameras/laptop/start", status_code=202)
def start_laptop_camera(body: CameraStart, request: Request):
    """Start detection on the laptop webcam (device ``index``) as camera "laptop"."""
    _check_camera_control(request)
    _require_pipeline()
    with _cameras_lock:
        status = _camera_state.get(LAPTOP_CAMERA, {}).get("status")
        if status in ("starting", "running"):
            return _camera_state[LAPTOP_CAMERA] | {"id": LAPTOP_CAMERA}
        _camera_state[LAPTOP_CAMERA] = {"status": "starting", "source": str(body.index)}
        thread = threading.Thread(
            target=_run_camera, args=(LAPTOP_CAMERA, _laptop_config(body.index)),
            daemon=True, name="camera-laptop",
        )
        _camera_threads[LAPTOP_CAMERA] = thread
    thread.start()
    logger.info("Laptop camera %d starting", body.index)
    return {"id": LAPTOP_CAMERA, "status": "starting", "source": str(body.index)}


@app.post("/api/cameras/laptop/stop")
def stop_laptop_camera(request: Request):
    """Stop the laptop webcam pipeline and release the camera."""
    _check_camera_control(request)
    with _cameras_lock:
        p = _cameras.get(LAPTOP_CAMERA)
        thread = _camera_threads.get(LAPTOP_CAMERA)
        _camera_state[LAPTOP_CAMERA] = {"status": "stopping" if p else "stopped"}
    if p is not None:
        p.video_source.stop()  # releases the device; the pipeline loop then ends and cleans up
    if thread is not None:
        thread.join(timeout=15)
    with _cameras_lock:
        if _camera_state.get(LAPTOP_CAMERA, {}).get("status") == "stopping":
            _camera_state[LAPTOP_CAMERA] = {"status": "stopped"}
    return _camera_info(LAPTOP_CAMERA)


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


# --- Search (plain-English questions over the event log) -------------------------------------

class SearchIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)


_search_times: deque = deque()
_search_lock = threading.Lock()


def _search_config():
    p = _require_pipeline()
    return p.config


def _zone_names() -> dict[str, str]:
    """Zone id -> display name across every running camera (for 'the loading dock')."""
    names: dict[str, str] = {}
    with _cameras_lock:
        pipelines = [pipeline, *_cameras.values()]
    for p in pipelines:
        engine = getattr(p, "anomaly_engine", None)
        for z in (getattr(engine, "zone_overlay_data", None) or []):
            if z.get("id"):
                names.setdefault(z["id"], z.get("name") or z["id"])
    return names


def _search_rate_ok(limit_per_min: int) -> bool:
    now = time.time()
    with _search_lock:
        while _search_times and now - _search_times[0] > 60:
            _search_times.popleft()
        if len(_search_times) >= max(1, limit_per_min):
            return False
        _search_times.append(now)
        return True


@app.get("/api/search/status")
def search_status():
    """Whether search is configured (provider, model); never returns keys."""
    from llm import DEFAULT_MODELS, LLMError, build_llm

    cfg = _search_config()
    try:
        client = build_llm(cfg.llm)
        return {"enabled": True, "provider": client.provider, "model": client.model, "reason": None}
    except LLMError as e:
        provider = cfg.llm.provider
        return {"enabled": False, "provider": provider, "model": cfg.llm.model or DEFAULT_MODELS.get(provider),
                "reason": str(e)}


@app.post("/api/search")
def search(body: SearchIn, request: Request):
    """Answer a question about recorded events. Streams Server-Sent Events: ``tool_call`` and
    ``tool_result`` steps as the agent works, then one ``done`` with the answer and the cited
    events (or ``error``). Read-only; only event metadata is sent to the LLM provider."""
    from llm import LLMError, build_llm
    from search import SearchAgent, SearchTools

    cfg = _search_config()
    _check_local_json(request, cfg.search.allow_remote, "search")
    store = _store()
    try:
        llm = build_llm(cfg.llm)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if not _search_rate_ok(cfg.search.rate_limit_per_min):
        raise HTTPException(status_code=429, detail="too many questions; try again in a minute")

    agent = SearchAgent(llm, SearchTools(store, _zone_names()), max_steps=cfg.search.max_steps,
                        max_tokens=cfg.search.max_tokens_per_question)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(to_serializable(payload))}\n\n"

    def stream():
        try:
            for step in agent.run(body.question):
                if step["type"] != "done":
                    yield sse(step)
                    continue
                r = step["result"]
                cited = [store.get(i) for i in r.citations]
                yield sse({
                    "type": "done", "answer": r.answer, "stop": r.stop, "error": r.error,
                    "citations": [_event_json(e) for e in cited if e is not None],
                    "removed_citations": r.removed_citations, "steps": r.steps,
                    "tool_calls": r.tool_calls, "latency_s": r.latency_s,
                    "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
                    "provider": r.provider, "model": r.model,
                })
        except Exception as e:  # never leave the client hanging mid-stream
            logger.exception("search failed")
            yield sse({"type": "error", "error": f"search failed: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/zones")
def get_zones(camera: str | None = None):
    return to_serializable(_pipeline_for(camera).anomaly_engine.zone_overlay_data)


@app.put("/api/zones")
def put_zones(body: ZonesIn, request: Request, camera: str = Query(...)):
    """Replace a camera's zones (the zone editor). Takes effect on the next frame and is saved.

    Only cameras whose zones live under the data directory (the laptop webcam) can be edited;
    the demo scenarios' zone files are part of the repository and stay read-only.
    """
    _check_camera_control(request)  # same rule: this computer only, JSON body
    from anomaly.zone_monitor import Zone

    p = _pipeline_for(camera)
    monitor = p.anomaly_engine.zone_monitor
    data_dir = Path(p.config.output.db_path).resolve().parent
    if data_dir not in Path(monitor.zones_file).resolve().parents:
        raise HTTPException(status_code=409, detail=f"camera {camera!r} uses read-only scenario zones")
    ids = [z.id for z in body.zones]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="zone ids must be unique")
    zones = [
        Zone(id=z.id, name=z.name, polygon=[tuple(pt) for pt in z.polygon], zone_type=z.zone_type,
             time_limit=z.time_limit, direction=z.direction, active=z.active)
        for z in (zone.checked() for zone in body.zones)
    ]
    monitor.replace_zones(zones)
    logger.info("camera %s: %d zone(s) saved to %s", camera, len(zones), monitor.zones_file)
    return to_serializable(monitor.get_zones_for_overlay())


@app.get("/api/tracks")
def get_tracks(camera: str | None = None):
    p = _pipeline_for(camera)
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
async def websocket_feed(websocket: WebSocket, camera: str | None = None):
    """Alerts from every camera; frames from the primary camera, or from ``?camera=<id>``."""
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

        last_sent = None  # the last frame message sent (a camera restart resets frame numbers)
        while True:
            # Alerts first, so none are delayed by frame throttling.
            while not alerts.empty():
                await websocket.send_text(alerts.get_nowait())
            with _frame_lock:
                if camera and not (pipeline and camera == pipeline.config.camera_id):
                    message = _camera_frames.get(camera, (None, -1))[0]
                else:
                    message = _latest_message
            if message is not None and message is not last_sent:
                await websocket.send_text(message)
                last_sent = message
            await asyncio.sleep(1 / MAX_FEED_FPS)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away; RuntimeError = send after close
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        with _clients_lock:
            _clients.pop(websocket, None)
        logger.info("Client disconnected. Total: %d", len(_clients))


def stop_cameras() -> None:
    """Release every extra camera (used at shutdown)."""
    with _cameras_lock:
        pipelines = list(_cameras.values())
    for p in pipelines:
        p.video_source.stop()




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
        # index.html names the current hashed bundles: never cache it, or browsers keep
        # loading the previous build after an update (the hashed assets can be cached).
        return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "Sentinel AI API. Build the dashboard with `npm run build` in "
                           "frontend/ to serve it here, or run `npm run dev`.",
                "docs": "/docs"}
