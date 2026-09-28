# Architecture

```text
 video source ──► VideoSource thread ──► SentinelPipeline.run (pipeline thread) ──► FastAPI (event loop)
 (file / webcam /  drop-oldest queue     detect+track ► pose ► anomaly engine        REST  /api/*
  RTSP / HTTP)     of 2 frames           ► annotate ► clips ► SQLite ► webhook       WS    /ws/feed
                                                                                     SPA   frontend/dist
```

## 1. Ingestion: `backend/core/video_source.py`
- A capture thread reads frames into a **queue of 2 that drops the oldest frame**. When
  processing falls behind, the dashboard shows the newest frame instead of lagging.
- **Files** are read at their native fps, and each frame's timestamp is its position on the
  video timeline. Zone timers and loitering therefore run in "video time", even when the
  machine is slower than real time. With `loop`, the file restarts and the timeline keeps
  increasing across loops.
- **Live sources** (a webcam index, RTSP, or HTTP) reconnect after failures and use
  wall-clock timestamps.

## 2. Perception: `backend/core/detector.py` and `pose_estimator.py`
- Detection and tracking use YOLOv8n with ByteTrack (`model.track(persist=True)`), which gives
  stable person IDs.
- YOLOv8n-pose supplies 17 COCO keypoints, matched to tracked boxes by IoU.
- Keypoints below 0.3 confidence count as missing. The hips then fall back to the box
  centre, and the head falls back to the top of the box.
- Each track keeps rolling features: centroid history, timestamps, and a calibrated standing
  height. That height is the median head-to-ankle distance over 10 upright frames in which
  the whole body is visible.

## 3. Anomaly engine: `backend/anomaly/`
| Detector | Rule | Output |
|---|---|---|
| Fall (`fall_detector.py`) | Per-person state machine: UPRIGHT → FALLING (hip descent faster than 1.2 standing heights per second) → FALLEN (lying and head dropped, within 1.5 s) → CONFIRMED (still for 1 s). Standing up resets it. Speeds are normalised by body height, so the thresholds don't depend on resolution, distance, or fps. | One `fall` alert per fall (critical), then a 30 s cooldown |
| Zones (`zone_monitor.py`) | Normalised polygons, hit-tested at the hip point. `restricted` alerts on entry. `time_limited` alerts once dwell exceeds its limit. `one_way` alerts on sustained movement against the allowed direction. | At most one alert per person per zone per cooldown (a re-entry from boundary jitter doesn't re-alert) |
| Loitering (`engine.py`) | Staying within `movement_threshold` px of one spot for longer than `time_threshold` seconds. Moving further away restarts the clock. | `loitering` (low), 60 s cooldown |
| Action LSTM (`temporal_model.py`) | Optional. It is **disabled unless trained weights exist** at `LSTM_MODEL_PATH` (see `training/`), so a randomly initialised network never raises alerts. | `action` alerts |

State for people who have left the frame is pruned every frame, so memory stays bounded.

## 4. Output: `backend/output/`
- **Forensic clips** (`clip_recorder.py`):
  - A rolling buffer holds the last 10 s of *annotated* frames, JPEG-compressed and
    downscaled.
  - On an alert, the clip keeps recording for another 5 s. It is then encoded on a background
    thread, to H.264 through imageio-ffmpeg so browsers can play it, falling back to mp4v.
  - Durations and playback speed follow the frame timestamps, so a clip plays in real time
    even when frames were dropped.
  - A snapshot JPEG of the alert frame is written straight away.
- **Event log** (`event_logger.py`): SQLite, which keeps history across restarts. It backs
  `/api/alerts`.
- **Webhook** (`webhook.py`): if `WEBHOOK_URL` is set, every alert is POSTed as JSON from a
  background thread. A slow endpoint never stalls the pipeline.

## 5. API and dashboard: `backend/api/server.py`, `frontend/`
- REST endpoints (all GET): `/api/health`, `/api/stats`, `/api/alerts`, `/api/zones`,
  `/api/tracks`, `/api/clips/{id}` and `/api/snapshots/{id}`. The clip and snapshot ids are
  validated and resolved inside `CLIPS_DIR`.
- WebSocket `/ws/feed`:
  1. first, `{"type":"history"}` with recent alerts;
  2. then `{"type":"alert"}` messages, one per alert, from a per-client queue, so they are
     never dropped;
  3. and `{"type":"frame","image":<jpeg b64>,"data":...}` at up to 15 fps. Each frame is
     serialised once and shared by all clients.
- The React dashboard (Vite, Tailwind v4, Recharts) is built into `frontend/dist` and served
  by the same FastAPI process. In development, `npm run dev` proxies `/api` and `/ws` to
  port 8000.

## Configuration
The settings come from three places:
1. `backend/config/settings.py` holds the defaults.
2. `.env` and environment variables override them (see `.env.example`).
3. CLI flags in `backend/main.py` override those.

A demo scenario (`config/demo/<name>_demo.json`) supplies its zones and an `overrides` block
that is applied to the config, for example `{"loiter": {"time_threshold": 10}}`.
