# Current state (before the safety-agent plan)

This is a snapshot of `main` at `8397c4b` (2026-09-28), written before Phase 0. Later phases
build on these structures and naming conventions.

## Module layout

```text
backend/                         # imported as top-level packages (backend/ is on sys.path)
  main.py                        # CLI entry: --demo [corridor|hallway], --source, --loop; starts API + pipeline
  config/settings.py             # dataclass config; SentinelConfig.from_env() reads .env; CLI overrides
  config/zones.json              # default zones (normalised 0-1 polygons)
  core/video_source.py           # threaded capture, drop-oldest queue (2), file timeline timestamps, loop
  core/detector.py               # YOLOv8n + ByteTrack (model.track) -> FrameDetections
  core/pose_estimator.py         # YOLOv8n-pose, IoU-matched to tracks -> PoseResult (17 COCO kpts); TrackFeatures
  core/pipeline.py               # SentinelPipeline: detect -> pose -> anomaly -> annotate -> clips/log/webhook
  core/samples.py                # sample videos (Intel, CC BY 4.0), sha256-verified download
  anomaly/fall_detector.py       # per-track state machine UPRIGHT->FALLING->FALLEN->CONFIRMED
  anomaly/zone_monitor.py        # restricted / time_limited / one_way zones, per-person cooldowns
  anomaly/engine.py              # AnomalyEngine: fall + zones + loitering (+ LSTM, off without weights)
  anomaly/temporal_model.py      # ActionLSTM; disabled unless trained weights load
  output/clip_recorder.py        # JPEG ring buffer, 10 s pre + 5 s post, H.264 via imageio-ffmpeg, snapshots
  output/event_logger.py         # SQLite event log (schema below)
  output/webhook.py              # background JSON POST of every alert (WEBHOOK_URL)
  api/server.py                  # FastAPI: REST, WebSocket /ws/feed, serves frontend/dist
frontend/src/                    # React 19 + Vite 8 + Tailwind 4 + Recharts, single page (no router)
  App.jsx -> components/HUD.jsx  # the whole dashboard
  components/                    # VideoFeed, StatsPanel, TrackList, AlertPanel, AlertTimeline, ZonePanel
  hooks/useWebSocket.js          # one socket, reconnect with backoff, history/alert/frame messages
  lib/api.js                     # apiUrl/wsUrl (same origin or VITE_*), usePoll, useNow
training/                        # LSTM pipeline (data_prep -> train_fall_detector -> evaluate); no weights shipped
scripts/                         # run_demo (offline annotated video + alerts JSON), run_all_demos, fetch_samples
config/demo/                     # scenario files: zones + setting overrides (corridor, hallway)
tests/                           # 63 pytest tests (+1 slow end-to-end on the corridor sample)
.github/workflows/ci.yml         # ruff F/E9, pytest, pytest -m slow, frontend lint+build, Docker smoke test
Dockerfile, docker-compose.yml   # multi-stage image with models + samples; webcam profile
```

## Data flow (per frame)

1. **Capture thread** (`VideoSource`): pushes `(frame, ts)` into a queue of 2, dropping the
   oldest. For files, `ts` is the position on the video timeline.
2. **Pipeline thread** (`SentinelPipeline.run` → `process_frame`) runs these steps in order:
   1. `Detector.detect_and_track`: detection and ByteTrack IDs in one ultralytics call, so
      detection and tracking time can't be split without changing that call.
   2. `PoseEstimator.estimate`: runs the pose model on the full frame, matches poses to tracks
      by IoU, and updates `TrackFeatures`.
   3. `AnomalyEngine.process(poses, features, ts)` returns `list[AnomalyAlert]`.
   4. `_annotate_frame` draws boxes, skeletons, zones and banners.
   5. For each alert, in order:
      - `ClipRecorder.save_clip(snapshot=annotated)`;
      - `EventLogger.log_event`;
      - `WebhookNotifier.send`;
      - the `on_alert` callback.
   6. `ClipRecorder.add_frame(annotated, ts)`.
   7. The `on_frame` callback runs.
3. **API** (asyncio): `on_frame` JPEG-encodes the annotated frame once and stores a pre-serialised
   message. Each WebSocket client gets frames at up to 15 fps. Alerts go through per-client queues
   and are never dropped.

The only timing recorded today is `processing_time_ms` (whole frame) and
`detections.inference_time_ms`. There's no per-layer breakdown.

## Event storage (SQLite, `DB_PATH`, default `data/events.db`)

```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  alert_id TEXT UNIQUE,      -- "ALT-XXXXXX"; also names the clip dir data/clips/<alert_id>/
  alert_type TEXT,           -- fall | zone_intrusion | time_exceeded | wrong_direction | loitering | action
  track_id INTEGER,
  timestamp REAL,            -- unix seconds (video timeline anchored to wall clock for files)
  severity TEXT,             -- low | medium | high | critical
  confidence REAL,
  message TEXT,
  details TEXT,              -- JSON: zone_id, duration, clip_path, dwell_seconds, fall signals, ...
  clip_path TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- **Files per alert:** `data/clips/<alert_id>/clip_<alert_id>.mp4` and `snapshot_<alert_id>.jpg`.
- **Not stored today:** camera id, zone id as a column (it's inside `details`), start/end times
  (single `timestamp`), thumbnail path, and verification. There's no migration mechanism.

## Alert object (`anomaly.engine.AnomalyAlert`)

`alert_id, alert_type, track_id, timestamp, confidence, severity, message, details{}`.

Severity map: `fall` is critical, `zone_intrusion` high, `time_exceeded` and `wrong_direction`
medium, and `loitering` low.

## WebSocket `/ws/feed` (server → client, JSON text)

| Message | Sent | Shape |
|---|---|---|
| `history` | once, on connect | `{"type":"history","alerts":[AlertDict…]}` (last 50, oldest first) |
| `alert` | per alert, never dropped | `{"type":"alert","alert":AlertDict}` |
| `frame` | ≤15 fps, only when new | `{"type":"frame","image":<base64 JPEG or null>,"data":{timestamp, frame_number, processing_time_ms, detections:{detections:[{track_id,bbox,confidence,class_id,class_name,centroid,width,height,aspect_ratio}], person_count, vehicle_count, inference_time_ms}, alerts:[…], stats:{person_count, active_tracks, alert_count, processing_time_ms, fps}}}` |

`AlertDict` is `AnomalyAlert.to_dict()`. The client sends nothing.

## REST (all GET)

| Endpoint | Returns |
|---|---|
| `/api/health` | `ok` / `degraded`, and the source |
| `/api/stats` | pipeline and source stats |
| `/api/alerts?limit&severity&alert_type` | from SQLite, with `has_clip` |
| `/api/zones` | overlay polygons |
| `/api/tracks` | speed, displacement, `time_tracked`, `fall_state` |
| `/api/clips/{alert_id}` | the MP4 clip |
| `/api/snapshots/{alert_id}` | the JPEG snapshot |

Unknown `/api/*` routes return 404. Anything else serves the SPA.

## Measured behaviour (from the Sep 28 upgrade)

- **Speed on CPU:** about 17 fps for YOLOv8n and pose together on 768×432 samples, with no GPU.
- **corridor sample (50 s):** 7 `zone_intrusion` alerts, 0 false falls.
- **hallway sample (139 s):** 6 `time_exceeded`, 6 `zone_intrusion` and 4 `loitering`
  alerts, 0 false falls.
- **Falls:** tested only with synthetic keypoint sequences. No freely licensed fall footage
  is bundled.

## Gaps relevant to the plan

- Everything runs on a single camera; there is no `camera_id` anywhere.
- There is no per-layer timing, no fall-detection accuracy metric, and no event migration.
- Notifications support webhooks only.
- The dashboard is a single page with no router.
- PPE and open-vocabulary detection are not implemented.
- There is no LLM integration.
