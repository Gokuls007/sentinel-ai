# Sentinel AI: Real-Time Video Anomaly Detection

[![CI](https://github.com/Gokuls007/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Gokuls007/sentinel-ai/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n%20%2B%20pose-brightgreen.svg)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Sentinel AI watches a video feed, tracks every person with a stable ID, and estimates each
person's 17-point skeleton. It raises alerts for **falls**, **restricted-zone intrusions**,
**time limits exceeded in a zone**, **walking the wrong way through one-way zones**, and
**loitering**. Every alert is saved with a snapshot and a playable clip (10 s before the alert
and 5 s after it), logged to SQLite, optionally sent to a webhook, and streamed live to a
browser dashboard.

**No camera needed.** It includes two sample scenarios that download automatically, so the
full system runs with one command.

![Corridor demo: restricted-door intrusions flagged live](docs/corridor_alert.jpg)

## Quick start (no camera)

**Docker**, the simplest route (the image already includes the models, sample videos and dashboard):
```bash
docker compose up --build
```
Then open **http://localhost:8000**.

**Or locally** (Python 3.11):
```bash
pip install -r backend/requirements.txt
cd frontend && npm ci && npm run build && cd ..
python backend/main.py --demo
```
Then open **http://localhost:8000**. The first run downloads YOLOv8n and YOLOv8n-pose (~13 MB)
and the sample video (checksum-verified).

| Scenario | Command | What you'll see |
|---|---|---|
| Office corridor | `python backend/main.py --demo corridor` | People walking past two restricted doorways: `zone_intrusion` alerts |
| Hallway workstation | `python backend/main.py --demo hallway` | People stopping at a table: `time_exceeded` (5 s zone limit), `loitering`, and staff-exit intrusions |

Measured with the full pipeline on a laptop CPU (no GPU) over each sample video, one pass
through the file (`python scripts/run_demo.py --demo <name>`):

| Scenario | Length | Alerts raised |
|---|---|---|
| corridor | 50 s | 7 `zone_intrusion` (door entries), 0 false falls |
| hallway | 139 s | 6 `time_exceeded`, 6 `zone_intrusion`, 4 `loitering` (6 people), 0 false falls |

The two YOLO models together run at about 17 fps on CPU, faster than both samples'
native frame rate (10–12 fps).

### Your own video or camera
```bash
python backend/main.py --source path/to/video.mp4 --loop
python backend/main.py --source 0                         # webcam
python backend/main.py --source rtsp://user:pass@cam/stream
docker compose --profile webcam up --build                # webcam in Docker (Linux)
```
Zones are normalised polygons (0–1) in `backend/config/zones.json`, or in a scenario file
like `config/demo/corridor_demo.json`. All settings can be set in `.env` (see
[`.env.example`](.env.example)), and CLI flags override them (`python backend/main.py --help`).

## What each detector does

| Alert | How it's decided | Severity |
|---|---|---|
| `possible_fall` | A rapid hip descent (faster than 1.2 body heights per second, calibrated to each person's height), then the person is on the ground. "On the ground" uses the torso angle plus direction-independent signals (hips near the floor, the skeleton collapsing), and it doesn't count while the hips are still high, as when bending to pick something up. Shown on the dashboard only, never notified. | medium |
| `fall` | The confirmed fall: still on the ground after 1 s of stillness. One alert per fall; notified (Telegram/email/webhook). | critical |
| `zone_intrusion` | The hip point enters a `restricted` polygon. | high |
| `time_exceeded` | Dwell in a `time_limited` zone passes its limit. | medium |
| `wrong_direction` | Sustained movement against a `one_way` zone's direction. | medium |
| `loitering` | Staying within a small radius for longer than a threshold (30 s by default). | low |

Every alert type has a per-person cooldown, so one event never becomes a burst of alerts.
[ARCHITECTURE.md](ARCHITECTURE.md) covers the threading model, the WebSocket protocol, and
each detector in detail.

### Fall detection on people it has never seen
These results come from CAUCAFall **subjects 6–10 only**: 25 falls and 25 daily activities
(walk, hop, pick up an object, sit down, kneel). Those five people were never used to design,
tune or train anything. Tuning and training used URFD and CAUCAFall subjects 1–5. The split
is **by subject**, so nobody appears on both sides. CAUCAFall was filmed in a home with
**occlusions, varied lighting and night-time infrared footage**, which makes it harder than
staged lab datasets.

| Method | Confirmed falls caught | Confirmed false alarms | Possible falls caught | Possible false alarms |
|---|---|---|---|---|
| Rules, original (torso angle only) | 2 / 25 (8%) | 4 | 7 / 25 (28%) | 6 |
| **Rules + direction-independent signals (default)** | **4 / 25 (16%)** | **0** | **10 / 25 (40%)** | **0** |
| Learned model: gradient boosting on 12 pose features (experiment, offline) | 14 / 25 (56%) | 3 | 18 / 25 (72%) | 7 |

The false alarms are counted over 4.5 min of no-fall video, so one false alarm is about 13 per
hour and one fall is 4 points of recall.

- **The default is the rules version with zero false alarms.** The "hips still high" veto
  removed every picking-up and sitting-down false alarm. Confirmed falls are sent as
  notifications, so a false alarm costs the most.
- **The learned model catches many more falls** but raises false alarms when people sit down
  or pick things up. It is kept as a documented experiment
  (`python training/fall_round3.py final`) and is not in the live pipeline.
- **Most missed falls are visibility problems**: the person is not detected during the fall, or
  never gets measured standing, in occluded or infrared footage. Fall direction is a smaller
  factor. Details, the confirmation-time sweep and the URFD numbers are in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Honest limitations
- **Falls are not in the bundled demos.** The sample videos have no falls. The fall datasets
  are downloaded by `training/eval_fall.py --download` (URFD) and
  `training/fetch_caucafall.py` (CAUCAFall), never committed. Unit tests cover synthetic pose
  sequences: falls, sitting down, crouching, bending, a webcam with no ankles in view, and
  people lost from view on the floor.
- **Fall recall is modest** (see above). Treat a missing alert as possible; the possible-fall
  level gives earlier, softer warnings on the dashboard.
- **The action LSTM is off by default.** `training/` can train a pose-sequence classifier. Until
  a trained `models/action_lstm.pt` exists, the classifier stays disabled rather than guessing
  with random weights.
- **PPE detection is not implemented.**
- Detection quality is YOLOv8n's: small or heavily occluded people can be missed. Try a
  larger model with `DETECTION_MODEL=yolov8s.pt`.

## Dashboard
The React dashboard shows:
- the annotated live feed;
- live person and track counts and pipeline latency;
- each track's fall state;
- the zones;
- a 30-minute alert timeline;
- an alert feed with snapshot thumbnails and inline clip playback.

Alert history comes from SQLite, so it survives restarts. When the backend is down the
dashboard says so; it never shows made-up data.

For frontend development, run `cd frontend && npm run dev` (http://localhost:5173). It
proxies to the backend on port 8000.

### Search: ask about events in plain English
The **Search** page answers questions like "how many falls were there at the loading dock
this week?" or "who went into chemical storage after 6pm yesterday?".

How it works:
- A language model calls read-only tools over the event log: count, find, look up one
  event, list zones and cameras.
- It answers with citations like `[#31]`, each linking to that event's clip.
- Any citation to an event the tools didn't return is removed, so links never point at
  made-up events.

Setup:
1. Put an API key in `.env`. The default, `LLM_PROVIDER=nvidia`, needs `NVIDIA_API_KEY`
   (free trial keys at build.nvidia.com) and uses `nvidia/nemotron-3-ultra-550b-a55b`.
   `LLM_PROVIDER=anthropic` needs `ANTHROPIC_API_KEY` and uses `claude-sonnet-5-5`.
2. Restart the server.

Privacy and cost:
- Only event metadata is sent to the provider: type, zone, time, track and message. Video
  frames and clips are never sent.
- The model never runs per frame.
- Search is allowed from this computer only (`ALLOW_REMOTE_SEARCH`) and is rate-limited.

Accuracy is measured by `python scripts/eval_search.py` (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

### Rules: write safety rules in plain English
On the **Rules** page you type a rule such as "alert if someone stays in the loading dock for
more than 30 seconds". It is compiled once into a structured rule, you check a plain-words
preview, and only then does it run on every frame.

How it works:
- **The compiler** (`backend/rules/compiler.py`) sends your sentence and the zone names to
  the LLM once. The model must either submit a rule from a fixed set of conditions, or refuse
  and say what's missing. Examples of refusals: helmets, distance to forklifts, "looks tired".
  - The rule is validated, including that every zone exists. Errors go back to the model
    for one repair attempt.
  - It never invents a zone or a condition.
- **The preview** is written by code, not the model, for example:
  "Person · in Loading Dock · for 30 s → medium alert, clip". It comes with warnings worth
  checking, such as a loosely matched zone name, or an instant rule that may be noisy.
- **The engine** (`backend/rules/engine.py`) runs in warehouse mode with no LLM involved. It
  keeps a timer and a cooldown per rule and per person, and fires once per episode. Rule
  events go through the same clips, store, notifications and dashboard as other alerts.
- **Conditions available now:**
  - being in or out of a zone;
  - a fall;
  - not moving for a while;
  - 2D REBA posture risk;
  - people counts, overall or per zone;
  - a time window;
  - holding a phone, laptop or book;
  - head turned;
  - looking down.

  Helmets, vests and forklifts need open-vocabulary detection (Phase 3b).
- **Cost:** a phone rule adds that class to the detector's existing pass, and only while such
  a rule exists. Ten rules add about 0.2 ms per frame.
- **Presets:** "Warehouse safety" loads a ready-made set of rules.
- **Built-in rules:** with `RULES_BUILTINS=true`, falls and restricted zones run as built-in
  rules instead of the original detectors. It's off until the parity check
  (`python scripts/rules_parity.py`) shows they match.

The compiler's accuracy is measured by `python scripts/eval_rules.py` (see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## API
| Endpoint | Returns |
|---|---|
| `GET /api/health` | `ok` / `degraded`, the source, and any source error |
| `GET /api/stats` | frames processed, fps, uptime, and source stats |
| `GET /api/alerts?limit=&severity=&alert_type=` | persisted alerts, newest first, with `has_clip` |
| `GET /api/zones`, `GET /api/tracks` | configured zones, and live tracks with fall state |
| `GET /api/clips/{alert_id}`, `GET /api/snapshots/{alert_id}` | the incident MP4 (H.264) and JPEG |
| `WS /ws/feed` | `history`, then `alert` and `frame` messages |
| `POST /api/search {"question"}` | Server-Sent Events: `tool_call` / `tool_result` steps, then `done` with the answer and cited events |
| `GET /api/search/status` | whether search is configured, plus the provider and model (never keys) |
| `POST /api/rules/compile {"text"}` | a draft rule with its preview and warnings, or a refusal (saves nothing) |
| `POST /api/rules`, `GET /api/rules`, `PATCH /api/rules/{id}`, `DELETE /api/rules/{id}` | confirm, list (with fire counts), edit or turn off, delete |
| `GET /api/presets`, `POST /api/presets/{name}/apply` | presets, and loading one (asks before replacing preset rules) |

Set `WEBHOOK_URL` to get every alert POSTed as JSON.

## Offline demo videos and reel
```bash
python scripts/run_demo.py --demo corridor    # writes outputs/corridor_demo.mp4 plus an alerts JSON
python scripts/run_all_demos.py               # both scenarios, stitched into assets/demo_reel.mp4
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q            # unit tests (~6 s): fall state machine, zones, loitering, config, API/WebSocket, clips
pytest -q -m slow    # end to end: the real models on the corridor sample must flag the doors and nothing else
```
CI runs both, plus the frontend lint and build, and a Docker build with a smoke test.

## Project layout
```text
backend/
  core/      video_source, detector (YOLOv8 + ByteTrack), pose_estimator, pipeline, samples
  anomaly/   fall_detector, zone_monitor, engine (loitering, alert routing), temporal_model
  rules/     plain-English rules: dsl (format), compiler (LLM), engine (per frame), presets, store
  output/    clip_recorder, event_logger (SQLite), webhook
  api/       FastAPI server (REST, WebSocket, serves the dashboard)
config/demo/ scenario zone files
frontend/    React + Vite + Tailwind dashboard
scripts/     run_demo, run_all_demos, fetch_samples, post_production
training/    pose-sequence LSTM data prep / training / evaluation
tests/       pytest suite
```

## Future work
- **Learned model for the "possible fall" level.** Gradient boosting on pose features catches
  far more falls on unseen subjects (56% vs 16% confirmed). Next step: add the same "hips still
  high" veto to cut its sitting and bending false alarms, tuned on CAUCAFall subjects 1–5 only,
  and use it for the dashboard-only possible-fall level. The rules stay in charge of the
  confirmed fall that sends notifications.
- **Keeping lying people detected.** Most missed falls happen because YOLOv8n stops detecting
  the person once they are on the floor. Re-finding them around their last box, including
  with the image rotated 90°, is implemented (`anomaly/fall_recovery.py`) but off by default:
  on unseen subjects it added false alarms.
- **More rule conditions:** helmets, vests and forklifts (Phase 3b, after the YOLO-World
  accuracy test), and the exam-hall preset with its review page (Phase 3d). See
  [docs/plans/phase-3-rules.md](docs/plans/phase-3-rules.md).

## Credits
Fall datasets (downloaded by the evaluation scripts, not redistributed):
[UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) (Kwolek & Kepski, 2014;
CC BY-NC-SA 4.0) and [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) (Eraso et al.,
Mendeley Data, doi:10.17632/7w7fccy7ky.4; CC BY 4.0).

The sample videos are by Intel Corporation
([intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos)) under
CC BY 4.0; see [demo_videos/ATTRIBUTION.md](demo_videos/ATTRIBUTION.md). Detection and pose
estimation use [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## License
MIT. See [LICENSE](LICENSE). Note that Ultralytics YOLOv8 is AGPL-3.0, which applies if you
distribute or serve a product built on it.
