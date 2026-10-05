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
Zones are normalised polygons (0–1). A live camera starts with **no zones**; draw your own
on the Camera page (they're saved to `backend/config/zones.json` or `data/zones_<camera>.json`).
The scenario demos keep their own zones in files like `config/demo/corridor_demo.json`. A zone
whose edges cross is rejected, and the editor offers a one-click fix (the points reordered
around the centre, or else the outline). All settings can be set in `.env` (see
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

### Live activity labels and the camera-view check
Each person gets a live activity label, drawn above them together with their REBA score, e.g.
`Bending · REBA 9 HIGH · back`. The labels are Standing, Walking, Sitting, Bending, Lifting,
Carrying, Reaching overhead, Lying down and Fallen. They come from pose rules
(`backend/activity/rules.py`): trunk and thigh angles, hip height against the person's own
upright reference, wrist height and speed. They are smoothed over 1 s, except Fallen, Lifting
and Bending, which show at once. Rules that need the legs are skipped when the legs aren't
visible ("Upper body only"), and Carrying without a detected bag or suitcase is marked low
confidence. The last two minutes of labels are kept per person.

If, for 5 s, everyone in view shows only their upper body, the dashboard says so: *"Only upper
body visible. Fall detection and ergonomics need a full-body view: place the camera 2–4 m away,
ideally side-on."* When REBA can't be scored, the panel says why (shoulders, hips or legs not
visible; too small or far away; facing the camera; keypoints unclear).

The rules were tuned **only on CAUCAFall subjects 1–5** and one hand-labelled webcam recording
of a lift and carry; subjects 6–10 stay held out (results in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)). A lift is a bend with the hands low followed by
rising **with the hands up in front**; Carrying then lasts while the hands stay up, so a bend
without picking anything up (arms hang afterwards) isn't a lift. Lifting and Carrying are
validated on that one recording only (every action caught), with no false lifts and 1 false
carry in 50 CAUCAFall clips; the CAUCAFall pick-ups (a quick squat for a small object, front-on)
are still missed.

### Test with demo footage
The Camera page has a **Test with demo footage** panel. It plays a clip on loop as a separate
`test` camera in warehouse mode, so you can watch activity labels and REBA without moving your
camera. The clips are:
- your own recordings in `data/recordings` (a side view of lifting and carrying works best);
- local CAUCAFall clips from subjects 1–5 (never bundled; fetch them with
  `training/fetch_caucafall.py`).

Test footage never sends notifications or webhooks and never enters the event history.

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

### Open issues
Tracked here until fixed; each links to the numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Lying on the floor: the person box is missing in 57% of frames** (YOLO11m at 0.25, own
  recording; it was 100% with YOLOv8n). Every missing frame is kept as "lost while lying", so the
  person isn't silently dropped, but there's no pose to judge. To revisit after the
  chair-slide/bed-exit work: the rotated-image recovery (`anomaly/fall_recovery.py`, off by
  default) or fine-tuning the detector on fall-dataset images.
- **Falls: the track id changes mid-fall** in 5 of 30 URFD falls (17 before the 2026-10-05
  tracker fix); URFD on-the-ground recall is 21/30.
- **Kneeling on all fours or crouching to the floor and holding still confirms a fall** (4
  CAUCAFall daily-activity clips). The steadier RTMPose keypoints pass the stillness check that
  the old model's jitter used to break.
- **Ceiling-height cameras:** walking is labelled Sitting in ~20% of frames (balance checks
  off), and seated people often read as Bending; the shin/thigh seated cue only works at eye level.
- **Fall signals are scaled by a standing height learned once:** walking toward a webcam reads as
  descending and the head as dropping (the frame-edge fix stops the false alarm, not the cause).
- **Slow slides out of a chair or off a bed raise no fall alert** (own chair/bed recording):
  the descent is too slow for the fall rules. That is the chair-slide/bed-exit step.

### Known dependency risks
- **The hosted LLM can disappear.** Search and the rule compiler use a hosted model, and
  providers retire models without notice: NVIDIA's `nemotron-3-super-120b-a12b`, the default
  until 2026-10-03, started returning HTTP 410 that day. Switching is one setting in `.env`:
  `LLM_MODEL=` for another model on the same provider, or `LLM_PROVIDER=anthropic` (with
  `ANTHROPIC_API_KEY`).
- **What keeps working without it.** Rules you've already confirmed keep running, because they
  are checked by local code with no LLM. Only writing new rules and asking search questions
  stop.
- **After switching models,** re-run `python scripts/eval_rules.py` and
  `python scripts/eval_search.py`. Accuracy is measured per model.

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

### Desk movement coach (laptop webcam)
Switch the mode to **Desk Posture Coach** and start your webcam. It is a *movement* coach,
not a posture police. Nobody holds one posture all day and shifting is normal; the real
problem is staying still too long.

- **What it shows:** one card with "Still for 23 min" (time since you last moved), the
  movement reminder and the stretch break, plus a line with today's breaks, static time and
  longest still stretch. Calibration, the posture details and the timeline are under
  **Advanced** (collapsed).
- **What counts as moving:**
  - getting up;
  - a stretch break;
  - a large position change held for about 8 s, such as moving your chair, leaning far back
    or standing partly.

  It is measured against your own position, so it works without any calibration.
- **What doesn't count:** typing, fidgeting, glancing around, and ordinary seated posture
  changes between upright, slouching and leaning.
  - On a laptop webcam, a small torso turn swings the shoulders' apparent width and angle a
    lot. A real session showed this, and the thresholds are set from it.
  - Per-second measures are logged locally to `data/posture_movement_log_laptop.jsonl` for
    tuning.
- **Static time** counts only still stretches longer than 10 minutes.
- **One reminder** comes after 30 minutes without moving, whatever your posture (15–60 min is
  configurable). It offers a 1-minute guided stretch break: neck tilts, shoulder shrugs and a
  stand-up, counted by the camera. Moving resets it.
- **Short-term posture is never flagged.** Warnings come only for sustained extremes: the head
  far down, or a strong lean, held for 20 minutes (configurable, or off).
  - These need a reference: an upright baseline, or the personal calibration below.
  - When one fires, a faint outline of your own good posture appears on the video, with
    "Back to good" once you've corrected.
- **Personal calibration (optional):** record a few postures, and a small local model learns
  them. It feeds the long-hold checks and the setup tips. Its moment-to-moment view is under
  Advanced.
- **Setup tips** come from your history, for example "your screen is probably too low".
- **Demo timings** (Settings, clearly labelled) shorten the reminder to 1 minute and the
  long-hold warnings to 2, for testing and recording a demo.
  - **They're on by default for now,** with a badge on the coach page. Turn them off in
    Settings for everyday use.
  - Once the demo is recorded, the default goes back to off.

Everything runs on this computer. No video, posture or movement data leaves it, and no LLM is
involved. The webcam gives 2D estimates; this is not a medical or ergonomic assessment.

## Privacy by design
Sentinel watches people, so it is built to keep as little about them as it can.

- **Video stays on this computer.** Detection, pose, falls, zones, activity labels, REBA and
  the posture coach all run locally. Frames are never uploaded.
- **The LLM sees metadata, never pixels or keypoints.** Search sends event fields to the
  configured LLM provider: id, type, severity, camera, start and end time, weekday,
  duration, zone name, track id, message, verified, and for one event its attributes and
  confidence (e.g. REBA scores, dwell time). The rule compiler sends your rule text, the zone
  names and types, and the object classes. Nothing is sent unless you configure a provider.
- **No face recognition, no identities.** People are anonymous track ids that change when
  someone leaves the view and comes back.
- **No rankings of individuals.** Analytics are per zone, per hour and per posture, never per
  person: there is no time-at-risk or event count per track id (schema v4 deleted the old
  per-track totals, after a backup). Each event still records its track id, and the live
  view shows the people in frame right now.
- **Retention.** History is kept for 30 days by default (Settings > Privacy: 7 days to a
  year, or forever). After that, events with their clips, snapshots and skeletons,
  recordings, time-at-risk totals, posture history, the movement-coach log and database
  backups are deleted, at startup and then hourly. Until you first open the Privacy panel,
  retention only does dry runs and logs what it would delete. The panel shows what is stored
  and has a confirmed **Delete now**.
- **Skeleton-only mode.** A switch in Settings > Privacy. When it's on, no clip, snapshot or
  recording is ever written. Each event keeps the keypoints from 10 s before to 5 s after
  (normalised, no pixels), played back as a stick figure, and notifications are text only.
- **Notifications send images off this computer.** Telegram and email alerts include the
  event snapshot, unless skeleton-only mode is on.
- **The exception: the desk posture coach.** It tracks one person, you, at your own desk,
  and keeps your own history (minutes, corrections, breaks, the movement log). It is for
  self-tracking only, stays on this computer, and follows the same retention.
- **Planned, not built:** privacy zones that blur part of the frame (Phase 3b) and face blur
  for exam-hall review (Phase 3d).

**Real deployments.** Watching employees or the public needs more than software settings:
tell people (notices at the camera), get consent or a lawful basis, limit who can see the
dashboard, and have the setup reviewed for your jurisdiction (e.g. GDPR, workplace
monitoring and works-council rules) before using it.

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
| `GET /api/demo/clips`, `POST /api/demo/play {"clip"}`, `POST /api/demo/stop` | demo-footage clips (no file paths), and playing or stopping one on the `test` camera |
| `GET /api/privacy`, `PUT /api/privacy`, `POST /api/privacy/delete-now` | what's stored, retention and skeleton-only settings, and deleting past the retention now |
| `GET /api/skeletons/{alert_id}` | skeleton-only mode's keypoints around an event (JSON) |

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
  activity/  live activity labels (pose rules) and the camera-view check
  anomaly/   fall_detector, zone_monitor, engine (loitering, alert routing), temporal_model
  rules/     plain-English rules: dsl (format), compiler (LLM), engine (per frame), presets, store
  output/    clip_recorder, skeleton_recorder, event_logger (SQLite), webhook
  privacy/   retention (what's kept, for how long)
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
- **Learned activity labels.** The pose-sequence ActionLSTM in `training/` could replace the
  hand-written rules where they are weakest, such as lifting and carrying, once there is
  side-view warehouse footage to train and test on. The rules would stay as the fallback.
- **Suggested zones** from open-vocabulary detection (YOLO-World), in Phase 3b.
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
