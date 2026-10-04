# Exam Hall Mode — Full Specification

Put this file at `docs/plans/exam-hall.md`. It replaces the Exam Hall placeholder
and extends section 3.6 of `docs/plans/phase-3-rules.md`. Where this file and the
Phase 3 plan disagree, this file wins.

Build it in the stages in section 12, one branch per stage, plan first, merged on
its checklist, exactly like earlier phases.

---

## 1. Purpose and principles (non-negotiable)

Exam Hall mode helps a human invigilator notice moments worth a second look during
an in-person exam filmed by one fixed camera at the front of the room.

1. **Flags for human review only.** The system never says anyone cheated. The words
   "cheating", "cheater", "suspicious student" or similar never appear in the UI,
   reports, events or logs. Use "flagged for review" and neutral behaviour
   descriptions ("head turned toward seat B2 for 9 s").
2. **No identity.** No face recognition, no names, no student IDs in the system.
   People are referred to by **seat label** (A1, A2, …) only. Linking a seat to a
   student happens outside the system, on paper, by the invigilator if needed.
3. **Low confidence never flags.** Every signal carries a confidence; below the
   threshold it is logged for debugging but never shown as a flag.
4. **Faces blurred by default** in every saved clip, thumbnail and in the review
   page. The head outline stays visible so head direction can still be judged.
   Unblur requires a typed reason and is logged (who/when/which flag/reason) and
   shown on the review page. No bulk unblur.
5. **Nothing automatic happens to a student.** No penalties, no scores per student,
   no ranking of students. The final report lists flags the reviewer chose to keep,
   with the reviewer's notes.
6. **Local only.** Video never leaves the machine. Exam mode never sends anything to
   an LLM except when the user compiles a rule (text only) or asks the search agent
   a question (event metadata only).
7. **Notice.** A printable "This exam is recorded" notice template is provided
   (section 9.4). The README says real use requires notice, consent where required,
   and institutional/legal review.

---

## 2. Camera setup and room assumptions

- One fixed camera at the front, ideally **raised (2–2.5 m)** and angled down,
  seeing every seated student's head, shoulders, arms and desk top.
- Designed for **small rooms: up to ~20 seats** in view. Beyond that, faces and
  keypoints are too small to be reliable. The setup check (section 3.2) must warn
  when people are too small.
- Works with the laptop webcam for testing with 1–3 people.

---

## 3. Exam session setup flow

### 3.1 Exam sessions
New concept: an **exam session** with name, room label, start and end time,
camera id, sensitivity preset, status (setup / calibrating / live / ended /
reviewed). Stored in SQLite (new schema version, with the usual backup before
migrating). All exam events belong to a session.

### 3.2 Setup check (before starting)
Show live video with a checklist that updates live:
- Camera view: upper bodies visible for most people; reuse the existing camera-view
  check. Warn if heads/shoulders are too small (e.g. shoulder width under a
  minimum in pixels) — "Students are too small in the image; move the camera
  closer or use fewer seats in view".
- Lighting: warn if keypoint confidence is low for most people.
- People count shown.

### 3.3 Seat map
- **Auto-detect seats:** when the user presses "Detect seats", find seated people
  (activity label Sitting, or upper body stable for 5 s) and create one seat region
  per person around their head/shoulder/desk area.
- Seats are labelled in reading order: rows front to back (A, B, C…), seats left to
  right (1, 2, 3…), from the invigilator's view. Show labels on the video.
- The user can drag, resize, delete, add and relabel seats. Seats are saved per
  session (and can be copied from a previous session in the same room).
- **Neighbour graph:** compute each seat's left/right/front/back neighbours from
  the seat layout. Used by the "toward neighbour" signals.
- Seat assignment each frame: a person belongs to the seat whose region contains
  their shoulder midpoint. Seat identity is more stable than tracker IDs, so all
  exam logic is keyed by **seat**, not track id.
- **Invigilator handling:** anyone standing or walking and not inside a seat is
  treated as staff/invigilator and ignored by all exam signals.

### 3.4 Calibration period (first 2–3 minutes after start)
Each seat's "normal" differs because students sit at different angles to the
camera. During calibration, record per seat:
- typical head yaw and pitch while writing/reading (median and spread),
- typical hand height relative to the desk/shoulders,
- typical posture.
All later signals compare a seat against **its own baseline**, not a global
threshold. Show a "Calibrating — 1:42 left" banner. No flags during calibration.
Seats that appear after calibration (late arrivals) calibrate individually over
their first 2 minutes.

---

## 4. Signals (what can be flagged)

Each signal produces: seat, start/end time, duration, confidence (0–1),
supporting measurements, and a short neutral description. A flag is created only
when confidence ≥ threshold and duration ≥ minimum (both per sensitivity preset).

| Signal | What it means | How |
|---|---|---|
| **Head turned toward neighbour** | Head turned clearly toward a neighbouring seat for a sustained time | Head yaw vs the seat's baseline beyond threshold, sign tells direction; map direction to the neighbour graph. Default min 6 s. |
| **Mutual turn** | Two neighbouring seats turned toward each other at the same time | Both seats have "head turned toward neighbour" pointing at each other, overlapping ≥ 3 s. Higher priority. |
| **Looking back** | Head turned strongly backward | Large yaw beyond side view / ears-only view; min 4 s. |
| **Phone visible** | A phone detected at the seat | COCO `cell phone` detection inside or overlapping the seat region, held ≥ 2 s (any confidence above detector threshold); no YOLO-World needed. High priority. |
| **Hands below desk for long** | Both hands hidden below desk level for an unusually long time | Wrists not visible or below the seat's baseline hand height for ≥ 20 s, compared with the seat's own calibration. |
| **Looking at lap** | Head pitched down toward lap (not desk) while hands are below desk | Head pitch beyond the seat's writing baseline **and** hands-below-desk at the same time, ≥ 8 s. |
| **Left seat** | Seat empty for a while during the exam | No person in the seat for ≥ 30 s (configurable). Low priority; often legitimate (bathroom with permission). |
| **Object passed** *(stretch goal)* | A hand crosses from one seat region into a neighbour's and back | Wrist enters neighbour's seat region and returns within 5 s. Mark experimental. |

**Important: looking down on its own is never a flag.** Students look down at
their paper almost all the time. Looking down only counts as part of "Looking at
lap" together with hands below the desk.

Do not include talking/mouth movement or gaze tracking: not reliable at this
resolution. List them under future work.

### 4.1 Sensitivity presets
Low / Medium (default) / High: each sets confidence thresholds, minimum durations
and which signals are on. Show the exact values in Settings. Custom values allowed.

### 4.2 Review priority
Each flag gets a priority (High / Medium / Low) from signal type, duration and
confidence. Phone visible and Mutual turn start High. Left seat starts Low.
Priority only orders the review list; it is never shown as a score per student.

### 4.3 Debounce and merging
- One flag per seat per signal per episode; consecutive episodes within 10 s are
  merged.
- A flag's clip covers 5 s before to 5 s after the episode (capped at 30 s), with
  faces blurred.

---

## 5. Live invigilator view (Exam Hall Live page)

- Video with seat outlines and labels. Seats with an active signal glow amber;
  High-priority signals glow red. No text over students' faces.
- **Room grid:** a schematic seat map (boxes in the room layout). Each box shows
  seat label and current state: normal / active signal (icon + seconds) /
  empty / not visible. Clicking a seat shows its last 5 minutes as a timeline.
- **Flag feed:** newest flags with seat, signal, duration, priority, and a
  "Review now" button.
- Session controls: Start exam (begins calibration), Pause, End exam.
- Optional sound for High-priority flags (off by default).
- Camera-view banner from warehouse mode reused when the view is poor.
- Live video face blur toggle (default **off** for the live view because the
  invigilator is physically present; **on** for everything saved).

---

## 6. Review page (Exam Review)

- Session picker, then a **timeline**: one row per seat across the exam duration,
  with flag markers coloured by priority.
- **Flag list**, sorted by priority then time, filterable by seat, signal,
  priority and review status.
- **Flag detail:** blurred clip with play/pause/scrub and 0.5× speed, the seat
  map with the seat highlighted, the signal description, measurements and
  confidence, and a mini chart of the relevant measurement over the clip
  (e.g. head yaw vs baseline).
- **Reviewer actions:** Dismiss (with optional reason), Keep for follow-up,
  Add note. Keyboard shortcuts: J/K next/previous, D dismiss, F keep, N note,
  Space play/pause.
- **Unblur:** button asks for a typed reason; logged; banner shows clip is
  unblurred; re-blurs when leaving the flag. Unblur log visible on the page.
- **Progress bar:** reviewed / total.
- **Export report** (after review): a printable HTML/PDF and a CSV listing only
  kept flags: seat, time, signal, duration, reviewer note, and a link to the clip
  if retained. Header states: "Moments flagged for human review. Not evidence of
  misconduct on their own." No dismissed flags, no per-student totals.

---

## 7. Integration with existing systems

- **Modes:** Exam Hall uses the existing mode switcher. Exam mode runs detection,
  tracking, pose, the exam signal engine, and (if enabled) exam rules. It does
  **not** run fall alerts to Telegram, warehouse zones, REBA or the posture coach.
- **Events:** exam flags are events with type `exam:<signal>` in the existing event
  store, linked to the session and seat, so search and analytics work. Analytics
  for exams show per session and per signal, never per seat over multiple sessions.
- **Rules engine:** add an **Exam hall preset** with rules equivalent to the
  default signals. Add DSL conditions `head_turned_toward_neighbour`,
  `mutual_turn`, `phone_visible`, `hands_below_desk_for`, `looking_at_lap`,
  `left_seat_for` with Pydantic validation and unit tests. The compiler maps
  phrases like "student looking at the person next to them for 5 seconds" to these
  and refuses anything about intent ("if they're cheating") or identity.
- **Search agent:** can answer "how many phone flags in this morning's exam?" and
  "show high-priority flags not reviewed yet". Add a `session` filter to the tools.
- **Privacy features:** retention applies to exam sessions (separate setting,
  default 14 days). Skeleton-only mode works in exam mode: clips become stick-figure
  playback with seat outlines. Notifications are off in exam mode.
- **Demo footage:** the demo-footage panel can play exam recordings from
  `data/recordings` as a test camera; test events are tagged TEST and not stored,
  as in warehouse mode. Allow running a full session on demo footage for testing.

---

## 8. Data model (suggested)

- `exam_sessions(id, name, room, camera_id, preset, status, started_at, ended_at, created_at)`
- `exam_seats(id, session_id, label, polygon_json, row, col, neighbours_json)`
- `exam_seat_baselines(seat_id, yaw_med, yaw_spread, pitch_med, pitch_spread, hand_height_med, samples, calibrated_at)`
- `exam_flags(id, session_id, seat_id, signal, start_ts, end_ts, confidence, priority, measurements_json, clip_path, status[open|dismissed|kept], reviewer_note, dismissed_reason, reviewed_at)`
- `exam_unblur_log(id, flag_id, reason, at)`
- Events table gets the flag via `type=exam:<signal>` and `attributes` with session and seat ids.

Adjust to the repo's existing conventions; follow the EventStore and migration
patterns already used.

---

## 9. Privacy and responsible use

### 9.1 Face blur
- Blur faces using the pose head keypoints (nose, eyes, ears) to place an elliptical
  blur over the face only, keeping the head outline and orientation visible.
- Blur is applied when writing clips and thumbnails (blurred files on disk), not
  only in the browser. Unblurred originals are **not** kept unless the user turns
  on "Keep unblurred originals for N days" (off by default). If off, unblur shows
  a message that no unblurred copy exists, and is still logged.

### 9.2 What is never stored
Face crops, embeddings, identity of any kind, per-student scores across sessions.

### 9.3 README section "Responsible use: Exam Hall"
Human review required; false-flag risk with concrete examples (stretching,
thinking with head turned, looking at the clock, left-handed writing posture);
notice to students; no automated penalties; bias considerations (camera angle,
seat position, lighting, disabilities and assistive devices affect signals);
evaluation numbers and their limits.

### 9.4 Notice template
`docs/exam_notice_template.md`: a short, plain notice that the exam is recorded,
what is analysed, that a human reviews any flagged moment, retention period, and
who to contact.

---

## 10. Evaluation

### 10.1 Footage
Staged sessions with volunteers who give consent:
- **Session A (tuning):** 3–6 people at desks, ~15 min.
- **Session B (held-out test):** different day or seating order, ~15 min, never
  used for tuning.

Each session includes long stretches of honest exam behaviour (writing, reading,
thinking, stretching, looking at the clock, dropping a pen) plus scripted actions:
head turn toward neighbour (5–15 s), mutual turn, looking back, phone in hand,
hands under desk with head to lap, leaving seat, passing an object.

### 10.2 Labelling tool
`scripts/label_exam.py`: step through the recording, mark start/end of each
scripted action per seat, and mark "honest" segments. Saves JSON next to the clip.

### 10.3 Metrics (in BENCHMARKS.md)
- **False flags per student-hour** on honest behaviour, per signal. This is the
  headline number.
- **Catch rate per scripted action**, with time-to-flag.
- Precision of flags (fraction of flags that match a scripted action).
- Separate tables for Session A (tuning) and Session B (held-out), clearly labelled.
- Report sample size and caveats; never round small samples into big claims.

### 10.4 Unit tests
Synthetic poses for every signal (positive and negative), seat assignment, the
neighbour graph, calibration baselines per seat, merging/debounce, priority,
review actions, unblur logging, export contents (no dismissed flags, no per-student
totals), invigilator exclusion, and the "never use the word cheating" check
(a test that scans UI strings and report templates).

---

## 11. Settings (Exam Hall section)

Sensitivity preset and custom thresholds; signals on/off; minimum durations;
calibration length; left-seat time; live face blur; keep-unblurred-originals
(off); exam retention days; high-priority sound.

---

## 12. Build stages

Each stage: short plan first, own branch from main, tests, CI green, then merge.

1. **E1 — Sessions, setup check, seat map, calibration.** Exam session model and
   migration; setup checklist; auto-detect seats with editing; neighbour graph;
   per-seat calibration. Live page shows seats and calibration progress.
2. **E2 — Signal engine.** All signals in section 4 except object passing;
   sensitivity presets; priority; merging; flags written as events with blurred
   clips (face blur from 9.1 built here).
3. **E3 — Review page.** Timeline, flag list, detail view, reviewer actions,
   shortcuts, unblur with log, export report.
4. **E4 — Live invigilator view.** Room grid, seat states, flag feed, session
   controls, sound.
5. **E5 — Integration.** Exam rules preset and new DSL conditions with compiler
   eval cases; search session filter; retention and skeleton-only support;
   demo-footage sessions.
6. **E6 — Evaluation and docs.** Labelling tool, eval script, BENCHMARKS tables,
   README responsible-use section, notice template, demo GIF.
7. **Stretch:** object passing (experimental flag).

### Done when
- A full mock exam with 3+ people can be set up, run, reviewed and exported
  end to end.
- Session B results are reported honestly in BENCHMARKS.md.
- The word "cheating" appears nowhere in the product (test enforced).
- All existing tests still pass; warehouse and posture modes are unaffected.

---

## Status

### E1 built (branch `exam-e1-sessions`)
Sessions, the live setup check, the seat map (detect, drag/resize/add/delete/relabel, copy from
another session), the neighbour graph, seat assignment with staff excluded, and per-seat
calibration with late arrivals. Schema v5 adds all five exam tables in one migration (backup
first). Small deviations, all following the repo's conventions:
- `exam_sessions` has a `calibration_s` column (per-session calibration length, 30-600 s) and
  `exam_unblur_log` an `actor` column for the "who" of section 1.4.
- Seats are rectangles in the UI, stored as 4-point polygons (`polygon_json`).
- Head pose sits behind `exam.head_pose.HeadPoseEstimator`; the default is a 2D estimate from
  the face keypoints (`KeypointHeadPose`). A dedicated model can replace it without other changes.
- Seats are drawn by the Exam page over the video (labels at the desk corner, never over a
  face), not burned into the live feed; drawing into saved clips comes with E2.
- Only one open (not ended) session per camera at a time.
