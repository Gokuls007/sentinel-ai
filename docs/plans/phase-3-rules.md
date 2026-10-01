# Phase 3 plan: plain-English rules and presets (draft for approval, revised)

**Status:** plan only. Implementation starts after **Phase 0 and Phase 1 are merged** and
the **fall confirmation sweep** is done.

## Positioning
Warehouse and industrial safety stays the main story. Plain-English rules let the same
system work in other settings, shown with two short demos: an **exam hall** and a **desk
posture coach**.

There is no separate feature per domain. Each demo is a *set of rules plus a preset*, built
from the same DSL, engine and dashboard.

Goal: you type "alert if someone is in the loading zone without a helmet for more than 5
seconds". It is compiled once into a validated rule, then checked on every frame by
deterministic code. There is no LLM in the frame loop.

## 3.1 Zones become optional
- **No rule or feature requires drawn zones.**
  - Fall detection, ergonomics and the posture coach run with zero zones. The zone-free
    case gets its own tests.
  - The compiler never invents a zone: "near the forklift" becomes an object-relative
    condition, not a zone.
- **Object-relative conditions.** Rules can refer to detected objects instead of drawn
  areas, for example `near_object("forklift", 2 m)`. The objects come from open-vocabulary
  detection (3.4).
  - Until ground-plane calibration exists (Phase 4), distances are measured in **body
    heights** of the nearest person, which works for any camera without calibration.
  - Metres are accepted only on a calibrated camera. Elsewhere the compiler refuses metres
    and offers body heights.
- **Auto-suggested zones.** On a camera's first setup, open-vocabulary detection looks for
  doors, machines, desks and floor markings, and proposes zones from what it finds.
  - You approve, edit or reject each suggestion. Nothing is enabled without your approval.
  - Floor markings are the least reliable class, so their suggestions are labelled
    low-confidence.
- **Privacy zones** stay, as an option.
  - Masked areas are blurred in the stream, clips and thumbnails.
  - Nothing inside them is analysed: the region is masked *before* detection and pose, and
    detections centred inside a mask are dropped.

## 3.2 Rule DSL (`backend/rules/dsl.py`)
- Rules are Pydantic models with these fields: `id`, `name`, `enabled`, `preset`,
  `camera_ids`, `subject.class`, `conditions` (all must hold), `duration_s`, `severity`,
  `cooldown_s`, `actions` (alert, record_clip, notify, review_flag, remind), `source_text`
  and `version`.
- The condition types are a closed, discriminated union. Unknown types and bad parameters
  are rejected with clear messages.
- **Every condition has its own Pydantic validation and unit tests.**

| Condition | Source | Notes |
|---|---|---|
| `in_zone(zone)` / `not_in_zone(zone)` | zone monitor | optional zones only |
| `fallen` | fall state machine | |
| `stationary_for(s)` | track history | |
| `posture_risk_at_least(level)` | ergonomics (REBA) | warehouse; side views |
| `count_in_zone_greater_than(zone, n)` / `count_greater_than(n)` | detector | the second needs no zone |
| `time_window(start, end, days?)` | clock | |
| `near_object(class, distance, unit=body_heights\|m)` | open-vocab (3.4) | metres only when calibrated |
| `missing_object(class, attached_to=head\|torso)` | open-vocab | helmet, vest |
| `head_turned(direction, min_angle, duration)` | pose: nose vs ear midpoint | carries a confidence; low confidence never fires |
| `looking_down_for(duration)` | pose: nose below ear line, head pitch | carries a confidence |
| `looking_at_seat(neighbour)` | pose + seat zones | optional; seat zones auto-suggested |
| `holding_object(class)` | open-vocab box near a wrist keypoint | e.g. phone |
| `object_passed_between(a, b)` | object track moving between two people | **stretch goal** |
| `posture_deviation(kind, threshold)` | posture coach baseline (3.7) | kind = head_drop, hunch, lean, too_close |

## 3.3 Compiler (`backend/rules/compiler.py`)
- It uses the same provider-agnostic LLM client as search: Nemotron by default, Anthropic
  as a switch.
- It is given the DSL schema and the real zone, camera, object-class and calibration
  lists. It returns JSON, which Pydantic validates, with up to one repair turn that feeds
  the errors back.
- It maps everyday phrasing onto conditions. For example:
  - "looks at a neighbour for 5 seconds" → `looking_at_seat(neighbour)` + `duration_s: 5`,
    or `head_turned(left|right, 30°, 5 s)` when there are no seat zones.
  - "slouching for 20 minutes" → `posture_deviation(head_drop)` + `duration_s: 1200`.
- It refuses clearly, naming what is missing, when no condition fits ("if someone looks
  tired", "if they are cheating"). It never invents a condition.
- The Rules page shows the compiled rule in plain words ("Person · near forklift (2 body
  heights) · for 3 s → high alert, clip"). You confirm or edit it. Only confirmed rules run.

## 3.4 Open-vocabulary detection
- The model is YOLO-World (`yolov8s-worldv2`) through `set_classes()` in Ultralytics. I'll
  confirm the current API and the weights' licence before using it.
- It runs only for classes that active rules need, and only where the rule's other
  conditions already hold (person crops, near-wrist crops). It is throttled per track.
- Auto-suggested zones run it once, at setup, not continuously.
- I'll measure FPS with 0, 3 and 10 rules, and report per-class accuracy (helmet, vest,
  forklift, phone) on labelled frames before any demo depends on a class.

## 3.5 Engine (`backend/rules/engine.py`)
- `evaluate(frame_state)` runs every frame. It keeps a duration timer and a cooldown per
  (rule, track), and emits `rule:<id>` events through the existing EventBus.
- Fall detection and zone intrusion become built-in rules. The old hard-coded alerts stay
  behind a flag until the built-ins match them on the sample videos, so nothing alerts
  twice.

## 3.6 Exam hall demo (preset "Exam hall")
**Design rule:** the system only **flags moments for a human to review**.
- It never decides anyone cheated.
- It never shows a "cheating" label.
- It never identifies anyone: track IDs only, no face recognition, and no names attached
  to tracks.

Conditions used: `head_turned`, `looking_down_for`, `holding_object("phone")`, optional
`looking_at_seat`, and stretch goal `object_passed_between`.
- Head yaw is a 2D estimate from where the nose sits relative to the ears. Each estimate
  has a confidence, and anything below the threshold is never flagged.
- Seat zones are auto-suggested at the start of the exam from detected desks or seated
  people, then approved by the invigilator.

UI: an **Exam review** page.
- A timeline of flagged moments, each with a short clip, the rule that fired and its
  confidence.
- **Dismiss** and **Keep** buttons on each flag. These record a human decision and nothing
  else.
- All wording is "flagged for review".

Eval, on staged footage with actors (you're arranging it):
- **flags per person per hour** during honest behaviour;
- **catch rate** on staged suspicious actions, per action type.

README: a **responsible-use** section covering:
- human review is required;
- false flags happen (with the measured rate);
- students must be told the system is in use;
- no automated penalties.

## 3.7 Desk posture coach demo (preset "Desk posture coach")
A laptop webcam sees you from the front, which the ergonomics module treats as low
confidence, so **REBA is not reused here**.
- **Calibration.** You sit upright and press **Set baseline**. It stores your shoulder
  width, head-to-shoulder distance, head height, shoulder tilt and face size (eye/ear
  span), using a median over about 3 s.
- **Deviations from baseline:**
  - head dropping: forward head or slouch;
  - shoulders hunching: shoulder width shrinking;
  - leaning sideways: shoulder tilt;
  - sitting too close to the screen: face size growing.

  Each has its own threshold and confidence.
- **Reminders** are gentle and come only after a posture has been held for a configurable
  time (default **5 minutes**). There is also a **daily summary** of time in good versus
  poor posture.
- **Optional RULA mode** for a side-mounted camera only. It implements RULA (McAtamney &
  Corlett, 1993), with its tables verified cell by cell against a published reference, as
  was done for REBA.
- **Privacy.** This preset runs fully locally. No events go to any LLM unless you ask the
  search agent a question, and video never leaves the machine.

Eval: your own webcam recording with marked good and slouched segments. I report agreement
per segment, overall and per deviation type.

## 3.8 Presets (Rules page)
- A **Presets** choice: **Warehouse safety** (default), **Exam hall**, **Desk posture
  coach**.
- A preset loads a set of rules plus the dashboard panels it needs: the review timeline
  for exams, the baseline and posture summary for the coach. You can edit any rule
  afterwards.
- Switching preset asks before replacing enabled rules.

## 3.9 Background workers (`backend/workers/`)
These are optional per rule and never delay an alert:
- `verify.py` sends 2–3 keyframes plus the rule text to a vision model and writes
  `events.verified`. It is not used by the desk posture coach, which stays local.
- `report.py` writes a short markdown incident report for high and critical events.

## 3.10 Storage, API, evaluation
- **Storage.** Rules, presets, baselines and review decisions are kept in SQLite (schema
  v3, after Phase 1's v2), with a backup before migrating.
- **API.** `POST /api/rules/compile`, `POST /api/rules`, plus list, enable/disable, delete,
  `POST /api/presets/{name}/apply`, `POST /api/posture/baseline`, and
  `POST /api/review/{event_id}` (dismiss or keep). All are local-only with JSON bodies.
- **Compiler eval.** The 40 cases are now 56, in `tests/rules/compile_cases.json`:
  warehouse, exam-hall and posture phrasings, and at least 10 that must be refused. I
  report exact and semantic match. Target ≥ 90%.
- **Benchmarks.** Unit tests for every condition, zone-free runs of falls, ergonomics and
  posture, FPS with 0/3/10 rules, and both demo evals in BENCHMARKS.md.

## Suggested build order (each a branch, merged on its checklist)
1. **3a:** DSL, engine, compiler and presets, using only data the pipeline already has.
   Zones become optional.
2. **3b:** open-vocabulary detection: `near_object`, `missing_object`, `holding_object`,
   auto-suggested zones, privacy zones.
3. **3c:** desk posture coach (no open-vocabulary detection needed, and you can test it
   alone).
4. **3d:** exam hall demo and review UI (needs 3b, plus your staged footage).
5. **3e:** background workers (verification, reports).

## Decisions needed from you
1. **Vision model for verification (3.9).** Nemotron is text-only. The options:
   - an NVIDIA vision model on build.nvidia.com (free trial);
   - Claude Sonnet 5.5 (paid; needs `ANTHROPIC_API_KEY`);
   - skip verification for now.
2. **YOLO-World for PPE, vehicles and objects (3.4).** It costs FPS and its accuracy is
   unmeasured. Object-relative rules, auto-suggested zones, `holding_object` and the exam
   demo now all depend on it, so skipping it now drops those too. The options:
   - go ahead, with per-class accuracy measured first;
   - build 3a and 3c first and decide after measuring.
3. **Branching.** Phase 3 needs ergonomics and schema v3, so it starts after Phases 0–2 are
   on main and Phase 1 is merged. That's already the gate above.

New questions raised by this revision:
4. **Exam clips.** Blur faces in flagged clips by default, so reviewers judge the action
   rather than the person?
5. **Distance units.** Is "body heights" an acceptable unit until Phase 4 calibration, or
   should `near_object` wait for metres?
