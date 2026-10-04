# Phase 3 plan: plain-English rules and presets (draft for approval, revised)

**Status:** plan **approved** (decisions recorded at the end). Implementation starts only
when the gate is met: **Phase 0 and Phase 1 merged** and the **fall confirmation sweep**
done.

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
  - The UI always shows body heights with an approximate conversion, marked as approximate:
    "1.5 body heights (≈2.5 m, approx.)". The conversion assumes a 1.7 m adult. This appears
    in the compiled-rule preview, the rule list and event details.
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
| `holding_object(class)` | object box near a wrist keypoint | phone = COCO `cell phone` (no YOLO-World) |
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

## 3.4 Object detection: COCO first, open-vocabulary only where needed
- **Standard COCO YOLOv8 classes are used wherever they exist**, instead of YOLO-World:
  `person`, `cell phone`, `laptop` and `book`. This includes `holding_object(phone)`.
  These come from the detector the pipeline already runs, so they cost little extra.
- YOLO-World covers only classes COCO lacks: helmet, safety vest, forklift, ladder, door
  and desk.
- **Accuracy test before 3b.** While 3a and 3c are built, I run a small per-class
  accuracy test of YOLO-World on helmet, safety vest, forklift, ladder, door and desk:
  - labelled frames from public, licence-checked images plus our own footage;
  - precision and recall per class at the chosen confidence threshold, plus FPS cost;
  - reported in BENCHMARKS.md before 3b starts.

  A class that tests poorly isn't offered in rules. The compiler refuses it and says why.
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
> **Superseded by [exam-hall.md](exam-hall.md)** (the full Exam Hall specification). Where they
> disagree, exam-hall.md wins.

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

**Faces are blurred by default** in flagged clips and thumbnails.
- The blur covers the face region only, from the face keypoints. The head outline stays
  visible, so a reviewer can still judge head direction.
- An **Unblur** action requires a typed reason. Each unblur is logged with timestamp,
  event id and reason, and the log is shown on the review page.
- There is no bulk unblur.

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
- no automated penalties;
- faces are blurred by default, and every unblur needs a logged reason.

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
- `verify.py` (vision verification) is **skipped for now**. It will be decided at 3e,
  behind the same provider-agnostic LLM client. It would never be used by the desk posture
  coach, which stays local.
- `report.py` writes a short markdown incident report for high and critical events.

## 3.10 Storage, API, evaluation
- **Storage.** Rules, presets, baselines and review decisions are kept in SQLite (schema
  v3, after Phase 1's v2), with a backup before migrating.
- **API.** `POST /api/rules/compile`, `POST /api/rules`, plus list, enable/disable, delete,
  `POST /api/presets/{name}/apply`, `POST /api/posture/baseline`,
  `POST /api/review/{event_id}` (dismiss or keep), and
  `POST /api/review/{event_id}/unblur {reason}` (logged). All are local-only with JSON
  bodies.
- **Compiler eval.** The 40 cases are now 56, in `tests/rules/compile_cases.json`:
  warehouse, exam-hall and posture phrasings, and at least 10 that must be refused. I
  report exact and semantic match. Target ≥ 90%.
- **Benchmarks.** Unit tests for every condition, zone-free runs of falls, ergonomics and
  posture, FPS with 0/3/10 rules, and both demo evals in BENCHMARKS.md.

## Suggested build order (each a branch, merged on its checklist)
1. **3a:** DSL, engine, compiler and presets, using only data the pipeline already has.
   Zones become optional. `holding_object` works with COCO classes.
2. **3c:** desk posture coach (no open-vocabulary detection needed, and you can test it
   alone).
3. In parallel with 3a and 3c: the **YOLO-World per-class accuracy test** (3.4), reported
   before 3b starts.
4. **3b:** open-vocabulary detection for the classes that passed: `near_object`,
   `missing_object`, auto-suggested zones, privacy zones.
5. **3d:** exam hall demo and review UI, with face blur and logged unblur (needs your
   staged footage).
6. **3e:** background workers. Reports, plus the decision on vision verification.

## Decisions (approved 2026-10-01)
1. **Vision verification:** skipped for now. It is decided at 3e, behind the same
   provider-agnostic client.
2. **YOLO-World:** 3a and 3c are built first. Meanwhile I run a small per-class accuracy
   test on helmet, safety vest, forklift, ladder, door and desk, and report it before 3b
   starts.
   - Standard COCO YOLOv8 classes are used wherever they exist (`cell phone`, `laptop`,
     `book`, `person`), including for `holding_object(phone)`.
   - So the exam demo's phone condition doesn't depend on YOLO-World. Only desk-based seat
     suggestions do, and suggestions from seated people work without it.
3. **Branching gate:** confirmed. No Phase 3 work starts until Phase 0 and Phase 1 are
   merged and the fall confirmation sweep is done.
4. **Exam clips:** faces are blurred by default, with the head outline kept visible. An
   Unblur action requires a reason and is logged with a timestamp. The responsible-use
   section says so (see 3.6).
5. **Distance units:** body heights are accepted until Phase 4, always shown with an
   approximate metre conversion marked as approximate (see 3.1).
## Status: 3a built (branch `phase-3a-rules`, 2026-10-03)
The gate was met by merging instead of waiving it: Phases 0–2, the laptop camera and the
posture coach went to `main` on 2026-10-03, after CI passed. The checks still open are tracked
as GitHub issues #1–#4. The fall confirmation sweep was done earlier (URFD and CAUCAFall).

Built: the DSL, engine, compiler, the Warehouse safety preset, storage (schema v3), the API, the
Rules page, the compiler eval, the built-in parity check and the rules cost benchmark.

Deviations from the plan above, each small:
- **Durations live on the rule.** `looking_down_for(duration)` became `looking_down`, an
  instant condition. Its duration comes from the rule's `duration_s`, as for every other
  condition, so "looks down for 10 s" is `looking_down` with `duration_s: 10`.
- **One alert per episode.** A rule fires once while its conditions keep holding for a person,
  then again only after they stop and the cooldown has passed. The original zone alerts
  instead repeated every cooldown while someone stayed inside. The parity check reports any
  difference this makes.
- **Not in 3a, as planned for later phases:** `near_object`, `missing_object`,
  `looking_at_seat` and `posture_deviation`. The compiler refuses them and names the phase.
  The desk posture coach stays its own mode rather than a preset.
- **The default NVIDIA model changed.** `nvidia/nemotron-3-super-120b-a12b` was retired on
  2026-10-03 (HTTP 410). The default is now `nvidia/nemotron-3-ultra-550b-a55b`. This also
  applies to search; its benchmark row still names the model it was measured with.
