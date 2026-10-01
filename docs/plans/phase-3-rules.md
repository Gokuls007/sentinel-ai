# Phase 3 plan: natural-language safety rules (draft for approval)

Goal: you type "alert if someone is in the loading zone without a helmet for more than 5
seconds"; it is compiled once to a validated rule and then enforced every frame by
deterministic code. No LLM in the frame loop.

## 3.1 Rule DSL (`backend/rules/dsl.py`)
- Rules are Pydantic models with these fields: `id`, `name`, `enabled`, `camera_ids`,
  `subject.class` (person, plus forklift, truck or car once vehicles are detected),
  `conditions` (all must hold, i.e. AND), `duration_s`, `severity`, `cooldown_s`,
  `actions` (alert, record_clip, notify), `source_text`, `version`.
- The condition types are a closed union. Unknown types are rejected.

| Condition | Uses data the pipeline already has? |
|---|---|
| `in_zone(zone_id)` / `not_in_zone(zone_id)` | yes (zone monitor) |
| `fallen` | yes (fall state machine) |
| `stationary_for(seconds)` | yes (track history) |
| `posture_risk_at_least(level)` | yes, once Phase 1 is merged |
| `count_in_zone_greater_than(zone_id, n)` | yes |
| `time_window(start, end, days?)` | clock only |
| `near_object(class, max_distance_m or px)` | needs vehicle/object detections; px until Phase 4 calibrates metres |
| `missing_object(class, attached_to=head/torso)` | needs open-vocabulary detection (3.3) |

## 3.2 Compiler (`backend/rules/compiler.py`)
- The same provider-agnostic LLM client as search (Nemotron by default, Anthropic as a
  switch).
- The compiler is given the DSL schema and the real zone, camera and class lists, and
  returns JSON. That JSON is validated by Pydantic, with up to one repair turn that
  feeds the validation errors back.
- If a request can't be expressed (for example "if someone looks tired"), the compiler
  must refuse and name the missing capability. It must not invent a condition.
- The Rules page shows the compiled rule in plain words ("Person · in Loading Dock · no
  helmet · for 5 s → high alert, clip, notify"). You confirm or edit it. Only confirmed
  rules run.

## 3.3 Open-vocabulary detection (helmet, vest, forklift, ladder)
- The model is YOLO-World (`yolov8s-worldv2`) through `set_classes()` in Ultralytics. I'll
  confirm the current API and the weights' licence before using it.
- It runs only for classes that an active rule needs, and only on person crops whose
  other conditions already hold (for example, inside the zone). It is throttled to N Hz
  per track.
- FPS is measured with 0, 3 and 10 rules active.

## 3.4 Engine (`backend/rules/engine.py`)
- An `evaluate(frame_state)` call runs each frame. It keeps a per-(rule, track) duration
  timer and a cooldown, and emits `rule:<id>` events through the existing EventBus.
- Fall detection and zone intrusion become built-in rules. The old hard-coded alerts sit
  behind a flag until the built-ins match them on the sample videos (no double alerts).

## 3.5 Background workers (`backend/workers/`)
These are optional per rule and never delay an alert:
- `verify.py` sends 2–3 keyframes plus the rule text to a vision model and writes
  `events.verified` (confirm, flag, or downgrade).
- `report.py` writes a short markdown incident report for high and critical events.

## 3.6 Storage, API and UI
- Rules are stored in SQLite (schema v3, after Phase 1's v2), with a backup before
  migrating.
- `POST /api/rules/compile`, then `POST /api/rules` to save, plus list, enable/disable
  and delete. All are local-only with JSON bodies.
- The Rules page has a text box, the compiled preview, confirm/edit, and a list with
  toggles and fire counts.

## 3.7 Evaluation
- 40 rule texts with expected DSL in `tests/rules/compile_cases.json`. I report both an
  exact match (after normalising) and a semantic match (same conditions, zones, duration
  and severity). 8 of the 40 are rules the compiler must refuse. Target: ≥ 90%.
- Unit tests cover every condition type, the durations and the cooldowns.
- FPS with 0, 3 and 10 rules goes into BENCHMARKS.md.

## Decisions needed from you
1. **Vision model for verification (3.5).** Nemotron is text-only. The options are an NVIDIA
   vision model on build.nvidia.com (free trial), or Claude Sonnet 5.5 (paid; needs
   `ANTHROPIC_API_KEY`). Or skip verification for now.
2. **YOLO-World for PPE/vehicles.** It costs FPS and its accuracy on helmets is
   unmeasured. Should I go ahead, or start with the conditions that use data the pipeline
   already has, and leave `missing_object` and `near_object` for later?
3. **Branching.** Phase 3 uses ergonomics (`posture_risk_at_least`) and schema v3, so it
   should start after Phases 0–2 are on main and Phase 1 is merged.
