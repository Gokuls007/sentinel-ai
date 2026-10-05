# Benchmarks

All numbers here are produced by scripts in this repo and can be re-run. Sections are
replaced in place by `--write docs/BENCHMARKS.md`.

| Section | Script |
|---|---|
| Latency and throughput | `python scripts/benchmark.py --write docs/BENCHMARKS.md` |
| Fall detection accuracy (URFD) | `python training/eval_fall.py --download --write docs/BENCHMARKS.md` |
| Fall detection: rules vs learned model, unseen subjects | `python training/fall_round3.py final --write docs/BENCHMARKS.md` |
| Fall confirmation time sweep | `python training/sweep_fall_confirm.py --write docs/BENCHMARKS.md` |
| False alarms on other non-fall footage | `python training/eval_false_alarms.py --write docs/BENCHMARKS.md` |
| Search accuracy | `python scripts/eval_search.py --write docs/BENCHMARKS.md` |

## Latency and throughput

<!-- benchmark:latency:start -->
_Measured 2026-09-30 23:49 with `python scripts/benchmark.py`_ on the `corridor` sample (768x432, 12 fps native), YOLOv8n + YOLOv8n-pose, torch 2.14.0+cu126, ultralytics 8.4.164, Python 3.11.9, Windows 11 (build 26200).

**CPU: Intel(R) Core(TM) i9-14900HX**: 576 frames after warm-up, **22.3 fps** end to end (unpaced), 7 alerts.

| Layer | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| decode | 0.40 | 0.36 | 0.63 |
| detect_track | 21.91 | 21.53 | 27.21 |
| pose | 19.81 | 19.46 | 24.72 |
| analytics | 0.05 | 0.01 | 0.18 |
| annotate | 1.04 | 0.98 | 1.36 |
| events_and_clips | 0.84 | 0.62 | 0.74 |
| stream | 0.78 | 0.73 | 0.89 |
| **total per frame** | **44.82** | **43.65** | **55.86** |

**GPU: NVIDIA GeForce RTX 4080 Laptop GPU**: 576 frames after warm-up, **54.3 fps** end to end (unpaced), 7 alerts.

| Layer | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| decode | 0.35 | 0.33 | 0.51 |
| detect_track | 9.15 | 8.46 | 12.52 |
| pose | 6.58 | 6.31 | 8.18 |
| analytics | 0.04 | 0.01 | 0.15 |
| annotate | 0.92 | 0.86 | 1.27 |
| events_and_clips | 0.74 | 0.53 | 0.61 |
| stream | 0.64 | 0.62 | 0.74 |
| **total per frame** | **18.42** | **17.77** | **24.55** |

- `detect_track` is YOLOv8n detection and ByteTrack together: ultralytics runs both in one `model.track` call, so they are not timed separately.
- `events_and_clips` is clip/snapshot saving, event publishing (store, notifications, WebSocket) and the clip ring buffer. Encoding runs on a background thread, not here.
- `stream` is the JPEG encode and JSON serialisation the server does once per frame for all dashboard clients.
- Capture waiting is excluded: live sources are paced by the camera, and files by their native fps.
<!-- benchmark:latency:end -->

## False alarms on other non-fall footage

<!-- benchmark:false-alarms:start -->
_Measured 2026-10-04 with `python training/eval_false_alarms.py` on auto._ These videos are separate from URFD and contain no falls, so every fall alert in them is a false alarm. Same production path and thresholds.

| Video | Source | Activities | Length | Fall alerts | Alerts/hour |
|---|---|---|---|---|---|
| Corridor sample (Intel, CC BY 4.0) | sample | walking, standing | 0.8 min | 0 | 0.00 |
| Hallway sample (Intel, CC BY 4.0) | sample | walking | 2.3 min | 0 | 0.00 |
| laptop_20261003_213146.mp4 | recording | n/a | 0.1 min | 0 | 0.00 |
| laptop_20261003_215620.mp4 | recording | n/a | 0.5 min | 0 | 0.00 |
| laptop_20261003_222917.mp4 | recording | n/a | 0.7 min | 0 | 0.00 |
| laptop_20261004_213121.mp4 | recording | n/a | 1.0 min | 3 | 175.14 |
| laptop_20261004_221430.mp4 | recording | n/a | 0.0 min | 0 | 0.00 |
| **All non-fall footage** | | | **0.092 h** | **3** | **32.63** |

**Hard negatives (squatting and kneeling).** These look the most like a fall to this detector, because the head drops and the box gets wider.

_No squat or kneel footage evaluated yet. Add the ergonomics clip to `data/recordings/` with a sidecar JSON naming those activities, then re-run._

This footage totals 5.5 minutes. That is far too little for a real field false-alarm rate, which needs hours of normal work at the target site.
<!-- benchmark:false-alarms:end -->

## Fall detection: rules vs learned model (unseen subjects)

<!-- benchmark:fall-final:start -->
_Measured 2026-10-04 with `python training/fall_round3.py final`._ **Test set: CAUCAFall subjects 6-10 only** (25 falls, 25 daily activities, 4.5 min of no-fall video). These people were never used to design, tune or train anything. Training and tuning used URFD, the sample clips and CAUCAFall subjects 1-5 (55 falls). Default 1 s confirmation.

| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) |
|---|---|---|---|---|
| Rules: baseline | 11 / 25 (44%) | 13.2 (1) | 11 / 25 (44%) | 13.2 (1) |
| Rules: frozen candidate (rotated retry + hysteresis, presence) | 11 / 25 (44%) | 13.2 (1) | 11 / 25 (44%) | 39.7 (3) |
| Rules: + direction-independent signals | 10 / 25 (40%) | 13.2 (1) | 10 / 25 (40%) | 13.2 (1) |
| Rules: + direction-independent signals + box calibration | 10 / 25 (40%) | 13.2 (1) | 10 / 25 (40%) | 13.2 (1) |
| Rules: candidate + direction-independent + box calibration | 11 / 25 (44%) | 26.5 (2) | 11 / 25 (44%) | 26.5 (2) |
| Learned: gradient boosting on pose features (threshold 0.5, chosen by cross-validation on training videos) | 25 / 25 (100%) | 39.7 (3) | 24 / 25 (96%) | 79.4 (6) |

False alarms on the test subjects by activity (confirmed / possible):

| Method | Hop | Kneel | Pick up object | Sit down | Walk |
|---|---|---|---|---|---|
| Rules: baseline | 0 / 0 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rules: frozen candidate | 0 / 0 | 1 / 2 | 0 / 0 | 0 / 1 | 0 / 0 |
| Rules: + direction-independent signals | 0 / 0 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rules: + direction-independent signals + box calibration | 0 / 0 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rules: candidate + direction-independent + box calibration | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 | 0 / 0 |
| Learned: gradient boosting on pose features | 0 / 0 | 2 / 2 | 0 / 1 | 1 / 3 | 0 / 0 |

Tuning half (CAUCAFall subjects 1-5), for reference. Rules only; the learned model was trained on these subjects:

| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) |
|---|---|---|---|---|
| Rules: baseline | 10 / 25 (40%) | 44.6 (3) | 14 / 25 (56%) | 59.5 (4) |
| Rules: frozen candidate (rotated retry + hysteresis, presence) | 12 / 25 (48%) | 44.6 (3) | 14 / 25 (56%) | 59.5 (4) |
| Rules: + direction-independent signals | 10 / 25 (40%) | 29.8 (2) | 12 / 25 (48%) | 29.8 (2) |
| Rules: + direction-independent signals + box calibration | 10 / 25 (40%) | 29.8 (2) | 12 / 25 (48%) | 29.8 (2) |
| Rules: candidate + direction-independent + box calibration | 12 / 25 (48%) | 29.8 (2) | 14 / 25 (56%) | 29.8 (2) |

Learned model: `HistGradientBoostingClassifier` (200 trees, depth 4) on 12 per-frame pose features over a 1 s window (descent_now, descent_max_1s, drop_from_start, shrink, shrink_min_1s, aspect, torso, spread, hip_height, head_drop, hip_std_05s, visible_kps). Possible = probability above the threshold for 0.3 s; confirmed = above it for a further 1 s. Threshold grid with grouped 5-fold cross-validation on training videos (recall minus false alarms per no-fall video): 0.5 → +0.43, 0.6 → +0.43, 0.7 → +0.43, 0.8 → +0.39, 0.9 → +0.40, 0.95 → +0.40.

Notes on the learned model:
- Its "confirmed" level means the probability stayed above the threshold for 1.3 s. It has no separate stillness check like the rules, so the two confirmed levels are not identical in meaning.
- It runs offline only (this script). It isn't wired into the live pipeline.
- An earlier run had a bug: a dip below the threshold shorter than 0.5 s didn't break a streak. These numbers are from the corrected code; the rule rows were unaffected.

With only 25 test falls and 4.5 min of no-fall video, one fall is 4 points of recall and one false alarm is about 13 per hour. Differences of one or two events are noise.

**Diagnosis (tuning subjects only): why falls never reached the on-the-ground stage** (baseline rules):

CAUCAFall tuning subjects 1-5: 25 falls, 14 reached the ground, 11 did not.

Why (missed falls):

- 6 x descent too slow
- 2 x person not tracked during the fall
- 2 x never calibrated (no standing height before the fall)
- 1 x descended, but the torso never looked horizontal

By on-screen direction (all falls: reached / total):

- not visible: 2 / 5
- sideways (rotates): 12 / 18
- toward/away (shortens): 0 / 1
- unclear: 0 / 1

By fall type (folder label):

- Fall backwards: 4 / 5
- Fall forward: 4 / 5
- Fall left: 2 / 5
- Fall right: 1 / 5
- Fall sitting: 3 / 5

Most misses are visibility problems (the person isn't tracked during the fall, or never calibrated), not fall direction: sideways falls reach the ground in most clips, and only one clip is clearly toward or away from the camera.
<!-- benchmark:fall-final:end -->

## Fall confirmation time sweep

<!-- benchmark:fall-sweep:start -->
_Measured 2026-10-04 with `python training/sweep_fall_confirm.py` on cuda._ The confirmation time is how long a person must lie still on the ground before the alert. The current default is **1 s**. Detection and pose ran once per video, and only the fall state machine was replayed at each setting. The "gave up" timeout is at least confirmation + 2 s. These rows use the earlier torso-only ground rule (`ground_mode="torso"`, the default before 2026-10-02). The current default is compared in "Fall detection: rules vs learned model (unseen subjects)" above.

No-fall footage: **13.3 min** (URFD ADL 5.0 min, sample clips 3.2 min, your recordings 5.2 min).

Two alert levels: a **possible fall** (yellow, dashboard only) when the person reaches the ground, and a **confirmed fall** (red, notifies) after the stillness check.

| Confirmation time | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) | Confirmable on URFD | Confirmable but missed (why) |
|---|---|---|---|---|---|---|
| 0.5 s | 6 / 30 (20%) | 4.5 (1) | 16 / 30 (53%) | 31.6 (7) | 14 / 30 | 6 lost from view, 1 not still long enough, 1 left the ground state |
| 1 s (default) | 3 / 30 (10%) | 4.5 (1) | 16 / 30 (53%) | 31.6 (7) | 11 / 30 | 4 lost from view, 3 not still long enough, 1 left the ground state |
| 2 s | 0 / 30 (0%) | 0.0 (0) | 16 / 30 (53%) | 31.6 (7) | 2 / 30 | 2 not still long enough |
| 3 s | 0 / 30 (0%) | 0.0 (0) | 16 / 30 (53%) | 31.6 (7) | 0 / 30 | none |
| 5 s | 0 / 30 (0%) | 0.0 (0) | 16 / 30 (53%) | 31.6 (7) | 0 / 30 | none |

**"On the ground" stage recall** (reached FALLEN, the step before confirmation; it is the same at every setting): **16 / 30 (53%)** of URFD falls. 6 of 49 no-fall videos also reached that stage without confirming.

**Fixes for losing the person on the floor**, each alone, at the default 1 s confirmation. These are replayed from the same pose cache. The region-local and rotated retries were run once per missing person and are used only where production would use them (falling or on the ground, within 3 s of the fall). The retry threshold is 0.15; normal is the pose model's threshold.

| Fix | Confirmed catches | Confirmed false alarms (/ h) | Possible catches | Possible false alarms (/ h) | Lost from view (of falls reaching the ground) |
|---|---|---|---|---|---|
| Baseline (no fixes) | 3 / 30 | 1 (4.5) | 16 / 30 | 7 (31.6) | 7 / 16 |
| 1. Hold lost track 5 s (last seen lying) | 5 / 30 | 2 +1 (9.0) | 18 / 30 | 7 (31.6) | 9 / 18 |
| 2. Region-local low threshold, as pose | 2 / 30 | 1 (4.5) | 22 / 30 | 11 (49.7) | 9 / 22 |
| 2. Region-local low threshold, as presence | 4 / 30 | 2 +1 (9.0) | 17 / 30 | 9 (40.6) | 3 / 17 |
| 3. Rotated fallback, as pose | 3 / 30 | 1 (4.5) | 22 / 30 | 21 (94.8) | 7 / 22 |
| 3. Rotated fallback, as presence | 5 / 30 | 2 +1 (9.0) | 18 / 30 | 9 (40.6) | 3 / 18 |
| 4. Ground-state hysteresis 0.5 s | 3 / 30 | 2 +1 (9.0) | 16 / 30 | 7 (31.6) | 7 / 16 |
| 2 + 3 + 4, as pose (no hold) | 2 / 30 | 1 (4.5) | 22 / 30 | 14 (63.2) | 1 / 22 |
| 3 + 4, as presence (no hold) | 5 / 30 | 4 +3 (18.1) | 18 / 30 | 9 (40.6) | 1 / 18 |
| 2 + 3 + 4, as presence (no hold) | 5 / 30 | 4 +3 (18.1) | 17 / 30 | 9 (40.6) | 0 / 17 |
| All four, as pose | 2 / 30 | 3 +2 (13.5) | 22 / 30 | 14 (63.2) | 1 / 22 |
| All four, as presence | 5 / 30 | 5 +4 (22.6) | 17 / 30 | 9 (40.6) | 0 / 17 |

*Lost from view*: the person's pose (tracked or recovered) was missing in most frames after reaching the ground. Fix 1 doesn't find the person; it keeps the fall alive while they are missing, so it raises catches without lowering this count.

*As pose*: a re-found person goes through the normal check. *As presence*: a re-found person who was last seen lying counts as still in place, unless clearly upright or moved more than half a body height. The re-found keypoints jitter too much to measure stillness directly.

**Caveat: these rows are optimistic.** The "last seen lying" gate and the presence mode were designed after inspecting these same URFD clips. Before choosing defaults they need confirming on footage not used here: your own recordings and a held-out dataset (CAUCAFall). All fixes stay off by default until one is chosen.

How to read it:
- URFD trims each fall clip about 1–2 s after the fall. So "Confirmable" caps recall, and at 2 s and above URFD can't tell you anything about recall.
- "Confirmable but missed": the clip had enough video after reaching the ground, but no alert. *Lost from view*: the person's pose was missing in most later frames. *Left the ground state*: the pose looked upright again, or the wait timed out. *Not still long enough*: the person kept moving on the ground.
- The false-alarm rate rests on only a few minutes of no-fall video. Treat the rows as a comparison between settings, not a field rate.
- Your own recordings (`data/recordings/`) are added automatically when this is re-run.

### Held-out test: CAUCAFall

CAUCAFall (CC BY 4.0) has 50 falls (5 types x 10 subjects) and 50 daily activities, including sitting down and kneeling: 8.6 min of no-fall video. Fall onset is the first frame labelled "fall". It was **not used** to design or tune any rule. Only the baseline and the candidate frozen beforehand are reported, at the default 1 s, run once.

| Setting | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) | Lost from view |
|---|---|---|---|---|---|
| Baseline (no fixes) | 21 / 50 (42%) | 28.0 (4) | 25 / 50 (50%) | 35.0 (5) | 2 / 25 |
| Candidate: rotated retry + hysteresis 0.5 s, as presence | 23 / 50 (46%) | 28.0 (4) | 25 / 50 (50%) | 49.0 (7) | 2 / 25 |

False alarms by daily activity (confirmed / possible):

| Setting | Hop | Kneel | Pick up object | Sit down | Walk |
|---|---|---|---|---|---|
| Baseline (no fixes) | 0 / 0 | 2 / 2 | 1 / 1 | 1 / 2 | 0 / 0 |
| Candidate: rotated retry + hysteresis 0.5 s, as presence | 0 / 0 | 2 / 3 | 1 / 1 | 1 / 3 | 0 / 0 |

Any further tuning will split CAUCAFall by video into a tuning half and a test half, and report test-half numbers only.
<!-- benchmark:fall-sweep:end -->

## Fall detection accuracy (UR Fall Detection dataset)

<!-- benchmark:falls:start -->
_Measured 2026-10-04 with `python training/eval_fall.py` on auto._ URFD camera 0 (RGB 640x480, 30 fps); production thresholds; onset tolerance 2.0 s.

| Metric | Value |
|---|---|
| Fall sequences / detected (recall) | 30 / 3 (10.0%) |
| Precision | 100.0% |
| F1 | 0.18 |
| False positives in fall sequences (early or repeated alerts) | 0 |
| False positives in ADL sequences | 0 over 40 sequences |
| **False alarms per hour of non-fall video** | **0.00**, based on **0.083 h (5.0 min, 8941 frames)** of ADL video |
| Alert latency after fall onset (median / max) | 2.47 s / 2.90 s |

The false-alarm rate rests on only 5.0 minutes of non-fall video (all that URFD provides), so treat it as a rough indicator, not a measured field rate. A reliable figure needs hours of normal-activity footage from the target site.

**Why alerts are missed here: stage diagnostics.** An alert is raised only after the state machine reaches FALLEN (a fast descent, then a lying pose) and the person then stays still for 1.0 s.

| Stage | Value |
|---|---|
| Fall sequences that reached FALLEN (on the ground, before confirmation) | 16 / 30 (53.3%) |
| Video left after fall onset (median) | 1.70 s |
| Video left after reaching FALLEN (median) | 1.50 s |
| ADL sequences that reached FALLEN (not confirmed) | 7 / 40 |

URFD trims each fall clip shortly after the fall. Where the detector reaches FALLEN, the median video left after that (1.50 s) is barely longer than the 1.0 s of stillness the detector waits for, so many clips end before an alert could fire. The confirmation-time sweep above breaks down the clips that had enough video but still got no alert (mostly the person is lost from view once on the floor). These numbers describe how the detector behaves on short, trimmed clips. They are not the recall you would see on continuous video, which needs longer fall recordings to measure.
<!-- benchmark:falls:end -->

## Search accuracy

<!-- benchmark:search:start -->
_Measured 2026-10-01 with `python scripts/eval_search.py` using **nvidia** `nvidia/nemotron-3-super-120b-a12b`._ The questions are in `tests/search/questions.json`, asked over the fixed event log in `tests/search/seed.py` (45 events; now = Wed 30 Sep 2026, 15:00). Grading is deterministic (numbers, cited event ids, labels), with no LLM judge.

| Metric | Value |
|---|---|
| **Accuracy** (3 runs of every question) | **66 / 66 (100.0%)**, 95% CI 94%–100% |
| Accuracy per run | 100.0%, 100.0%, 100.0% |
| Questions that passed in some runs only (flaky) | none |
| Questions that failed in every run | none |
| Questions written by the user | 0 |
| Latency per question p50 / p95 | 8.7 s / 28.8 s |
| Model turns per question (mean) | 2.4 |
| Tokens per question (mean, input + output) | 5,659 |
| Runs that ended without an answer (error or budget) | 0 |
| Made-up citations removed | 0 |

| Category | Passed |
|---|---|
| breakdown | 3 / 3 |
| count | 27 / 27 |
| list | 15 / 15 |
| lookup | 9 / 9 |
| none | 6 / 6 |
| top-n | 6 / 6 |

The target is ≥ 90%. With 30 or fewer questions, the confidence interval is wide.
<!-- benchmark:search:end -->

**How to read this.**
- These 22 questions were written by the developer (Claude). The 8–10 questions the user writes in casual phrasing (`tests/search/user_questions.md`) are not in yet, so treat this as an upper bound until they are.
- The confidence interval treats the 66 attempts as independent. They are really 22 questions asked 3 times, so the interval over distinct questions is wider (22/22: about 85–100%).
- Earlier runs on 2026-10-01, before two fixes, scored 21/22 and 19/22:
  - One failure was a correct answer that cited events as `**#4**` instead of `[#4]`. Bare ids of events the tools returned are now turned into links.
  - Two failures were real: the model got `this week` wrong, calling Wed 23 Sep a Monday. The agent now gets pre-computed date ranges.
  - One failure was a grader bug: a no-break space in `Loading Dock`.
- Each fix generalises beyond these questions, but they were made after seeing these questions fail.

## Rule compiler accuracy

<!-- benchmark:rules:start -->
_Measured 2026-10-03 with `python scripts/eval_rules.py` using **nvidia** `nvidia/nemotron-3-ultra-550b-a55b`._ Cases are in `tests/rules/compile_cases.json` (fixed zones: Loading Dock, Forklift Lane, Chemical Storage, Yard Gate). Grading is deterministic: the compiled conditions and duration must match the expected rule (or an accepted equivalent), and must-refuse cases must be refused.

| Metric | Value |
|---|---|
| **Pass rate** (semantic match or correct refusal) | **43 / 46 (93.5%)**, 95% CI 82%–98% |
| Exact match (no alternative needed) | 42 / 46 |
| Rules compiled correctly | 29 / 32 |
| Must-refuse cases refused | 14 / 14 |
| **Rules compiled that should have been refused** | **0** |
| Valid requests wrongly refused | 2 |
| Errors (invalid after the repair turn, or provider errors) | 0 |
| Needed the repair turn | 0 |
| Severity as stated | 4 / 4 |
| Notify when asked (and only then) | 3 / 3 |
| Cases written by the user | 0 |
| Latency p50 / tokens per case (mean) | 15.3 s / 3,152 |
| Flaky cases (passed in some runs only) | none |

| Domain | Passed |
|---|---|
| exam | 7 / 8 |
| posture | 1 / 1 |
| warehouse | 35 / 37 |

Failures: `w13` (refusal: The rule needs a time window for the weekend, but the time_window condition requires explicit start and end times (e.g., "00:00" to "23:59"). Please provide the exact hours you want to cover on Saturday and Sunday.); `w22` (got in_zone(zone=lane), holding_object(object=book), looking_down() for 0 s); `e06` (refusal: The rule refers to "the hall" as a specific area, but the available zones are only: dock_1 (Loading Dock), lane (Forklift Lane), chem (Chemical Storage), and gate (Yard Gate). There is no "hall" zone defined, so I cannot limit the count to that area.)

All cases so far were written by the developer (Claude); treat this as an upper bound until the user's own sentences (`tests/rules/user_rules.md`) are in.

**Re-run after two prompt fixes (2026-10-03, affected cases only, one run each).** The fixes: days without hours mean all day on those days, and no conditions the sentence doesn't ask for.
- `w13` and `w22` now pass.
- Guard cases `r09` (must still refuse "after hours" with no times), `w10` and `w19` still pass.
- `e06` is unchanged: "the hall" is not a zone at the eval site.

On these 46 cases that is 45 / 46, but the full set hasn't been re-run with the fixes: the table above is from before them.

The target is ≥ 90%.
<!-- benchmark:rules:end -->

## Built-in rules vs original alerts (parity)

<!-- benchmark:rules-parity:start -->
_Measured 2026-10-03 with `python scripts/rules_parity.py` on cuda._ 91 clips, 15,067 frames. The same poses feed both paths; alerts match by type and track within 0.5 s.

| Alert | Original | Built-in rules | Matched |
|---|---|---|---|
| fall | 11 | 11 | 11 |
| zone_intrusion | 7 | 7 | 7 |

**Parity: 18 of 18 alerts match.** No differences.

RULES_BUILTINS stays off until this shows no differences that matter.
<!-- benchmark:rules-parity:end -->

## Cost of rules

<!-- benchmark:rules-cost:start -->
_Measured 2026-10-03 with `python scripts/benchmark_rules.py`_, 5 people in view, 2000 frames, real pipeline code on synthetic poses. FPS combines this with the measured pipeline total of 18.42 ms per frame (above).

| Rules | Rules cost per frame | Pipeline FPS |
|---|---|---|
| 0 | 0.00 ms | 54.3 (-0.0%) |
| 3 | 0.16 ms | 53.8 (-0.9%) |
| 10 | 0.20 ms | 53.7 (-1.1%) |

Rules that need objects (phone, laptop, book) add those classes to the detector's existing pass; any change in detector time from that is not included here.
<!-- benchmark:rules-cost:end -->

## Activity labels (CAUCAFall subjects 1-5, tuning set)

<!-- benchmark:activity-1-5:start -->
_Measured 2026-10-04 with `python scripts/eval_activity.py --subjects 1-5`._ CAUCAFall subjects 1-5 (the tuning set), 50 clips. The rules were tuned **only on subjects 1-5** (plus one own lift-and-carry recording, below); subjects 6-10 are held out (they are also the fall test set). A clip passes when its expected label shows for at least 1 s (main person, smoothed labels).

| Clip type | Expected | Passed |
|---|---|---|
| Fall (any direction) | Fallen or Lying down | 23 / 25 |
| Hop | no false Fallen | 5 / 5 |
| Kneel | no false Fallen | 4 / 5 |
| Pick up object | Bending, then Lifting | 0 / 5 |
| Sit down | Sitting | 5 / 5 |
| Walk | Walking | 4 / 5 |

False "Lifting" (shown for 1 s or more in Walk, Sit down, Hop or Kneel clips): 0 / 20 clips.

False "Carrying" (shown for 1 s or more; nobody carries anything in these clips): 1 / 50 clips.

Failures: `Subject.1/Kneel` (showed Fallen); `Subject.1/Pick up object` (no Bending, no Lifting); `Subject.2/Pick up object` (no Bending, no Lifting); `Subject.2/Walk` (no Walking); `Subject.3/Fall left` (no Fallen or Lying down); `Subject.3/Pick up object` (no Bending, no Lifting); `Subject.4/Fall sitting` (no Fallen or Lying down); `Subject.4/Pick up object` (no Bending, no Lifting); `Subject.5/Pick up object` (no Lifting)

Front-on room-scale clips, not warehouse footage: this checks the rules end to end, not field accuracy.
<!-- benchmark:activity-1-5:end -->

## Activity labels (CAUCAFall subjects 6-10, held out)

<!-- benchmark:activity-6-10:start -->
_Measured 2026-10-04 with `python scripts/eval_activity.py --subjects 6-10`._ CAUCAFall subjects 6-10 (held out: never used for tuning), 50 clips. The rules were tuned **only on subjects 1-5** (plus one own lift-and-carry recording, below); subjects 6-10 are held out (they are also the fall test set). A clip passes when its expected label shows for at least 1 s (main person, smoothed labels).

| Clip type | Expected | Passed |
|---|---|---|
| Fall (any direction) | Fallen or Lying down | 25 / 25 |
| Hop | no false Fallen | 5 / 5 |
| Kneel | no false Fallen | 4 / 5 |
| Pick up object | Bending, then Lifting | 0 / 5 |
| Sit down | Sitting | 4 / 5 |
| Walk | Walking | 4 / 5 |

False "Lifting" (shown for 1 s or more in Walk, Sit down, Hop or Kneel clips): 0 / 20 clips.

False "Carrying" (shown for 1 s or more; nobody carries anything in these clips): 0 / 50 clips.

Failures: `Subject.10/Kneel` (showed Fallen); `Subject.10/Pick up object` (no Bending, no Lifting); `Subject.6/Pick up object` (no Bending, no Lifting); `Subject.6/Walk` (no Walking); `Subject.7/Pick up object` (no Lifting); `Subject.8/Pick up object` (no Bending, no Lifting); `Subject.8/Sit down` (no Sitting); `Subject.9/Pick up object` (no Lifting)

Front-on room-scale clips, not warehouse footage: this checks the rules end to end, not field accuracy.
<!-- benchmark:activity-6-10:end -->

## Activity labels (own lift-and-carry recording)

<!-- benchmark:activity-own-laptop_20261003_215620:start -->
_Measured 2026-10-04 with `python scripts/eval_activity_own.py data/recordings/laptop_20261003_215620.mp4`._ One hand-labelled webcam recording (not committed), side/oblique view; it was used for tuning together with CAUCAFall subjects 1-5, so these are training-set numbers, not an accuracy estimate.

| Action | Time (s) | Expected | Result | Labels shown (frames) |
|---|---|---|---|---|
| walk | 8.5-9.3 | Walking | pass | Walking 16, Bending 7 |
| bend to pick up | 9.4-10.9 | Bending | pass | Bending 22, Lifting 9, Walking 3 |
| lift | 10.9-11.8 | Lifting | pass | Lifting 22, Bending 2 |
| carry | 11.8-22.2 | Carrying | pass | Carrying 154, Lifting 12, Standing 1 |
| put down | 22.3-23.7 | (reported only) | not scored | Carrying 32 |
| walk back | 24.0-26.0 | Walking | pass | Walking 15, Standing 12, Carrying 11, Sitting 3 |
<!-- benchmark:activity-own-laptop_20261003_215620:end -->

Not covered yet: the clip has no reach overhead, squat, or bend without lifting, so those are
untested on real footage (the bend-without-lift case is covered by unit tests only). "Put down"
has no label of its own: Carrying ends about 1 s after both arms drop (0.5 s hang plus
smoothing). The view is side/oblique and partly front-on, with a soft bag rather than a box.

## Activity labels (own recording, held out)

<!-- benchmark:activity-own-laptop_20261003_222917:start -->
_Measured 2026-10-04 with `python scripts/eval_activity_own.py data/recordings/laptop_20261003_222917.mp4`._ One hand-labelled webcam recording (not committed), **held out**: labelled before the first run, evaluated once, never used for tuning.

| Action | Time (s) | Expected | Result | Labels shown (frames) |
|---|---|---|---|---|
| carry | 2.8-12.6 | Carrying | missed | Standing 106, Lying down 35, Walking 17 |
| reach overhead | 13.1-17.3 | Reaching overhead | pass | Reaching overhead 36, Standing 27, Lying down 11 |
| put down | 17.7-18.9 | (reported only) | not scored | Lying down 14, Standing 13, Reaching overhead 1 |
| handle bag low | 19.1-24.3 | (reported only) | not scored | Lying down 50, Standing 39 |
| lift | 26.6-27.6 | Lifting | missed | Lying down 22, Walking 2, Upper body only 1, Standing 1 |
| carry one hand | 28.0-30.3 | Carrying | missed | Lying down 41, Standing 5 |
| raise to chest | 30.3-31.3 | (reported only) | not scored | Lying down 22, Standing 4 |
| carry | 31.3-35.0 | Carrying | missed | Lying down 38, Standing 28 |
| put down | 35.3-36.4 | (reported only) | not scored | Standing 20, Lying down 7 |
| walk | 36.6-37.6 | Walking | pass | Standing 12, Walking 11, Lying down 3 |
<!-- benchmark:activity-own-laptop_20261003_222917:end -->

**Read this with the table.** This clip is mostly **front-on** and dim, with a soft bag rather
than a box. It starts already carrying, and it has no squat or bend without lifting. Result:
1 of 5 scored actions caught (reach overhead), plus walking. Why the rest failed (diagnosed
after the run; nothing was tuned on it):

- **A false "person" (the gaming chair next to the bright monitor)** was detected in about 178
  of 597 frames. The evaluation follows the largest person box, so in those frames it scored
  the chair, whose wide box reads as Lying down. Live, it would show as a phantom person.
- **A false confirmed fall** on the real person (21.5-23.1 s) while swinging the bag low near the
  left edge of the frame. Live, this would have sent a fall alert.
- **Carrying needs a lift first**, so carrying from the first frame isn't recognised, and a bag
  held low in one hand at the side isn't covered by the carry rule.
- **The one lift (26.6-27.6 s)** happened partly out of frame, one-handed.

What would help (not done; to decide together): a third, side-on clip to tune on; ignoring
person boxes with implausible keypoints (the chair); following the person the event is about
rather than the largest box in the evaluation; and checking the fall detector on bag handling.

## Pose models: YOLOv8n-pose vs RTMPose-m

<!-- benchmark:pose-compare:start -->
Clip `laptop_20261004_221104.mp4`, 983 frames at 15 fps; same frames and tracker boxes for both models; main (largest) person.

| Segment | Model | Frames | Box missing | No pose | Mean conf | Low conf (<0.3) | Jitter (% box h) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| standing | yolov8n-pose | 86 | 0% | 0% | 0.91 | 0% | 1.05 | 12.0 |
| standing | rtmpose-m | 86 | 0% | 0% | 0.84 | 0% | 0.85 | 8.8 |
| lying across couch | yolov8n-pose | 37 | 24% | 5% | 0.51 | 34% | 2.89 | 11.3 |
| lying across couch | rtmpose-m | 37 | 24% | 0% | 0.26 | 70% | 1.42 | 8.7 |
| sitting on couch edge | yolov8n-pose | 45 | 0% | 0% | 0.85 | 3% | 1.83 | 12.7 |
| sitting on couch edge | rtmpose-m | 45 | 0% | 0% | 0.72 | 0% | 1.63 | 9.2 |
| leaning forward to stand | yolov8n-pose | 28 | 7% | 0% | 0.67 | 19% | 6.15 | 13.0 |
| leaning forward to stand | rtmpose-m | 28 | 7% | 0% | 0.52 | 16% | 4.09 | 9.0 |
| crouching on floor | yolov8n-pose | 80 | 16% | 0% | 0.64 | 21% | 3.54 | 11.5 |
| crouching on floor | rtmpose-m | 80 | 16% | 0% | 0.51 | 19% | 2.50 | 8.5 |
| lying on floor | yolov8n-pose | 14 | 100% | 0% | -- | -- | -- | -- |
| lying on floor | rtmpose-m | 14 | 100% | 0% | -- | -- | -- | -- |
| getting up from floor | yolov8n-pose | 42 | 60% | 0% | 0.68 | 18% | 5.32 | 12.8 |
| getting up from floor | rtmpose-m | 42 | 60% | 0% | 0.57 | 6% | 4.47 | 9.7 |

Weakest keypoints (mean confidence):
- standing, yolov8n-pose: weakest l_ear 0.68, l_ankle 0.72, r_ear 0.74, r_ankle 0.74
- standing, rtmpose-m: weakest nose 0.75, r_hip 0.76, l_hip 0.76, l_eye 0.80
- lying across couch, yolov8n-pose: weakest l_eye 0.02, l_ear 0.04, nose 0.06, r_eye 0.07
- lying across couch, rtmpose-m: weakest l_wrist 0.18, l_elbow 0.18, l_hip 0.23, r_eye 0.23
- sitting on couch edge, yolov8n-pose: weakest l_ear 0.44, l_ankle 0.63, r_ankle 0.64, r_ear 0.83
- sitting on couch edge, rtmpose-m: weakest r_ankle 0.51, r_knee 0.57, l_wrist 0.61, l_hip 0.63
- leaning forward to stand, yolov8n-pose: weakest r_ankle 0.14, l_ankle 0.15, r_knee 0.31, l_knee 0.32
- leaning forward to stand, rtmpose-m: weakest r_ankle 0.23, l_ankle 0.31, r_knee 0.31, l_knee 0.41
- crouching on floor, yolov8n-pose: weakest r_ankle 0.08, l_ankle 0.08, r_knee 0.23, l_knee 0.23
- crouching on floor, rtmpose-m: weakest r_ankle 0.22, l_ankle 0.26, l_wrist 0.39, r_wrist 0.39
- getting up from floor, yolov8n-pose: weakest l_ankle 0.20, r_ankle 0.21, l_knee 0.36, r_knee 0.38
- getting up from floor, rtmpose-m: weakest r_ankle 0.33, l_ankle 0.39, r_knee 0.42, l_knee 0.44
<!-- benchmark:pose-compare:end -->

**Reading it.** One 65 s webcam clip of one person, poses shorter than planned (1-5 s each),
boundaries marked from the frames. Small sample: a direction, not a measurement.

- **The person box is the bottleneck.** Both pose models are top-down on the tracker's boxes
  (YOLOv8n + ByteTrack), and the box is missing in 24% of the frames lying across the couch,
  16% crouching, 60% getting up from the floor and 100% of the (short) time lying on the floor.
  No pose model helps there; fall prediction needs a person detector that keeps lying people.
- **RTMPose-m is steadier.** Lower jitter in every segment (standing 0.85 vs 1.05% of box
  height, leaning to stand 4.09 vs 6.15, crouching 2.50 vs 3.54), and legs are found more often in
  the hard poses (ankles while crouching 0.22-0.26 vs 0.08; low-confidence keypoints while getting
  up 6% vs 18%).
- **Confidences aren't comparable across models.** RTMPose's SimCC scores run lower than the
  YOLO pose scores for the same quality (standing: 0.84 vs 0.91), so compare jitter and the
  low-confidence share, not mean confidence. Lying across the couch, RTMPose marked 70% of
  keypoints low: the body was cut off by the frame edge there.
- **Speed.** Pose time per frame: RTMPose-m ~9-10 ms vs YOLOv8n-pose ~12 ms (which detects on the
  whole frame). Live on the webcam with RTMPose-m, while recording: 17.3 fps for the whole
  pipeline (person tracker, YOLO11m + YOLO-World objects, pose, rules, recording).

## Person detection in hard poses (lying, sliding, getting up)

<!-- benchmark:person-detection:start -->
**laptop_20261004_221434.mp4**: share of frames with no person found, per segment.

| Detector | sitting in chair | sliding out of chair | on floor by chair | getting up | back in chair | standing by bed | lying on bed | rolling off bed to floor | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n @0.5 (current) | 0% | 21% | 4% | 4% | 4% | 2% | 83% | 13% | 9.4 |
| yolov8n @0.25 | 0% | 3% | 1% | 2% | 4% | 0% | 63% | 10% | 8.6 |
| yolo11m @0.5 | 0% | 4% | 0% | 3% | 0% | 0% | 2% | 9% | 12.1 |
| yolo11m @0.25 | 0% | 2% | 0% | 3% | 0% | 0% | 1% | 7% | 12.3 |
| rtmo-m @0.7 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 17.9 |
| rtmo-m @0.4 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 17.9 |
| rtmo-l @0.7 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 23.2 |
| rtmo-l @0.4 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 24.6 |

**laptop_20261004_221104.mp4**: share of frames with no person found, per segment.

| Detector | standing | lying across couch | sitting on couch edge | leaning forward to stand | crouching on floor | lying on floor | getting up from floor | ms/frame |
|---|---|---|---|---|---|---|---|---|
| yolov8n @0.5 (current) | 0% | 54% | 0% | 11% | 18% | 100% | 62% | 10.4 |
| yolov8n @0.25 | 0% | 3% | 0% | 4% | 16% | 86% | 57% | 9.4 |
| yolo11m @0.5 | 0% | 0% | 0% | 4% | 12% | 79% | 19% | 13.7 |
| yolo11m @0.25 | 0% | 0% | 0% | 0% | 8% | 21% | 7% | 13.4 |
| rtmo-m @0.7 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 18.4 |
| rtmo-m @0.4 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 18.7 |
| rtmo-l @0.7 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 24.3 |
| rtmo-l @0.4 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 25.7 |
<!-- benchmark:person-detection:end -->

**Usable pose** (at least 8 of 17 keypoints at confidence >= 0.3, main person) on the same
frames, which is what fall prediction actually needs:

| Segment | YOLOv8n@0.5 + RTMPose-m (old) | YOLO11m@0.25 + RTMPose-m | RTMO-m |
|---|---|---|---|
| Lying on bed | 17% | **99%** | 100% |
| Sliding out of chair | 79% | **98%** | 91% |
| Rolling off bed to floor | 87% | **93%** | 93% |
| Lying across couch | 35% | **68%** | 22% |
| Lying on floor | 0% | **57%** | 0% |
| Getting up from floor | 38% | **93%** | 64% |

RTMO finds a person in every frame, but in the hardest poses its keypoints are mostly unusable
(lying on the floor: 0% usable). **Chosen: YOLO11m at 0.25 as the person detector, with
RTMPose-m.** In the full pipeline (tracker included), no person box on the bed went from 70% to
0% (one identity throughout); lying on the floor from 100% to 57%, with every missing frame kept
as "lost while lying". Cost: ~4-6 ms more per frame. One person, two clips, short segments: a
direction, not a measurement. The fall and activity benchmarks above were measured with
YOLOv8n and need re-running.
