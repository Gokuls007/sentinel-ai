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
_Measured 2026-10-01 with `python training/eval_false_alarms.py` on cuda._ These videos are separate from URFD and contain no falls, so every fall alert in them is a false alarm. Same production path and thresholds.

| Video | Source | Activities | Length | Fall alerts | Alerts/hour |
|---|---|---|---|---|---|
| Corridor sample (Intel, CC BY 4.0) | sample | walking, standing | 0.8 min | 0 | 0.00 |
| Hallway sample (Intel, CC BY 4.0) | sample | walking | 2.3 min | 0 | 0.00 |
| **All non-fall footage** | | | **0.053 h** | **0** | **0.00** |

**Hard negatives (squatting and kneeling).** These look the most like a fall to this detector, because the head drops and the box gets wider.

_No squat or kneel footage evaluated yet. Add the ergonomics clip to `data/recordings/` with a sidecar JSON naming those activities, then re-run._

This footage totals 3.2 minutes. That is far too little for a real field false-alarm rate, which needs hours of normal work at the target site.
<!-- benchmark:false-alarms:end -->

## Fall detection: rules vs learned model (unseen subjects)

<!-- benchmark:fall-final:start -->
_Measured 2026-10-02 with `python training/fall_round3.py final`._ **Test set: CAUCAFall subjects 6-10 only** (25 falls, 25 daily activities, 4.5 min of no-fall video). These people were never used to design, tune or train anything. Training and tuning used URFD, the sample clips and CAUCAFall subjects 1-5 (55 falls). Default 1 s confirmation.

| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) |
|---|---|---|---|---|
| Rules: baseline | 2 / 25 (8%) | 52.9 (4) | 7 / 25 (28%) | 79.4 (6) |
| Rules: frozen candidate (rotated retry + hysteresis, presence) | 4 / 25 (16%) | 66.2 (5) | 8 / 25 (32%) | 79.4 (6) |
| Rules: + direction-independent signals | 4 / 25 (16%) | 0.0 (0) | 10 / 25 (40%) | 0.0 (0) |
| Rules: + direction-independent signals + box calibration | 4 / 25 (16%) | 0.0 (0) | 10 / 25 (40%) | 0.0 (0) |
| Rules: candidate + direction-independent + box calibration | 5 / 25 (20%) | 39.7 (3) | 10 / 25 (40%) | 52.9 (4) |
| Learned: gradient boosting on pose features (threshold 0.5, chosen by cross-validation on training videos) | 14 / 25 (56%) | 39.7 (3) | 18 / 25 (72%) | 92.6 (7) |

False alarms on the test subjects by activity (confirmed / possible):

| Method | Hop | Kneel | Pick up object | Sit down | Walk |
|---|---|---|---|---|---|
| Rules: baseline | 0 / 0 | 0 / 0 | 3 / 4 | 1 / 2 | 0 / 0 |
| Rules: frozen candidate | 0 / 0 | 0 / 0 | 4 / 4 | 1 / 2 | 0 / 0 |
| Rules: + direction-independent signals | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rules: + direction-independent signals + box calibration | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rules: candidate + direction-independent + box calibration | 0 / 0 | 0 / 0 | 3 / 3 | 0 / 1 | 0 / 0 |
| Learned: gradient boosting on pose features | 0 / 0 | 0 / 2 | 1 / 2 | 2 / 3 | 0 / 0 |

Tuning half (CAUCAFall subjects 1-5), for reference. Rules only; the learned model was trained on these subjects:

| Method | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) |
|---|---|---|---|---|
| Rules: baseline | 5 / 25 (20%) | 0.0 (0) | 14 / 25 (56%) | 44.6 (3) |
| Rules: frozen candidate (rotated retry + hysteresis, presence) | 8 / 25 (32%) | 0.0 (0) | 14 / 25 (56%) | 59.5 (4) |
| Rules: + direction-independent signals | 7 / 25 (28%) | 0.0 (0) | 15 / 25 (60%) | 0.0 (0) |
| Rules: + direction-independent signals + box calibration | 7 / 25 (28%) | 0.0 (0) | 15 / 25 (60%) | 0.0 (0) |
| Rules: candidate + direction-independent + box calibration | 8 / 25 (32%) | 0.0 (0) | 15 / 25 (60%) | 29.8 (2) |

Learned model: `HistGradientBoostingClassifier` (200 trees, depth 4) on 12 per-frame pose features over a 1 s window (descent_now, descent_max_1s, drop_from_start, shrink, shrink_min_1s, aspect, torso, spread, hip_height, head_drop, hip_std_05s, visible_kps). Possible = probability above the threshold for 0.3 s; confirmed = above it for a further 1 s. Threshold grid with grouped 5-fold cross-validation on training videos (recall minus false alarms per no-fall video): 0.5 → +0.48, 0.6 → +0.44, 0.7 → +0.39, 0.8 → +0.30, 0.9 → +0.26, 0.95 → +0.19.

Notes on the learned model:
- Its "confirmed" level means the probability stayed above the threshold for 1.3 s. It has no separate stillness check like the rules, so the two confirmed levels are not identical in meaning.
- It runs offline only (this script). It isn't wired into the live pipeline.
- An earlier run had a bug: a dip below the threshold shorter than 0.5 s didn't break a streak. These numbers are from the corrected code; the rule rows were unaffected.

With only 25 test falls and 4.5 min of no-fall video, one fall is 4 points of recall and one false alarm is about 13 per hour. Differences of one or two events are noise.

**Diagnosis (tuning subjects only): why falls never reached the on-the-ground stage** (baseline rules):

CAUCAFall tuning subjects 1-5: 25 falls, 14 reached the ground, 11 did not.

Why (missed falls):

- 6 x person not tracked during the fall
- 3 x never calibrated (no standing height before the fall)
- 1 x descent too slow
- 1 x descended, but the torso never looked horizontal

By on-screen direction (all falls: reached / total):

- not visible: 2 / 8
- sideways (rotates): 12 / 14
- toward/away (shortens): 0 / 1
- unclear: 0 / 2

By fall type (folder label):

- Fall backwards: 4 / 5
- Fall forward: 2 / 5
- Fall left: 3 / 5
- Fall right: 2 / 5
- Fall sitting: 3 / 5

Most misses are visibility problems (the person isn't tracked during the fall, or never calibrated), not fall direction: sideways falls reach the ground in most clips, and only one clip is clearly toward or away from the camera.
<!-- benchmark:fall-final:end -->

## Fall confirmation time sweep

<!-- benchmark:fall-sweep:start -->
_Measured 2026-10-02 with `python training/sweep_fall_confirm.py` on cuda._ The confirmation time is how long a person must lie still on the ground before the alert. The current default is **1 s**. Detection and pose ran once per video, and only the fall state machine was replayed at each setting. The "gave up" timeout is at least confirmation + 2 s.

No-fall footage: **8.1 min** (URFD ADL 5.0 min, sample clips 3.2 min).

Two alert levels: a **possible fall** (yellow, dashboard only) when the person reaches the ground, and a **confirmed fall** (red, notifies) after the stillness check.

| Confirmation time | Confirmed recall | Confirmed false alarms / h (count) | Possible recall | Possible false alarms / h (count) | Confirmable on URFD | Confirmable but missed (why) |
|---|---|---|---|---|---|---|
| 0.5 s | 3 / 30 (10%) | 14.8 (2) | 25 / 30 (83%) | 81.3 (11) | 23 / 30 | 15 lost from view, 3 left the ground state, 2 not still long enough |
| 1 s (default) | 0 / 30 (0%) | 7.4 (1) | 25 / 30 (83%) | 81.3 (11) | 14 / 30 | 7 lost from view, 4 not still long enough, 3 left the ground state |
| 2 s | 0 / 30 (0%) | 0.0 (0) | 25 / 30 (83%) | 81.3 (11) | 2 / 30 | 1 lost from view, 1 not still long enough |
| 3 s | 0 / 30 (0%) | 0.0 (0) | 25 / 30 (83%) | 81.3 (11) | 0 / 30 | none |
| 5 s | 0 / 30 (0%) | 0.0 (0) | 25 / 30 (83%) | 81.3 (11) | 0 / 30 | none |

**"On the ground" stage recall** (reached FALLEN, the step before confirmation; it is the same at every setting): **25 / 30 (83%)** of URFD falls. 11 of 42 no-fall videos also reached that stage without confirming.

**Fixes for losing the person on the floor**, each alone, at the default 1 s confirmation. These are replayed from the same pose cache. The region-local and rotated retries were run once per missing person and are used only where production would use them (falling or on the ground, within 3 s of the fall). The retry threshold is 0.15; normal is the pose model's threshold.

| Fix | Confirmed catches | Confirmed false alarms (/ h) | Possible catches | Possible false alarms (/ h) | Lost from view (of falls reaching the ground) |
|---|---|---|---|---|---|
| Baseline (no fixes) | 0 / 30 | 1 (7.4) | 25 / 30 | 11 (81.3) | 16 / 25 |
| 1. Hold lost track 5 s (last seen lying) | 6 / 30 | 4 +3 (29.6) | 26 / 30 | 11 (81.3) | 17 / 26 |
| 2. Region-local low threshold, as pose | 0 / 30 | 1 (7.4) | 27 / 30 | 14 (103.5) | 13 / 27 |
| 2. Region-local low threshold, as presence | 0 / 30 | 1 (7.4) | 26 / 30 | 12 (88.7) | 10 / 26 |
| 3. Rotated fallback, as pose | 0 / 30 | 1 (7.4) | 27 / 30 | 19 (140.4) | 7 / 27 |
| 3. Rotated fallback, as presence | 5 / 30 | 1 (7.4) | 26 / 30 | 12 (88.7) | 5 / 26 |
| 4. Ground-state hysteresis 0.5 s | 0 / 30 | 1 (7.4) | 25 / 30 | 11 (81.3) | 16 / 25 |
| 2 + 3 + 4, as pose (no hold) | 0 / 30 | 1 (7.4) | 27 / 30 | 15 (110.9) | 0 / 27 |
| 3 + 4, as presence (no hold) | 5 / 30 | 1 (7.4) | 26 / 30 | 12 (88.7) | 3 / 26 |
| 2 + 3 + 4, as presence (no hold) | 3 / 30 | 2 +1 (14.8) | 26 / 30 | 12 (88.7) | 0 / 26 |
| All four, as pose | 0 / 30 | 8 +7 (59.1) | 27 / 30 | 15 (110.9) | 0 / 27 |
| All four, as presence | 3 / 30 | 4 +3 (29.6) | 26 / 30 | 12 (88.7) | 0 / 26 |

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
| Baseline (no fixes) | 7 / 50 (14%) | 28.0 (4) | 21 / 50 (42%) | 63.0 (9) | 5 / 21 |
| Candidate: rotated retry + hysteresis 0.5 s, as presence | 12 / 50 (24%) | 35.0 (5) | 22 / 50 (44%) | 70.0 (10) | 5 / 22 |

False alarms by daily activity (confirmed / possible):

| Setting | Hop | Kneel | Pick up object | Sit down | Walk |
|---|---|---|---|---|---|
| Baseline (no fixes) | 0 / 0 | 0 / 1 | 3 / 5 | 1 / 2 | 0 / 1 |
| Candidate: rotated retry + hysteresis 0.5 s, as presence | 0 / 0 | 0 / 1 | 4 / 5 | 1 / 3 | 0 / 1 |

Any further tuning will split CAUCAFall by video into a tuning half and a test half, and report test-half numbers only.
<!-- benchmark:fall-sweep:end -->

## Fall detection accuracy (UR Fall Detection dataset)

<!-- benchmark:falls:start -->
_Measured 2026-10-01 with `python training/eval_fall.py` on cuda._ URFD camera 0 (RGB 640x480, 30 fps); production thresholds; onset tolerance 2.0 s.

| Metric | Value |
|---|---|
| Fall sequences / detected (recall) | 30 / 0 (0.0%) |
| Precision | 0.0% |
| F1 | n/a |
| False positives in fall sequences (early or repeated alerts) | 0 |
| False positives in ADL sequences | 1 over 40 sequences |
| **False alarms per hour of non-fall video** | **12.08**, based on **0.083 h (5.0 min, 8941 frames)** of ADL video |
| Alert latency after fall onset (median / max) | n/a s / n/a s |

The false-alarm rate rests on only 5.0 minutes of non-fall video (all that URFD provides), so treat it as a rough indicator, not a measured field rate. A reliable figure needs hours of normal-activity footage from the target site.

**Why alerts are missed here: stage diagnostics.** An alert is raised only after the state machine reaches FALLEN (a fast descent, then a lying pose) and the person then stays still for 1.0 s.

| Stage | Value |
|---|---|
| Fall sequences that reached FALLEN (on the ground, before confirmation) | 25 / 30 (83.3%) |
| Video left after fall onset (median) | 1.70 s |
| Video left after reaching FALLEN (median) | 1.03 s |
| ADL sequences that reached FALLEN (not confirmed) | 11 / 40 |

URFD trims each fall clip shortly after the fall. Where the detector reaches FALLEN, the median video left after that (1.03 s) is barely longer than the 1.0 s of stillness the detector waits for, so many clips end before an alert could fire. The confirmation-time sweep above breaks down the clips that had enough video but still got no alert (mostly the person is lost from view once on the floor). These numbers describe how the detector behaves on short, trimmed clips. They are not the recall you would see on continuous video, which needs longer fall recordings to measure.
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
