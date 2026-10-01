# Benchmarks

All numbers here are produced by scripts in this repo and can be re-run. Sections are
replaced in place by `--write docs/BENCHMARKS.md`.

| Section | Script |
|---|---|
| Latency and throughput | `python scripts/benchmark.py --write docs/BENCHMARKS.md` |
| Fall detection accuracy (URFD) | `python training/eval_fall.py --download --write docs/BENCHMARKS.md` |
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

## Fall confirmation time sweep

<!-- benchmark:fall-sweep:start -->
_Measured 2026-10-01 with `python training/sweep_fall_confirm.py` on cuda._ The confirmation time is how long a person must lie still on the ground before the alert. The current default is **1 s**. Detection and pose ran once per video, and only the fall state machine was replayed at each setting. The "gave up" timeout is at least confirmation + 2 s.

No-fall footage: **8.1 min** (URFD ADL 5.0 min, sample clips 3.2 min).

| Confirmation time | URFD recall (confirmed alerts) | Confirmable on URFD | Confirmable but missed (why) | False alarms / hour (no-fall footage) | False alarms (count) |
|---|---|---|---|---|---|
| 0.5 s | 3 / 30 (10%) | 23 / 30 | 15 lost from view, 3 left the ground state, 2 not still long enough | 14.8 | 2 |
| 1 s (default) | 0 / 30 (0%) | 14 / 30 | 7 lost from view, 4 not still long enough, 3 left the ground state | 7.4 | 1 |
| 2 s | 0 / 30 (0%) | 2 / 30 | 1 lost from view, 1 not still long enough | 0.0 | 0 |
| 3 s | 0 / 30 (0%) | 0 / 30 | none | 0.0 | 0 |
| 5 s | 0 / 30 (0%) | 0 / 30 | none | 0.0 | 0 |

**"On the ground" stage recall** (reached FALLEN, the step before confirmation; it is the same at every setting): **25 / 30 (83%)** of URFD falls. 11 of 42 no-fall videos also reached that stage without confirming.

**Fixes for losing the person on the floor**, each alone, at the default 1 s confirmation. These are replayed from the same pose cache. The region-local and rotated retries were run once per missing person and are used only where production would use them (falling or on the ground, within 3 s of the fall). The retry threshold is 0.15; normal is the pose model's threshold.

| Fix | URFD catches | Lost from view (of falls reaching the ground) | Reached the ground | False alarms (no-fall footage) | False alarms / hour |
|---|---|---|---|---|---|
| Baseline (no fixes) | 0 / 30 | 16 / 25 | 25 / 30 | 1 | 7.4 |
| 1. Hold lost track 5 s (last seen lying) | 6 / 30 | 17 / 26 | 26 / 30 | 4 (+3) | 29.6 |
| 2. Region-local low threshold, as pose | 0 / 30 | 13 / 27 | 27 / 30 | 1 | 7.4 |
| 2. Region-local low threshold, as presence | 0 / 30 | 10 / 26 | 26 / 30 | 1 | 7.4 |
| 3. Rotated fallback, as pose | 0 / 30 | 7 / 27 | 27 / 30 | 1 | 7.4 |
| 3. Rotated fallback, as presence | 5 / 30 | 5 / 26 | 26 / 30 | 1 | 7.4 |
| 4. Ground-state hysteresis 0.5 s | 0 / 30 | 16 / 25 | 25 / 30 | 1 | 7.4 |
| 2 + 3 + 4, as pose (no hold) | 0 / 30 | 0 / 27 | 27 / 30 | 1 | 7.4 |
| 3 + 4, as presence (no hold) | 5 / 30 | 3 / 26 | 26 / 30 | 1 | 7.4 |
| 2 + 3 + 4, as presence (no hold) | 3 / 30 | 0 / 26 | 26 / 30 | 2 (+1) | 14.8 |
| All four, as pose | 0 / 30 | 0 / 27 | 27 / 30 | 8 (+7) | 59.1 |
| All four, as presence | 3 / 30 | 0 / 26 | 26 / 30 | 4 (+3) | 29.6 |

*Lost from view*: the person's pose (tracked or recovered) was missing in most frames after reaching the ground. Fix 1 doesn't find the person; it keeps the fall alive while they are missing, so it raises catches without lowering this count.

*As pose*: a re-found person goes through the normal check. *As presence*: a re-found person who was last seen lying counts as still in place, unless clearly upright or moved more than half a body height. The re-found keypoints jitter too much to measure stillness directly.

**Caveat: these rows are optimistic.** The "last seen lying" gate and the presence mode were designed after inspecting these same URFD clips. Before choosing defaults they need confirming on footage not used here: your own recordings and a held-out dataset (CAUCAFall). All fixes stay off by default until one is chosen.

How to read it:
- URFD trims each fall clip about 1–2 s after the fall. So "Confirmable" caps recall, and at 2 s and above URFD can't tell you anything about recall.
- "Confirmable but missed": the clip had enough video after reaching the ground, but no alert. *Lost from view*: the person's pose was missing in most later frames. *Left the ground state*: the pose looked upright again, or the wait timed out. *Not still long enough*: the person kept moving on the ground.
- The false-alarm rate rests on only a few minutes of no-fall video. Treat the rows as a comparison between settings, not a field rate.
- Your own recordings (`data/recordings/`) are added automatically when this is re-run.
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
