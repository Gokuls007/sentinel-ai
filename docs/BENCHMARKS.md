# Benchmarks

All numbers here are produced by scripts in this repo and can be re-run. Sections are
replaced in place by `--write docs/BENCHMARKS.md`.

| Section | Script |
|---|---|
| Latency and throughput | `python scripts/benchmark.py --write docs/BENCHMARKS.md` |
| Fall detection accuracy (URFD) | `python training/eval_fall.py --download --write docs/BENCHMARKS.md` |
| False alarms on other non-fall footage | `python training/eval_false_alarms.py --write docs/BENCHMARKS.md` |

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
| Fall sequences that reached FALLEN (on the ground, before confirmation) | 20 / 30 (66.7%) |
| Video left after fall onset (median) | 1.70 s |
| Video left after reaching FALLEN (median) | 0.62 s |
| ADL sequences that reached FALLEN (not confirmed) | 3 / 40 |

URFD trims each fall clip shortly after the fall. Where the detector does reach FALLEN, the median video left after that is shorter than the stillness the detector waits for, so the clip ends before an alert could fire. In about half of the clips that never reach FALLEN, the person stops being detected once on the floor (YOLOv8n misses many lying people at this camera angle). In the rest, the lying pose never crosses the aspect-ratio or head-drop threshold. These numbers describe how the detector behaves on short, trimmed clips. They are not the recall you would see on continuous video, which needs longer fall recordings to measure.
<!-- benchmark:falls:end -->
