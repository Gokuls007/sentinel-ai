# Benchmarks

All numbers here are produced by scripts in this repo and can be re-run. Sections are
replaced in place by `--write docs/BENCHMARKS.md`.

| Section | Script |
|---|---|
| Latency and throughput | `python scripts/benchmark.py --write docs/BENCHMARKS.md` |
| Fall detection accuracy (URFD) | `python training/eval_fall.py --download --write docs/BENCHMARKS.md` |

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
