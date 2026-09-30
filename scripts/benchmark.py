"""Per-layer latency and throughput of the full pipeline on a fixed sample video.

    python scripts/benchmark.py                       # corridor sample, CPU (+ GPU if available)
    python scripts/benchmark.py --device cpu --frames 300
    python scripts/benchmark.py --write docs/BENCHMARKS.md   # replace the latency section there

Every frame goes through exactly what the live server runs: decode, detection + tracking
(one ultralytics call, so they are timed together), pose, analytics, annotation, event
publishing and clip buffering, and the WebSocket frame encoding. Events go to a throwaway
database and clip folder, and notifications are switched off. The first ``--warmup``
frames (model warm-up, CUDA init) are excluded from the statistics.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

from api.server import encode_frame_message
from config.settings import SentinelConfig
from core.pipeline import SentinelPipeline
from core.samples import SAMPLES, ensure_sample
from main import apply_demo_config

LAYERS = ["decode", "detect_track", "pose", "analytics", "annotate", "events_and_clips", "stream"]
SECTION_START = "<!-- benchmark:latency:start -->"
SECTION_END = "<!-- benchmark:latency:end -->"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def run(video: str, config_path: str | None, device: str, frames: int | None, warmup: int) -> dict:
    tmp = tempfile.mkdtemp(prefix="sentinel-bench-")
    cfg = SentinelConfig.from_env()
    cfg.source = video
    if config_path:
        apply_demo_config(cfg, config_path)
    cap = cv2.VideoCapture(video)
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25
    cfg.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cfg.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfg.detector.device = device
    cfg.output.db_path = os.path.join(tmp, "events.db")
    cfg.output.clips_dir = os.path.join(tmp, "clips")
    cfg.output.webhook_url = ""
    cfg.notifications.telegram_bot_token = cfg.notifications.smtp_host = ""
    pipeline = SentinelPipeline(cfg)
    pipeline.clip_recorder.set_fps(fps_native)
    pipeline.on_frame(lambda result: encode_frame_message(result, 0.0))

    per_layer: dict[str, list[float]] = {name: [] for name in LAYERS}
    totals: list[float] = []
    alerts = 0
    i = 0
    wall_start = None
    while frames is None or i < frames + warmup:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        decode_ms = (time.perf_counter() - t0) * 1000
        if not ok:
            break
        result = pipeline.process_frame(frame, 1_000_000.0 + i / fps_native)
        alerts += len(result.alerts)
        if i == warmup:
            wall_start = time.perf_counter()
        if i >= warmup:
            per_layer["decode"].append(decode_ms)
            for name in LAYERS[1:]:
                per_layer[name].append(result.timings_ms.get(name, 0.0))
            totals.append(decode_ms + sum(result.timings_ms.values()))
        i += 1
    wall = time.perf_counter() - (wall_start or time.perf_counter())
    cap.release()
    pipeline.clip_recorder.flush()
    pipeline.notifier.close()
    n = len(totals)
    return {
        "device": device,
        "frames": n,
        "native_fps": fps_native,
        "resolution": f"{cfg.frame_width}x{cfg.frame_height}",
        "alerts": alerts,
        "fps": n / wall if wall > 0 else 0.0,
        "layers": {
            name: (statistics.mean(v), percentile(v, 50), percentile(v, 95)) for name, v in per_layer.items() if v
        },
        "total": (statistics.mean(totals), percentile(totals, 50), percentile(totals, 95)) if totals else None,
    }


def hardware(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return f"GPU: {torch.cuda.get_device_name(0)}"
    try:
        import subprocess

        name = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() if platform.system() == "Windows" else platform.processor()
    except Exception:
        name = platform.processor()
    return f"CPU: {name or platform.machine()}"


def markdown(results: list[dict], sample: str) -> str:
    import torch
    import ultralytics

    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d %H:%M} with `python scripts/benchmark.py`_ on the "
        f"`{sample}` sample ({results[0]['resolution']}, {results[0]['native_fps']:.0f} fps native), "
        f"YOLOv8n + YOLOv8n-pose, torch {torch.__version__}, ultralytics {ultralytics.__version__}, "
        f"Python {platform.python_version()}, {platform.system()} {platform.release()}.",
        "",
    ]
    for r in results:
        lines += [
            f"**{hardware(r['device'])}**: {r['frames']} frames after warm-up, "
            f"**{r['fps']:.1f} fps** end to end (unpaced), {r['alerts']} alerts.",
            "",
            "| Layer | mean ms | p50 ms | p95 ms |",
            "|---|---|---|---|",
        ]
        for name, (mean, p50, p95) in r["layers"].items():
            lines.append(f"| {name} | {mean:.2f} | {p50:.2f} | {p95:.2f} |")
        mean, p50, p95 = r["total"]
        lines += [f"| **total per frame** | **{mean:.2f}** | **{p50:.2f}** | **{p95:.2f}** |", ""]
    lines += [
        "- `detect_track` is YOLOv8n detection and ByteTrack together: ultralytics runs both in one "
        "`model.track` call, so they are not timed separately.",
        "- `events_and_clips` is clip/snapshot saving, event publishing (store, notifications, "
        "WebSocket) and the clip ring buffer. Encoding runs on a background thread, not here.",
        "- `stream` is the JPEG encode and JSON serialisation the server does once per frame for "
        "all dashboard clients.",
        "- Capture waiting is excluded: live sources are paced by the camera, and files by their "
        "native fps.",
        SECTION_END,
    ]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    text = "# Benchmarks\n\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        after = rest.split(SECTION_END, 1)[1]
        text = before + section + after
    else:
        text = text.rstrip() + "\n\n## Latency and throughput\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", choices=sorted(SAMPLES), default="corridor")
    parser.add_argument("--source", help="benchmark this video file instead of a sample")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto = CPU, plus GPU if one is available")
    parser.add_argument("--frames", type=int, default=None, help="frames to measure (default: whole video)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--write", metavar="MD", help="replace the latency section of this markdown file")
    args = parser.parse_args()

    video = args.source or str(ensure_sample(args.demo))
    config_path = None if args.source else str(SAMPLES[args.demo].config_path)
    devices = [args.device]
    if args.device == "auto":
        import torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = []
    for device in devices:
        print(f"benchmarking on {device} ...", flush=True)
        results.append(run(video, config_path, device, args.frames, args.warmup))
    section = markdown(results, os.path.basename(video) if args.source else args.demo)
    print("\n" + section)
    if args.write:
        write_section(args.write, section)
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
