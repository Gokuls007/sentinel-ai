"""Sentinel AI entry point: runs the video pipeline and the API/dashboard server.

Examples:
    python backend/main.py --demo                 # no camera needed: looped sample video
    python backend/main.py --demo hallway         # another demo scenario
    python backend/main.py --source 0             # webcam
    python backend/main.py --source clip.mp4 --loop
    python backend/main.py --source rtsp://user:pass@cam/stream
Settings come from .env / environment variables (see .env.example); flags override them.
"""

import argparse
import json
import logging
import sys

# Logging must be configured before any module logs during import.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

import uvicorn  # noqa: E402

from config.settings import SentinelConfig  # noqa: E402
from core.samples import DEFAULT_SAMPLE, SAMPLES, ensure_sample  # noqa: E402

logger = logging.getLogger("sentinel.main")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sentinel AI - Video Intelligence Pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: webcam index, file path, RTSP/HTTP URL, or 'demo'")
    parser.add_argument("--demo", nargs="?", const=DEFAULT_SAMPLE, default=None,
                        choices=sorted(SAMPLES),
                        help="Run on a bundled sample video (no camera needed); loops forever")
    parser.add_argument("--loop", dest="loop", action="store_true", default=None,
                        help="Restart video files when they end")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Stop at the end of a video file")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Compute device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--host", type=str, default=None, help="Server host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port for the web server")
    return parser.parse_args(argv)


def apply_demo_config(config: SentinelConfig, path) -> None:
    """Point zones at the scenario file and apply its setting overrides, if any."""
    config.zone.zones_file = str(path)
    with open(path, encoding="utf-8") as f:
        demo = json.load(f)
    for section, values in demo.get("overrides", {}).items():
        target = getattr(config, section, None)
        for key, value in values.items():
            if target is not None and hasattr(target, key):
                setattr(target, key, value)
            else:
                logger.warning("Ignoring unknown demo override %s.%s", section, key)


def build_config(args) -> SentinelConfig:
    config = SentinelConfig.from_env()
    if args.source is not None:
        config.source = args.source
    if str(config.source).lower() == "demo" and args.demo is None:
        args.demo = DEFAULT_SAMPLE
    if args.demo:
        sample = SAMPLES[args.demo]
        config.source = str(ensure_sample(args.demo))
        config.loop = True
        if sample.config_path.is_file():
            apply_demo_config(config, sample.config_path)
    if args.loop is not None:
        config.loop = args.loop
    if args.confidence is not None:
        config.detector.confidence_threshold = args.confidence
    if args.device is not None:
        config.detector.device = args.device
    if args.host is not None:
        config.server.host = args.host
    if args.port is not None:
        config.server.port = args.port
    return config


def main(argv=None):
    args = parse_args(argv)
    try:
        config = build_config(args)
    except Exception as e:  # e.g. sample download failed / checksum mismatch
        logger.error("Could not prepare the video source: %s", e)
        sys.exit(1)

    # Imported after logging/config so model loading messages use our format.
    from api.server import app, start_pipeline

    logger.info("=" * 60)
    logger.info("  SENTINEL AI — Real-Time Video Intelligence")
    logger.info("=" * 60)
    logger.info("  Source:    %s%s", config.source, " (looping)" if config.loop else "")
    logger.info("  Device:    %s", config.detector.device)
    logger.info("  Zones:     %s", config.zone.zones_file)
    logger.info("  Dashboard: http://localhost:%d", config.server.port)
    logger.info("=" * 60)

    try:
        start_pipeline(config)
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")


if __name__ == "__main__":
    main()
