import argparse
import logging
import uvicorn
from config.settings import SentinelConfig
from api.server import app, start_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", 
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("sentinel.main")

def parse_args():
    parser = argparse.ArgumentParser(description="Sentinel AI - Video Intelligence Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Configuration
    config = SentinelConfig()
    config.source = args.source
    config.detector.confidence_threshold = args.confidence
    config.detector.device = args.device
    config.server.port = args.port
    
    logger.info("=" * 60)
    logger.info("  SENTINEL AI — Real-Time Video Intelligence")
    logger.info("=" * 60)
    logger.info(f"  Source: {config.source}")
    logger.info(f"  Device: {config.detector.device}")
    logger.info(f"  API:    http://localhost:{config.server.port}")
    logger.info("=" * 60)
    
    # Start Pipeline and Web Server
    try:
        start_pipeline(config)
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
