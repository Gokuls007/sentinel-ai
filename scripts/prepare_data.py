"""
Sentinel AI — Demo Video Asset Downloader
Downloads surveillance-style sample videos for pipeline demos.
"""
import urllib.request
import os
import sys
import hashlib

VIDEOS = {
    "fall_sample.mp4": {
        "url": "https://github.com/computationalcore/fall-detection/raw/master/example/demo.mp4",
        "description": "Indoor person falling (UR Fall style)",
        "min_size": 500_000  # 500KB minimum to consider valid
    },
    "walking_sample.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4",
        "description": "Single person walking through corridor",
        "min_size": 1_000_000
    },
    "multi_person_sample.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "description": "Multi-person busy scene",
        "min_size": 2_000_000
    },
    "corridor_sample.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "description": "Outdoor corridor / people detection",
        "min_size": 2_000_000
    }
}

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_videos")


def verify_video(path: str, min_size: int) -> bool:
    """Verify a downloaded video exists and meets minimum size."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < min_size:
        print(f"  WARNING: {path} is only {size} bytes (minimum: {min_size}). Will re-download.")
        return False
    return True


def download_with_retry(url: str, dest: str, retries: int = 3) -> bool:
    """Download a URL with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries}: {url}")
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            print(f"  Error on attempt {attempt}: {e}")
            if attempt == retries:
                return False
    return False


def download_data(force: bool = False):
    """Download all demo videos. Skips if valid file already exists."""
    os.makedirs(DEST_DIR, exist_ok=True)
    
    success = 0
    skipped = 0
    failed = 0
    
    print("=" * 60)
    print("  SENTINEL AI — Demo Video Asset Manager")
    print("=" * 60)
    
    for name, info in VIDEOS.items():
        dest = os.path.join(DEST_DIR, name)
        print(f"\n[{name}] {info['description']}")
        
        # Skip if already valid
        if not force and verify_video(dest, info["min_size"]):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  EXISTS — {size_mb:.1f} MB (valid). Skipping.")
            skipped += 1
            continue
        
        # Download
        print(f"  Downloading...")
        if download_with_retry(info["url"], dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  SUCCESS — Saved {name} ({size_mb:.1f} MB)")
            success += 1
        else:
            print(f"  FAILED — Could not download {name}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Results: {success} downloaded, {skipped} skipped, {failed} failed")
    print(f"  Output: {os.path.abspath(DEST_DIR)}")
    print(f"{'=' * 60}")
    
    return failed == 0


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        print("Force mode: re-downloading all clips.\n")
    success = download_data(force=force)
    sys.exit(0 if success else 1)
