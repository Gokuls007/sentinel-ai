"""Sample videos for running Sentinel AI without a camera.

The clips are downloaded on demand (they are not stored in git) and verified by
SHA-256. All are from Intel's IoT DevKit sample-videos repository, licensed CC BY 4.0:
https://github.com/intel-iot-devkit/sample-videos (credit: Intel Corporation).
"""

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

logger = logging.getLogger("sentinel.samples")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "demo_videos"
DEMO_CONFIG_DIR = REPO_ROOT / "config" / "demo"
_BASE = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
LICENSE = "CC BY 4.0, Intel Corporation (github.com/intel-iot-devkit/sample-videos)"


@dataclass(frozen=True)
class Sample:
    name: str          # scenario name used on the command line
    filename: str
    url: str
    sha256: str
    description: str

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.filename

    @property
    def config_path(self) -> Path:
        return DEMO_CONFIG_DIR / f"{self.name}_demo.json"


SAMPLES: Dict[str, Sample] = {s.name: s for s in [
    Sample("corridor", "corridor_sample.mp4", _BASE + "people-detection.mp4",
           "18ffe8672d741e3e29c9d891d22c59d453720b086c25b35c88b393d55f92f693",
           "People walking through an office corridor past a restricted doorway"),
    Sample("hallway", "walking_sample.mp4", _BASE + "one-by-one-person-detection.mp4",
           "a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c",
           "People taking turns standing at a table (time-limited zone, loitering)"),
]}

DEFAULT_SAMPLE = "corridor"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_sample(name: str = DEFAULT_SAMPLE) -> Path:
    """Return the local path of a sample video, downloading and verifying it if needed."""
    if name not in SAMPLES:
        raise ValueError(f"Unknown sample {name!r}; choose from {', '.join(SAMPLES)}")
    sample = SAMPLES[name]
    if sample.path.is_file() and _sha256(sample.path) == sample.sha256:
        return sample.path

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = sample.path.with_suffix(".part")
    logger.info("Downloading sample video %s (%s) ...", sample.filename, LICENSE)
    urllib.request.urlretrieve(sample.url, tmp)
    digest = _sha256(tmp)
    if digest != sample.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {sample.filename}: got {digest}")
    os.replace(tmp, sample.path)
    return sample.path


def ensure_all() -> Dict[str, Path]:
    return {name: ensure_sample(name) for name in SAMPLES}
