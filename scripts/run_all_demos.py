"""
Sentinel AI - Master Demo Runner
Runs every bundled sample scenario through the pipeline, then stitches a demo reel.

Usage:
    python scripts/run_all_demos.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
PYTHON = sys.executable

sys.path.append(os.path.join(ROOT, "backend"))
from core.samples import SAMPLES  # noqa: E402

SECTORS = {"corridor": "SECTOR-01-CORRIDOR", "hallway": "SECTOR-02-HALLWAY"}


def run_demo(name: str) -> bool:
    print(f"\n{'-' * 60}\n  Running: {name} ({SAMPLES[name].description})\n{'-' * 60}")
    cmd = [PYTHON, os.path.join(SCRIPTS, "run_demo.py"), "--demo", name,
           "--sector", SECTORS.get(name, "SECTOR-01")]
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main():
    print("=" * 60)
    print("  SENTINEL AI - Master Demo Pipeline")
    print("=" * 60)

    start = time.time()
    results = {name: "OK" if run_demo(name) else "FAIL" for name in SAMPLES}

    print(f"\n{'=' * 60}\n  Post-Production\n{'=' * 60}")
    post = subprocess.run([PYTHON, os.path.join(SCRIPTS, "post_production.py")], cwd=ROOT)
    results["post-production"] = "OK" if post.returncode == 0 else "FAIL"

    print(f"\n{'=' * 60}\n  PIPELINE COMPLETE - {time.time() - start:.0f}s total\n{'=' * 60}")
    for name, status in results.items():
        print(f"  [{status}] {name}")

    print("\n  Outputs:")
    for d in ("outputs", "assets"):
        full_dir = os.path.join(ROOT, d)
        if os.path.isdir(full_dir):
            for f in sorted(os.listdir(full_dir)):
                full = os.path.join(full_dir, f)
                if os.path.isfile(full):
                    print(f"    {d}/{f} ({os.path.getsize(full) / 1e6:.1f} MB)")
    sys.exit(0 if all(s == "OK" for s in results.values()) else 1)


if __name__ == "__main__":
    main()
