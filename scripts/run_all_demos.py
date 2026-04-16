"""
Sentinel AI — Master Demo Runner
Runs all 4 demo scenarios through the pipeline, then post-produces into a reel.

Usage:
    python scripts/run_all_demos.py
"""
import os
import sys
import time
import subprocess

# Resolve paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
PYTHON = sys.executable

DEMOS = [
    {
        "name": "PERSISTENT TRACKING",
        "source": "demo_videos/walking_sample.mp4",
        "output": "outputs/walking_demo.mp4",
        "config": "config/demo/hallway_demo.json",
        "scenario": "PERSISTENT TRACKING",
        "sector": "SECTOR-01-MAIN"
    },
    {
        "name": "FALL DETECTION",
        "source": "demo_videos/fall_sample.mp4",
        "output": "outputs/fall_demo.mp4",
        "config": "config/demo/fall_demo.json",
        "scenario": "FALL DETECTION",
        "sector": "SECTOR-03-RESIDENTIAL"
    },
    {
        "name": "ZONE MONITORING",
        "source": "demo_videos/multi_person_sample.mp4",
        "output": "outputs/multi_person_demo.mp4",
        "config": "config/demo/multi_person_demo.json",
        "scenario": "ZONE MONITORING",
        "sector": "SECTOR-02-LOBBY"
    },
    {
        "name": "INTRUSION DETECTION",
        "source": "demo_videos/corridor_sample.mp4",
        "output": "outputs/corridor_demo.mp4",
        "config": "config/demo/corridor_demo.json",
        "scenario": "INTRUSION DETECTION",
        "sector": "SECTOR-04-NORTH"
    }
]


def run_demo(demo: dict) -> bool:
    """Run a single demo scenario."""
    print(f"\n{'─' * 60}")
    print(f"  Running: {demo['name']}")
    print(f"{'─' * 60}")
    
    source = os.path.join(ROOT, demo["source"])
    if not os.path.exists(source):
        print(f"  SKIP: Source not found: {source}")
        return False
    
    cmd = [
        PYTHON, os.path.join(SCRIPTS, "run_demo.py"),
        "--source", source,
        "--output", os.path.join(ROOT, demo["output"]),
        "--sector", demo["sector"],
        "--scenario", demo["scenario"]
    ]
    
    if demo.get("config"):
        config_path = os.path.join(ROOT, demo["config"])
        if os.path.exists(config_path):
            cmd.extend(["--config", config_path])
    
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  SENTINEL AI — Master Demo Pipeline")
    print("  Running all scenarios...")
    print("=" * 60)
    
    start = time.time()
    results = {}
    
    # Phase 1: Run all demos
    for demo in DEMOS:
        success = run_demo(demo)
        results[demo["name"]] = "OK" if success else "FAIL"
    
    # Phase 2: Post-production
    print(f"\n{'=' * 60}")
    print("  Phase 2: Post-Production")
    print(f"{'=' * 60}")
    
    post_result = subprocess.run(
        [PYTHON, os.path.join(SCRIPTS, "post_production.py")],
        cwd=ROOT
    )
    results["POST-PRODUCTION"] = "OK" if post_result.returncode == 0 else "FAIL"
    
    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {elapsed:.0f}s total")
    print(f"{'=' * 60}")
    for name, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name}: {status}")
    
    # List outputs
    outputs_dir = os.path.join(ROOT, "outputs")
    assets_dir = os.path.join(ROOT, "assets")
    print(f"\n  Outputs:")
    for d in [outputs_dir, assets_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                full = os.path.join(d, f)
                if os.path.isfile(full):
                    size = os.path.getsize(full) / (1024 * 1024)
                    print(f"    {os.path.relpath(full, ROOT)} ({size:.1f} MB)")
    
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
