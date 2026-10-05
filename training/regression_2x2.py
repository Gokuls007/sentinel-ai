"""Why the fall benchmarks moved when the person detector and pose model changed: each change alone.

    python training/regression_2x2.py            # caches poses per setup (~4 min each), then reports

Four setups, on the clips that moved (URFD falls 1-30; CAUCAFall daily activities of subjects 1-5,
plus Subject.10/Kneel, the held-out clip that newly alerted):

    detector YOLOv8n @ 0.5 or YOLO11m @ 0.25   x   pose YOLOv8n-pose or RTMPose-m

Same fall rules and replay as training/sweep_fall_confirm.py (baseline fixes, 1 s confirmation).
Per setup: URFD falls that reach the on-the-ground stage (and the diagnosis of those that don't,
from fall_round3.diagnose_clip), falls where the person comes back under a new track id after
the onset, and CAUCAFall daily activities that raise a possible / confirmed fall.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "regression_2x2")
SETUPS = {  # tag: (detector, confidence, pose backend)
    "v8n@0.5 + yolo-pose": ("yolov8n.pt", "0.5", "yolo"),
    "v8n@0.5 + rtmpose": ("yolov8n.pt", "0.5", "rtmpose"),
    "11m@0.25 + yolo-pose": ("yolo11m.pt", "0.25", "yolo"),
    "11m@0.25 + rtmpose": ("yolo11m.pt", "0.25", "rtmpose"),
}


def cache_path(tag: str) -> str:
    return os.path.join(OUT, tag.replace(" ", "").replace("+", "_").replace("@", "-") + ".pkl")


def build(path: str) -> None:
    """Run in a child process: the setup comes from the environment (production Models)."""
    import eval_fall as ef
    import sweep_fall_confirm as sw

    models = ef.Models("auto")
    size = (models.cfg.frame_width, models.cfg.frame_height)
    onsets = ef.fall_onsets(os.path.join(ef.DATA_DIR, "urfall-cam0-falls.csv"))
    videos = []
    for i in range(1, ef.N_FALLS + 1):
        name = f"fall-{i:02d}"
        videos.append(sw.cache_frames(models, ef.frames_from_zip(os.path.join(ef.DATA_DIR, f"{name}-cam0-rgb.zip")),
                                      name, "fall", ef.FPS, onsets[name]))
    for video, _labels, rel in sw.caucafall_videos():
        subject = int(rel.split("/")[0].split(".")[1])
        is_fall = os.path.basename(os.path.dirname(video)).lower().startswith("fall")
        if is_fall or not (subject <= 5 or rel == "Subject.10/Kneel"):
            continue
        frames, fps = sw.video_frames(video)
        videos.append(sw.cache_frames(models, frames, rel, "caucafall_adl", fps, resize=size))
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(videos, f)


def new_id_after_onset(v) -> bool:
    first = next((tid for _ts, _a, rows, *_ in v.frames for tid, _p, _h in rows), None)
    onset = (v.onset_frame - 1) / v.fps
    return any(tid != first for ts, _a, rows, *_ in v.frames if ts >= onset for tid, _p, _h in rows)


def report() -> None:
    import fall_round3 as r3
    import sweep_fall_confirm as sw
    from config.settings import FallDetectorConfig

    cfg = FallDetectorConfig()
    for tag in SETUPS:
        with open(cache_path(tag), "rb") as f:
            videos = pickle.load(f)
        falls = [v for v in videos if v.kind == "fall"]
        daily = [v for v in videos if v.kind != "fall"]
        why = [r3.diagnose_clip(v, cfg)["reason"].split(" (")[0] for v in falls]
        switched = sum(new_id_after_onset(v) for v in falls)
        possible, confirmed = [], []
        for v in daily:
            r = sw.replay(v, sw.make_detector(cfg, 1.0, sw.BASELINE), sw.BASELINE)
            possible += [v.name] if r["possible"] else []
            confirmed += [v.name] if r["alerts"] else []
        reached = why.count("reached the ground")
        print(f"\n{tag}\n  URFD reached the ground: {reached}/{len(falls)}; new track id after onset: "
              f"{switched}/{len(falls)}")
        print("  misses: " + ", ".join(f"{n} x {k}" for k, n in Counter(why).items() if k != "reached the ground"))
        print(f"  CAUCAFall daily activities, possible fall: {possible or 'none'}")
        print(f"  CAUCAFall daily activities, confirmed fall: {confirmed or 'none'}")


def main() -> int:
    sys.path[:0] = [os.path.join(ROOT, "backend"), os.path.join(ROOT, "training")]
    os.chdir(ROOT)
    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        build(sys.argv[2])
        return 0
    for tag, (model, conf, pose) in SETUPS.items():
        if not os.path.isfile(cache_path(tag)):
            print(f"caching {tag} ...", flush=True)
            env = {**os.environ, "DETECTION_MODEL": model, "DETECTION_CONFIDENCE": conf, "POSE_BACKEND": pose}
            subprocess.run([sys.executable, __file__, "--build", cache_path(tag)], env=env, check=True)
    import sweep_fall_confirm as sw

    sys.modules.setdefault("sweep_fall_confirm", sw)  # the pickles name this module
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
