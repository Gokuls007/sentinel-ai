"""Count fall alerts on footage with no falls in it, beyond URFD's ADL clips.

    python training/eval_false_alarms.py                          # every video below that exists
    python training/eval_false_alarms.py --video clip.mp4 --activities squat,kneel
    python training/eval_false_alarms.py --write docs/BENCHMARKS.md

Videos come from three places:
- ``training/nonfall_videos.json``: the committed sample clips (corridor and hallway).
- ``data/recordings/*.mp4|*.avi|*.mov|*.mkv``: your own recordings (data/ is not committed),
  for example the ergonomics clip or webcam recordings. An optional sidecar
  ``<clip>.json`` describes what happens in it::

      {"label": "Ergonomics clip", "activities": ["lifting", "squat", "kneel"],
       "segments": [{"start_s": 12.0, "end_s": 20.5, "activity": "squat"}]}

- ``--video PATH`` on the command line (with ``--activities``).

Every video must contain **no real fall**. Each fall alert is then a false alarm.

**Hard negatives** are squats and kneeling. They look like a fall to a simple detector, because
the head drops and the box gets wider. Two cases:
- With ``segments``, only the labelled stretches count as hard-negative time, and alerts
  inside them count as hard-negative alerts.
- Without segments, a video whose ``activities`` include squat or kneel counts as
  hard-negative footage as a whole. The report says which of the two it used.

This runs the same production path as eval_fall.py: YOLOv8n + ByteTrack, then YOLOv8n-pose,
then the fall state machine, with thresholds from config/settings.py. Frames are resized to
the configured frame size, as VideoSource does. Timestamps follow the video's own timeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MANIFEST = os.path.join(ROOT, "training", "nonfall_videos.json")
RECORDINGS_DIR = os.path.join(ROOT, "data", "recordings")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
HARD_NEGATIVES = ("squat", "kneel")
SECTION_START = "<!-- benchmark:false-alarms:start -->"
SECTION_END = "<!-- benchmark:false-alarms:end -->"


@dataclass
class Video:
    path: str
    label: str
    activities: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    source: str = ""

    @property
    def hard_negative_activities(self) -> list[str]:
        acts = set(self.activities) | {s.get("activity", "") for s in self.segments}
        return sorted(a for a in acts if a in HARD_NEGATIVES)


@dataclass
class VideoResult:
    label: str
    path: str
    source: str
    activities: list[str]
    seconds: float = 0.0
    frames: int = 0
    alert_times_s: list[float] = field(default_factory=list)
    hard_negative_seconds: float = 0.0
    hard_negative_alerts: int = 0
    hard_negative_basis: str = ""  # "segments", "whole video" or ""
    missing: bool = False


# --- finding videos ------------------------------------------------------------------------

def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_manifest(path: str = MANIFEST) -> list[Video]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Video(path=v["path"], label=v.get("label", os.path.basename(v["path"])),
                  activities=list(v.get("activities", [])), segments=list(v.get("segments", [])),
                  source="sample") for v in data.get("videos", [])]


STAGED = ("lying", "fall", "rolling", "on floor")


def staged_fall(path: str) -> bool:
    """A recording whose ``.labels.json`` segments include lying down, falling or rolling off
    something: deliberate fall-like footage, so an alert in it is not a false alarm."""
    labels = os.path.splitext(path)[0] + ".labels.json"
    if not os.path.isfile(labels):
        return False
    with open(labels, encoding="utf-8") as f:
        segments = json.load(f).get("segments", [])
    return any(any(w in str(s.get("action", "")).lower() for w in STAGED) for s in segments)


def load_recordings(directory: str = RECORDINGS_DIR) -> list[Video]:
    videos = []
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if not path.lower().endswith(VIDEO_EXTS):
            continue
        meta: dict = {}
        sidecar = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(sidecar):
            with open(sidecar, encoding="utf-8") as f:
                meta = json.load(f)
        if staged_fall(path):
            print(f"skipping {os.path.basename(path)}: its labels include deliberate lying/falls", flush=True)
            continue
        videos.append(Video(path=path, label=meta.get("label", os.path.basename(path)),
                            activities=list(meta.get("activities", [])),
                            segments=list(meta.get("segments", [])), source="recording"))
    return videos


# --- scoring (pure; unit-tested) ---------------------------------------------------------------

def attribute(result: VideoResult, video: Video) -> VideoResult:
    """Fill the hard-negative time and alert counts from the segments or whole-video activities."""
    hard_segments = [s for s in video.segments if s.get("activity") in HARD_NEGATIVES]
    if hard_segments:
        result.hard_negative_basis = "segments"
        result.hard_negative_seconds = sum(
            max(0.0, min(float(s["end_s"]), result.seconds) - max(0.0, float(s["start_s"])))
            for s in hard_segments)
        result.hard_negative_alerts = sum(
            any(float(s["start_s"]) <= t <= float(s["end_s"]) for s in hard_segments)
            for t in result.alert_times_s)
    elif any(a in HARD_NEGATIVES for a in video.activities):
        result.hard_negative_basis = "whole video"
        result.hard_negative_seconds = result.seconds
        result.hard_negative_alerts = len(result.alert_times_s)
    return result


def summarize(results: list[VideoResult]) -> dict:
    ran = [r for r in results if not r.missing]
    hours = sum(r.seconds for r in ran) / 3600
    alerts = sum(len(r.alert_times_s) for r in ran)
    hn_hours = sum(r.hard_negative_seconds for r in ran) / 3600
    hn_alerts = sum(r.hard_negative_alerts for r in ran)
    return {
        "videos": len(ran), "missing": [r.path for r in results if r.missing],
        "hours": hours, "alerts": alerts, "alerts_per_hour": alerts / hours if hours else None,
        "hard_negative_hours": hn_hours, "hard_negative_alerts": hn_alerts,
        "hard_negative_alerts_per_hour": hn_alerts / hn_hours if hn_hours else None,
    }


# --- running ----------------------------------------------------------------------------------

def run_video(models, video: Video) -> VideoResult:
    result = VideoResult(label=video.label, path=os.path.relpath(_abs(video.path), ROOT),
                         source=video.source, activities=video.activities)
    cap = cv2.VideoCapture(_abs(video.path))
    if not cap.isOpened():
        result.missing = True
        return result
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 240 else 25.0
    width, height = models.cfg.frame_width, models.cfg.frame_height
    detector, pose, falls = models.detector, models.pose, models.fresh()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        ts = idx / fps
        dets = detector.detect_and_track(frame)
        people = [d for d in dets.detections if d.class_name == "person" and d.track_id is not None]
        poses = pose.estimate(frame, [d.track_id for d in people], [d.bbox for d in people], ts)
        feats = pose.get_all_features()
        falls.prune(feats.keys())
        for tid, p in poses.items():
            if tid in feats and falls.check(tid, p, feats[tid], ts):
                result.alert_times_s.append(round(ts, 2))
        idx += 1
    cap.release()
    result.frames = idx
    result.seconds = idx / fps
    return attribute(result, video)


# --- report -----------------------------------------------------------------------------------

def _rate(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def markdown(results: list[VideoResult], s: dict, device: str) -> str:
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python training/eval_false_alarms.py` on {device}._ "
        "These videos are separate from URFD and contain no falls, so every fall alert in them is a "
        "false alarm. Same production path and thresholds.",
        "",
        "| Video | Source | Activities | Length | Fall alerts | Alerts/hour |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.missing:
            continue
        rate = len(r.alert_times_s) / (r.seconds / 3600) if r.seconds else None
        lines.append(f"| {r.label} | {r.source} | {', '.join(r.activities) or 'n/a'} | "
                     f"{r.seconds / 60:.1f} min | {len(r.alert_times_s)} | {_rate(rate)} |")
    lines += [
        f"| **All non-fall footage** | | | **{s['hours']:.3f} h** | **{s['alerts']}** | "
        f"**{_rate(s['alerts_per_hour'])}** |",
        "",
        "**Hard negatives (squatting and kneeling).** These look the most like a fall to this "
        "detector, because the head drops and the box gets wider.",
        "",
    ]
    hn = [r for r in results if not r.missing and r.hard_negative_basis]
    if hn:
        lines += ["| Video | Counted from | Hard-negative time | Alerts in it |", "|---|---|---|---|"]
        lines += [f"| {r.label} | {r.hard_negative_basis} | {r.hard_negative_seconds / 60:.1f} min | "
                  f"{r.hard_negative_alerts} |" for r in hn]
        lines += ["", f"Total: {s['hard_negative_alerts']} false alarms in "
                      f"{s['hard_negative_hours'] * 60:.1f} min of squatting/kneeling "
                      f"({_rate(s['hard_negative_alerts_per_hour'])} per hour)."]
    else:
        lines.append("_No squat or kneel footage evaluated yet. Add the ergonomics clip to "
                     "`data/recordings/` with a sidecar JSON naming those activities, then re-run._")
    if s["missing"]:
        lines += ["", "Listed but not found (skipped): " + ", ".join(f"`{p}`" for p in s["missing"])]
    lines += [
        "",
        f"This footage totals {s['hours'] * 60:.1f} minutes. That is far too little for a real "
        "field false-alarm rate, which needs hours of normal work at the target site.",
        SECTION_END,
    ]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## False alarms on other non-fall footage\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[], help="extra non-fall video (repeatable)")
    ap.add_argument("--activities", default="", help="comma-separated activities for --video clips")
    ap.add_argument("--no-samples", action="store_true", help="skip training/nonfall_videos.json")
    ap.add_argument("--no-recordings", action="store_true", help="skip data/recordings/")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", metavar="MD", help="replace the false-alarm section of this markdown file")
    args = ap.parse_args()

    videos: list[Video] = []
    if not args.no_samples:
        videos += load_manifest()
    if not args.no_recordings:
        videos += load_recordings()
    acts = [a.strip() for a in args.activities.split(",") if a.strip()]
    videos += [Video(path=p, label=os.path.basename(p), activities=acts, source="command line")
               for p in args.video]
    if not videos:
        print("no videos to evaluate")
        return 1

    from eval_fall import Models

    models = Models(args.device)
    results = []
    for v in videos:
        t0 = time.time()
        r = run_video(models, v)
        results.append(r)
        if r.missing:
            print(f"skip {v.path}: not found")
        else:
            print(f"{r.label}: {r.seconds:.0f}s, fall alerts at {r.alert_times_s} [{time.time() - t0:.0f}s]",
                  flush=True)
    summary = summarize(results)
    out = os.path.join(ROOT, "outputs", "eval_false_alarms.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": [asdict(r) for r in results]}, f, indent=2)
    section = markdown(results, summary, args.device)
    print("\n" + section + f"\n\nper-video results: {out}")
    if args.write:
        write_section(args.write, section)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
