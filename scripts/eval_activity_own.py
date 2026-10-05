"""Score the live activity labels on your own annotated recording.

    python scripts/eval_activity_own.py data/recordings/<clip>.mp4

Ground truth comes from ``<clip>.labels.json`` next to the video (time segments you mark by
hand, e.g. ``{"segments": [{"action": "lift", "start": 10.9, "end": 11.8}, ...]}``). The clip
and its labels stay local (``data/`` is git-ignored); keypoints are cached in
``outputs/activity_cache`` so rule changes can be re-scored without the models.

For each segment it prints the labels shown (main person, smoothed) and whether the expected
label appeared. Expected labels per action:

| Action            | Must show                  | Must not show (after the first 1 s) |
|-------------------|----------------------------|-------------------------------------|
| bend to pick up   | Bending                    |                                     |
| lift              | Lifting                    |                                     |
| carry             | Carrying                   | Lifting                             |
| put down          | (reported only)            |                                     |
| reach overhead    | Reaching overhead          | Lifting                             |
| squat             | Sitting or Bending         | Lifting, Carrying                   |
| bend without lift | Bending                    | Lifting, Carrying                   |
| walk, walk back   | Walking                    | Lifting                             |

Labels lag the pose (smoothing over 1 s), so a segment's expected label may show up to
0.7 s after its end, and forbidden labels are only counted from 1 s after its start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.join(ROOT, "training"))
sys.path.append(os.path.join(ROOT, "scripts"))

# action -> (labels that must appear, labels that must not appear)
EXPECT = {
    "bend to pick up": (["Bending"], []),
    "lift": (["Lifting"], []),
    "carry": (["Carrying"], ["Lifting"]),
    "put down": ([], []),
    "reach overhead": (["Reaching overhead"], ["Lifting"]),
    "squat": (["Sitting", "Bending"], ["Lifting", "Carrying"]),
    "bend without lift": (["Bending"], ["Lifting", "Carrying"]),
    "walk": (["Walking"], ["Lifting"]),
    "walk back": (["Walking"], ["Lifting"]),
    "carry one hand": (["Carrying"], ["Lifting"]),  # a load held low at the side
    "raise to chest": ([], []),
    "handle bag low": ([], []),
}


def segment_report(labels: list[tuple[float, str]], segments: list[dict], slack: float = 0.7,
                   lag: float = 1.0) -> list[dict]:
    """Per segment: labels shown inside it (plus ``slack`` s after, since labels lag the pose a
    little), whether an expected label appeared, and whether a forbidden one did (counted from
    ``lag`` s after the start)."""
    rows = []
    for s in segments:
        shown = Counter(lbl for t, lbl in labels if s["start"] <= t <= s["end"] + slack)
        settled = Counter(lbl for t, lbl in labels if s["start"] + lag <= t <= s["end"])
        must, never = EXPECT.get(s["action"], ([], []))
        hit = not must or any(shown[m] for m in must)
        bad = [n for n in never if settled[n] >= 3]  # 3 frames = 0.2 s at 15 fps
        rows.append({**s, "shown": shown.most_common(4), "ok": hit and not bad, "hit": hit, "bad": bad,
                     "expected": " or ".join(must) or "(reported only)"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", default=None, help="default outputs/activity_cache/<detector-conf-pose>")
    ap.add_argument("--write", metavar="MD", help="add or replace this clip's section in a markdown file")
    ap.add_argument("--held-out", action="store_true", help="this clip was never used for tuning (say so)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from eval_activity import extract, score_frames

    stem = os.path.splitext(os.path.basename(args.clip))[0]
    with open(os.path.splitext(args.clip)[0] + ".labels.json", encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    if args.cache is None:
        from eval_fall import setup_tag

        args.cache = os.path.join(ROOT, "outputs", "activity_cache", setup_tag())
    os.makedirs(args.cache, exist_ok=True)
    cache = os.path.join(args.cache, f"own__{stem}.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            frames = json.load(f)
    else:
        from eval_fall import Models

        frames = extract(Models(args.device), args.clip)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(frames, f)
    labels = score_frames(frames)
    rows = segment_report(labels, segments)
    for r in rows:
        shown = ", ".join(f"{k} {v}" for k, v in r["shown"]) or "(no person)"
        flag = "PASS" if r["ok"] else "FAIL"
        extra = f"  showed {', '.join(r['bad'])}" if r["bad"] else ""
        span = f"{r['start']:5.1f}-{r['end']:5.1f}s"
        print(f"{flag} {r['action']:<18} {span}  expected {r['expected']:<20} [{shown}]{extra}")
    if args.write:
        write_section(args.write, stem, rows, args.held_out)
    return 0


def write_section(md_path: str, stem: str, rows: list[dict], held_out: bool = False) -> None:
    from datetime import datetime

    start, end = f"<!-- benchmark:activity-own-{stem}:start -->", f"<!-- benchmark:activity-own-{stem}:end -->"
    lines = [start,
             f"_Measured {datetime.now():%Y-%m-%d} with "
             f"`python scripts/eval_activity_own.py data/recordings/{stem}.mp4`._ "
             + ("One hand-labelled webcam recording (not committed), **held out**: labelled before the first run, "
              "evaluated once, never used for tuning."
              if held_out else
              "One hand-labelled webcam recording (not committed), side/oblique view; it was used for tuning "
              "together with CAUCAFall subjects 1-5, so these are training-set numbers, not an accuracy estimate."),
             "", "| Action | Time (s) | Expected | Result | Labels shown (frames) |", "|---|---|---|---|---|"]
    for r in rows:
        shown = ", ".join(f"{k} {v}" for k, v in r["shown"])
        result = "pass" if r["ok"] else ("missed" if not r["hit"] else f"showed {', '.join(r['bad'])}")
        if r["expected"] == "(reported only)":
            result = "not scored"
        lines.append(f"| {r['action']} | {r['start']:.1f}-{r['end']:.1f} | {r['expected']} | {result} | {shown} |")
    lines.append(end)
    md = "\n".join(lines)
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    if start in text:
        before, rest = text.split(start, 1)
        text = before + md + rest.split(end, 1)[1]
    else:
        title = "own recording, held out" if held_out else "own lift-and-carry recording"
        text = text.rstrip() + f"\n\n## Activity labels ({title})\n\n" + md + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    sys.exit(main())
