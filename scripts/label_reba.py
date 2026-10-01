"""Label REBA risk levels on sampled frames of a video (for the 2D-REBA agreement check).

    python scripts/label_reba.py my_clip.mp4                 # a frame every 2 s
    python scripts/label_reba.py my_clip.mp4 --every 1.0 --out labels/my_clip.json

Labelling is blind: you see the raw frame, never Sentinel's skeleton or score, so the labels
aren't biased toward what the system says. Label the main person in the frame.

Keys (in the video window):
    1 negligible   2 low   3 medium   4 high   5 very high
    0 or x         can't tell (recorded as "unsure", excluded from agreement)
    n or space     next frame, no label      b  previous frame
    q or Esc       save and quit
Labels are saved after every key press, and re-running resumes where you left off.
Then run:  python scripts/eval_ergo.py my_clip.mp4 --labels <the json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2

LEVELS = {ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4, ord("5"): 5}
NAMES = {1: "negligible", 2: "low", 3: "medium", 4: "high", 5: "very high", None: "unsure"}
WINDOW = "REBA labelling (1-5 level, 0 unsure, n next, b back, q quit)"


def load(path: str, video: str, every: float) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if os.path.basename(data.get("video", "")) != os.path.basename(video):
            sys.exit(f"{path} belongs to {data.get('video')}, not {video}")
        return data
    return {"video": os.path.abspath(video), "every_s": every, "labels": {}}


def save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def overlay(frame, text_lines):
    view = frame.copy()
    y = 28
    for line in text_lines:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28
    return view


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--out", help="labels JSON (default: <video>.reba_labels.json)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(args.every * fps))
    frames = list(range(0, total, step))
    out = args.out or os.path.splitext(args.video)[0] + ".reba_labels.json"
    data = load(out, args.video, args.every)
    data.update(fps=fps, total_frames=total)
    labels: dict = data["labels"]

    # Resume at the first unlabelled frame.
    i = next((k for k, f in enumerate(frames) if str(f) not in labels), 0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while 0 <= i < len(frames):
        idx = frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            i += 1
            continue
        current = labels.get(str(idx), {}).get("level", "-")
        done = sum(1 for v in labels.values() if v.get("level") is not None)
        lines = [
            f"Frame {idx} ({idx / fps:.1f}s)   sample {i + 1}/{len(frames)}   labelled {done}",
            f"Current: {NAMES.get(current, current) if current != '-' else '-'}",
            "1 negl  2 low  3 med  4 high  5 very high  0 unsure  n next  b back  q quit",
        ]
        cv2.imshow(WINDOW, overlay(frame, lines))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            i -= 1
            continue
        if key in LEVELS or key in (ord("0"), ord("x")):
            level = LEVELS.get(key)
            labels[str(idx)] = {"frame": idx, "time_s": round(idx / fps, 2), "level": level,
                                "level_name": NAMES[level]}
            save(out, data)
        i += 1
    save(out, data)
    cap.release()
    cv2.destroyAllWindows()
    n = sum(1 for v in labels.values() if v.get("level") is not None)
    print(f"{n} labelled frame(s) ({len(labels) - n} unsure) saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
