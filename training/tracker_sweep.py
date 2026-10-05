"""Person-tracker settings vs identity switches during falls, replayed offline.

    python training/tracker_sweep.py           # caches YOLO11m person boxes once (~10 min), then sweeps

During a fall the person's box turns from tall to wide and its detection score drops for a few
frames; if the tracker then starts a new track, the new one has no standing height and the fall is
never measured (training/regression_2x2.py). This replays ByteTrack (ultralytics) over cached raw
detections (YOLO11m, every person box down to confidence 0.1, as the Detector asks for) with
different settings, so each setting costs seconds, not a model run.

Clips: URFD falls 1-30 and daily activities, CAUCAFall subjects 1-5 (6-10 are the held-out fall
test set and aren't used), and the two multi-person corridor samples. Per group:

- **switch**: falls where a track id appears after the onset that wasn't there before it;
- **gap**: share of frames after the onset with no tracked box;
- **ids/clip**: track ids that last 5+ frames (single-person clips should have 1);
- **jumps**: a track's box centre moving more than 0.6 box heights in one frame (one track taking
  over another person's box; also counts a very fast fall).

Also tried: re-attaching a confident unmatched box to a track lost within 1 s by centre distance or
an expanded box (``RescueTracker``), before it starts a new track.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "outputs", "tracker_sweep_dets.pkl")
SAMPLES = ("demo_videos/corridor_sample.mp4", "demo_videos/walking_sample.mp4")
# The production thresholds at DETECTION_CONFIDENCE 0.25 (core.detector._tracker_config), before this sweep.
BEFORE = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.35, track_buffer=90,
              match_thresh=0.8, fuse_score=True)
GRID = [
    {}, {"fuse_score": False}, {"match_thresh": 0.9}, {"match_thresh": 0.95},
    {"fuse_score": False, "match_thresh": 0.9}, {"fuse_score": False, "match_thresh": 0.95},
    {"track_high_thresh": 0.15}, {"new_track_thresh": 0.5},
    {"new_track_thresh": 0.5, "fuse_score": False, "match_thresh": 0.9},
    {"rescue": {"mode": "center", "radius": 0.5}}, {"rescue": {"mode": "center", "radius": 0.75}},
    {"rescue": {"mode": "center", "radius": 1.0}}, {"rescue": {"mode": "expand", "radius": 1.5}},
    {"rescue": {"mode": "expand", "radius": 2.0}}, {"fuse_score": False, "rescue": {"mode": "center", "radius": 0.75}},
]
GROUPS = {
    "URFD falls": lambda c: c["kind"] == "fall" and c["name"].startswith("fall-"),
    "CAUCAFall 1-5 falls": lambda c: c["kind"] == "fall" and c["name"].startswith("Subject"),
    "URFD daily": lambda c: c["kind"] == "adl",
    "CAUCAFall 1-5 daily": lambda c: c["kind"] == "cauca_adl",
    "corridor (multi-person)": lambda c: c["kind"] == "multi",
}


def build_cache() -> list[dict]:
    import cv2
    from ultralytics import YOLO

    import eval_fall as ef
    import sweep_fall_confirm as sw
    from config.settings import SentinelConfig

    cfg = SentinelConfig()
    size = (cfg.frame_width, cfg.frame_height)
    model = YOLO(os.path.join(ROOT, "yolo11m.pt"))
    onsets = ef.fall_onsets(os.path.join(ef.DATA_DIR, "urfall-cam0-falls.csv"))
    out = []

    def run(name, kind, frames, fps, onset=None, resize=False):
        dets, shape = [], None
        for _i, f in frames:
            if resize and (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size)
            shape = f.shape[:2]
            r = model.predict(f, conf=0.1, iou=cfg.detector.iou_threshold, classes=[0], verbose=False)[0]
            dets.append(r.boxes.data.cpu().numpy().copy())
        out.append({"name": name, "kind": kind, "fps": fps, "onset": onset, "dets": dets, "shape": shape})

    for i in range(1, ef.N_FALLS + 1):  # URFD at its native 640x480, as the fall benchmarks run it
        n = f"fall-{i:02d}"
        run(n, "fall", ef.frames_from_zip(os.path.join(ef.DATA_DIR, f"{n}-cam0-rgb.zip")), ef.FPS, onsets[n])
    for i in range(1, ef.N_ADLS + 1):
        path = os.path.join(ef.DATA_DIR, f"adl-{i:02d}-cam0-rgb.zip")
        if os.path.isfile(path):
            run(f"adl-{i:02d}", "adl", ef.frames_from_zip(path), ef.FPS)
    for video, labels, rel in sw.caucafall_videos():
        if int(rel.split("/")[0].split(".")[1]) > 5:
            continue
        frames, fps = sw.video_frames(video)
        fall = rel.split("/")[1].lower().startswith("fall")
        run(rel, "fall" if fall else "cauca_adl", frames, fps, sw.caucafall_onset(labels) if fall else None,
            resize=True)
    for rel in SAMPLES:
        frames, fps = sw.video_frames(os.path.join(ROOT, rel))
        run(rel, "multi", frames, fps, resize=True)
    return out


def rescue_tracker_class():
    from ultralytics.trackers.byte_tracker import BYTETracker

    class RescueTracker(BYTETracker):
        """Before a confident unmatched box starts a new track, re-attach it to a track lost within
        ``max_gap_s`` whose last seen box is near (one-to-one, nearest first)."""

        def __init__(self, args, frame_rate=30, mode="center", radius=0.75, max_gap_s=1.0):
            super().__init__(args)
            self.mode, self.radius, self.max_gap = mode, radius, max(1, round(max_gap_s * frame_rate))
            self.last_obs: dict[int, tuple] = {}

        def update(self, results, img=None, feats=None, **kw):
            out = super().update(results, img, feats, **kw)
            for row in out:
                self.last_obs[int(row[4])] = (row[:4].copy(), self.frame_id)
            return out

        def _near(self, last, det) -> float | None:
            lc = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
            dc = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
            size = max(last[2] - last[0], last[3] - last[1])
            d = float(np.linalg.norm(dc - lc))
            if self.mode == "center":
                return d / size if d <= self.radius * size else None
            inside = (abs(dc[0] - lc[0]) <= (last[2] - last[0]) / 2 * self.radius
                      and abs(dc[1] - lc[1]) <= (last[3] - last[1]) / 2 * self.radius)
            return d / size if inside else None

        def _init_new_tracks(self, u_detection, detections, activated, refind=None):
            lost = [t for t in self.lost_stracks if t.track_id in self.last_obs
                    and self.frame_id - self.last_obs[t.track_id][1] <= self.max_gap]
            pairs = [(d, i, t) for i in u_detection if detections[i].score >= self.args.new_track_thresh
                     for t in lost for d in [self._near(self.last_obs[t.track_id][0], detections[i].xyxy)]
                     if d is not None]
            used_d, used_t = set(), set()
            for _d, i, t in sorted(pairs, key=lambda p: p[0]):
                if i in used_d or t.track_id in used_t:
                    continue
                used_d.add(i)
                used_t.add(t.track_id)
                self._apply_match(t, detections[i], activated, refind if refind is not None else activated)
            super()._init_new_tracks([i for i in u_detection if i not in used_d], detections, activated, refind)

    return RescueTracker


def track(clip: dict, cfg: dict) -> list:
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = dict(cfg)
    rescue = cfg.pop("rescue", None)
    tracker = (rescue_tracker_class()(SimpleNamespace(**cfg), round(clip["fps"]), **rescue) if rescue
               else BYTETracker(SimpleNamespace(**cfg)))
    out = []
    for d in clip["dets"]:
        rows = tracker.update(Boxes(d if len(d) else np.zeros((0, 6), np.float32), clip["shape"]))
        out.append([(int(x[4]), x[:4]) for x in rows])
    return out


def metrics(clip: dict, frames: list) -> dict:
    counts: dict[int, int] = {}
    for fr in frames:
        for tid, _b in fr:
            counts[tid] = counts.get(tid, 0) + 1
    res = {"ids": sum(1 for n in counts.values() if n >= 5)}
    if clip["onset"]:
        on = clip["onset"] - 1
        before = {tid for fr in frames[:on] for tid, _b in fr}
        res["switch"] = bool(before) and any(tid not in before for fr in frames[on:] for tid, _b in fr)
        res["gap"] = sum(1 for fr in frames[on:] if not fr) / max(len(frames) - on, 1)
    last, jumps = {}, 0
    for fr in frames:
        for tid, b in fr:
            c, h = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), max(b[3] - b[1], 1.0)
            if tid in last and np.linalg.norm(c - last[tid][0]) > 0.6 * max(h, last[tid][1]):
                jumps += 1
            last[tid] = (c, h)
    res["jumps"] = jumps
    return res


def main() -> int:
    sys.path[:0] = [os.path.join(ROOT, "backend"), os.path.join(ROOT, "training")]
    os.chdir(ROOT)
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as f:
            clips = pickle.load(f)
    else:
        t0 = time.time()
        clips = build_cache()
        with open(CACHE, "wb") as f:
            pickle.dump(clips, f)
        print(f"cached {len(clips)} clips in {time.time() - t0:.0f}s")
    for change in GRID:
        cfg = {**BEFORE, **change}
        print(change or "before (production until 2026-10-05)")
        for name, member in GROUPS.items():
            ms = [metrics(c, track(c, cfg)) for c in clips if member(c)]
            if not ms:
                continue
            line = f"   {name}: ids/clip {np.mean([m['ids'] for m in ms]):.2f}, jumps {sum(m['jumps'] for m in ms)}"
            if "switch" in ms[0]:
                line += (f", switch {sum(m['switch'] for m in ms)}/{len(ms)}, "
                         f"gap {np.mean([m['gap'] for m in ms]):.2f}")
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
