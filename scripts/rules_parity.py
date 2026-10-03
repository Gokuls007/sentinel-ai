"""Parity check: do the built-in rules raise the same fall and zone-intrusion alerts as the
hard-coded detectors they would replace (RULES_BUILTINS=true)?

    python scripts/rules_parity.py                       # URFD + CAUCAFall falls, corridor sample with zones
    python scripts/rules_parity.py --urfd 10 --write docs/BENCHMARKS.md

Each clip runs through detection, tracking and pose **once**; the same poses then feed two
analytics paths, so any difference comes from the rules, not from detection noise:
- **original**: AnomalyEngine's fall and zone_intrusion alerts;
- **rules**: a second AnomalyEngine (for its fall and zone state, its own alerts of those two
  types dropped) plus the RuleEngine with the built-in rules.
Alerts are matched by type and track within ``--tolerance`` seconds. Fall recovery for people
the detector loses is skipped on both paths (it needs the frame-level recovery step), so
both see the same inputs.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.join(ROOT, "training"))

SECTION_START = "<!-- benchmark:rules-parity:start -->"
SECTION_END = "<!-- benchmark:rules-parity:end -->"
TYPES = ("fall", "zone_intrusion")


def match(a: list[tuple], b: list[tuple], tol: float) -> tuple[int, list, list]:
    """Greedy one-to-one matching of (type, track, ts) by type and track within ``tol`` s.
    Returns (matched, only_in_a, only_in_b)."""
    left = list(b)
    matched, only_a = 0, []
    for t, tid, ts in sorted(a, key=lambda x: x[2]):
        hit = next((x for x in left if x[0] == t and x[1] == tid and abs(x[2] - ts) <= tol), None)
        if hit:
            left.remove(hit)
            matched += 1
        else:
            only_a.append((t, tid, ts))
    return matched, only_a, left


class Paths:
    def __init__(self, cfg):
        from anomaly.engine import AnomalyEngine
        from core.pipeline import SentinelPipeline
        from rules.engine import RuleEngine

        self.original = AnomalyEngine(cfg)
        rcfg = copy.deepcopy(cfg)
        rcfg.rules.builtins = True
        self.p = SentinelPipeline.__new__(SentinelPipeline)
        self.p.config = rcfg
        self.p.anomaly_engine = AnomalyEngine(rcfg)
        self.p.rules = RuleEngine()
        self.p.rule_store = None
        from types import SimpleNamespace

        self.p.detector = SimpleNamespace(classes=[0])
        self.p.reload_rules()

    def step(self, poses, feats, dets, ts) -> tuple[list, list]:
        a = [(x.alert_type, x.track_id, ts) for x in self.original.process(poses, feats, ts) if x.alert_type in TYPES]
        self.p.anomaly_engine.process(poses, feats, ts)
        b = [(x.alert_type, x.track_id, ts) for x in self.p._evaluate_rules(poses, feats, dets, ts)
             if x.alert_type in TYPES]
        return a, b


def run_clip(models, frames, cfg, fps: float) -> tuple[list, list, int]:
    models.detector.reset_tracker()
    models.pose.track_features.clear()
    paths = Paths(cfg)
    alerts_a, alerts_b, n = [], [], 0
    for idx, frame in frames:
        ts = idx / fps
        dets = models.detector.detect_and_track(frame)
        people = [d for d in dets.detections if d.class_name == "person" and d.track_id is not None]
        poses = models.pose.estimate(frame, [d.track_id for d in people], [d.bbox for d in people], ts)
        a, b = paths.step(poses, models.pose.get_all_features(), dets, ts)
        alerts_a += a
        alerts_b += b
        n += 1
    return alerts_a, alerts_b, n


def video_frames(path: str, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        yield idx, frame
        idx += 1
    cap.release()


def markdown(rows: list[dict], tol: float, device: str) -> str:
    tot = {k: sum(r[k] for r in rows) for k in ("original", "rules", "matched", "frames")}
    by_type = {}
    for t in TYPES:
        o = sum(sum(1 for x in r["a"] if x[0] == t) for r in rows)
        ru = sum(sum(1 for x in r["b"] if x[0] == t) for r in rows)
        m = sum(r["matched_by_type"].get(t, 0) for r in rows)
        by_type[t] = (o, ru, m)
    diffs = [r for r in rows if r["only_a"] or r["only_b"]]
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/rules_parity.py` on {device}._ "
        f"{len(rows)} clips, {tot['frames']:,} frames. The same poses feed both paths; alerts match by type and "
        f"track within {tol:g} s.",
        "",
        "| Alert | Original | Built-in rules | Matched |",
        "|---|---|---|---|",
    ]
    lines += [f"| {t} | {o} | {ru} | {m} |" for t, (o, ru, m) in by_type.items()]
    lines += ["", f"**Parity: {tot['matched']} of {max(tot['original'], tot['rules'])} alerts match.** "
              + ("No differences." if not diffs else "Differences: " + "; ".join(
                  f"`{r['clip']}` original-only {[(x[0], round(x[2], 1)) for x in r['only_a']]}, "
                  f"rules-only {[(x[0], round(x[2], 1)) for x in r['only_b']]}" for r in diffs)),
              "", "RULES_BUILTINS stays off until this shows no differences that matter.", SECTION_END]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Built-in rules vs original alerts (parity)\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urfd", type=int, default=30, help="URFD fall sequences to use (ADLs: a third as many)")
    ap.add_argument("--caucafall", type=int, default=50,
                    help="CAUCAFall fall videos to use (longer than URFD's, so falls can be confirmed)")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", metavar="MD")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from eval_fall import DATA_DIR, FPS, Models, frames_from_zip

    models = Models(args.device)
    clips = []
    for kind, n in (("fall", args.urfd), ("adl", max(1, args.urfd // 3))):
        for i in range(1, n + 1):
            z = os.path.join(DATA_DIR, f"{kind}-{i:02d}-cam0-rgb.zip")
            if os.path.isfile(z):
                clips.append((f"{kind}-{i:02d}", lambda z=z: frames_from_zip(z), FPS, None))
    if args.caucafall:
        import cv2

        from sweep_fall_confirm import caucafall_videos

        falls = [v for v in caucafall_videos() if v[2].split("/")[-1].lower().startswith("fall")][:args.caucafall]
        for path, _labels, rel in falls:
            cap = cv2.VideoCapture(path)
            vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            clips.append((f"cauca {rel}", lambda path=path: video_frames(path, models.cfg.frame_width,
                                                                          models.cfg.frame_height), vfps, None))
    corridor = os.path.join(ROOT, "demo_videos", "corridor_sample.mp4")
    if os.path.isfile(corridor):
        clips.append(("corridor (zones)", None, None, os.path.join(ROOT, "config", "demo", "corridor_demo.json")))

    rows = []
    for name, frames_fn, fps, zones_file in clips:
        cfg = copy.deepcopy(models.cfg)
        cfg.output.db_path = os.path.join(ROOT, "outputs", "parity", "events.db")
        cfg.anomaly.lstm_model_path = os.path.join(ROOT, "outputs", "parity", "none.pt")
        if zones_file:
            import cv2

            cfg.zone.zones_file = zones_file
            cap = cv2.VideoCapture(corridor)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            frames = video_frames(corridor, cfg.frame_width, cfg.frame_height)
        else:
            cfg.zone.zones_file = os.path.join(ROOT, "outputs", "parity", "no_zones.json")
            frames = frames_fn()
        a, b, n = run_clip(models, frames, cfg, fps)
        matched, only_a, only_b = match(a, b, args.tolerance)
        by_type = {t: match([x for x in a if x[0] == t], [x for x in b if x[0] == t], args.tolerance)[0] for t in TYPES}
        rows.append({"clip": name, "frames": n, "original": len(a), "rules": len(b), "matched": matched,
                     "only_a": only_a, "only_b": only_b, "a": a, "b": b, "matched_by_type": by_type})
        print(f"{name}: {n} frames, original {len(a)}, rules {len(b)}, matched {matched}"
              + (f"  DIFF original-only {only_a} rules-only {only_b}" if only_a or only_b else ""), flush=True)
    md = markdown(rows, args.tolerance, models.detector.device)
    print(md)
    if args.write:
        write_section(args.write, md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
