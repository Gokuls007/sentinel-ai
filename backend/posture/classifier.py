"""Personal posture classifier: learns *your* postures from a short guided calibration.

Fixed thresholds on noisy front-view keypoints flicker: a real slouch looks a lot like looking
down at the keyboard, and keypoint jitter pushes values back and forth across any fixed
limit. So instead the user records each posture for ~20 s while moving naturally, and a small
model is trained on those frames.

Features (all from the pose keypoints; no extra model):
- the coach's scale-invariant measures: head height, face/shoulder width, shoulder tilt,
  head offset (ears, else eyes);
- head pitch: the nose's height relative to the ear line (else the eye line), in eye spans
  (it drops when the head tilts forward / down);
- head yaw: the nose's sideways offset from the eye midpoint, in eye spans;
- head roll: the eye line's angle;
- distance: eye span and shoulder width relative to the Good recording (too close).

The model (logistic regression vs gradient boosting, whichever scores better) is judged on the
**last 25% of each posture's recording**, held out by time so neighbouring frames can't leak
into the score. A separate "test my calibration" run scores it on brand-new data.
"""

from __future__ import annotations

import math
import pickle
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

POSTURES = ("good", "slouching", "leaning_left", "leaning_right", "too_close", "looking_down")
POSTURE_LABELS = {
    "good": "Good", "slouching": "Slouching", "leaning_left": "Leaning left", "leaning_right": "Leaning right",
    "too_close": "Too close", "looking_down": "Looking down",
}
RAW_FEATURES = ("head_ratio", "face_ratio", "tilt_deg", "lateral", "pitch", "yaw", "roll_deg", "eye_span",
                "shoulder_width")
# Distances are divided by the Good recording's median; everything else has it subtracted.
RELATIVE_BY_RATIO = ("eye_span", "shoulder_width")
HOLDOUT_FRACTION = 0.25

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6


def raw_features(keypoints: np.ndarray, min_conf: float = 0.4) -> dict | None:
    """Per-frame features, or None unless the nose, both eyes and both shoulders are visible."""
    kp = np.asarray(keypoints, float)
    if kp.shape[0] < 7 or min(kp[i, 2] for i in (NOSE, L_EYE, R_EYE, L_SH, R_SH)) < min_conf:
        return None
    ls, rs, nose = kp[L_SH, :2], kp[R_SH, :2], kp[NOSE, :2]
    le, re = kp[L_EYE, :2], kp[R_EYE, :2]
    width = float(np.hypot(*(ls - rs)))
    eye_span = float(np.hypot(*(le - re)))
    if width < 10 or eye_span < 3:
        return None
    mid = (ls + rs) / 2
    eye_mid = (le + re) / 2
    a, b = (ls, rs) if ls[0] <= rs[0] else (rs, ls)
    ea, eb = (le, re) if le[0] <= re[0] else (re, le)
    ears = min(kp[L_EAR, 2], kp[R_EAR, 2]) >= min_conf
    ear_mid = (kp[L_EAR, :2] + kp[R_EAR, :2]) / 2 if ears else None
    head_ref = ear_mid if ears else eye_mid
    return {
        "head_ratio": float(mid[1] - nose[1]) / width,
        "face_ratio": eye_span / width,
        "tilt_deg": math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
        "lateral": float(head_ref[0] - mid[0]) / width,
        "pitch": float(nose[1] - head_ref[1]) / eye_span,
        "yaw": float(nose[0] - eye_mid[0]) / eye_span,
        "roll_deg": math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0])),
        "eye_span": eye_span,
        "shoulder_width": width,
    }


def reference(good_frames: list[dict]) -> dict:
    """Median of each raw feature over the Good recording."""
    return {k: float(np.median([f[k] for f in good_frames])) for k in RAW_FEATURES}


def vector(raw: dict, ref: dict) -> np.ndarray:
    """Raw features made relative to the Good reference (the model's input)."""
    out = []
    for k in RAW_FEATURES:
        if k in RELATIVE_BY_RATIO:
            out.append(raw[k] / ref[k] if ref[k] else 1.0)
        else:
            out.append(raw[k] - ref[k])
    return np.array(out, float)


def split_by_time(recordings: dict[str, list], holdout: float = HOLDOUT_FRACTION):
    """(train, test) as lists of (posture, raw) with the last ``holdout`` of each posture's
    recording (by time) in test."""
    train, test = [], []
    for posture, frames in recordings.items():
        frames = sorted(frames, key=lambda f: f["t"])
        cut = round(len(frames) * (1 - holdout))
        train += [(posture, f["features"]) for f in frames[:cut]]
        test += [(posture, f["features"]) for f in frames[cut:]]
    return train, test


def _matrix(classes, y_true, y_pred) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred, strict=True):
        m[idx[t]][idx[p]] += 1
    return m


def score_report(classes, y_true, y_pred) -> dict:
    """Accuracy, balanced accuracy, per-class recall and the confusion matrix (rows = true)."""
    y_true, y_pred = list(y_true), list(y_pred)
    m = _matrix(classes, y_true, y_pred)
    per_class = {}
    for i, c in enumerate(classes):
        n = sum(m[i])
        per_class[c] = {"n": n, "recall": (m[i][i] / n) if n else None}
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    return {
        "frames": len(y_true),
        "accuracy": correct / len(y_true) if y_true else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "per_class": per_class,
        "confusion": {"classes": list(classes), "matrix": m},
    }


def _candidates(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=150, max_depth=3, learning_rate=0.08, random_state=seed),
    }


@dataclass
class PostureModel:
    model: object
    classes: tuple
    ref: dict
    name: str
    report: dict
    trained_at: float = field(default_factory=time.time)
    test_report: dict | None = None  # from "Test my calibration" (new data)

    def predict_proba(self, raw: dict) -> dict:
        p = self.model.predict_proba(vector(raw, self.ref).reshape(1, -1))[0]
        order = list(self.model.classes_) if hasattr(self.model, "classes_") else list(self.model[-1].classes_)
        probs = {c: 0.0 for c in self.classes}
        for c, v in zip(order, p, strict=True):
            probs[str(c)] = float(v)
        return probs

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> PostureModel | None:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, PostureModel) else None
        except (OSError, pickle.UnpicklingError, AttributeError, EOFError, ImportError):
            return None


def train(recordings: dict[str, list], min_frames: int = 20, seed: int = 0) -> PostureModel:
    """Train on calibration recordings ({posture: [{"t", "features"}]}). Every posture in
    POSTURES must have at least ``min_frames`` frames. Picks the candidate with the best
    balanced accuracy on the time-held-out part, then refits it on all frames."""
    missing = [p for p in POSTURES if len(recordings.get(p, [])) < min_frames]
    if missing:
        raise ValueError(f"record these postures first (at least {min_frames} frames each): {', '.join(missing)}")
    ref = reference([f["features"] for f in recordings["good"]])
    train_rows, test_rows = split_by_time(recordings)
    Xtr = np.array([vector(r, ref) for _p, r in train_rows])
    ytr = np.array([p for p, _r in train_rows])
    Xte = np.array([vector(r, ref) for _p, r in test_rows])
    yte = [p for p, _r in test_rows]
    results = {}
    for name, make in _candidates(seed).items():
        model = make().fit(Xtr, ytr)
        results[name] = score_report(POSTURES, yte, list(model.predict(Xte)))
    best = max(results, key=lambda n: results[n]["balanced_accuracy"] or 0.0)
    all_rows = train_rows + test_rows
    final = _candidates(seed)[best]().fit(np.array([vector(r, ref) for _p, r in all_rows]),
                                          np.array([p for p, _r in all_rows]))
    report = {**results[best], "model": best,
              "compared": {n: {"accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"]}
                           for n, r in results.items()},
              "holdout": f"last {HOLDOUT_FRACTION:.0%} of each posture's recording (by time)",
              "train_frames": len(train_rows)}
    return PostureModel(model=final, classes=POSTURES, ref=ref, name=best, report=report)


def evaluate(model: PostureModel, labelled: list[tuple[str, dict]]) -> dict:
    """Score a saved model on new labelled frames (the "Test my calibration" run)."""
    y_true = [p for p, _r in labelled]
    y_pred = [max(probs, key=probs.get) for probs in (model.predict_proba(r) for _p, r in labelled)]
    return score_report(model.classes, y_true, y_pred)


class StableStatus:
    """Turns noisy per-frame probabilities into a steady status.

    Probabilities are averaged over ``window_s``. The shown posture changes only when another
    posture's averaged probability stays at or above ``min_prob`` for ``hold_s``; otherwise the
    current one stays (so glancing at a phone for a few seconds changes nothing)."""

    def __init__(self, window_s: float = 10.0, hold_s: float = 5.0, min_prob: float = 0.7):
        self.window_s, self.hold_s, self.min_prob = window_s, hold_s, min_prob
        self._probs: deque[tuple[float, dict]] = deque()
        self.current: str | None = None
        self._candidate: str | None = None
        self._since = 0.0
        self.mean: dict = {}

    def reset(self) -> None:
        self._probs.clear()
        self.current = self._candidate = None
        self.mean = {}

    def update(self, probs: dict, ts: float) -> str | None:
        self._probs.append((ts, probs))
        while self._probs and ts - self._probs[0][0] > self.window_s:
            self._probs.popleft()
        keys = probs.keys()
        self.mean = {k: float(np.mean([p[k] for _t, p in self._probs])) for k in keys}
        best = max(self.mean, key=self.mean.get)
        if best == self.current or self.mean[best] < self.min_prob:
            self._candidate = None
            return self.current
        if best != self._candidate:
            self._candidate, self._since = best, ts
        if ts - self._since >= self.hold_s:
            self.current, self._candidate = best, None
        return self.current
