"""What the camera learns about the room from the people it watches.

Per camera, on a coarse grid over the picture:

- **Floor**: where people stand with both feet seen inside the picture (torso upright, a leg
  straight). Their ankles mark the floor.
- **Furniture that hides legs**: where an upright person's box ends above the bottom of the
  picture with no feet seen: something (a couch back, a counter) stands in front of the legs.
  On the 2026-10-12 kitchen recording, two minutes of people at the sink traced the top edge of
  the couch in front of them; that couch made bending over the sink read as a fall.
- **Standing height by position**: a standing person's height in pixels against where the feet
  are (further away = higher in the picture = smaller), fitted as a line once there is enough
  spread (own bedroom recording: median error 2.3%). Used only when the fit is good.

The fall detector uses "legs hidden by furniture" to stop judging hips, feet and box shape for
that person (they are guesses there) and asks for a clearly lying torso with the head down.

The model is saved per camera and reloaded at start, unless the camera now sees a different
view (a laptop gets moved): then it starts again rather than trust a map of another room.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque

import cv2
import numpy as np

from anomaly.geometry import box_too_close

logger = logging.getLogger("sentinel.scene")

C = 0.5  # keypoint confidence for "seen"
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16


def torso_deg(kp) -> float | None:
    if min(kp[L_SH, 2], kp[R_SH, 2], kp[L_HIP, 2], kp[R_HIP, 2]) < C:
        return None
    v = kp[[L_SH, R_SH], :2].mean(0) - kp[[L_HIP, R_HIP], :2].mean(0)
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


def straight_leg(kp) -> bool:
    """Hips above the ankle by at least 0.85 of the leg's own length (thigh + shin)."""
    for h, k, a in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if min(kp[h, 2], kp[k, 2], kp[a, 2]) < C:
            continue
        length = float(np.hypot(*(kp[k, :2] - kp[h, :2])) + np.hypot(*(kp[a, :2] - kp[k, :2])))
        if length > 1.0 and (kp[a, 1] - kp[h, 1]) / length >= 0.85:
            return True
    return False


class RoomModel:
    CELL = 40                # px of a 1280x720 picture
    UPRIGHT_DEG = 25.0
    OCCLUDER_MIN = 12        # legs-hidden samples in a cell before it counts as furniture
    FIT_MIN_SAMPLES = 60
    FIT_MIN_SPREAD = 0.08    # foot positions must span this much of the picture height
    FIT_MAX_ERROR = 0.08     # median relative error of the height line
    SAVE_EVERY_S = 30.0
    VIEW_MAX_DIFF = 28.0     # mean grey-level difference (0-255) of a 64x36 thumbnail: another view
    MAX_COUNT = 1000.0       # per cell, so old evidence can be outweighed

    def __init__(self, width: int = 1280, height: int = 720, path: str | None = None):
        self.width, self.height, self.path = width, height, path
        self.cols, self.rows = -(-width // self.CELL), -(-height // self.CELL)
        self.reset()
        self._last_save = None
        self._loaded_view: np.ndarray | None = None
        self._view_checked = path is None
        if path:
            self._load()

    def reset(self) -> None:
        self.floor = np.zeros((self.rows, self.cols), np.float32)
        self.occluder = np.zeros((self.rows, self.cols), np.float32)
        self.heights: deque = deque(maxlen=3000)  # (foot_y, standing height px)
        self.fit: tuple[float, float, float] | None = None  # (a, b, median error): h = a*y + b
        self.view: np.ndarray | None = None
        self.samples = 0

    # -- learning -----------------------------------------------------------------------------

    def update(self, poses: dict, timestamp: float, frame: np.ndarray | None = None) -> None:
        if frame is not None:
            self._check_view(frame)
        W, H = self.width, self.height
        for pose in poses.values():
            kp = np.asarray(pose.keypoints, float)
            x1, y1, x2, y2 = (float(v) for v in pose.bbox[:4])
            t = torso_deg(kp)
            if t is None or t > self.UPRIGHT_DEG or y1 <= 0.01 * H or box_too_close(pose.bbox, W, H):
                continue
            feet = kp[L_ANK, 2] >= C and kp[R_ANK, 2] >= C
            foot_y = max(kp[L_ANK, 1], kp[R_ANK, 1])
            if feet and foot_y < 0.97 * H and straight_leg(kp):
                self._add(self.floor, float(kp[[L_ANK, R_ANK], 0].mean()), foot_y)
                self.heights.append((foot_y, foot_y - y1))
                self.samples += 1
            elif kp[L_ANK, 2] < 0.3 and kp[R_ANK, 2] < 0.3 and y2 < 0.97 * H:
                # The box bottom is where the furniture's edge cuts the body: mark the middle of it.
                for x in np.linspace(x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1), 5):
                    self._add(self.occluder, float(x), y2, 0.2)
                self.samples += 1
        if self.samples and self.samples % 50 == 0:
            self._fit()
        if self.path and (self._last_save is None or timestamp - self._last_save >= self.SAVE_EVERY_S):
            self._last_save = timestamp
            self.save()

    def _add(self, grid, x: float, y: float, w: float = 1.0) -> None:
        c, r = int(x // self.CELL), int(y // self.CELL)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            grid[r, c] = min(self.MAX_COUNT, grid[r, c] + w)

    def _fit(self) -> None:
        self.fit = None
        if len(self.heights) < self.FIT_MIN_SAMPLES:
            return
        a_ = np.array(self.heights, float)
        ys, hs = a_[:, 0], a_[:, 1]
        if ys.max() - ys.min() < self.FIT_MIN_SPREAD * self.height:
            return  # everyone stood at the same distance: a line through one spot means nothing
        keep = np.ones(len(ys), bool)
        for _ in range(3):  # trim the worst 20% (a crouch, a hand raised over the head)
            a, b = np.polyfit(ys[keep], hs[keep], 1)
            err = np.abs(hs - (a * ys + b)) / np.maximum(hs, 1.0)
            keep = err <= np.percentile(err, 80)
        med = float(np.median(np.abs(hs - (a * ys + b)) / np.maximum(hs, 1.0)))
        if a > 0 and med <= self.FIT_MAX_ERROR:
            self.fit = (float(a), float(b), med)

    # -- questions ----------------------------------------------------------------------------

    def legs_hidden(self, box) -> bool:
        """The box ends on learned furniture: the legs below it are out of sight, not missing."""
        x1, _y1, x2, y2 = (float(v) for v in box[:4])
        r = int(y2 // self.CELL)
        c1, c2 = int((x1 + 0.2 * (x2 - x1)) // self.CELL), int((x2 - 0.2 * (x2 - x1)) // self.CELL)
        rows = [rr for rr in (r - 1, r, r + 1) if 0 <= rr < self.rows]
        cols = list(range(max(0, c1), min(self.cols, c2 + 1)))
        if not rows or not cols:
            return False
        occ = self.occluder[np.ix_(rows, cols)]
        flo = self.floor[np.ix_(rows, cols)]
        return bool(occ.max() >= self.OCCLUDER_MIN and occ.sum() > flo.sum())

    def standing_height(self, foot_y: float) -> float | None:
        """Expected standing height (px) for feet at image row ``foot_y``, if the fit is good."""
        if self.fit is None:
            return None
        a, b, _ = self.fit
        h = a * foot_y + b
        return h if h > 0 else None

    def status(self) -> dict:
        ys = [y for y, _h in self.heights]
        spread = (max(ys) - min(ys)) / self.height if ys else 0.0
        return {
            # Progress of the standing-height fit: enough full-body standing samples, near and far.
            "height_samples": len(self.heights),
            "height_samples_needed": self.FIT_MIN_SAMPLES,
            "depth_spread": round(spread, 3),
            "depth_spread_needed": self.FIT_MIN_SPREAD,
            "samples": int(self.samples),
            "floor_cells": int((self.floor >= 3).sum()),
            "furniture_cells": int((self.occluder >= self.OCCLUDER_MIN).sum()),
            "height_ready": self.fit is not None,
            "height_error": round(self.fit[2], 3) if self.fit else None,
        }

    def draw(self, frame: np.ndarray) -> None:
        """Floor cells green, leg-hiding furniture red, a standing-height ruler if fitted."""
        overlay = frame.copy()
        sx, sy = frame.shape[1] / self.width, frame.shape[0] / self.height
        for grid, colour, low in ((self.floor, (60, 200, 60), 3), (self.occluder, (40, 40, 230), self.OCCLUDER_MIN)):
            for r, c in zip(*np.nonzero(grid >= low), strict=True):
                p1 = (int(c * self.CELL * sx), int(r * self.CELL * sy))
                p2 = (int((c + 1) * self.CELL * sx), int((r + 1) * self.CELL * sy))
                cv2.rectangle(overlay, p1, p2, colour, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for grid, colour, low in ((self.occluder, (40, 40, 255), self.OCCLUDER_MIN),):
            for r, c in zip(*np.nonzero(grid >= low), strict=True):  # outlined: the part that changes judgement
                p1 = (int(c * self.CELL * sx), int(r * self.CELL * sy))
                p2 = (int((c + 1) * self.CELL * sx), int((r + 1) * self.CELL * sy))
                cv2.rectangle(frame, p1, p2, colour, 2)
        if self.fit is not None:
            ys = np.array(self.heights, float)[:, 0]
            for fy in np.linspace(ys.min(), ys.max(), 3):
                h = self.standing_height(fy)
                x = int(0.06 * frame.shape[1])
                cv2.line(frame, (x, int(fy * sy)), (x, int((fy - h) * sy)), (255, 200, 0), 2)
        s = self.status()
        cv2.putText(frame, f"Room map: floor {s['floor_cells']}  furniture {s['furniture_cells']}  "
                    f"height {'ready' if s['height_ready'] else 'learning'}",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # -- persistence --------------------------------------------------------------------------

    @staticmethod
    def _thumb(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _check_view(self, frame: np.ndarray) -> None:
        thumb = self._thumb(frame)
        if not self._view_checked:
            self._view_checked = True
            if self._loaded_view is not None:
                diff = float(np.abs(thumb - self._loaded_view).mean())
                if diff > self.VIEW_MAX_DIFF:
                    logger.info("room map: the camera sees a different view (diff %.0f), starting again", diff)
                    self.reset()
        # A slow running average: people pass through, the room stays.
        self.view = thumb if self.view is None else 0.98 * self.view + 0.02 * thumb

    def save(self) -> None:
        if not self.path:
            return
        data = {"cell": self.CELL, "size": [self.width, self.height], "samples": int(self.samples),
                "floor": self.floor.round(2).tolist(), "occluder": self.occluder.round(2).tolist(),
                "heights": [[round(y, 1), round(h, 1)] for y, h in list(self.heights)[-1500:]],
                "view": self.view.round(1).tolist() if self.view is not None else None}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning("room map not saved: %s", e)

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("cell") != self.CELL or data.get("size") != [self.width, self.height]:
                return
            floor, occ = np.array(data["floor"], np.float32), np.array(data["occluder"], np.float32)
            if floor.shape != self.floor.shape or occ.shape != self.occluder.shape:
                return
            self.floor, self.occluder = floor, occ
            self.heights.extend(tuple(x) for x in data.get("heights", []))
            self.samples = int(data.get("samples", 0))
            if data.get("view") is not None:
                self._loaded_view = np.array(data["view"], np.float32)
                self.view = self._loaded_view.copy()
            self._fit()
        except (OSError, ValueError, KeyError, TypeError) as e:
            if not isinstance(e, FileNotFoundError):
                logger.warning("room map not loaded (%s): starting empty", e)
