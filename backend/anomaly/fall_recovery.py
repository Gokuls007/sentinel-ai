"""Find a fallen person the detector lost.

YOLOv8n misses many people lying on the floor: low confidence, and a lying body looks
nothing like the upright people it mostly saw in training. For a track that is falling or
on the ground and has no pose this frame, two cheap retries run on the region around its
last box only:

1. **Region-local low threshold:** the pose model on that crop, accepting detections down
   to ``low_conf`` (the normal threshold applies everywhere else).
2. **Rotated fallback:** the same crop rotated 90° (both ways). A lying person then looks
   upright to the model. Keypoints and box are mapped back to frame coordinates.

This only runs for tracks that are already down, so normal frames cost nothing extra.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.pose_estimator import PoseResult

# The crop around the last box: lying people spread sideways, so widen more than heighten.
PAD_X, PAD_Y = 0.75, 0.5


def crop_region(bbox, frame_shape) -> tuple[int, int, int, int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = max(bw, bh)  # a lying person's box may be short; use the longer side for both pads
    cx1 = int(max(0, x1 - PAD_X * side))
    cy1 = int(max(0, y1 - PAD_Y * side))
    cx2 = int(min(w, x2 + PAD_X * side))
    cy2 = int(min(h, y2 + PAD_Y * side))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None
    return cx1, cy1, cx2, cy2


def unrotate_points(pts: np.ndarray, rotation: int, crop_w: int, crop_h: int) -> np.ndarray:
    """Map (x, y) points from a rotated crop back to the unrotated crop.

    ``cv2.ROTATE_90_CLOCKWISE`` maps crop (x, y) to rotated (crop_h - 1 - y, x); its inverse
    is x = y_r, y = crop_h - 1 - x_r. Counter-clockwise maps (x, y) to (y, crop_w - 1 - x);
    its inverse is x = crop_w - 1 - y_r, y = x_r.
    """
    xr, yr = pts[:, 0], pts[:, 1]
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return np.stack([yr, crop_h - 1 - xr], axis=1)
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return np.stack([crop_w - 1 - yr, xr], axis=1)
    return pts.copy()


def _best_pose(result, last_bbox_crop) -> tuple[np.ndarray, np.ndarray] | None:
    """The detection closest to the last box (by centre), as (keypoints, xyxy)."""
    if not result or result[0].keypoints is None or result[0].boxes is None or len(result[0].boxes) == 0:
        return None
    kps = result[0].keypoints.data.cpu().numpy()
    boxes = result[0].boxes.xyxy.cpu().numpy()
    lx = (last_bbox_crop[0] + last_bbox_crop[2]) / 2
    ly = (last_bbox_crop[1] + last_bbox_crop[3]) / 2
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    i = int(np.argmin(np.hypot(centres[:, 0] - lx, centres[:, 1] - ly)))
    return kps[i].astype(np.float32), boxes[i].astype(np.float32)


def recover_pose(pose_model, frame: np.ndarray, track_id: int, last_bbox, low_conf: float,
                 try_low: bool = True, try_rotated: bool = True) -> tuple[PoseResult | None, str | None]:
    """Look for the lost person near ``last_bbox``. Returns (pose, how) with how = "low" or
    "rotated", or (None, None)."""
    region = crop_region(last_bbox, frame.shape)
    if region is None:
        return None, None
    x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    last_crop = np.array([last_bbox[0] - x1, last_bbox[1] - y1, last_bbox[2] - x1, last_bbox[3] - y1])

    if try_low:
        found = _best_pose(pose_model(crop, conf=low_conf, verbose=False), last_crop)
        if found is not None:
            kps, box = found
            kps[:, 0] += x1
            kps[:, 1] += y1
            kps[kps[:, 2] <= 0, :2] = 0  # undetected keypoints stay (0, 0, 0)
            return PoseResult(track_id, kps, box + np.array([x1, y1, x1, y1], np.float32)), "low"

    if try_rotated:
        for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rotated = cv2.rotate(crop, rotation)
            # The last box, rotated the same way, to pick the nearest detection.
            corners = np.array([[last_crop[0], last_crop[1]], [last_crop[2], last_crop[3]]], np.float32)
            if rotation == cv2.ROTATE_90_CLOCKWISE:
                rc = np.stack([ch - 1 - corners[:, 1], corners[:, 0]], axis=1)
            else:
                rc = np.stack([corners[:, 1], cw - 1 - corners[:, 0]], axis=1)
            last_rot = np.array([rc[:, 0].min(), rc[:, 1].min(), rc[:, 0].max(), rc[:, 1].max()])
            found = _best_pose(pose_model(rotated, conf=low_conf, verbose=False), last_rot)
            if found is None:
                continue
            kps, box = found
            seen = kps[:, 2] > 0
            kps[seen, :2] = unrotate_points(kps[seen, :2], rotation, cw, ch)
            kps[~seen, :2] = 0
            kps[seen, 0] += x1
            kps[seen, 1] += y1
            bc = unrotate_points(np.array([[box[0], box[1]], [box[2], box[3]]], np.float32), rotation, cw, ch)
            box_full = np.array([bc[:, 0].min() + x1, bc[:, 1].min() + y1, bc[:, 0].max() + x1, bc[:, 1].max() + y1],
                                np.float32)
            return PoseResult(track_id, kps, box_full), "rotated"
    return None, None
