from __future__ import annotations

import cv2
import numpy as np

from face_processing.models import FrameData


def rotate_landmarks(
    lmks_2d: np.ndarray,
    angle_deg: float,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Rotate 2D landmark points (N, 2) by angle_deg around frame center."""
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    ones = np.ones((lmks_2d.shape[0], 1), dtype=np.float64)
    pts = np.hstack([lmks_2d, ones])  # (N, 3)
    return (M @ pts.T).T  # (N, 2)


def compute_raw_crop_geometry(
    fd: FrameData,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float] | None:
    """Return (cx_rot, cy_rot, w_rot, h_rot) from roll-corrected landmarks.

    Returns None if fd.landmarks is None.
    """
    if fd.landmarks is None:
        return None

    lmks_px = fd.landmarks.copy()
    lmks_px[:, 0] *= frame_w
    lmks_px[:, 1] *= frame_h

    roll = fd.roll if fd.pose_valid else 0.0
    rotated = rotate_landmarks(lmks_px[:, :2], -roll, frame_w, frame_h)

    xs = rotated[:, 0]
    ys = rotated[:, 1]
    return (
        float(np.mean(xs)),
        float(np.mean(ys)),
        float(np.max(xs) - np.min(xs)),
        float(np.max(ys) - np.min(ys)),
    )
