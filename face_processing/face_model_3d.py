from __future__ import annotations

import math

import cv2
import numpy as np

# Canonical 3D face model points (model-space coordinates).
# 6 key landmarks: nose tip, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner.
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip       — MediaPipe lmk 1
        (0.0, -330.0, -65.0),     # chin            — MediaPipe lmk 152
        (-225.0, 170.0, -135.0),  # left eye outer  — MediaPipe lmk 33
        (225.0, 170.0, -135.0),   # right eye outer — MediaPipe lmk 263
        (-150.0, -150.0, -125.0), # left mouth      — MediaPipe lmk 61
        (150.0, -150.0, -125.0),  # right mouth     — MediaPipe lmk 291
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe 478-mesh landmark indices
LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


def _build_camera_matrix(w: int, h: int) -> np.ndarray:
    focal_length = float(w)
    cx = w / 2.0
    cy = h / 2.0
    return np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
        dtype=np.float64,
    )


def estimate_head_pose(
    landmarks_px: np.ndarray,
    image_size: tuple[int, int],
    landmark_indices: tuple[int, ...] = LANDMARK_INDICES,
) -> tuple[float, float, float, bool, float]:
    """Estimate yaw, pitch, roll from 2D landmarks via solvePnP.

    Args:
        landmarks_px: (N, 2|3) array of pixel-space landmarks (at least 478 rows).
        image_size: (width, height) of the frame.
        landmark_indices: which landmark rows to pick for the 6-point model.

    Returns:
        (yaw, pitch, roll, pose_valid, reprojection_error)
    """
    w, h = image_size

    # Select 2D points corresponding to the 3D model
    image_points = np.array(
        [landmarks_px[i, :2] for i in landmark_indices],
        dtype=np.float64,
    )

    camera_matrix = _build_camera_matrix(w, h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0, False, float("inf")

    # Euler angles from rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler(rotation_matrix)

    # Reprojection error
    projected, _ = cv2.projectPoints(
        MODEL_POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(image_points - projected, axis=1)
    reprojection_error = float(np.sqrt(np.mean(errors**2)))

    return yaw, pitch, roll, True, reprojection_error


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll (degrees) from a 3x3 rotation matrix.

    Convention: Rz(yaw) * Ry(pitch) * Rx(roll) — extrinsic ZYX.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )
