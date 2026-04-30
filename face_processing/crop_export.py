from __future__ import annotations

import logging
import subprocess

import cv2
import numpy as np

from face_processing.config import ExportConfig
from face_processing.geometry import compute_raw_crop_geometry, rotate_landmarks
from face_processing.models import FrameData, Segment

logger = logging.getLogger(__name__)

LEFT_EYE_INDICES = (33, 133, 159, 145, 158, 153, 160, 144)
RIGHT_EYE_INDICES = (362, 263, 386, 374, 385, 380, 387, 373)
MOUTH_INDICES = (13, 14, 61, 291, 78, 308)
SCALE_OUTLIER_THRESHOLD_RATIO = 0.04


def compute_output_size(
    segment: Segment,
    frame_w: int,
    frame_h: int,
) -> int:
    """Compute output square size from median tilt-aware face height.

    Dataset exports crop a median-width by median-height face rectangle and
    resize it to a square. The square side is the segment median face height.
    """
    heights: list[float] = []

    for fd in segment.frame_data:
        if fd.landmarks is None:
            continue
        geom = compute_raw_crop_geometry(fd, frame_w, frame_h)
        if geom is not None:
            heights.append(geom[3])

    if not heights:
        return 2
    return _even_round(float(np.median(heights)))


def export_segment(
    segment: Segment,
    video_path: str,
    output_path: str,
    frame_w: int,
    frame_h: int,
    output_size: int,
    config: ExportConfig | None = None,
    source_video_path: str | None = None,
) -> str:
    """Export a segment as a square face video.

    For each frame:
    1. Rotate by -roll
    2. Compute crop around rotated face
    3. Resize the median reference face rectangle to output_size x output_size
    4. Pipe to ffmpeg
    """
    if config is None:
        config = ExportConfig()

    S = output_size
    logger.info(
        "Exporting segment %d: frames %d-%d, size %dx%d -> %s",
        segment.segment_id, segment.start_frame, segment.end_frame, S, S, output_path,
    )

    # Audio timing from segment boundaries
    start_sec = segment.start_frame / config.fps
    duration_sec = segment.length / config.fps

    cmd = [
        config.ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{S}x{S}",
        "-r", str(config.fps),
        "-i", "pipe:0",
    ]
    if config.ffmpeg_threads and config.ffmpeg_threads > 0:
        cmd[1:1] = ["-threads", str(int(config.ffmpeg_threads))]
    if source_video_path:
        cmd += [
            "-ss", f"{start_sec:.4f}",
            "-t", f"{duration_sec:.4f}",
            "-i", source_video_path,
            "-c:v", config.codec,
            "-b:v", config.bitrate,
            "-pix_fmt", config.pixel_format,
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
        ]
    else:
        cmd += [
            "-c:v", config.codec,
            "-b:v", config.bitrate,
            "-pix_fmt", config.pixel_format,
            "-an",
        ]
    cmd.append(output_path)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)

    try:
        for fd in segment.frame_data:
            ret, frame_bgr = cap.read()
            if not ret or fd.landmarks is None:
                black = np.zeros((S, S, 3), dtype=np.uint8)
                proc.stdin.write(black.tobytes())
                continue

            roll = fd.roll if fd.pose_valid else 0.0
            cropped = _crop_face_rotated(
                frame_bgr, fd, roll, frame_w, frame_h, S,
            )
            assert cropped.shape == (S, S, 3), f"Bad crop shape {cropped.shape}, expected ({S},{S},3)"
            proc.stdin.write(cropped.tobytes())
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass

    try:
        proc.wait(timeout=config.ffmpeg_timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RuntimeError(
            f"ffmpeg export timed out after {config.ffmpeg_timeout}s for {output_path}"
        ) from exc
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.returncode != 0:
        logger.error("ffmpeg stderr: %s", stderr.decode())
        raise RuntimeError(
            f"ffmpeg export failed (code {proc.returncode}):\n{stderr.decode()}"
        )

    logger.info("Exported segment %d to %s", segment.segment_id, output_path)
    return output_path


def _crop_face_rotated(
    frame_bgr: np.ndarray,
    fd: FrameData,
    roll: float,
    frame_w: int,
    frame_h: int,
    S: int,
) -> np.ndarray:
    """Rotate frame by -roll, crop the segment reference face rect, resize to SxS."""
    # Rotation center at frame center
    center = (frame_w / 2.0, frame_h / 2.0)
    M = cv2.getRotationMatrix2D(center, -roll, 1.0)
    rotated = cv2.warpAffine(
        frame_bgr, M, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    if fd.crop_cx_rot is not None and fd.crop_w_rot is not None:
        cx_rot = fd.crop_cx_rot
        cy_rot = fd.crop_cy_rot if fd.crop_cy_rot is not None else fd.cy
        face_w_rot = fd.crop_w_rot
        face_h_rot = fd.crop_h_rot if fd.crop_h_rot is not None else face_w_rot
    else:
        if fd.raw_crop_cx_rot is not None and fd.raw_crop_w_rot is not None:
            cx_rot = fd.raw_crop_cx_rot
            cy_rot = fd.raw_crop_cy_rot if fd.raw_crop_cy_rot is not None else fd.cy
            face_w_rot = fd.raw_crop_w_rot
            face_h_rot = fd.raw_crop_h_rot if fd.raw_crop_h_rot is not None else face_w_rot
        else:
            geom = compute_raw_crop_geometry(fd, frame_w, frame_h)
            cx_rot, cy_rot, face_w_rot, face_h_rot = geom if geom is not None else (fd.cx, fd.cy, fd.face_w, fd.face_h)

    return _extract_reference_crop(rotated, cx_rot, cy_rot, face_w_rot, face_h_rot, S, frame_w, frame_h)


def prepare_segment_crop_geometry(
    segment: Segment,
    frame_w: int,
    frame_h: int,
) -> None:
    raw_cx: list[float] = []
    raw_cy: list[float] = []
    raw_w: list[float] = []
    raw_h: list[float] = []
    valid: list[bool] = []

    for fd in segment.frame_data:
        if fd.landmarks is None:
            raw_cx.append(float("nan"))
            raw_cy.append(float("nan"))
            raw_w.append(float("nan"))
            raw_h.append(float("nan"))
            valid.append(False)
            continue

        roll = fd.roll if fd.pose_valid else 0.0
        geom = compute_raw_crop_geometry(fd, frame_w, frame_h)
        if geom is None:
            raw_cx.append(float("nan"))
            raw_cy.append(float("nan"))
            raw_w.append(float("nan"))
            raw_h.append(float("nan"))
            valid.append(False)
            continue

        cx_rot, cy_rot, face_w_rot, face_h_rot = geom
        fd.raw_crop_cx_rot = cx_rot
        fd.raw_crop_cy_rot = cy_rot
        fd.raw_crop_w_rot = face_w_rot
        fd.raw_crop_h_rot = face_h_rot

        eye_dist, eye_mouth_dist = _compute_anchor_distances(fd, roll, frame_w, frame_h)
        fd.eye_dist = eye_dist
        fd.eye_mouth_dist = eye_mouth_dist

        raw_cx.append(cx_rot)
        raw_cy.append(cy_rot)
        raw_w.append(face_w_rot)
        raw_h.append(face_h_rot)
        valid.append(True)

    raw_cx_arr = np.array(raw_cx, dtype=np.float64)
    raw_cy_arr = np.array(raw_cy, dtype=np.float64)
    raw_w_arr = np.array(raw_w, dtype=np.float64)
    raw_h_arr = np.array(raw_h, dtype=np.float64)
    valid_arr = np.array(valid, dtype=bool)

    if np.any(valid_arr):
        median_face_w = float(np.nanmedian(raw_w_arr[valid_arr]))
        median_face_h = float(np.nanmedian(raw_h_arr[valid_arr]))
        ref_w = _even_round(median_face_w)
        ref_h = _even_round(median_face_h)
        segment.reference_crop_w = ref_w
        segment.reference_crop_h = ref_h
        segment.output_size = ref_h
    else:
        segment.reference_crop_w = 2
        segment.reference_crop_h = 2
        segment.output_size = 2

    ref_w_float = float(segment.reference_crop_w or 2)
    ref_h_float = float(segment.reference_crop_h or 2)

    for idx, fd in enumerate(segment.frame_data):
        if not valid[idx]:
            continue
        fd.stable_crop_cx_rot = float(raw_cx_arr[idx])
        fd.stable_crop_cy_rot = float(raw_cy_arr[idx])
        fd.stable_crop_w_rot = ref_w_float
        fd.stable_crop_h_rot = ref_h_float
        fd.crop_cx_rot = fd.stable_crop_cx_rot
        fd.crop_cy_rot = fd.stable_crop_cy_rot
        fd.crop_w_rot = fd.stable_crop_w_rot
        fd.crop_h_rot = fd.stable_crop_h_rot
        if np.isfinite(raw_w_arr[idx]) and np.isfinite(raw_h_arr[idx]) and ref_w_float > 0 and ref_h_float > 0:
            w_ratio = abs((raw_w_arr[idx] / ref_w_float) - 1.0)
            h_ratio = abs((raw_h_arr[idx] / ref_h_float) - 1.0)
            ratio = max(w_ratio, h_ratio)
            fd.scale_deviation_ratio = ratio if ratio > SCALE_OUTLIER_THRESHOLD_RATIO else 0.0
        else:
            fd.scale_deviation_ratio = 0.0


def _compute_anchor_distances(
    fd: FrameData,
    roll: float,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    if fd.landmarks is None:
        return 0.0, 0.0

    lmks_px = fd.landmarks.copy()
    lmks_px[:, 0] *= frame_w
    lmks_px[:, 1] *= frame_h
    rotated_lmks = rotate_landmarks(lmks_px[:, :2], -roll, frame_w, frame_h)

    left_eye = np.mean(rotated_lmks[list(LEFT_EYE_INDICES)], axis=0)
    right_eye = np.mean(rotated_lmks[list(RIGHT_EYE_INDICES)], axis=0)
    mouth = np.mean(rotated_lmks[list(MOUTH_INDICES)], axis=0)

    eye_mid = 0.5 * (left_eye + right_eye)
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    eye_mouth_dist = float(np.linalg.norm(mouth - eye_mid))
    return eye_dist, eye_mouth_dist



def _extract_reference_crop(
    frame: np.ndarray,
    cx: float,
    cy: float,
    face_w: float,
    face_h: float,
    S: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Crop a reference face rectangle centered on face, then resize to SxS."""
    half_w = face_w / 2.0
    half_h = face_h / 2.0

    x1 = int(round(cx - half_w))
    x2 = int(round(cx + half_w))
    y1 = int(round(cy - half_h))
    y2 = int(round(cy + half_h))

    # Clamp and pad
    crop = _safe_crop(frame, x1, y1, x2, y2, frame_w, frame_h)

    # Dataset exports are square, so the median reference rectangle is resized to SxS.
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
    else:
        crop = np.zeros((S, S, 3), dtype=np.uint8)

    return crop


def _even_round(value: float) -> int:
    result = max(2, int(round(value / 2.0) * 2))
    return result


def _safe_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
) -> np.ndarray:
    """Crop with zero-padding for out-of-bounds regions."""
    h_crop = y2 - y1
    w_crop = x2 - x1
    if h_crop <= 0 or w_crop <= 0:
        return np.zeros((max(h_crop, 1), max(w_crop, 1), 3), dtype=np.uint8)

    result = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)

    # Source region (clamped to frame bounds)
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(frame_w, x2)
    sy2 = min(frame_h, y2)

    if sx1 >= sx2 or sy1 >= sy2:
        return result

    # Destination offsets
    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    result[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    return result
