from __future__ import annotations

import logging

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_processing.config import DetectionConfig
from face_processing.face_model_3d import extract_euler_from_transform
from face_processing.models import FrameData

logger = logging.getLogger(__name__)


def _create_detector(config: DetectionConfig) -> vision.FaceLandmarker:
    delegate = (
        mp.tasks.BaseOptions.Delegate.GPU
        if config.use_gpu
        else mp.tasks.BaseOptions.Delegate.CPU
    )
    delegate_name = "GPU (Metal)" if config.use_gpu else "CPU"
    logger.info("MediaPipe delegate: %s", delegate_name)
    base_options = python.BaseOptions(
        model_asset_path=config.model_path,
        delegate=delegate,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=config.num_faces,
        min_face_detection_confidence=config.min_detection_confidence,
        min_face_presence_confidence=config.min_presence_confidence,
    )
    return vision.FaceLandmarker.create_from_options(options)


def analyze_frames(
    video_path: str,
    detection_config: DetectionConfig | None = None,
) -> list[FrameData]:
    if detection_config is None:
        detection_config = DetectionConfig()

    detector = _create_detector(detection_config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = frame_w * frame_h

    logger.info(
        "Analyzing %d frames (%dx%d) from %s",
        total_frames, frame_w, frame_h, video_path,
    )

    results: list[FrameData] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        fd = _process_frame(
            frame_bgr, frame_idx, frame_w, frame_h, frame_area,
            detector, detection_config.use_gpu,
        )
        results.append(fd)

        if frame_idx % 500 == 0:
            logger.info("  frame %d / %d", frame_idx, total_frames)
        logger.debug(
            "  frame %d: faces=%d size=%.0fx%.0f yaw=%.1f pitch=%.1f roll=%.1f conf=%.2f",
            frame_idx, fd.num_faces, fd.face_w, fd.face_h,
            fd.yaw, fd.pitch, fd.roll, fd.confidence,
        )
        frame_idx += 1

    cap.release()
    detector.close()
    logger.info("Analysis complete: %d frames processed", len(results))
    return results


def _process_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    frame_w: int,
    frame_h: int,
    frame_area: int,
    detector: vision.FaceLandmarker,
    use_gpu: bool = False,
) -> FrameData:
    fd = FrameData(frame_idx=frame_idx)

    if use_gpu:
        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
    else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    try:
        detection_result = detector.detect(mp_image)
    except Exception:
        logger.debug("Detection failed on frame %d", frame_idx)
        return fd

    num_faces = len(detection_result.face_landmarks)
    fd.num_faces = num_faces

    if num_faces == 0:
        return fd

    fd.face_detected = True

    if num_faces > 1:
        # Still extract first face for metrics, but mark as multi-face
        pass

    # Use first face
    face_landmarks = detection_result.face_landmarks[0]
    fd.confidence = _detection_confidence(detection_result, 0)

    # Extract landmarks as pixel coordinates
    lmks_norm = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float64
    )
    fd.landmarks = lmks_norm

    lmks_px = lmks_norm.copy()
    lmks_px[:, 0] *= frame_w
    lmks_px[:, 1] *= frame_h

    # Bounding box from landmarks
    xs = lmks_px[:, 0]
    ys = lmks_px[:, 1]
    x1, y1 = float(np.min(xs)), float(np.min(ys))
    x2, y2 = float(np.max(xs)), float(np.max(ys))
    fd.bbox = (int(x1), int(y1), int(x2), int(y2))
    fd.face_w = x2 - x1
    fd.face_h = y2 - y1
    fd.cx = (x1 + x2) / 2.0
    fd.cy = (y1 + y2) / 2.0
    fd.face_area_ratio = (fd.face_w * fd.face_h) / frame_area if frame_area > 0 else 0.0

    # Head pose from MediaPipe transformation matrix (uses full 478-point mesh)
    if detection_result.facial_transformation_matrixes:
        tm = np.array(detection_result.facial_transformation_matrixes[0])
        fd.transform_matrix = tm
        yaw, pitch, roll = extract_euler_from_transform(tm)
        fd.yaw = yaw
        fd.pitch = pitch
        fd.roll = roll
        fd.pose_valid = True


    return fd


def _detection_confidence(result, face_idx: int) -> float:
    """Extract detection confidence for a face from FaceLandmarkerResult."""
    # MediaPipe FaceLandmarkerResult doesn't directly expose a single
    # confidence score. We use the face_blendshapes or fall back to 1.0.
    # The actual confidence comes from the internal detection — we can
    # approximate using the landmark presence.
    # For the Tasks API, we rely on min_detection_confidence filtering.
    # Return 1.0 as baseline; the reprojection error is our real quality signal.
    try:
        if hasattr(result, 'face_blendshapes') and result.face_blendshapes:
            bs = result.face_blendshapes[face_idx]
            if len(bs) > 0:
                # The first blendshape is "_neutral" with a score 0..1
                return float(bs[0].score)
    except (IndexError, AttributeError):
        pass
    return 1.0
