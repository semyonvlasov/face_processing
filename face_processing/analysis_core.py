from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2

from face_processing.config import PipelineConfig
from face_processing.face_analysis import analyze_frames
from face_processing.models import FrameData
from face_processing.normalize import normalize_video

logger = logging.getLogger(__name__)


@dataclass
class CoreAnalysisResult:
    normalized_path: str
    frame_w: int
    frame_h: int
    total_frames: int
    frame_data: list[FrameData]


def run_core_analysis(
    input_path: str,
    out_dir: str,
    config: PipelineConfig,
) -> CoreAnalysisResult:
    """Normalize video and extract per-frame face landmarks + pose.

    Shared by both face_processing and face_framedata pipelines.
    """
    logger.info("=== Stage 1: Normalizing video ===")
    normalized_path = os.path.join(out_dir, "normalized.mp4")
    normalize_video(input_path, normalized_path, config.normalization)

    cap = cv2.VideoCapture(normalized_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.info("=== Stage 2: Analyzing frames ===")
    frame_data = analyze_frames(normalized_path, config.detection)

    return CoreAnalysisResult(
        normalized_path=normalized_path,
        frame_w=frame_w,
        frame_h=frame_h,
        total_frames=total_frames,
        frame_data=frame_data,
    )
