from __future__ import annotations

import numpy as np

from face_processing.config import RankingThresholds
from face_processing.models import Segment, SegmentMetrics


def compute_segment_metrics(
    segment: Segment,
    total_frames_in_source: int = 0,
) -> SegmentMetrics:
    """Compute aggregate quality metrics for a segment."""
    fds = segment.frame_data
    n = len(fds)
    if n == 0:
        return SegmentMetrics()

    yaws = np.array([abs(fd.yaw) for fd in fds])
    pitches = np.array([abs(fd.pitch) for fd in fds])
    rolls = np.array([abs(fd.roll) for fd in fds])
    face_hs = np.array([fd.face_h for fd in fds])
    face_ws = np.array([fd.face_w for fd in fds])
    cxs = np.array([fd.cx for fd in fds])
    cys = np.array([fd.cy for fd in fds])

    eye_dists = np.array([fd.eye_dist for fd in fds if fd.eye_dist and fd.eye_dist > 0], dtype=np.float64)
    eye_mouth_dists = np.array(
        [fd.eye_mouth_dist for fd in fds if fd.eye_mouth_dist and fd.eye_mouth_dist > 0],
        dtype=np.float64,
    )
    scale_dev = np.array(
        [fd.scale_deviation_ratio for fd in fds if fd.scale_deviation_ratio is not None],
        dtype=np.float64,
    )

    mean_face_h = float(np.mean(face_hs)) if n > 0 else 1.0
    std_face_h = float(np.std(face_hs)) if n > 1 else 0.0
    mean_face_w = float(np.mean(face_ws)) if n > 0 else 1.0
    std_face_w = float(np.std(face_ws)) if n > 1 else 0.0

    # Jump ratio: frames with frame_jumps reason / total
    jump_count = sum(1 for fd in fds if "frame_jumps" in fd.bad_reasons)
    scale_outlier_count = sum(1 for fd in fds if (fd.scale_deviation_ratio or 0.0) > 0.0)
    # Missing ratio: frames with no face / total
    missing_count = sum(1 for fd in fds if not fd.face_detected)
    # Low confidence ratio
    low_conf_count = sum(1 for fd in fds if "low_face_confidence" in fd.bad_reasons)

    return SegmentMetrics(
        mean_abs_yaw=float(np.mean(yaws)),
        max_abs_yaw=float(np.max(yaws)),
        mean_abs_pitch=float(np.mean(pitches)),
        max_abs_pitch=float(np.max(pitches)),
        mean_abs_roll=float(np.mean(rolls)),
        max_abs_roll=float(np.max(rolls)),
        mean_face_h=mean_face_h,
        min_face_h=float(np.min(face_hs)),
        face_size_std_ratio=std_face_h / mean_face_h if mean_face_h > 0 else 0.0,
        mean_face_w=mean_face_w,
        min_face_w=float(np.min(face_ws)),
        face_width_std_ratio=std_face_w / mean_face_w if mean_face_w > 0 else 0.0,
        std_cx=float(np.std(cxs)),
        std_cy=float(np.std(cys)),
        eye_dist_std_ratio=float(np.std(eye_dists) / np.mean(eye_dists)) if eye_dists.size > 1 else 0.0,
        eye_mouth_std_ratio=float(np.std(eye_mouth_dists) / np.mean(eye_mouth_dists)) if eye_mouth_dists.size > 1 else 0.0,
        scale_outlier_ratio=scale_outlier_count / n,
        jump_ratio=jump_count / n,
        missing_ratio=missing_count / n,
        low_conf_ratio=low_conf_count / n,
    )


def rank_segment(
    metrics: SegmentMetrics,
    thresholds: RankingThresholds,
    output_size: int,
) -> str:
    """Assign rank: 'confident', 'medium', or 'unconfident'."""
    if _meets_confident(metrics, thresholds, output_size):
        return "confident"
    if _meets_medium(metrics, thresholds, output_size):
        return "medium"
    return "unconfident"


def _meets_confident(m: SegmentMetrics, t: RankingThresholds, s: int) -> bool:
    return (
        m.mean_abs_yaw <= t.conf_mean_abs_yaw
        and m.mean_abs_pitch <= t.conf_mean_abs_pitch
        and m.mean_abs_roll <= t.conf_mean_abs_roll
        and m.max_abs_yaw <= t.conf_max_abs_yaw
        and m.max_abs_pitch <= t.conf_max_abs_pitch
        and m.max_abs_roll <= t.conf_max_abs_roll
        and m.face_size_std_ratio <= t.conf_face_size_std_ratio
        and m.face_width_std_ratio <= t.conf_face_width_std_ratio
        and m.std_cx <= t.conf_std_cx_ratio * s
        and m.std_cy <= t.conf_std_cy_ratio * s
        and m.eye_dist_std_ratio <= t.conf_eye_dist_std_ratio
        and m.eye_mouth_std_ratio <= t.conf_eye_mouth_std_ratio
        and m.scale_outlier_ratio <= t.conf_scale_outlier_ratio
        and m.jump_ratio <= t.conf_jump_ratio
        and m.low_conf_ratio <= t.conf_low_conf_ratio
    )


def _meets_medium(m: SegmentMetrics, t: RankingThresholds, s: int) -> bool:
    return (
        m.mean_abs_yaw <= t.med_mean_abs_yaw
        and m.mean_abs_pitch <= t.med_mean_abs_pitch
        and m.mean_abs_roll <= t.med_mean_abs_roll
        and m.max_abs_yaw <= t.med_max_abs_yaw
        and m.max_abs_pitch <= t.med_max_abs_pitch
        and m.max_abs_roll <= t.med_max_abs_roll
        and m.face_size_std_ratio <= t.med_face_size_std_ratio
        and m.face_width_std_ratio <= t.med_face_width_std_ratio
        and m.std_cx <= t.med_std_cx_ratio * s
        and m.std_cy <= t.med_std_cy_ratio * s
        and m.eye_dist_std_ratio <= t.med_eye_dist_std_ratio
        and m.eye_mouth_std_ratio <= t.med_eye_mouth_std_ratio
        and m.scale_outlier_ratio <= t.med_scale_outlier_ratio
        and m.jump_ratio <= t.med_jump_ratio
        and m.low_conf_ratio <= t.med_low_conf_ratio
    )
