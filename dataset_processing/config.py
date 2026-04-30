from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from face_processing.config import PipelineConfig


_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dataset_config(path: str | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if path is None:
        return apply_dataset_config(cfg, {})
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    section = data.get("dataset_processing", data)
    if not isinstance(section, dict):
        raise ValueError(f"dataset_processing config must be a mapping: {path}")
    apply_dataset_config(cfg, section)
    return cfg


def apply_dataset_config(cfg: PipelineConfig, values: dict[str, Any]) -> PipelineConfig:
    _set_if_present(cfg, values, "save_frame_log")
    _set_if_present(cfg, values, "keep_normalized")
    _set_if_present(cfg, values, "output_dir")

    _set_if_present(cfg.normalization, values, "normalization_fps", "fps")
    _set_if_present(cfg.normalization, values, "normalization_bitrate", "bitrate")
    _set_if_present(cfg.normalization, values, "normalization_codec", "codec")
    _set_if_present(cfg.normalization, values, "pixel_format")
    _set_if_present(cfg.normalization, values, "ffmpeg_bin")
    _set_if_present(cfg.normalization, values, "ffmpeg_threads")
    _set_if_present(cfg.normalization, values, "ffmpeg_timeout")

    _set_if_present(cfg.detection, values, "model_path", transform=_resolve_model_path)
    for key in (
        "num_faces",
        "min_detection_confidence",
        "min_presence_confidence",
        "use_gpu",
        "roi_top_ratio",
        "roi_bottom_ratio",
    ):
        _set_if_present(cfg.detection, values, key)

    for key in (
        "min_face_h",
        "max_abs_yaw",
        "max_abs_pitch",
        "max_abs_roll",
        "max_delta_cx_ratio",
        "max_delta_cy_ratio",
        "max_delta_yaw",
        "max_delta_pitch",
        "max_delta_roll",
        "max_face_h_ratio_deviation",
        "max_face_w_ratio_deviation",
        "min_confidence",
        "min_segment_length",
        "max_segment_length",
        "motion_window_frames",
        "max_cumulative_motion_ratio",
    ):
        _set_if_present(cfg.bad_frame, values, key)

    for key in (
        "conf_mean_abs_yaw",
        "conf_mean_abs_pitch",
        "conf_mean_abs_roll",
        "conf_max_abs_yaw",
        "conf_max_abs_pitch",
        "conf_max_abs_roll",
        "conf_face_size_std_ratio",
        "conf_face_width_std_ratio",
        "conf_std_cx_ratio",
        "conf_std_cy_ratio",
        "conf_eye_dist_std_ratio",
        "conf_eye_mouth_std_ratio",
        "conf_scale_outlier_ratio",
        "conf_jump_ratio",
        "conf_low_conf_ratio",
        "med_mean_abs_yaw",
        "med_mean_abs_pitch",
        "med_mean_abs_roll",
        "med_max_abs_yaw",
        "med_max_abs_pitch",
        "med_max_abs_roll",
        "med_face_size_std_ratio",
        "med_face_width_std_ratio",
        "med_std_cx_ratio",
        "med_std_cy_ratio",
        "med_eye_dist_std_ratio",
        "med_eye_mouth_std_ratio",
        "med_scale_outlier_ratio",
        "med_jump_ratio",
        "med_low_conf_ratio",
    ):
        _set_if_present(cfg.ranking, values, key)

    _set_if_present(cfg.export, values, "export_fps", "fps")
    _set_if_present(cfg.export, values, "export_bitrate", "bitrate")
    _set_if_present(cfg.export, values, "export_codec", "codec")
    _set_if_present(cfg.export, values, "export_pixel_format", "pixel_format")
    _set_if_present(cfg.export, values, "ffmpeg_bin")
    _set_if_present(cfg.export, values, "ffmpeg_threads")
    _set_if_present(cfg.export, values, "ffmpeg_timeout")
    cfg.export.mode = "median_face_rect"
    return cfg


def _set_if_present(
    target: object,
    values: dict[str, Any],
    source_key: str,
    target_key: str | None = None,
    *,
    transform=None,
) -> None:
    if source_key not in values:
        return
    value = values[source_key]
    if transform is not None:
        value = transform(value)
    setattr(target, target_key or source_key, value)


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())
