from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from face_processing.config import PipelineConfig


_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_call_video_config(path: str | None = None) -> tuple[PipelineConfig, dict[str, Any]]:
    cfg = PipelineConfig()
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Call video config must be a mapping: {path}")
        section = data.get("call_video_preparation", data)
        if not isinstance(section, dict):
            raise ValueError(f"call_video_preparation config must be a mapping: {path}")
        values.update(section)

    if "model_path" in values:
        cfg.detection.model_path = _resolve_model_path(str(values["model_path"]))
    for key in (
        "num_faces",
        "min_detection_confidence",
        "min_presence_confidence",
        "use_gpu",
    ):
        if key in values:
            setattr(cfg.detection, key, values[key])
    return cfg, values


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())
