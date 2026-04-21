from __future__ import annotations

import json
import logging
import os

import numpy as np

from face_processing.analysis_core import run_core_analysis
from face_processing.config import PipelineConfig
from face_processing.geometry import compute_raw_crop_geometry

logger = logging.getLogger(__name__)

def _smooth_geometry(
    frames_out: list[dict],
    frame_data_valid: list,
    frame_w: int,
    frame_h: int,
    window: int = 5,
) -> None:
    """Add consistent smoothed geometry fields to each valid frame.

    Two-pass algorithm:
      1. Smooth the raw roll values (5-frame centered MA) → sroll.
      2. Recompute (scx, scy, sw, sh) for each valid frame using sroll as
         roll_override so that all smoothed fields share the same rotation angle.

    This ensures that cut.py and restore.py can use sroll for frame rotation
    and scx/scy/sw/sh for crop bounds without geometric inconsistency.
    Mutates frames_out in place.
    """
    half = window // 2
    valid_indices = [i for i, fr in enumerate(frames_out) if "status" not in fr]
    if not valid_indices:
        return

    n = len(valid_indices)
    raw_rolls = [float(frames_out[i]["roll"]) for i in valid_indices]

    # Pass 1: smooth roll with centered MA
    srolls: list[float] = []
    for k in range(n):
        lo = max(0, k - half)
        hi = min(n, k + half + 1)
        srolls.append(float(np.mean(raw_rolls[lo:hi])))

    # Pass 2: recompute geometry using sroll so rotation and coords are consistent
    for k, list_idx in enumerate(valid_indices):
        fr = frames_out[list_idx]
        fd = frame_data_valid[k]
        sroll = srolls[k]
        geom = compute_raw_crop_geometry(fd, frame_w, frame_h, roll_override=sroll)
        if geom is not None:
            scx, scy, sw, sh = geom
            fr["sroll"] = sroll
            fr["scx"]   = scx
            fr["scy"]   = scy
            fr["sw"]    = sw
            fr["sh"]    = sh


def process_video_framedata(
    input_path: str,
    config: PipelineConfig | None = None,
    smooth_window: int = 5,
) -> dict:
    """Run the framedata pipeline: normalize, analyze, write per-frame geometry JSON.

    Returns a summary dict with total/valid/fail counts.
    """
    if config is None:
        config = PipelineConfig()

    source_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(config.output_dir, source_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Stages 1+2: Normalize + face analysis ---
    analysis = run_core_analysis(input_path, out_dir, config)
    frame_w, frame_h = analysis.frame_w, analysis.frame_h

    # --- Stage 3: Compute per-frame crop geometry ---
    logger.info("=== Stage 3: Computing per-frame crop geometry ===")
    frames_out: list[dict] = []
    frame_data_valid: list = []  # FrameData objects for valid frames (parallel to valid entries)
    fail_count = 0

    for fd in analysis.frame_data:
        geom = compute_raw_crop_geometry(fd, frame_w, frame_h)
        if geom is None:
            frames_out.append({"i": fd.frame_idx, "status": "fail"})
            fail_count += 1
        else:
            cx, cy, w, h = geom
            roll = fd.roll if fd.pose_valid else 0.0
            frames_out.append({
                "i": fd.frame_idx,
                "roll": roll,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            })
            frame_data_valid.append(fd)

    # --- Stage 3b: Smooth geometry (two-pass: smooth roll, recompute coords) ---
    if smooth_window > 1:
        _smooth_geometry(frames_out, frame_data_valid, frame_w, frame_h, window=smooth_window)

    valid_count = len(frames_out) - fail_count
    logger.info(
        "Frames: total=%d  valid=%d  fail=%d",
        analysis.total_frames, valid_count, fail_count,
    )

    # --- Stage 4: Write output ---
    framedata_path = os.path.join(out_dir, f"{source_name}_framedata.json")
    framedata = {
        "source_video": os.path.basename(input_path),
        "total_frames": analysis.total_frames,
        "frames": frames_out,
    }
    with open(framedata_path, "w") as f:
        json.dump(framedata, f, separators=(",", ":"))
    logger.info("Wrote framedata -> %s", framedata_path)

    report = {
        "source_video": os.path.basename(input_path),
        "status": "processed",
        "total_frames": analysis.total_frames,
        "valid_frames": valid_count,
        "fail_frames": fail_count,
    }
    report_path = os.path.join(out_dir, f"{source_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if not config.keep_normalized:
        _cleanup(analysis.normalized_path)

    return report


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
        logger.info("Cleaned up normalized video: %s", path)
    except OSError:
        pass
