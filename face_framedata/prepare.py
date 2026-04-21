"""Prepare a large square clip and generate 1080p + 540p framedata.

Pipeline:
  1. Validate: source width >= 1080 and height >= 1920.
  2. Center-crop to 9:16 aspect ratio, scale to 1080×1920 at 25 fps, 8 Mbps.
  3. Analyze the 1080p video (ROI [0.1, 0.4] by default) and write framedata JSON.
  4. Scale 1080p → 540×960 at 4 Mbps.
  5. Derive 540p framedata by halving all pixel coordinates (cx, cy, w, h).

Usage:
    face-framedata-prepare --input clip.mp4 --output-dir output/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

import cv2

from face_processing.config import PipelineConfig
from face_processing.face_analysis import analyze_frames
from face_processing.geometry import compute_raw_crop_geometry
from face_framedata.pipeline import _smooth_geometry

logger = logging.getLogger(__name__)

_HD_W, _HD_H = 1080, 1920
_SD_W, _SD_H = 540, 960
_HD_BITRATE = "8M"
_SD_BITRATE = "4M"
_TARGET_FPS = 25
_DEFAULT_ROI_TOP = 0.1
_DEFAULT_ROI_BOTTOM = 0.4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_and_analyze(
    input_path: str,
    output_dir: str,
    config: PipelineConfig | None = None,
    smooth_window: int = 5,
    roi_top: float = _DEFAULT_ROI_TOP,
    roi_bottom: float = _DEFAULT_ROI_BOTTOM,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_timeout: int = 600,
) -> dict:
    """Prepare a large clip to 1080p + 540p portrait and generate framedata.

    Returns a summary dict with paths and frame counts.
    """
    if config is None:
        config = PipelineConfig()

    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(output_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Validate source dimensions ────────────────────────────────
    src_w, src_h = _probe_dimensions(input_path)
    if src_w < _HD_W or src_h < _HD_H:
        raise ValueError(
            f"Source too small: {src_w}×{src_h}. "
            f"Need at least {_HD_W} wide and {_HD_H} tall."
        )
    logger.info("Source: %dx%d — %s", src_w, src_h, input_path)

    # ── Step 2: Center-crop 9:16, scale to 1080×1920 @ 25 fps ────────────
    hd_path = os.path.join(out_dir, f"{stem}_1080p.mp4")
    _prepare_hd(
        input_path, hd_path, src_w, src_h,
        _HD_W, _HD_H, _TARGET_FPS, _HD_BITRATE,
        ffmpeg_bin, ffmpeg_timeout,
    )

    # ── Step 3: Face analysis on 1080p ────────────────────────────────────
    detection_cfg = config.detection.__class__(
        model_path=config.detection.model_path,
        num_faces=config.detection.num_faces,
        min_detection_confidence=config.detection.min_detection_confidence,
        min_presence_confidence=config.detection.min_presence_confidence,
        use_gpu=config.detection.use_gpu,
        roi_top_ratio=roi_top,
        roi_bottom_ratio=roi_bottom,
    )

    logger.info("=== Analyzing 1080p video (ROI [%.2f, %.2f]) ===", roi_top, roi_bottom)
    frame_data = analyze_frames(hd_path, detection_cfg)

    # ── Step 4: Compute per-frame geometry + smooth ───────────────────────
    frames_out: list[dict] = []
    frame_data_valid: list = []
    fail_count = 0
    for fd in frame_data:
        geom = compute_raw_crop_geometry(fd, _HD_W, _HD_H)
        if geom is None:
            frames_out.append({"i": fd.frame_idx, "status": "fail"})
            fail_count += 1
        else:
            cx, cy, w, h = geom
            roll = fd.roll if fd.pose_valid else 0.0
            frames_out.append({
                "i": fd.frame_idx,
                "roll": roll, "cx": cx, "cy": cy, "w": w, "h": h,
            })
            frame_data_valid.append(fd)

    if smooth_window > 1:
        _smooth_geometry(frames_out, frame_data_valid, _HD_W, _HD_H, window=smooth_window)

    total = len(frame_data)
    valid_count = total - fail_count
    logger.info("Frames: total=%d  valid=%d  fail=%d", total, valid_count, fail_count)

    # ── Step 5: Write 1080p framedata ────────────────────────────────────
    hd_framedata: dict = {
        "source_video": os.path.basename(hd_path),
        "total_frames": total,
        "frames": frames_out,
    }
    hd_fd_path = os.path.join(out_dir, f"{stem}_1080p_framedata.json")
    with open(hd_fd_path, "w") as fh:
        json.dump(hd_framedata, fh, separators=(",", ":"))
    logger.info("Wrote 1080p framedata -> %s", hd_fd_path)

    # ── Step 6: Scale to 540p ─────────────────────────────────────────────
    sd_path = os.path.join(out_dir, f"{stem}_540p.mp4")
    _scale_video(
        hd_path, sd_path, _SD_W, _SD_H, _SD_BITRATE,
        ffmpeg_bin, ffmpeg_timeout,
    )

    # ── Step 7: Derive 540p framedata (pixel coords × 0.5) ───────────────
    scale = _SD_W / _HD_W  # 0.5
    sd_framedata = _scale_framedata(hd_framedata, scale)
    sd_framedata["source_video"] = os.path.basename(sd_path)
    sd_fd_path = os.path.join(out_dir, f"{stem}_540p_framedata.json")
    with open(sd_fd_path, "w") as fh:
        json.dump(sd_framedata, fh, separators=(",", ":"))
    logger.info("Wrote 540p framedata -> %s", sd_fd_path)

    return {
        "source_video": os.path.basename(input_path),
        "total_frames": total,
        "valid_frames": valid_count,
        "fail_frames": fail_count,
        "1080p_video": hd_path,
        "1080p_framedata": hd_fd_path,
        "540p_video": sd_path,
        "540p_framedata": sd_fd_path,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_dimensions(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or h == 0:
        raise RuntimeError(f"Cannot read video dimensions: {path}")
    return w, h


def _prepare_hd(
    input_path: str,
    output_path: str,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    fps: int,
    bitrate: str,
    ffmpeg_bin: str,
    timeout: int,
) -> None:
    """Center-crop to target aspect ratio and scale to out_w × out_h."""
    target_ar = out_w / out_h  # 9/16

    if src_w / src_h > target_ar:
        # Source is wider → crop width, keep full height
        crop_h = src_h
        crop_w = _even(round(src_h * out_w / out_h))
        x_off = (src_w - crop_w) // 2
        y_off = 0
    else:
        # Source is taller or exact → crop height, keep full width
        crop_w = src_w
        crop_h = _even(round(src_w * out_h / out_w))
        x_off = 0
        y_off = (src_h - crop_h) // 2

    vf = (
        f"crop={crop_w}:{crop_h}:{x_off}:{y_off},"
        f"scale={out_w}:{out_h}:flags=lanczos"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264", "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    logger.info(
        "Crop %dx%d+%d+%d → scale %dx%d: %s",
        crop_w, crop_h, x_off, y_off, out_w, out_h, output_path,
    )
    _run_ffmpeg(cmd, output_path, timeout)


def _scale_video(
    input_path: str,
    output_path: str,
    w: int,
    h: int,
    bitrate: str,
    ffmpeg_bin: str,
    timeout: int,
) -> None:
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-vf", f"scale={w}:{h}:flags=lanczos",
        "-c:v", "libx264", "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    logger.info("Scale to %dx%d → %s", w, h, output_path)
    _run_ffmpeg(cmd, output_path, timeout)


def _run_ffmpeg(cmd: list[str], output_path: str, timeout: int) -> None:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}:\n{proc.stderr.decode()}"
        )


def _scale_framedata(framedata: dict, scale: float) -> dict:
    """Return new framedata with pixel coords multiplied by *scale*; angles unchanged."""
    pixel_keys = ("cx", "cy", "w", "h", "scx", "scy", "sw", "sh")
    scaled = []
    for f in framedata["frames"]:
        if "status" in f:
            scaled.append(dict(f))
            continue
        sf: dict = {}
        for k, v in f.items():
            sf[k] = v * scale if k in pixel_keys else v
        scaled.append(sf)
    return {
        "source_video": framedata["source_video"],
        "total_frames": framedata["total_frames"],
        "frames": scaled,
    }


def _even(n: int) -> int:
    """Round down to nearest even integer."""
    return n if n % 2 == 0 else n - 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="face-framedata-prepare",
        description=(
            "Prepare a large clip to 1080p+540p portrait and generate framedata. "
            "Input must be at least 1080 wide and 1920 tall."
        ),
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Source video (square or larger, e.g. 3800×3800)")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Root output directory (default: output/)")
    parser.add_argument("--roi-top", type=float, default=_DEFAULT_ROI_TOP,
                        help=f"ROI top fraction for face detection (default: {_DEFAULT_ROI_TOP})")
    parser.add_argument("--roi-bottom", type=float, default=_DEFAULT_ROI_BOTTOM,
                        help=f"ROI bottom fraction for face detection (default: {_DEFAULT_ROI_BOTTOM})")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (Metal) for MediaPipe inference")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig()
    config.detection.use_gpu = args.gpu

    report = prepare_and_analyze(
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
    )

    print(f"\nDone:")
    print(f"  1080p video:     {report['1080p_video']}")
    print(f"  1080p framedata: {report['1080p_framedata']}")
    print(f"  540p  video:     {report['540p_video']}")
    print(f"  540p  framedata: {report['540p_framedata']}")
    print(f"  Frames: {report['valid_frames']}/{report['total_frames']} valid")


if __name__ == "__main__":
    main()
