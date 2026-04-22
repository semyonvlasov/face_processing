"""Prepare a large square clip and generate native + 1080p + 540p framedata.

Pipeline:
  1. Validate: source width >= 1080 and height >= 1920.
  2. Center-crop to 9:16 at native resolution, 25 fps, lossless H264.
  3. Analyze the native crop (ROI [0.1, 0.4] by default) and compute framedata.
  4. Scale native framedata → 1080p and 540p by proportional coordinate scaling.
  5. Scale native crop → 1080×1920 @ 8 Mbps and 540×960 @ 4 Mbps.
  6. Cut face clips from native video: 192×192 and 96×96 in one pass
     using stretch_to_square_mean_width (median face width).

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
from face_framedata.cut import cut_face_clips_from_native

logger = logging.getLogger(__name__)

_HD_W, _HD_H = 1080, 1920
_SD_W, _SD_H = 540, 960
_NATIVE_CRF  = 0       # lossless H264 for intermediate native crop
_HD_BITRATE  = "8M"
_SD_BITRATE  = "4M"
_TARGET_FPS  = 25
_DEFAULT_ROI_TOP    = 0.1
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
    produce_faceclip: bool = True,
) -> dict:
    """Prepare a large clip: native crop → framedata → 1080p/540p → face clips.

    By default produces face clips: 192×192 for 1080p, 96×96 for 540p,
    cut from the native-resolution crop to minimise blur.
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

    # ── Step 2: Center-crop to 9:16 at native resolution (lossless) ───────
    native_path = os.path.join(out_dir, f"{stem}_native_crop.mp4")
    native_w, native_h = _crop_native(
        input_path, native_path, src_w, src_h,
        _TARGET_FPS, _NATIVE_CRF,
        ffmpeg_bin, ffmpeg_timeout,
    )

    # ── Step 3: Face analysis on native crop ──────────────────────────────
    detection_cfg = config.detection.__class__(
        model_path=config.detection.model_path,
        num_faces=config.detection.num_faces,
        min_detection_confidence=config.detection.min_detection_confidence,
        min_presence_confidence=config.detection.min_presence_confidence,
        use_gpu=config.detection.use_gpu,
        roi_top_ratio=roi_top,
        roi_bottom_ratio=roi_bottom,
    )

    logger.info(
        "=== Analyzing native crop %dx%d (ROI [%.2f, %.2f]) ===",
        native_w, native_h, roi_top, roi_bottom,
    )
    frame_data = analyze_frames(native_path, detection_cfg)

    # ── Step 4: Compute per-frame geometry + smooth (native coords) ────────
    frames_out: list[dict] = []
    frame_data_valid: list = []
    fail_count = 0
    for fd in frame_data:
        geom = compute_raw_crop_geometry(fd, native_w, native_h)
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
        _smooth_geometry(frames_out, frame_data_valid, native_w, native_h, window=smooth_window)

    total = len(frame_data)
    valid_count = total - fail_count
    logger.info("Frames: total=%d  valid=%d  fail=%d", total, valid_count, fail_count)

    # ── Step 5: Write native framedata ────────────────────────────────────
    native_framedata: dict = {
        "source_video": os.path.basename(native_path),
        "total_frames": total,
        "frames": frames_out,
    }
    native_fd_path = os.path.join(out_dir, f"{stem}_native_framedata.json")
    with open(native_fd_path, "w") as fh:
        json.dump(native_framedata, fh, separators=(",", ":"))
    logger.info("Wrote native framedata -> %s", native_fd_path)

    # ── Step 6: Derive 1080p framedata (scale native coords) ─────────────
    scale_hd = _HD_W / native_w
    hd_framedata = _scale_framedata(native_framedata, scale_hd)
    hd_framedata["source_video"] = f"{stem}_1080p.mp4"
    hd_fd_path = os.path.join(out_dir, f"{stem}_1080p_framedata.json")
    with open(hd_fd_path, "w") as fh:
        json.dump(hd_framedata, fh, separators=(",", ":"))
    logger.info("Wrote 1080p framedata -> %s", hd_fd_path)

    # ── Step 7: Derive 540p framedata ─────────────────────────────────────
    scale_sd = _SD_W / native_w
    sd_framedata = _scale_framedata(native_framedata, scale_sd)
    sd_framedata["source_video"] = f"{stem}_540p.mp4"
    sd_fd_path = os.path.join(out_dir, f"{stem}_540p_framedata.json")
    with open(sd_fd_path, "w") as fh:
        json.dump(sd_framedata, fh, separators=(",", ":"))
    logger.info("Wrote 540p framedata -> %s", sd_fd_path)

    # ── Step 8: Scale native crop → 1080p ─────────────────────────────────
    hd_path = os.path.join(out_dir, f"{stem}_1080p.mp4")
    _scale_video(
        native_path, hd_path, _HD_W, _HD_H, _HD_BITRATE,
        ffmpeg_bin, ffmpeg_timeout,
    )

    # ── Step 9: Scale native crop → 540p ──────────────────────────────────
    sd_path = os.path.join(out_dir, f"{stem}_540p.mp4")
    _scale_video(
        native_path, sd_path, _SD_W, _SD_H, _SD_BITRATE,
        ffmpeg_bin, ffmpeg_timeout,
    )

    result = {
        "source_video": os.path.basename(input_path),
        "total_frames": total,
        "valid_frames": valid_count,
        "fail_frames": fail_count,
        "native_crop": native_path,
        "native_framedata": native_fd_path,
        "1080p_video": hd_path,
        "1080p_framedata": hd_fd_path,
        "540p_video": sd_path,
        "540p_framedata": sd_fd_path,
    }

    # ── Step 10: Cut face clips from native (192×192 + 96×96, one pass) ───
    if produce_faceclip:
        hd_face_path = os.path.join(out_dir, f"{stem}_1080p_face.mp4")
        sd_face_path = os.path.join(out_dir, f"{stem}_540p_face.mp4")
        logger.info("=== Cutting face clips from native (192x192 + 96x96) ===")
        cut_face_clips_from_native(
            native_framedata_path=native_fd_path,
            native_video_path=native_path,
            output_hd_path=hd_face_path,
            output_sd_path=sd_face_path,
            ffmpeg_bin=ffmpeg_bin,
            ffmpeg_timeout=ffmpeg_timeout,
        )
        result["1080p_face_video"] = hd_face_path
        result["540p_face_video"] = sd_face_path

    return result


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


def _crop_native(
    input_path: str,
    output_path: str,
    src_w: int,
    src_h: int,
    fps: int,
    crf: int,
    ffmpeg_bin: str,
    timeout: int,
) -> tuple[int, int]:
    """Center-crop source to 9:16 at native resolution, lossless H264.

    Returns (out_w, out_h) of the produced video.
    """
    target_ar = 9 / 16

    if src_w / src_h > target_ar:
        crop_h = src_h
        crop_w = _even(round(src_h * 9 / 16))
        x_off = (src_w - crop_w) // 2
        y_off = 0
    else:
        crop_w = src_w
        crop_h = _even(round(src_w * 16 / 9))
        x_off = 0
        y_off = (src_h - crop_h) // 2

    vf = f"crop={crop_w}:{crop_h}:{x_off}:{y_off}"
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    logger.info(
        "Native crop %dx%d+%d+%d → %s (crf=%d)",
        crop_w, crop_h, x_off, y_off, output_path, crf,
    )
    _run_ffmpeg(cmd, output_path, timeout)
    return crop_w, crop_h


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
            "Prepare a large clip: native 9:16 crop → framedata → "
            "1080p/540p videos + face clips. "
            "Input must be at least 1080 wide and 1920 tall."
        ),
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Source video (square or larger, e.g. 3840×3840)")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Root output directory (default: output/)")
    parser.add_argument("--roi-top", type=float, default=_DEFAULT_ROI_TOP,
                        help=f"ROI top fraction for face detection (default: {_DEFAULT_ROI_TOP})")
    parser.add_argument("--roi-bottom", type=float, default=_DEFAULT_ROI_BOTTOM,
                        help=f"ROI bottom fraction for face detection (default: {_DEFAULT_ROI_BOTTOM})")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (Metal) for MediaPipe inference")
    parser.add_argument("--no-faceclip", action="store_true",
                        help="Skip face clip generation (192x192 @ 1080p, 96x96 @ 540p)")
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
        produce_faceclip=not args.no_faceclip,
    )

    print(f"\nDone:")
    print(f"  native crop:     {report['native_crop']}")
    print(f"  native framedata:{report['native_framedata']}")
    print(f"  1080p video:     {report['1080p_video']}")
    print(f"  1080p framedata: {report['1080p_framedata']}")
    print(f"  540p  video:     {report['540p_video']}")
    print(f"  540p  framedata: {report['540p_framedata']}")
    if "1080p_face_video" in report:
        print(f"  1080p face:      {report['1080p_face_video']}")
    if "540p_face_video" in report:
        print(f"  540p  face:      {report['540p_face_video']}")
    print(f"  Frames: {report['valid_frames']}/{report['total_frames']} valid")


if __name__ == "__main__":
    main()
