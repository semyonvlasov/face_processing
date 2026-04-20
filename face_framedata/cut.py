"""Extract per-frame face crops from a normalized video using framedata JSON.

For each frame the pipeline:
  1. Rotates the full frame by -roll around its center.
  2. Crops a (face_w × output_size) rect centered at (cx, cy) using the same
     two-boundary integer rounding as crop_export._extract_crop_stretch.
  3. Stretches to output_size × output_size.

Fail frames (no face detected) are written as black frames so the output
video stays in sync with the source frame indices.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

import cv2
import numpy as np

from face_processing.geometry import rotate_landmarks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def cut_face_video(
    framedata_path: str,
    video_path: str,
    output_path: str,
    output_size: int | None = None,
    fps: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_timeout: int = 300,
) -> int:
    """Extract face crops from *video_path* and encode to *output_path*.

    Returns the output_size (S) that was used.
    """
    with open(framedata_path) as f:
        framedata = json.load(f)

    frames_by_idx: dict[int, dict] = {fr["i"]: fr for fr in framedata["frames"]}
    total = framedata["total_frames"]

    # Compute output_size = floor(min face height) across valid frames,
    # rounded down to even — same logic as crop_export.compute_output_size.
    if output_size is None:
        valid_heights = [fr["h"] for fr in framedata["frames"] if "status" not in fr]
        if not valid_heights:
            raise ValueError(f"No valid frames in {framedata_path}")
        S = max(2, int(min(valid_heights)))
        if S % 2:
            S -= 1
    else:
        S = output_size

    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    out_fps = fps if fps is not None else src_fps

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{S}x{S}", "-r", str(out_fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-b:v", "1M",
        "-pix_fmt", "yuv420p", "-an",
        output_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    black = np.zeros((S, S, 3), dtype=np.uint8)

    try:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            meta = frames_by_idx.get(frame_idx)
            if meta is None or "status" in meta:
                proc.stdin.write(black.tobytes())
                continue

            roll: float = meta.get("sroll", meta["roll"])
            cx: float = meta.get("scx",  meta["cx"])
            cy: float = meta.get("scy",  meta["cy"])
            w: float = meta.get("sw",   meta["w"])

            # Rotate full frame by -roll
            M = cv2.getRotationMatrix2D(
                (frame_w / 2.0, frame_h / 2.0), -roll, 1.0,
            )
            rotated = cv2.warpAffine(
                frame, M, (frame_w, frame_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )

            # Two-boundary rounding — matches crop_export._extract_crop_stretch
            x1 = int(round(cx - w / 2))
            x2 = int(round(cx + w / 2))
            y1 = int(round(cy - S / 2.0))
            y2 = int(round(cy + S / 2.0))

            crop = _safe_crop(rotated, x1, y1, x2, y2, frame_w, frame_h)
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
            else:
                crop = black

            proc.stdin.write(crop.tobytes())

    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass

    try:
        proc.wait(timeout=ffmpeg_timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RuntimeError(f"ffmpeg timed out for {output_path}") from exc

    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.returncode != 0:
        logger.error("ffmpeg stderr: %s", stderr.decode())
        raise RuntimeError(
            f"ffmpeg failed (code {proc.returncode}):\n{stderr.decode()}"
        )

    logger.info("Cut face video (%dx%d, %d frames) -> %s", S, S, total, output_path)
    return S


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    sx1 = max(0, x1); sy1 = max(0, y1)
    sx2 = min(frame_w, x2); sy2 = min(frame_h, y2)
    if sx1 >= sx2 or sy1 >= sy2:
        return result

    dx1 = sx1 - x1; dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1); dy2 = dy1 + (sy2 - sy1)
    result[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="face-framedata-cut",
        description="Extract face crops from a normalized video using framedata JSON.",
    )
    parser.add_argument("--framedata", "-d", required=True,
                        help="Path to *_framedata.json produced by face-framedata")
    parser.add_argument("--normalized", "-n", required=True,
                        help="Normalized source video (keep with --keep-normalized)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output face video path (.mp4)")
    parser.add_argument("--output-size", type=int, default=None,
                        help="Override output square size S (default: min face height)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    S = cut_face_video(
        framedata_path=args.framedata,
        video_path=args.normalized,
        output_path=args.output,
        output_size=args.output_size,
    )
    print(f"\nDone: face video at {S}x{S} -> {args.output}")


if __name__ == "__main__":
    main()
