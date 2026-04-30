"""Extract per-frame face crops from a normalized video using framedata JSON.

For each frame the pipeline:
  1. Rotates the full frame by -roll around its center.
  2. Extracts a fixed median(sw) × median(sh) rectangle centered at (cx, cy).
  3. Resizes that rectangle to the requested S×S face clip.

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

    ref_w, ref_h = _reference_crop_size(framedata["frames"], framedata_path)
    S = output_size if output_size is not None else ref_h

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

            # Rotate full frame by -roll
            M = cv2.getRotationMatrix2D(
                (frame_w / 2.0, frame_h / 2.0), -roll, 1.0,
            )
            rotated = cv2.warpAffine(
                frame, M, (frame_w, frame_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )

            patch = cv2.getRectSubPix(rotated, (ref_w, ref_h), (cx, cy))
            crop = cv2.resize(patch, (S, S), interpolation=cv2.INTER_LINEAR)

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

    logger.info(
        "Cut face video (%dx%d patch -> %dx%d, %d frames) -> %s",
        ref_w, ref_h, S, S, total, output_path,
    )
    return S


# ---------------------------------------------------------------------------
# Native-resolution face clip extraction (two sizes, one pass)
# ---------------------------------------------------------------------------

def cut_face_clips_from_native(
    native_framedata_path: str,
    native_video_path: str,
    output_hd_path: str,
    output_sd_path: str,
    hd_size: int = 192,
    sd_size: int = 96,
    fps: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_timeout: int = 300,
) -> None:
    """Cut face clips from native-resolution video producing two sizes in one pass.

    Uses a single reference crop rectangle: even-rounded median(sw) ×
    median(sh). Both output clips are produced in a single read of the source
    video.

    output_hd_path: hd_size × hd_size clip (default 192×192)
    output_sd_path: sd_size × sd_size clip (default 96×96)
    """
    with open(native_framedata_path) as f:
        framedata = json.load(f)

    frames_by_idx: dict[int, dict] = {fr["i"]: fr for fr in framedata["frames"]}
    total = framedata["total_frames"]

    ref_w, ref_h = _reference_crop_size(framedata["frames"], native_framedata_path)

    cap = cv2.VideoCapture(native_video_path)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    out_fps = fps if fps is not None else src_fps

    def _make_pipe(out_path: str, size: int) -> subprocess.Popen:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{size}x{size}", "-r", str(out_fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-b:v", "1M",
            "-pix_fmt", "yuv420p", "-an",
            out_path,
        ]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    proc_hd = _make_pipe(output_hd_path, hd_size)
    proc_sd = _make_pipe(output_sd_path, sd_size)

    black_hd = np.zeros((hd_size, hd_size, 3), dtype=np.uint8)
    black_sd = np.zeros((sd_size, sd_size, 3), dtype=np.uint8)

    try:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            meta = frames_by_idx.get(frame_idx)
            if meta is None or "status" in meta:
                proc_hd.stdin.write(black_hd.tobytes())
                proc_sd.stdin.write(black_sd.tobytes())
                continue

            roll: float = float(meta.get("sroll", meta["roll"]))
            cx: float = float(meta.get("scx", meta["cx"]))
            cy: float = float(meta.get("scy", meta["cy"]))

            M = cv2.getRotationMatrix2D(
                (native_w / 2.0, native_h / 2.0), -roll, 1.0,
            )
            rotated = cv2.warpAffine(
                frame, M, (native_w, native_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )

            patch = cv2.getRectSubPix(rotated, (ref_w, ref_h), (cx, cy))

            proc_hd.stdin.write(
                cv2.resize(patch, (hd_size, hd_size), interpolation=cv2.INTER_LINEAR).tobytes()
            )
            proc_sd.stdin.write(
                cv2.resize(patch, (sd_size, sd_size), interpolation=cv2.INTER_LINEAR).tobytes()
            )

    except BrokenPipeError:
        pass
    finally:
        cap.release()
        for proc in (proc_hd, proc_sd):
            try:
                proc.stdin.close()
            except OSError:
                pass

    errors: list[str] = []
    for proc, out_path in ((proc_hd, output_hd_path), (proc_sd, output_sd_path)):
        try:
            proc.wait(timeout=ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"ffmpeg timed out for {out_path}")
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.returncode != 0:
            errors.append(f"{out_path}: {stderr.decode()}")
    if errors:
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(errors))

    logger.info(
        "Cut native face clips (%dx%d patch → %dx%d + %dx%d, %d frames)",
        ref_w, ref_h, hd_size, hd_size, sd_size, sd_size, total,
    )


def _reference_crop_size(frames: list[dict], source: str) -> tuple[int, int]:
    widths = [float(fr.get("sw", fr["w"])) for fr in frames if "status" not in fr]
    heights = [float(fr.get("sh", fr["h"])) for fr in frames if "status" not in fr]
    if not widths or not heights:
        raise ValueError(f"No valid frames in {source}")
    return _even_round(float(np.median(widths))), _even_round(float(np.median(heights)))


def _even_round(value: float) -> int:
    return max(2, int(round(value / 2.0) * 2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="call-video-cut",
        description="Extract face crops from a normalized video using framedata JSON.",
    )
    parser.add_argument("--framedata", "-d", required=True,
                        help="Path to framedata JSON produced by call-video-preparation")
    parser.add_argument("--normalized", "-n", required=True,
                        help="Normalized source video (keep with --keep-normalized)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output face video path (.mp4)")
    parser.add_argument("--output-size", type=int, default=None,
                        help="Override output square size S (default: median smoothed face height)")
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
