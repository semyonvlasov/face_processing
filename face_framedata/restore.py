"""Restore a processed face video back into the normalized source video.

For each frame:
  - Valid frames: use geometry (roll, cx, cy) from the framedata JSON.
  - Fail frames (no face detected during analysis): geometry is linearly
    interpolated between the nearest valid neighbours so the face region
    is still blended back rather than left as the original.

The inverse transform mirrors cut.py exactly:
  1. The face video contains S×S frames resized from median(sw)×median(sh).
  2. Resize S×S back to that reference rectangle.
  3. Rotate corners back by +roll → original frame coordinates.
  4. warpAffine + feathered alpha-blend.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

import cv2
import numpy as np

from face_processing.restore import make_feather_mask, warp_face_into_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def restore_video(
    framedata_path: str,
    face_video_path: str,
    normalized_path: str,
    output_path: str,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_timeout: int = 300,
) -> None:
    """Warp each face-crop frame back into the normalized source video."""
    with open(framedata_path) as f:
        framedata = json.load(f)

    total = framedata["total_frames"]
    frames_list: list[dict] = framedata["frames"]

    widths = [float(fr.get("sw", fr["w"])) for fr in frames_list if "status" not in fr]
    heights = [float(fr.get("sh", fr["h"])) for fr in frames_list if "status" not in fr]
    if not widths or not heights:
        raise ValueError(f"No valid frames in {framedata_path}")
    ref_w = max(2, int(round(float(np.median(widths)) / 2.0) * 2))
    ref_h = max(2, int(round(float(np.median(heights)) / 2.0) * 2))

    # Build per-frame geometry arrays with linear interpolation for fail frames.
    geom = _build_interpolated_geometry(frames_list, total)
    # geom[i] = (roll, cx, cy, w, h) or None when there are no valid frames at all

    cap_norm = cv2.VideoCapture(normalized_path)
    cap_face = cv2.VideoCapture(face_video_path)

    frame_w = int(cap_norm.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_norm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_norm.get(cv2.CAP_PROP_FPS)) or 25
    # S is inferred from the face video dimensions (it's always S×S square)
    S = int(cap_face.get(cv2.CAP_PROP_FRAME_WIDTH))

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-b:v", "4M",
        "-pix_fmt", "yuv420p", "-an",
        output_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        for frame_idx in range(total):
            ret_norm, frame_orig = cap_norm.read()
            ret_face, face_crop = cap_face.read()

            if not ret_norm:
                break

            g = geom[frame_idx] if geom is not None else None
            if not ret_face or g is None:
                # No face at all — pass through unchanged
                proc.stdin.write(frame_orig.tobytes())
                continue

            roll, cx, cy, w, h = g

            # Undo cut: resize S×S → median(sw) × median(sh), then warp.
            unscaled = cv2.resize(face_crop, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            restored = warp_face_into_frame(
                frame_orig, unscaled, roll,
                ref_w, ref_h, frame_w, frame_h,
                cx, cy,
            )
            proc.stdin.write(restored.tobytes())

    except BrokenPipeError:
        pass
    finally:
        cap_norm.release()
        cap_face.release()
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

    logger.info("Restored video -> %s", output_path)


# ---------------------------------------------------------------------------
# Geometry interpolation
# ---------------------------------------------------------------------------

def _build_interpolated_geometry(
    frames_list: list[dict],
    total: int,
) -> list[tuple[float, float, float, float, float] | None] | None:
    """Return per-frame (roll, cx, cy, w, h), linearly interpolating fail frames.

    Prefers smoothed fields (sroll, scx, scy, sw, sh) when present, falls back
    to raw fields (roll, cx, cy, w, h).  Returns None if there are no valid frames.
    """
    def _pick(fr: dict, raw: str, smooth: str) -> float:
        return float(fr[smooth]) if smooth in fr else float(fr[raw])

    valid = [
        (
            fr["i"],
            _pick(fr, "roll", "sroll"),
            _pick(fr, "cx", "scx"),
            _pick(fr, "cy", "scy"),
            _pick(fr, "w", "sw"),
            _pick(fr, "h", "sh"),
        )
        for fr in frames_list if "status" not in fr
    ]

    if not valid:
        return None

    all_idx = np.arange(total, dtype=np.float64)
    v_idx = np.array([v[0] for v in valid], dtype=np.float64)

    interp_fields = []
    for col in range(1, 6):  # roll, cx, cy, w, h
        vals = np.array([v[col] for v in valid], dtype=np.float64)
        interp_fields.append(np.interp(all_idx, v_idx, vals))

    result: list[tuple | None] = []
    for i in range(total):
        result.append(tuple(float(interp_fields[c][i]) for c in range(5)))  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="call-video-restore",
        description="Restore a processed face video back into the source using framedata JSON.",
    )
    parser.add_argument("--framedata", "-d", required=True,
                        help="Path to *_framedata.json")
    parser.add_argument("--face-video", "-f", required=True,
                        help="Processed face video (output of call-video-cut + inference)")
    parser.add_argument("--normalized", "-n", required=True,
                        help="Normalized source video")
    parser.add_argument("--output", "-o", required=True,
                        help="Output restored video path (.mp4)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    restore_video(
        framedata_path=args.framedata,
        face_video_path=args.face_video,
        normalized_path=args.normalized,
        output_path=args.output,
    )
    print(f"\nDone: restored -> {args.output}")


if __name__ == "__main__":
    main()
