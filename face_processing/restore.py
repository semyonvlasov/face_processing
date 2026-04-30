"""Restore face back into original (normalized) video frames.

Takes an exported face segment video + segment JSON + frame log CSV +
normalized source video, and produces a video where the processed face
is pasted back into the original frames.

Usage:
    python -m face_processing.restore \
        --segment-json output/vid/vid_seg_000.json \
        --face-video   output/vid/vid_seg_000.mp4 \
        --frame-log    output/vid/vid_frame_log.csv \
        --normalized   output/vid/normalized.mp4 \
        --output       output/vid/vid_seg_000_restored.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def restore_segment(
    segment_json_path: str,
    face_video_path: str,
    frame_log_path: str,
    normalized_video_path: str,
    output_path: str,
    source_audio_path: str | None = None,
) -> str:
    # Load segment metadata
    with open(segment_json_path) as f:
        seg_meta = json.load(f)

    start_frame = seg_meta["start_frame"]
    end_frame = seg_meta["end_frame"]
    length = seg_meta["length_frames"]
    S = seg_meta["output_size"]
    export_mode = seg_meta.get("export_mode", "median_face_rect")

    logger.info(
        "Restoring segment %d: frames %d-%d, size=%d, mode=%s",
        seg_meta["segment_id"], start_frame, end_frame, S, export_mode,
    )

    # Load per-frame data from CSV (only frames in this segment)
    frame_rows = _load_frame_rows(frame_log_path, start_frame, end_frame)

    # Open videos
    cap_norm = cv2.VideoCapture(normalized_video_path)
    frame_w = int(cap_norm.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_norm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_norm.get(cv2.CAP_PROP_FPS)) or 25

    cap_face = cv2.VideoCapture(face_video_path)

    # Seek to start of segment in normalized video
    cap_norm.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Setup ffmpeg output (with optional audio)
    audio_start = start_frame / fps
    audio_duration = length / fps

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if source_audio_path:
        cmd += [
            "-ss", f"{audio_start:.4f}",
            "-t", f"{audio_duration:.4f}",
            "-i", source_audio_path,
            "-c:v", "libx264", "-b:v", "20M", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
        ]
    else:
        cmd += [
            "-c:v", "libx264", "-b:v", "20M", "-pix_fmt", "yuv420p",
            "-an",
        ]
    cmd.append(output_path)

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        for i in range(length):
            ret_norm, frame_orig = cap_norm.read()
            ret_face, face_crop = cap_face.read()

            if not ret_norm:
                break

            if not ret_face or i >= len(frame_rows):
                # No face frame available, pass original through
                proc.stdin.write(frame_orig.tobytes())
                continue

            row = frame_rows[i]
            roll = row["roll"]
            cx_rot, cy_rot, crop_w_rot, crop_h_rot = _resolve_restore_geometry(row, S)

            if export_mode != "median_face_rect":
                raise RuntimeError(f"Unsupported dataset export_mode: {export_mode}")

            # Resize the square face video frame back to its exported reference rect.
            # Use the same two-boundary rounding as _extract_reference_crop so
            # that crop_w/crop_h match the pixel dimensions that were actually
            # exported.  Plain int(round(w)) can differ by ±1 from
            # round(cx+w/2)-round(cx-w/2) when cx has a fractional part near
            # 0.5, which causes a per-frame flip in the affine matrix → jitter.
            x1_rot = int(round(cx_rot - crop_w_rot / 2))
            x2_rot = int(round(cx_rot + crop_w_rot / 2))
            y1_rot = int(round(cy_rot - crop_h_rot / 2.0))
            y2_rot = int(round(cy_rot + crop_h_rot / 2.0))
            crop_w = max(1, x2_rot - x1_rot)
            crop_h = max(1, y2_rot - y1_rot)
            # Align center to the actual pixel-centre of the crop so the
            # affine corners land exactly on the exported pixel boundaries.
            cx_rot = (x1_rot + x2_rot) / 2.0
            cy_rot = (y1_rot + y2_rot) / 2.0
            face_rect = cv2.resize(face_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            # Warp face patch directly into original frame using inverse
            # of the export transform (no double-rotation of the full frame).
            restored = warp_face_into_frame(
                frame_orig, face_rect, roll,
                crop_w, crop_h, frame_w, frame_h, cx_rot, cy_rot,
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

    proc.wait()
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr.decode()}")

    logger.info("Restored segment -> %s", output_path)
    return output_path


def _load_frame_rows(
    csv_path: str, start_frame: int, end_frame: int,
) -> list[dict]:
    def _parse_optional_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            idx = int(r["frame_idx"])
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            rows.append({
                "frame_idx": idx,
                "roll": float(r["roll"]) if r["roll"] else 0.0,
                "cx": float(r["cx"]) if r["cx"] else 0.0,
                "cy": float(r["cy"]) if r["cy"] else 0.0,
                "face_w": float(r["face_w"]) if r["face_w"] else 0.0,
                "face_h": float(r["face_h"]) if r["face_h"] else 0.0,
                "raw_crop_cx_rot": _parse_optional_float(r.get("raw_crop_cx_rot")),
                "raw_crop_cy_rot": _parse_optional_float(r.get("raw_crop_cy_rot")),
                "raw_crop_w_rot": _parse_optional_float(r.get("raw_crop_w_rot")),
                "raw_crop_h_rot": _parse_optional_float(r.get("raw_crop_h_rot")),
                "crop_cx_rot": _parse_optional_float(r.get("crop_cx_rot")),
                "crop_cy_rot": _parse_optional_float(r.get("crop_cy_rot")),
                "crop_w_rot": _parse_optional_float(r.get("crop_w_rot")),
                "crop_h_rot": _parse_optional_float(r.get("crop_h_rot")),
                "stable_crop_cx_rot": _parse_optional_float(r.get("stable_crop_cx_rot")),
                "stable_crop_cy_rot": _parse_optional_float(r.get("stable_crop_cy_rot")),
                "stable_crop_w_rot": _parse_optional_float(r.get("stable_crop_w_rot")),
                "stable_crop_h_rot": _parse_optional_float(r.get("stable_crop_h_rot")),
            })
    return rows


def _resolve_restore_geometry(
    row: dict,
    output_size: int,
) -> tuple[float, float, float, float]:
    has_raw_columns = any(
        row.get(key) is not None
        for key in ("raw_crop_cx_rot", "raw_crop_cy_rot", "raw_crop_w_rot", "raw_crop_h_rot")
    )

    cx_rot = row["crop_cx_rot"]
    if cx_rot is None:
        cx_rot = row["stable_crop_cx_rot"]
    if cx_rot is None:
        cx_rot = row["raw_crop_cx_rot"]

    cy_rot = row["crop_cy_rot"]
    if cy_rot is None:
        cy_rot = row["stable_crop_cy_rot"]
    if cy_rot is None:
        cy_rot = row["raw_crop_cy_rot"]

    crop_w_rot = row["crop_w_rot"]
    if crop_w_rot is None:
        crop_w_rot = row["stable_crop_w_rot"]
    if crop_w_rot is None:
        crop_w_rot = row["raw_crop_w_rot"]

    crop_h_rot = row["crop_h_rot"]
    if crop_h_rot is None:
        crop_h_rot = row["stable_crop_h_rot"]
    if crop_h_rot is None:
        crop_h_rot = row["raw_crop_h_rot"]
    if crop_h_rot is None:
        crop_h_rot = float(output_size)

    # Backward compatibility for older logs where crop_* was raw and stable_* held the effective geometry.
    if not has_raw_columns and row.get("stable_crop_cx_rot") is not None and row.get("stable_crop_w_rot") is not None:
        cx_rot = float(row["stable_crop_cx_rot"])
        cy_rot = float(row["stable_crop_cy_rot"]) if row.get("stable_crop_cy_rot") is not None else float(cy_rot)
        crop_w_rot = float(row["stable_crop_w_rot"])
        crop_h_rot = (
            float(row["stable_crop_h_rot"])
            if row.get("stable_crop_h_rot") is not None
            else float(crop_h_rot)
        )

    if cx_rot is None or cy_rot is None or crop_w_rot is None:
        raise RuntimeError("Missing crop geometry required for restore")

    return float(cx_rot), float(cy_rot), float(crop_w_rot), float(crop_h_rot)




def warp_face_into_frame(
    frame_orig: np.ndarray,
    face_patch: np.ndarray,
    roll: float,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
    cx_rot: float,
    cy_rot: float,
) -> np.ndarray:
    """Warp face patch back into the original (non-rotated) frame.

    During export the pipeline did:
      1. Rotate frame by -roll around frame center
      2. Crop a (crop_w x crop_h) rect centered at (cx_rot, cy_rot)

    To reverse with a single affine warp of just the face patch:
      - Map 3 corners of the face patch to where they sit in the
        original frame by applying the INVERSE rotation (+roll).
    """
    fc = frame_w / 2.0, frame_h / 2.0

    # 3 corners of the face patch in patch-local coords
    #   top-left, top-right, bottom-left
    src_pts = np.array([
        [0, 0],
        [crop_w, 0],
        [0, crop_h],
    ], dtype=np.float32)

    # Same 3 corners in the rotated frame (centered on cx_rot, cy_rot)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    dst_rot = np.array([
        [cx_rot - half_w, cy_rot - half_h],
        [cx_rot + half_w, cy_rot - half_h],
        [cx_rot - half_w, cy_rot + half_h],
    ], dtype=np.float64)

    # Now rotate these points BACK by +roll to get positions in the original frame
    M_inv = cv2.getRotationMatrix2D(fc, roll, 1.0)
    ones = np.ones((3, 1), dtype=np.float64)
    dst_rot_h = np.hstack([dst_rot, ones])
    dst_orig = (M_inv @ dst_rot_h.T).T  # (3, 2)

    # Affine: patch coords → original frame coords
    M_warp = cv2.getAffineTransform(src_pts, dst_orig.astype(np.float32))

    # Warp face patch into full-frame size
    warped_face = cv2.warpAffine(
        face_patch, M_warp, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Feathered mask: solid white with smooth fade at edges
    feather = max(1, int(min(crop_w, crop_h) * 0.08))
    mask_patch = make_feather_mask(crop_w, crop_h, feather)
    mask_warped = cv2.warpAffine(
        mask_patch, M_warp, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    # Alpha-blend: result = orig * (1 - alpha) + face * alpha
    alpha = mask_warped[:, :, np.newaxis].astype(np.float32) / 255.0
    result = (
        frame_orig.astype(np.float32) * (1.0 - alpha)
        + warped_face.astype(np.float32) * alpha
    )
    return result.clip(0, 255).astype(np.uint8)


def make_feather_mask(w: int, h: int, feather: int) -> np.ndarray:
    """Create a single-channel mask: 255 inside, smooth linear fade at edges."""
    mask = np.ones((h, w), dtype=np.float32) * 255.0
    for i in range(feather):
        t = (i + 1) / (feather + 1)
        val = t * 255.0
        mask[i, :] = np.minimum(mask[i, :], val)        # top
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], val)  # bottom
        mask[:, i] = np.minimum(mask[:, i], val)         # left
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], val)  # right
    return mask.astype(np.uint8)




def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-processing-restore",
        description="Restore processed face back into original video frames.",
    )
    parser.add_argument("--segment-json", "-s", required=True, help="Segment metadata JSON")
    parser.add_argument("--face-video", "-f", required=True, help="Exported face segment video")
    parser.add_argument("--frame-log", "-l", required=True, help="Per-frame CSV log")
    parser.add_argument("--normalized", "-n", required=True, help="Normalized source video")
    parser.add_argument("--output", "-o", required=True, help="Output restored video path")
    parser.add_argument("--audio-source", "-a", default=None, help="Original video for audio track")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    restore_segment(
        segment_json_path=args.segment_json,
        face_video_path=args.face_video,
        frame_log_path=args.frame_log,
        normalized_video_path=args.normalized,
        output_path=args.output,
        source_audio_path=args.audio_source,
    )


if __name__ == "__main__":
    main()
