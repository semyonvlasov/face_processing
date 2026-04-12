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
    export_mode = seg_meta.get("export_mode", "stretch_to_square")

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
            cx = row["cx"]
            cy = row["cy"]
            face_w_orig = row["face_w"]

            # Resize face crop back to the crop dimensions before stretch
            if export_mode == "stretch_to_square":
                # Was: crop (face_w x S) then stretched to (S x S)
                # Reverse: un-stretch from (S x S) to (face_w_rot x S)
                face_w_rot = _estimate_rotated_face_w(row, frame_w, frame_h, roll)
                crop_w = max(1, int(round(face_w_rot)))
                crop_h = S
                unstretched = cv2.resize(face_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            else:
                unstretched = face_crop
                crop_w = S
                crop_h = S

            # Compute rotated face center (same math as crop_export)
            cx_rot, cy_rot = _rotate_point(cx, cy, -roll, frame_w, frame_h)

            # Paste into rotated frame
            rotated_frame = _rotate_frame(frame_orig, -roll, frame_w, frame_h)
            _paste_crop(rotated_frame, unstretched, cx_rot, cy_rot, crop_w, crop_h)

            # Un-rotate back to original orientation
            restored = _rotate_frame(rotated_frame, roll, frame_w, frame_h)

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
                "bbox_x1": int(r["bbox_x1"]) if r["bbox_x1"] else 0,
                "bbox_y1": int(r["bbox_y1"]) if r["bbox_y1"] else 0,
                "bbox_x2": int(r["bbox_x2"]) if r["bbox_x2"] else 0,
                "bbox_y2": int(r["bbox_y2"]) if r["bbox_y2"] else 0,
            })
    return rows


def _estimate_rotated_face_w(
    row: dict, frame_w: int, frame_h: int, roll: float,
) -> float:
    """Estimate face width in the rotated frame from bbox corners."""
    x1, y1 = row["bbox_x1"], row["bbox_y1"]
    x2, y2 = row["bbox_x2"], row["bbox_y2"]
    # Rotate all 4 corners of the bbox
    corners = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float64)
    rotated = _rotate_points(corners, -roll, frame_w, frame_h)
    return float(np.max(rotated[:, 0]) - np.min(rotated[:, 0]))


def _rotate_point(
    x: float, y: float, angle_deg: float, frame_w: int, frame_h: int,
) -> tuple[float, float]:
    cx, cy = frame_w / 2.0, frame_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    pt = np.array([[x, y, 1.0]])
    rotated = (M @ pt.T).T
    return float(rotated[0, 0]), float(rotated[0, 1])


def _rotate_points(
    pts: np.ndarray, angle_deg: float, frame_w: int, frame_h: int,
) -> np.ndarray:
    cx, cy = frame_w / 2.0, frame_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    return (M @ pts_h.T).T


def _rotate_frame(
    frame: np.ndarray, angle_deg: float, frame_w: int, frame_h: int,
) -> np.ndarray:
    center = (frame_w / 2.0, frame_h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        frame, M, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def _paste_crop(
    canvas: np.ndarray,
    crop: np.ndarray,
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
) -> None:
    """Paste crop centered at (cx, cy) onto canvas (mutates in place)."""
    h_canvas, w_canvas = canvas.shape[:2]
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Source region in crop
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = crop_w - max(0, x2 - w_canvas)
    sy2 = crop_h - max(0, y2 - h_canvas)

    # Dest region in canvas
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(w_canvas, x2)
    dy2 = min(h_canvas, y2)

    if dx1 < dx2 and dy1 < dy2 and sx1 < sx2 and sy1 < sy2:
        canvas[dy1:dy2, dx1:dx2] = crop[sy1:sy2, sx1:sx2]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="face-processing-restore",
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
