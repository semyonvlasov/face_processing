"""End-to-end test for the call_video_preparation internals.

Analogous to the manual dataset pipeline test:
  dataset-processing  →  dataset-processing-restore

This test runs:
  call-video-preparation internals  →  call-video-cut  →  call-video-restore

Usage:
    python tests/test_framedata_pipeline.py

The test passes when all output files are created and non-empty.
"""
from __future__ import annotations

import json
import os
import sys

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from face_processing.config import PipelineConfig
from face_framedata.pipeline import process_video_framedata
from face_framedata.cut import cut_face_video
from face_framedata.restore import restore_video


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_test_videos() -> list[tuple[str, str]]:
    """Return (path, label) pairs, preferring test_sample/ then output_test/ normalized."""
    candidates = [
        # Original source videos (if present)
        (os.path.join(_REPO_ROOT, "test_sample", "portrait_avatar.mp4"), "portrait_avatar"),
        (os.path.join(_REPO_ROOT, "test_sample", "portrait_rama.mp4"),   "portrait_rama"),
        # Fallback: normalized videos from a previous dataset-processing run
        (os.path.join(_REPO_ROOT, "output_test", "portrait_avatar", "normalized.mp4"), "portrait_avatar"),
        (os.path.join(_REPO_ROOT, "output_test", "portrait_rama",   "normalized.mp4"), "portrait_rama"),
    ]
    seen_labels: set[str] = set()
    result = []
    for path, label in candidates:
        if os.path.isfile(path) and label not in seen_labels:
            result.append((path, label))
            seen_labels.add(label)
    return result


TEST_VIDEOS = _find_test_videos()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    return condition


def run_one(input_path: str, label: str, out_dir: str) -> bool:
    video_out_dir = os.path.join(out_dir, label)
    os.makedirs(video_out_dir, exist_ok=True)
    ok = True

    print(f"\n=== {label} ({os.path.basename(input_path)}) ===")

    # ── Stage 1: analyze → framedata JSON ──────────────────────────────────
    config = PipelineConfig()
    config.output_dir = video_out_dir   # write directly into the label dir
    config.keep_normalized = True        # needed for cut + restore

    report = process_video_framedata(input_path, config)

    # process_video_framedata creates {output_dir}/{stem}/ — find it
    stem = os.path.splitext(os.path.basename(input_path))[0]
    inner_dir = os.path.join(video_out_dir, stem)

    framedata_path = os.path.join(inner_dir, f"{stem}_framedata.json")
    normalized_path = os.path.join(inner_dir, "normalized.mp4")

    ok &= check(os.path.isfile(framedata_path), f"framedata JSON exists: {framedata_path}")
    ok &= check(os.path.getsize(framedata_path) > 0, "framedata JSON non-empty")
    ok &= check(os.path.isfile(normalized_path), "normalized video kept")

    with open(framedata_path) as fh:
        fd = json.load(fh)
    total = fd["total_frames"]
    valid = sum(1 for f in fd["frames"] if "status" not in f)
    ok &= check(total > 0, f"total_frames={total}")
    ok &= check(valid > 0, f"valid frames={valid}")
    print(f"    report: {report}")

    if not ok:
        return False

    # ── Stage 2: cut → face video ───────────────────────────────────────────
    face_video_path = os.path.join(inner_dir, f"{stem}_face.mp4")
    S = cut_face_video(
        framedata_path=framedata_path,
        video_path=normalized_path,
        output_path=face_video_path,
    )

    ok &= check(os.path.isfile(face_video_path), f"face video exists: {face_video_path}")
    ok &= check(os.path.getsize(face_video_path) > 0, "face video non-empty")
    ok &= check(S > 0 and S % 2 == 0, f"output_size={S} (positive even)")
    print(f"    output_size={S}")

    if not ok:
        return False

    # ── Stage 3: restore → restored video ──────────────────────────────────
    restored_path = os.path.join(inner_dir, f"{stem}_restored.mp4")
    restore_video(
        framedata_path=framedata_path,
        face_video_path=face_video_path,
        normalized_path=normalized_path,
        output_path=restored_path,
    )

    ok &= check(os.path.isfile(restored_path), f"restored video exists: {restored_path}")
    ok &= check(os.path.getsize(restored_path) > 0, "restored video non-empty")

    return ok


def main() -> None:
    out_dir = os.path.join(_REPO_ROOT, "output_test_framedata")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    if not TEST_VIDEOS:
        print("[SKIP] No test videos found")
        return

    all_ok = True
    for path, label in TEST_VIDEOS:
        all_ok &= run_one(path, label, out_dir)

    print("\n" + "=" * 50)
    if all_ok:
        print(f"[{PASS}] All checks passed")
    else:
        print(f"[{FAIL}] Some checks failed — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
