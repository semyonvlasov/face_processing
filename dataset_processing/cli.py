from __future__ import annotations

import argparse
import logging

from dataset_processing.config import load_dataset_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-processing",
        description="Dataset faceclip export pipeline.",
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--config", "-c", default=None, help="Flat dataset-processing YAML config")
    parser.add_argument("--save-frame-log", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for MediaPipe inference")
    parser.add_argument("--keep-normalized", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_dataset_config(args.config)
    config.output_dir = args.output_dir
    if args.save_frame_log:
        config.save_frame_log = True
    if args.gpu:
        config.detection.use_gpu = True
    if args.keep_normalized:
        config.keep_normalized = True

    from dataset_processing.pipeline import process_video

    result = process_video(args.input, config)
    if result.status == "dropped":
        print(f"\nVideo DROPPED: {result.drop_reason}")
        return

    exported = [s for s in result.segments if s.status == "exported"]
    print(f"\nVideo processed: {len(exported)} segments exported")
    for segment in exported:
        print(
            f"  Segment {segment.segment_id}: {segment.length} frames, "
            f"rank={segment.rank}, size={segment.output_size}, "
            f"ref={segment.reference_crop_w}x{segment.reference_crop_h}"
        )


if __name__ == "__main__":
    main()
