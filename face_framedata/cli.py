from __future__ import annotations

import argparse
import logging

from face_processing.config import PipelineConfig
from face_framedata.pipeline import process_video_framedata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="face-framedata",
        description="Per-frame face geometry extraction pipeline.",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input video (.mp4, .mov, .mkv)")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to JSON config file to override defaults")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (Metal) for MediaPipe inference")
    parser.add_argument("--keep-normalized", action="store_true",
                        help="Keep normalized video after processing")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    config.output_dir = args.output_dir
    if args.gpu:
        config.detection.use_gpu = True
    if args.keep_normalized:
        config.keep_normalized = True

    report = process_video_framedata(args.input, config)
    print(
        f"\nDone: {report['valid_frames']} valid frames, "
        f"{report['fail_frames']} fail frames "
        f"(total {report['total_frames']})"
    )


if __name__ == "__main__":
    main()
