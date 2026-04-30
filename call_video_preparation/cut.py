from __future__ import annotations

import argparse
import logging

from face_framedata.cut import cut_face_clips_from_native, cut_face_video


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="call-video-cut",
        description="Extract call-video face crops from framedata JSON.",
    )
    parser.add_argument("--framedata", "-d", required=True, help="Path to framedata JSON")
    parser.add_argument("--normalized", "-n", required=True, help="Normalized source video")
    parser.add_argument("--output", "-o", required=True, help="Output face video path")
    parser.add_argument("--output-size", type=int, default=None, help="Override output square size")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    size = cut_face_video(
        framedata_path=args.framedata,
        video_path=args.normalized,
        output_path=args.output,
        output_size=args.output_size,
    )
    print(f"\nDone: face video at {size}x{size} -> {args.output}")


__all__ = ["cut_face_clips_from_native", "cut_face_video", "main"]


if __name__ == "__main__":
    main()
