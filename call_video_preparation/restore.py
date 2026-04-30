from __future__ import annotations

import argparse
import logging

from face_framedata.restore import restore_video


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="call-video-restore",
        description="Restore a processed call-video face crop back into the source video.",
    )
    parser.add_argument("--framedata", "-d", required=True, help="Path to framedata JSON")
    parser.add_argument("--face-video", "-f", required=True, help="Processed face video")
    parser.add_argument("--normalized", "-n", required=True, help="Normalized source video")
    parser.add_argument("--output", "-o", required=True, help="Output restored video path")
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


__all__ = ["main", "restore_video"]


if __name__ == "__main__":
    main()
