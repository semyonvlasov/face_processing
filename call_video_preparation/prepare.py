from __future__ import annotations

import argparse
import logging

from call_video_preparation.config import load_call_video_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="call-video-preparation",
        description="Prepare call video artifacts: compressed videos, faceclips, and restore JSON.",
    )
    parser.add_argument("--input", "-i", required=True, help="Source video")
    parser.add_argument("--output-dir", "-o", default="output", help="Root output directory")
    parser.add_argument("--config", "-c", default=None, help="Flat call-video-preparation YAML config")
    parser.add_argument("--roi-top", type=float, default=None, help="ROI top fraction")
    parser.add_argument("--roi-bottom", type=float, default=None, help="ROI bottom fraction")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for MediaPipe inference")
    parser.add_argument("--no-faceclip", action="store_true", help="Skip face clip generation")
    parser.add_argument("--keep-native", action="store_true", help="Keep native intermediate files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config, values = load_call_video_config(args.config)
    if args.gpu:
        config.detection.use_gpu = True

    roi_top = args.roi_top if args.roi_top is not None else float(values.get("roi_top", 0.1))
    roi_bottom = args.roi_bottom if args.roi_bottom is not None else float(values.get("roi_bottom", 0.4))
    produce_faceclip = not args.no_faceclip and bool(values.get("produce_faceclip", True))
    keep_native = args.keep_native or bool(values.get("keep_native", False))

    from face_framedata.prepare import prepare_and_analyze

    report = prepare_and_analyze(
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        produce_faceclip=produce_faceclip,
        keep_native=keep_native,
    )

    print("\nDone:")
    if "native_crop" in report:
        print(f"  native crop:      {report['native_crop']}")
        print(f"  native framedata: {report['native_framedata']}")
    print(f"  1080p video:      {report['1080p_video']}")
    print(f"  1080p framedata:  {report['1080p_framedata']}")
    print(f"  540p video:       {report['540p_video']}")
    print(f"  540p framedata:   {report['540p_framedata']}")
    if "1080p_face_video" in report:
        print(f"  1080p face:       {report['1080p_face_video']}")
    if "540p_face_video" in report:
        print(f"  540p face:        {report['540p_face_video']}")
    print(f"  Frames: {report['valid_frames']}/{report['total_frames']} valid")


if __name__ == "__main__":
    main()
