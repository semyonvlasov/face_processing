from __future__ import annotations

from face_processing.models import FrameData, Segment


def split_into_segments(
    frame_data: list[FrameData],
    min_length: int = 50,
) -> tuple[list[Segment], list[Segment]]:
    """Split frame_data into continuous runs of good frames.

    Returns:
        (exportable, dropped) — exportable segments have length >= min_length,
        dropped segments are too short.
    """
    segments: list[Segment] = []
    segment_id = 0
    run_start: int | None = None

    for i, fd in enumerate(frame_data):
        if not fd.is_bad:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                seg = Segment(
                    segment_id=segment_id,
                    start_frame=run_start,
                    end_frame=i,
                    length=i - run_start,
                    frame_data=frame_data[run_start:i],
                )
                segments.append(seg)
                segment_id += 1
                run_start = None

    # Handle trailing good run
    if run_start is not None:
        seg = Segment(
            segment_id=segment_id,
            start_frame=run_start,
            end_frame=len(frame_data),
            length=len(frame_data) - run_start,
            frame_data=frame_data[run_start:],
        )
        segments.append(seg)

    exportable: list[Segment] = []
    dropped: list[Segment] = []
    for seg in segments:
        if seg.length >= min_length:
            exportable.append(seg)
        else:
            seg.status = "dropped"
            seg.drop_reason = "segment_too_short"
            dropped.append(seg)

    return exportable, dropped
