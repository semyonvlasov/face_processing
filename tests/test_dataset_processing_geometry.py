from __future__ import annotations

import unittest

import numpy as np

from face_processing.config import BadFrameThresholds
from face_processing.crop_export import prepare_segment_crop_geometry
from face_processing.frame_quality import classify_frames, compute_deltas
from face_processing.models import FrameData, Segment


def _frame_with_box(frame_idx: int, w: float, h: float, frame_w: int = 1000, frame_h: int = 1000) -> FrameData:
    cx = 500.0
    cy = 500.0
    landmarks = np.zeros((478, 3), dtype=np.float64)
    landmarks[:, 0] = cx / frame_w
    landmarks[:, 1] = cy / frame_h
    landmarks[0, 0] = (cx - w / 2.0) / frame_w
    landmarks[1, 0] = (cx + w / 2.0) / frame_w
    landmarks[2, 1] = (cy - h / 2.0) / frame_h
    landmarks[3, 1] = (cy + h / 2.0) / frame_h
    return FrameData(
        frame_idx=frame_idx,
        num_faces=1,
        face_detected=True,
        confidence=1.0,
        face_w=w,
        face_h=h,
        cx=cx,
        cy=cy,
        landmarks=landmarks,
        pose_valid=True,
    )


class DatasetProcessingGeometryTests(unittest.TestCase):
    def test_segment_reference_crop_uses_segment_median_width_and_height(self) -> None:
        frames = [
            _frame_with_box(0, 100, 200),
            _frame_with_box(1, 200, 300),
            _frame_with_box(2, 300, 400),
        ]
        segment = Segment(segment_id=0, start_frame=0, end_frame=3, length=3, frame_data=frames)

        prepare_segment_crop_geometry(segment, frame_w=1000, frame_h=1000)

        self.assertEqual(segment.reference_crop_w, 200)
        self.assertEqual(segment.reference_crop_h, 300)
        self.assertEqual(segment.output_size, 300)
        self.assertTrue(all(fd.crop_w_rot == 200 for fd in frames))
        self.assertTrue(all(fd.crop_h_rot == 300 for fd in frames))

    def test_width_ratio_deviation_marks_bad_frame(self) -> None:
        frames = [
            _frame_with_box(0, 100, 200),
            _frame_with_box(1, 130, 200),
        ]
        thresholds = BadFrameThresholds(
            max_face_h_ratio_deviation=0.12,
            max_face_w_ratio_deviation=0.12,
        )

        compute_deltas(frames)
        classify_frames(frames, thresholds, frame_w=1000, frame_h=1000)

        self.assertFalse(frames[0].is_bad)
        self.assertTrue(frames[1].is_bad)
        self.assertIn("strong_face_scale_change", frames[1].bad_reasons)


if __name__ == "__main__":
    unittest.main()
