# Face Video Preprocessing Pipeline — Technical Specification

## 1. Goal

Build a preprocessing pipeline for raw talking-head FullHD videos.

The pipeline must:
- take an input video;
- normalize it to **25 fps** and **20 Mb/s**;
- detect and track a single face through the video;
- estimate head pose (`yaw`, `pitch`, `roll`);
- correct head tilt by rotating each frame by `-roll`;
- split the source into valid continuous face segments;
- export processed square face videos with a quality rank;
- or drop the whole video / a segment with a single primary reason.

The pipeline is intended for dataset preparation, where geometric stability of the face region matters more than preserving the full original frame.

---

## 2. Recommended stack

- **FFmpeg** for decode / re-encode / fps normalization / bitrate normalization.
- **MediaPipe Face Landmarker** for per-frame face landmarks and face presence in image/video processing.
- **OpenCV** for:
  - head pose estimation via `solvePnP`,
  - per-frame roll correction via `getRotationMatrix2D` + `warpAffine`,
  - crop geometry utilities.

---

## 3. Scope

### In scope
- single-input video processing;
- face-based frame quality filtering;
- segmentation into good continuous subsequences;
- export of per-segment square face videos;
- export of rank and drop reasons;
- drop if more than one face is present;
- drop or split around face loss, tracking instability, motion jumps, size jumps, excessive head rotation/tilt, or low face confidence.

### Out of scope
- audio-based filtering;
- identity clustering;
- choosing the “main face” among multiple faces;
- face restoration / super-resolution;
- inpainting overlays over the face;
- occlusion removal.

---

## 4. Input

### 4.1 Supported formats
Input video formats:
- `.mp4`
- `.mov`
- `.mkv`

### 4.2 Assumptions
Expected content:
- one dominant human face;
- talking-head style framing;
- no need to preserve the original background;
- downstream consumer accepts square face videos.

---

## 5. Normalization stage

### 5.1 Intermediate normalized video
Every input video must first be converted into a normalized intermediate representation:

- codec: `H.264`
- frame rate: **25 fps**
- target video bitrate: **20 Mb/s**
- pixel format: `yuv420p`

### 5.2 Purpose
Normalization ensures that:
- thresholds are stable across all inputs;
- segment length filtering is deterministic;
- downstream exported videos share the same time base.

### 5.3 Example FFmpeg command
```bash
ffmpeg -y -i input.mp4 \
  -r 25 \
  -c:v libx264 -b:v 20M -pix_fmt yuv420p \
  -an \
  normalized.mp4
```

---

## 6. Face analysis stage

### 6.1 Per-frame outputs
For each frame, the pipeline must compute:

- `num_faces`
- `face_detected` flag
- face landmarks
- face bounding box
- face center `(cx, cy)`
- face width `face_w`
- face height `face_h`
- normalized face area ratio
- face confidence / tracking confidence
- optional facial transformation matrix
- `yaw`, `pitch`, `roll`
- pose validity flag
- reprojection error

### 6.2 Multi-face frame rule
If **more than one face** is detected on a frame, that frame must be marked as **bad** with reason:

- `multiple_faces`

Such frames must act as **segment boundaries**.

The source video must **not** be dropped solely because multiple faces appear on some frames.
The whole video is dropped only if, after segmentation and filtering, no exportable segment of at least 50 frames remains.

### 6.3 Face confidence / face integrity rule
The pipeline must evaluate confidence that the detected region is a clean readable face.

This rule is needed because overlays, captions, stickers, graphics, hands, or partial occlusions may overlap the face and make the landmarks unreliable even if a face is still nominally detected.

A frame must be marked **bad** if one or more of the following is true:
- face confidence / tracking confidence is below threshold;
- landmark fit is unstable or invalid;
- reprojection error is above threshold;
- too many required landmarks are missing or unreliable.

Primary reason:
- `low_face_confidence`

---

## 7. Head pose estimation

### 7.1 Method
Head pose must be estimated per frame from landmarks using a 2D-to-3D face model and `solvePnP`.

### 7.2 Output
For every valid frame:
- `yaw`
- `pitch`
- `roll`

### 7.3 Pose-based invalidity
A frame must be marked **bad** if head orientation exceeds allowed thresholds and facial motion quality becomes unsuitable.

Primary reason:
- `head_pose_extreme`

Recommended interpretation:
- extreme yaw = face turned too far left/right;
- extreme pitch = face tilted too far up/down;
- extreme roll = face slanted too much even before correction.

---

## 8. Roll correction

### 8.1 Rule
The pipeline must correct **only roll**, not yaw/pitch.

For each valid frame:
1. estimate `roll`;
2. rotate the original frame by `-roll`;
3. transform the face crop coordinates into the rotated frame;
4. perform crop extraction on the rotated frame.

### 8.2 Motivation
This preserves facial geometry better than aggressive affine normalization, while removing obvious head slant.

---

## 9. Per-frame quality metrics

For each frame, compute at least:

### 9.1 Face size
- `face_w`
- `face_h`
- `face_area_ratio`

### 9.2 Face position
- `cx`
- `cy`

### 9.3 Inter-frame deltas
Between frame `t-1` and `t`:
- `delta_cx`
- `delta_cy`
- `delta_face_w`
- `delta_face_h`
- `delta_yaw`
- `delta_pitch`
- `delta_roll`

### 9.4 Relative scale change
- `face_h_ratio = face_h_t / face_h_(t-1)`
- `face_w_ratio = face_w_t / face_w_(t-1)`

---

## 10. Bad-frame rules

A frame must be marked as **bad** if any of the following is true.

### 10.1 Face missing
- no face detected;
- pose invalid;
- landmarks invalid.

Reason:
- `face_missing_or_tracking_lost`

### 10.2 Face too small
Reason:
- `face_too_small`

Default starting threshold for FullHD:
- `face_h < 180 px`

### 10.3 Low confidence in detected face
Reason:
- `low_face_confidence`

Triggers may include:
- low detector/tracker confidence;
- excessive landmark instability;
- reprojection error above threshold;
- suspected overlays or occlusion over the face region.

### 10.4 Extreme head rotation / tilt
Reason:
- `head_pose_extreme`

Recommended starting per-frame hard thresholds:
- `abs(yaw) > 30°`
- `abs(pitch) > 22°`
- `abs(roll) > 20°`

### 10.5 Frame jump
Reason:
- `frame_jumps`

Recommended starting thresholds:
- `abs(delta_cx) > 0.08 * frame_width`
- `abs(delta_cy) > 0.08 * frame_height`
- `abs(delta_yaw) > 10°`
- `abs(delta_pitch) > 10°`
- `abs(delta_roll) > 12°`

### 10.6 Excessive face motion
Reason:
- `excessive_face_motion`

This is a softer motion criterion accumulated across a segment, not only a single-frame jump.
It is triggered when the face center moves too much over time even if no single jump exceeds the frame-jump threshold.

### 10.7 Strong face scale change
Reason:
- `strong_face_scale_change`

Recommended starting threshold:
- `abs(face_h_ratio - 1.0) > 0.12`
- `abs(face_w_ratio - 1.0) > 0.12`

This covers abrupt height or width changes between adjacent frames.

---

## 11. Good/bad sequence segmentation

### 11.1 Definitions
- **good frame**: a frame that passes all frame-level checks.
- **bad frame**: a frame that fails at least one check.

### 11.2 Segment splitting
The normalized video must be split into continuous runs of good frames.

A new segment boundary must be created whenever:
- face disappears;
- multiple faces appear;
- low face confidence occurs;
- head pose becomes extreme;
- frame jump occurs;
- abrupt scale jump occurs;
- tracking becomes invalid;
- a scene cut / discontinuity is detected.

Scene-cut handling may use FFmpeg scene-change metadata (for example via `scdet`) together with face-geometry discontinuity checks.

### 11.3 Short segment rule
Segments shorter than **50 frames** must not be exported.

At 25 fps this means segments shorter than **2.0 seconds** are discarded.

Reason:
- `segment_too_short`

---

## 12. Video-level drop rules

A source video must be dropped completely if any of the following is true:

1. **more than one face** appears on any frame  
   → `multiple_faces`

2. after segmentation, **no valid segment of at least 50 frames remains**  
   → `segment_too_short`

3. face is too small for most of the video and no exportable segment remains  
   → `face_too_small`

4. face confidence is too poor and no exportable segment remains  
   → `low_face_confidence`

5. head pose is too extreme for most of the video and no exportable segment remains  
   → `head_pose_extreme`

6. tracking repeatedly breaks and no exportable segment remains  
   → `face_missing_or_tracking_lost`

---

## 13. Primary drop reason priority

If several reasons are possible, the system must output **one primary reason** using this priority order:

1. `low_face_confidence`
2. `head_pose_extreme`
3. `face_missing_or_tracking_lost`
4. `frame_jumps`
5. `strong_face_scale_change`
6. `excessive_face_motion`
7. `face_too_small`
8. `multiple_faces`
9. `segment_too_short`

This applies both to whole-video drops and to segment-level rejection reporting.

---

## 14. Ranking exported segments

Each exported segment must receive one rank:

- `confident`
- `medium`
- `unconfident`

### 14.1 Aggregate metrics per segment
For each segment compute:
- `mean_abs_yaw`, `max_abs_yaw`
- `mean_abs_pitch`, `max_abs_pitch`
- `mean_abs_roll`, `max_abs_roll`
- `mean_face_h`, `min_face_h`
- `mean_face_w`, `min_face_w`
- `std_face_h / mean_face_h`
- `std_face_w / mean_face_w`
- `std_cx`, `std_cy`
- `jump_ratio`
- `missing_ratio`
- `low_conf_ratio`

### 14.2 Suggested initial ranking thresholds

#### confident
- `mean_abs_yaw <= 12°`
- `mean_abs_pitch <= 10°`
- `mean_abs_roll <= 8°`
- `max_abs_yaw <= 20°`
- `max_abs_pitch <= 16°`
- `max_abs_roll <= 12°`
- `std_face_h / mean_face_h <= 0.08`
- `std_face_w / mean_face_w <= 0.08`
- `std_cx <= 0.04 * output_reference_size`
- `std_cy <= 0.04 * output_reference_size`
- `jump_ratio <= 0.02`
- `low_conf_ratio <= 0.02`

#### medium
- `mean_abs_yaw <= 18°`
- `mean_abs_pitch <= 14°`
- `mean_abs_roll <= 12°`
- `max_abs_yaw <= 28°`
- `max_abs_pitch <= 20°`
- `max_abs_roll <= 18°`
- `std_face_h / mean_face_h <= 0.15`
- `std_face_w / mean_face_w <= 0.15`
- `std_cx <= 0.08 * output_reference_size`
- `std_cy <= 0.08 * output_reference_size`
- `jump_ratio <= 0.08`
- `low_conf_ratio <= 0.08`

#### unconfident
Exportable segment that does not satisfy `confident` or `medium`, but still passes minimal export criteria.

---

## 15. Output video geometry

### 15.1 Segment-level reference geometry
For each exportable segment, compute:

- `reference_crop_w = median(face_w_rotated_good_frames)`
- `reference_crop_h = median(face_h_rotated_good_frames)`
- `S = reference_crop_h`

Where:
- `face_w_rotated_good_frames` is the face width measured on the rotated, valid frames of that segment.
- `face_h_rotated_good_frames` is the face height measured on the rotated, valid frames of that segment.

The output segment resolution must be:

- `S x S`

### 15.2 Crop logic
For each valid frame in the segment:
1. use the roll-corrected frame;
2. define a `reference_crop_w x reference_crop_h` crop around the rotated face;
3. resize that reference rectangle to square output size `S x S`.

### 15.3 Required export mode
Per user requirement, the dataset pipeline must use **median face rectangle export**:

- the output is always square;
- side length = median rotated face height `S`;
- crop width = median rotated face width;
- crop height = median rotated face height.

This mode must be named:

- `median_face_rect`

### 15.4 Recommended optional mode
No alternate dataset export mode is required.

---

## 16. Export rules

### 16.1 Exportable segment
A segment is exportable if:
- it contains only good frames;
- its length is at least 50 frames.

### 16.2 Exported file
Each exportable segment must be saved as a separate video:
- codec: `H.264`
- fps: `25`
- bitrate: `20 Mb/s`
- resolution: `S x S`

### 16.3 Segment metadata
Each exported segment must have a sidecar JSON manifest.

Example:
```json
{
  "source_video": "input_001.mp4",
  "segment_id": 3,
  "status": "exported",
  "start_frame": 1240,
  "end_frame": 1689,
  "length_frames": 450,
  "rank": "medium",
  "drop_reason": null,
  "output_size": 312,
  "reference_crop_w": 296,
  "reference_crop_h": 312,
  "export_mode": "median_face_rect",
  "metrics": {
    "mean_abs_yaw": 9.8,
    "mean_abs_pitch": 6.2,
    "mean_abs_roll": 3.1,
    "max_abs_yaw": 19.4,
    "min_face_h": 312,
    "mean_face_h": 338,
    "face_size_std_ratio": 0.07,
    "jump_ratio": 0.01,
    "missing_ratio": 0.0,
    "low_conf_ratio": 0.0
  }
}
```

### 16.4 Whole-video drop report
If the whole source video is dropped, export:
```json
{
  "source_video": "input_001.mp4",
  "status": "dropped",
  "reason": "multiple_faces"
}
```

---

## 17. Segment rejection examples

### 17.1 Segment rejected due to length
A subsequence created after splitting must not be exported if:
- `length_frames < 50`

Reason:
- `segment_too_short`

### 17.2 Segment rejected due to extreme head pose
If a continuous subsequence exists but a large fraction of frames violates pose limits:
- either split around bad frames,
- or reject residual short pieces,
- using:
  - `head_pose_extreme`
  - or `segment_too_short` if nothing usable remains.

### 17.3 Segment rejected due to overlays / captions
If a talking-head video contains captions, stickers, logos, emojis, or graphics on top of the face and this significantly lowers face confidence / landmark validity:
- mark affected frames as bad;
- split around them;
- reject resulting too-short pieces if needed.

Reason:
- `low_face_confidence`

---

## 18. Acceptance criteria

The implementation is accepted if it satisfies all of the following:

1. every input video is normalized to **25 fps** and **20 Mb/s** before analysis;
2. frames containing more than one face are marked bad with `multiple_faces` and split segments rather than immediately dropping the whole video;
3. bad frames are identified using at least:
   - face missing,
   - face too small,
   - low face confidence,
   - extreme head pose,
   - frame jumps,
   - strong face scale changes;
4. the video is split into continuous good segments;
5. segments shorter than **50 frames** are never exported;
6. exported segments are rotated by `-roll` before crop;
7. exported videos are square with side length equal to the **median rotated face height** of the segment;
8. export mode `median_face_rect` is supported;
9. each exported segment has rank `confident`, `medium`, or `unconfident`;
10. each dropped video has exactly one primary reason.

---

## 19. Notes for implementation

### 19.1 Threshold tuning
All numeric thresholds in this document are **initial defaults** and must be validated on a representative subset of the target dataset.

### 19.2 Logging
For debugging and threshold tuning, the pipeline should optionally save per-frame CSV/Parquet metrics:
- frame index
- face present
- confidence
- bbox
- yaw/pitch/roll
- deltas
- bad-frame reason

### 19.3 Recommended future extensions
Potential later additions:
- blur / sharpness filter;
- mouth-visibility filter;
- eye-visibility filter;
- occlusion classifier;
- face quality model learned from labeled examples.

---

## 20. Short implementation summary

1. Normalize input video to `25 fps`, `20 Mb/s`, `H.264`.
2. Run face landmark detection on each frame.
3. If more than one face appears on any frame, drop video.
4. Estimate head pose (`yaw/pitch/roll`) from landmarks.
5. Rotate each valid frame by `-roll`.
6. Mark frames bad if:
   - face missing,
   - face too small,
   - low confidence,
   - extreme pose,
   - frame jump,
   - strong zoom in/out.
7. Split into continuous good segments.
8. Drop segments shorter than `50` frames.
9. Export each remaining segment as square face video with side `S = min(face_h_rotated_good_frames)`.
10. Assign `confident` / `medium` / `unconfident`.
11. Save JSON metadata.
