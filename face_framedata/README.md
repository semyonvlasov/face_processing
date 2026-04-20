# face_framedata

Compact per-frame geometry pipeline for face crop / restore workflows.

Produces a single `*_framedata.json` per video instead of encoded faceclip
segments.  Downstream tools (lipsync, etc.) use the JSON to extract exactly
the face region needed, run inference, then warp the result back.

## Pipeline stages

### Prepare pipeline (large square clips → 1080p + 540p)

For clips that arrive as large squares (e.g. 3800×3800), use the all-in-one
prepare command:

```
face-framedata-prepare  →  {stem}_1080p.mp4  +  {stem}_1080p_framedata.json
                        →  {stem}_540p.mp4   +  {stem}_540p_framedata.json
```

What it does internally:
1. Validates source is at least 1080 wide and 1920 tall.
2. Center-crops to 9:16 aspect ratio, scales to 1080×1920 at 25 fps / 8 Mbps.
3. Analyzes faces on the 1080p video with ROI [0.1, 0.4] (default) and writes framedata.
4. Scales 1080p → 540×960 at 4 Mbps.
5. Derives 540p framedata by halving all pixel coordinates (cx, cy, w, h).

### Base pipeline (already-normalized video)

```
face-framedata          →  *_framedata.json + normalized.mp4
    ↓
face-framedata-cut      →  *_face.mp4  (S×S square crops, one frame per source frame)
    ↓
<inference>             →  *_face_processed.mp4  (replace face content)
    ↓
face-framedata-restore  →  *_restored.mp4  (warp processed face back into source)
```

Geometry is smoothed with a 5-frame centred moving average; both raw
(`roll/cx/cy/w/h`) and smoothed (`sroll/scx/scy/sw/sh`) values are stored.
Cut and restore both use the smoothed values, so there is no inter-frame
jitter from the geometry source.

---

## Docker

### Build

```bash
git clone https://github.com/semyonvlasov/face_processing.git
cd face_processing
docker build -f docker/Dockerfile.framedata -t face-framedata:latest .
```

The MediaPipe model (`face_landmarker_v2_with_blendshapes.task`) is downloaded
automatically during the build — no manual asset setup needed.

### Run — prepare pipeline (large square clip → 1080p + 540p)

One command, all outputs:

```bash
docker run --rm \
  -v /path/to/your/videos:/data \
  face-framedata:latest \
  face-framedata-prepare \
    --input      /data/clip_3800x3800.mp4 \
    --output-dir /data/output
```

Output in `/data/output/<stem>/`:
```
<stem>_1080p.mp4               — 1080×1920, 25 fps, 8 Mbps
<stem>_1080p_framedata.json    — per-frame geometry for 1080p
<stem>_540p.mp4                — 540×960,  25 fps, 4 Mbps
<stem>_540p_framedata.json     — per-frame geometry for 540p (coords ÷ 2)
```

Optional flags:
```
--roi-top    0.1   vertical ROI start (fraction of frame height)
--roi-bottom 0.4   vertical ROI end   (face must be in this band)
--gpu              use GPU (Metal) for MediaPipe
--verbose / -v
```

### Run — base pipeline (already-normalized video)

Mount a local directory as `/data` and run each stage:

```bash
# 1. Analyze — produces /data/output/<stem>/<stem>_framedata.json + normalized.mp4
docker run --rm \
  -v /path/to/your/videos:/data \
  face-framedata:latest \
  face-framedata \
    --input  /data/input.mp4 \
    --output-dir /data/output \
    --keep-normalized

# 2. Cut face crops — produces /data/output/<stem>/<stem>_face.mp4
docker run --rm \
  -v /path/to/your/videos:/data \
  face-framedata:latest \
  face-framedata-cut \
    --framedata  /data/output/<stem>/<stem>_framedata.json \
    --normalized /data/output/<stem>/normalized.mp4 \
    --output     /data/output/<stem>/<stem>_face.mp4

# 3. <run your inference on <stem>_face.mp4 — produces <stem>_face_processed.mp4>

# 4. Restore — warps processed face back into source
docker run --rm \
  -v /path/to/your/videos:/data \
  face-framedata:latest \
  face-framedata-restore \
    --framedata  /data/output/<stem>/<stem>_framedata.json \
    --face-video /data/output/<stem>/<stem>_face_processed.mp4 \
    --normalized /data/output/<stem>/normalized.mp4 \
    --output     /data/output/<stem>/<stem>_restored.mp4
```

### Run — interactive shell

```bash
docker run --rm -it \
  -v /path/to/your/videos:/data \
  face-framedata:latest \
  bash
```

All CLI commands are on PATH inside the container:
`face-framedata-prepare`, `face-framedata`, `face-framedata-cut`, `face-framedata-restore`.

### CLI flags

**`face-framedata-prepare`**
```
--input / -i       Large source video (≥1080 wide, ≥1920 tall)
--output-dir / -o  Root output directory (default: output/)
--roi-top          ROI top fraction for face detection (default: 0.1)
--roi-bottom       ROI bottom fraction for face detection (default: 0.4)
--gpu              Use GPU (Metal) for MediaPipe inference
--verbose / -v     Debug logging
```

**`face-framedata`**
```
--input / -i       Source video (any format ffmpeg can read)
--output-dir / -o  Root output directory  (default: output)
--keep-normalized  Keep normalized.mp4 alongside framedata JSON
--verbose / -v     Debug logging
```

**`face-framedata-cut`**
```
--framedata / -d   Path to *_framedata.json
--normalized / -n  Normalized source video
--output / -o      Output face video path (.mp4)
--output-size      Override output square size S (default: min face height, even)
--verbose / -v
```

**`face-framedata-restore`**
```
--framedata  / -d  Path to *_framedata.json
--face-video / -f  Processed face video (S×S, same frame count as source)
--normalized / -n  Normalized source video
--output / -o      Output restored video path (.mp4)
--verbose / -v
```

---

## Local dev / testing

### Requirements

```bash
pip install -e .
```

The mediapipe model is expected at `assets/face_landmarker_v2_with_blendshapes.task`
(repo root).

### End-to-end test

Place source videos in `test_sample/` (or reuse a previous run's
`output_test/*/normalized.mp4`):

```bash
python tests/test_framedata_pipeline.py
```

Artifacts land in `output_test_framedata/<label>/<stem>/`:
- `<stem>_framedata.json` — per-frame geometry (raw + smoothed)
- `normalized.mp4` — 25 fps re-encoded source
- `<stem>_face.mp4` — S×S face crop video
- `<stem>_restored.mp4` — face warped back into source

Expected output:
```
=== portrait_avatar (portrait_avatar.mp4) ===
  [PASS] framedata JSON exists
  [PASS] total_frames=…  valid frames=…
  [PASS] face video exists  output_size=… (positive even)
  [PASS] restored video exists

[PASS] All checks passed
```

### Inspect framedata JSON

```python
import json

d = json.load(open("output_test_framedata/portrait_avatar/.../..._framedata.json"))
f = d["frames"][0]
print("raw:    ", f["roll"], f["cx"], f["cy"], f["w"], f["h"])
print("smooth: ", f["sroll"], f["scx"], f["scy"], f["sw"], f["sh"])
```

---

## framedata JSON format

```json
{
  "source_video": "input.mp4",
  "total_frames": 1250,
  "frames": [
    {
      "i": 0,
      "roll": 2.345678901234,  "cx": 320.5012, "cy": 240.3456, "w": 280.01, "h": 350.09,
      "sroll": 2.312345678901, "scx": 320.491, "scy": 240.321, "sw": 280.05, "sh": 350.11
    },
    { "i": 1, "status": "fail" },
    ...
  ]
}
```

- Valid frame: `i`, raw geometry (`roll/cx/cy/w/h`), smoothed geometry (`sroll/scx/scy/sw/sh`)
- Fail frame (no face detected): `i` + `"status": "fail"`
- All float values are full float64 precision — no rounding
- Smoothed fields use a 5-frame centred moving average over valid frames only
- Restore interpolates fail frames linearly between nearest valid neighbours
