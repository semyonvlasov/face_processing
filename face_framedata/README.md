# face_framedata

Compact per-frame geometry pipeline for face crop / restore workflows.

Produces a single `*_framedata.json` per video instead of encoded faceclip
segments.  Downstream tools (lipsync, etc.) use the JSON to extract exactly
the face region needed, run inference, then warp the result back.

## Pipeline stages

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
# from repo root
docker build -f docker/Dockerfile.framedata -t face-framedata:latest .
```

### Run — single video, all stages

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

All three CLI commands are on PATH inside the container:
`face-framedata`, `face-framedata-cut`, `face-framedata-restore`.

### CLI flags

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
