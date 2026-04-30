# call_video_preparation

Call-video preparation produces compressed call videos, face clips, and compact
framedata JSON for restore. The public commands are:

```bash
call-video-preparation --input clip.mp4 --output-dir output/
call-video-cut --framedata output/clip/clip_1080p_framedata.json --normalized output/clip/clip_1080p.mp4 --output output/clip/clip_face.mp4
call-video-restore --framedata output/clip/clip_1080p_framedata.json --face-video output/clip/clip_face_processed.mp4 --normalized output/clip/clip_1080p.mp4 --output output/clip/clip_restored.mp4
```

The underlying framedata stores raw geometry (`roll/cx/cy/w/h`) and smoothed
geometry (`sroll/scx/scy/sw/sh`). Cut and restore use a single reference crop
rectangle per clip:

```python
ref_w = even_round(median(sw))
ref_h = even_round(median(sh))
```

Failed face-detection frames remain in the timeline as `"status": "fail"`.
Restore interpolates geometry across those frames so frame indexes stay aligned.

Docker:

```bash
docker build -f docker/Dockerfile.call_video_preparation -t call-video-preparation:latest .
```
