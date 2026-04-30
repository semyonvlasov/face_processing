#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dataset-processing-cpu:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-dataset-processing-cpu}"
DATA_ROOT="${DATA_ROOT:-$HOME/.cache/face_processing/process-docker}"
RCLONE_CONFIG_PATH="${RCLONE_CONFIG_PATH:-$HOME/.config/rclone/rclone.conf}"
CONFIG_PATH="${CONFIG_PATH:-configs/gdrive_container_cpu.yaml}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/assets/face_landmarker_v2_with_blendshapes.task}"
RCLONE_CONFIG_DIR="$(dirname "$RCLONE_CONFIG_PATH")"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [ ! -f "$RCLONE_CONFIG_PATH" ]; then
  echo "[docker-process] missing rclone config: $RCLONE_CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[docker-process] missing MediaPipe model: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/process/logs"

if [ "$SKIP_BUILD" != "1" ]; then
  docker build \
    -f "$REPO_ROOT/docker/Dockerfile.dataset_processing_cpu" \
    -t "$IMAGE_TAG" \
    "$REPO_ROOT"
fi

docker run --rm --init \
  --name "$CONTAINER_NAME" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO_ROOT:/workspace/repo:ro" \
  -v "$DATA_ROOT:/workspace-data" \
  -v "$RCLONE_CONFIG_DIR:/root/.config/rclone" \
  -w /workspace/repo \
  "$IMAGE_TAG" \
  python3 batch/gdrive_processor.py \
    "$CONFIG_PATH" \
    "$@" \
  2>&1 | tee -a "$DATA_ROOT/process/logs/raw_drive.log"
