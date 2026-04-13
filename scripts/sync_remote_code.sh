#!/bin/bash
set -euo pipefail

PORT="${PORT:-22}"
REMOTE="${REMOTE:-root@127.0.0.1}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/lipsync_test/face_processing}"
REMOTE_GIT_URL="${REMOTE_GIT_URL:-}"
REMOTE_GIT_BRANCH="${REMOTE_GIT_BRANCH:-main}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT")
SSH_CMD=(ssh "${SSH_OPTS[@]}")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

origin_url="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
if [ -z "$REMOTE_GIT_URL" ]; then
  REMOTE_GIT_URL="$origin_url"
fi

to_https_url() {
  local url="$1"
  if [[ "$url" =~ ^git@github.com:(.+)$ ]]; then
    echo "https://github.com/${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$url" =~ ^ssh://git@github.com/(.+)$ ]]; then
    echo "https://github.com/${BASH_REMATCH[1]}"
    return 0
  fi
  echo "$url"
}

REMOTE_GIT_URL="$(to_https_url "$REMOTE_GIT_URL")"

echo "[sync] Remote: $REMOTE:$REMOTE_ROOT"
echo "[sync] Git URL: $REMOTE_GIT_URL"
echo "[sync] Git branch: $REMOTE_GIT_BRANCH"

"${SSH_CMD[@]}" "$REMOTE" \
  "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_GIT_URL='$REMOTE_GIT_URL' REMOTE_GIT_BRANCH='$REMOTE_GIT_BRANCH' bash -s" <<'EOS'
set -euo pipefail

mkdir -p "$(dirname "$REMOTE_ROOT")"

if ! command -v git >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git ca-certificates
fi

if [ -d "$REMOTE_ROOT/.git" ]; then
  cd "$REMOTE_ROOT"
  git remote set-url origin "$REMOTE_GIT_URL"
  git fetch --depth 1 origin "$REMOTE_GIT_BRANCH"
  git checkout -B "$REMOTE_GIT_BRANCH" "origin/$REMOTE_GIT_BRANCH"
  git reset --hard "origin/$REMOTE_GIT_BRANCH"
else
  rm -rf "$REMOTE_ROOT"
  git clone --branch "$REMOTE_GIT_BRANCH" --depth 1 "$REMOTE_GIT_URL" "$REMOTE_ROOT"
fi
EOS

echo "[sync] Done."
