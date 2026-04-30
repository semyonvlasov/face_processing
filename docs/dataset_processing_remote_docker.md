# Dataset Processing Remote Docker Runbook

CPU-only dataset processor image:

```text
semyonvlasov/dataset-processing-cpu:latest
```

The image contains the code, MediaPipe model, and container config at:

```text
/workspace/repo/configs/gdrive_container_cpu.yaml
```

Only the working data directory and rclone credentials are mounted from the host.
The rclone config must be writable because Google Drive OAuth tokens can be
refreshed by rclone during long downloads/uploads.

## Start Remote From Local SSH

Use this from the local machine when the remote host is `192.168.1.34`:

```bash
ssh root@192.168.1.34 '
  set -e

  docker pull semyonvlasov/dataset-processing-cpu:latest

  mkdir -p /root/.cache/face_processing/dataset-processing/logs
  test -f /root/.config/rclone/rclone.conf

  docker run -d --rm --init \
    --name dataset-processing-cpu \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v /root/.cache/face_processing/dataset-processing:/workspace-data \
    -v /root/.config/rclone:/root/.config/rclone \
    semyonvlasov/dataset-processing-cpu:latest \
    python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml
'
```

Follow logs from the local machine:

```bash
ssh root@192.168.1.34 'docker logs -f dataset-processing-cpu'
```

Stop from the local machine:

```bash
ssh root@192.168.1.34 'docker stop dataset-processing-cpu'
```

## Start On Remote Host Manually

SSH into the host:

```bash
ssh root@192.168.1.34
```

Pull the current image:

```bash
docker pull semyonvlasov/dataset-processing-cpu:latest
```

Check the pulled platform:

```bash
docker image inspect semyonvlasov/dataset-processing-cpu:latest \
  --format '{{.Id}} {{.Architecture}} {{.Os}}'
```

Prepare persistent worker state:

```bash
mkdir -p /root/.cache/face_processing/dataset-processing/logs
```

Check that rclone credentials exist:

```bash
test -f /root/.config/rclone/rclone.conf && echo "rclone config ok"
```

Start the worker in the background:

```bash
docker run -d --rm --init \
  --name dataset-processing-cpu \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/.cache/face_processing/dataset-processing:/workspace-data \
  -v /root/.config/rclone:/root/.config/rclone \
  semyonvlasov/dataset-processing-cpu:latest \
  python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml
```

Watch logs:

```bash
docker logs -f dataset-processing-cpu
```

Check container status:

```bash
docker ps --filter name=dataset-processing-cpu
```

Stop the worker:

```bash
docker stop dataset-processing-cpu
```

The container is removed after stop because it is started with `--rm`.

## Start On Local Mac

Use host paths from the macOS user home. Do not use `/root/...` on macOS.

Pull the current image:

```bash
docker pull semyonvlasov/dataset-processing-cpu:latest
```

Check rclone config:

```bash
test -f "$HOME/.config/rclone/rclone.conf" && echo "rclone config ok"
```

Start one local worker:

```bash
mkdir -p "$HOME/.cache/face_processing/dataset-processing-1"

docker run -d --rm --init \
  --name dataset-processing-cpu-1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$HOME/.cache/face_processing/dataset-processing-1:/workspace-data" \
  -v "$HOME/.config/rclone:/root/.config/rclone" \
  semyonvlasov/dataset-processing-cpu:latest \
  python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml
```

Start a second local worker with isolated state and an isolated rclone config copy:

```bash
mkdir -p \
  "$HOME/.cache/face_processing/dataset-processing-2" \
  "$HOME/.cache/face_processing/rclone-worker-2"

cp "$HOME/.config/rclone/rclone.conf" \
  "$HOME/.cache/face_processing/rclone-worker-2/rclone.conf"

docker run -d --rm --init \
  --name dataset-processing-cpu-2 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$HOME/.cache/face_processing/dataset-processing-2:/workspace-data" \
  -v "$HOME/.cache/face_processing/rclone-worker-2:/root/.config/rclone" \
  semyonvlasov/dataset-processing-cpu:latest \
  python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml
```

Watch local logs:

```bash
docker logs -f dataset-processing-cpu-1
docker logs -f dataset-processing-cpu-2
```

Check local load:

```bash
docker stats
```

Keep the Mac awake while the workers run:

```bash
caffeinate -dimsu docker wait dataset-processing-cpu-1 dataset-processing-cpu-2
```

## Run One Archive For Test

Use the same command with an exact archive glob and `--max-archives 1`:

```bash
docker run -d --rm --init \
  --name dataset-processing-cpu-test \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/.cache/face_processing/dataset-processing-test:/workspace-data \
  -v /root/.config/rclone:/root/.config/rclone \
  semyonvlasov/dataset-processing-cpu:latest \
  python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml \
    --archive-glob "ARCHIVE_NAME.tar" \
    --max-archives 1
```

Logs:

```bash
docker logs -f dataset-processing-cpu-test
```

Use a separate test workdir so stale `active_archive_state.json` from a previous run cannot affect the main worker.

## Multiple Workers

Each worker must use a separate container name and a separate `/workspace-data`
mount. For rclone, prefer a per-worker copy of the config so concurrent token
refresh writes cannot race on one `rclone.conf`.

Example for worker 2:

```bash
mkdir -p \
  /root/.cache/face_processing/dataset-processing-2 \
  /root/.cache/face_processing/rclone-worker-2

cp /root/.config/rclone/rclone.conf \
  /root/.cache/face_processing/rclone-worker-2/rclone.conf

docker run -d --rm --init \
  --name dataset-processing-cpu-2 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/.cache/face_processing/dataset-processing-2:/workspace-data \
  -v /root/.cache/face_processing/rclone-worker-2:/root/.config/rclone \
  semyonvlasov/dataset-processing-cpu:latest \
  python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml
```

## Important Details

- The config path in the command is inside the container. It is not a file that must exist on the remote host.
- rclone config is not baked into the image. The host must provide `/root/.config/rclone/rclone.conf`.
- Do not mount rclone config read-only for Google Drive workers. rclone may refresh OAuth tokens and write the updated token back to `rclone.conf`.
- The worker is CPU-only and does not require GPU access.
- The image is multiarch. Linux x86 hosts pull `linux/amd64`; Apple Silicon hosts pull `linux/arm64`.
- Runtime state persists under the host path mounted to `/workspace-data`.
- On the remote Linux host, the default state path is `/root/.cache/face_processing/dataset-processing`.
- On local macOS runs, use `$HOME/.cache/face_processing/...`.
- Container paths under `/workspace-data` map to that host state directory.
- Main manifest path inside the mounted workdir:

```text
/root/.cache/face_processing/dataset-processing/process/raw_faceclips_cycle/archive_manifest.jsonl
```

- Active resume state, if any:

```text
/root/.cache/face_processing/dataset-processing/process/raw_faceclips_cycle/active_archive_state.json
```

- The processor claims a raw Drive archive by renaming `ARCHIVE.tar` to `ARCHIVE.tar.processed` before download.
- The output Drive folder is taken from the baked config. Current processed folder id:

```text
1D6vtNpRmZabqnlW4598X6lqgFETZ9RvO
```

- If a run fails, inspect `active_archive_state.json` and `archive_manifest.jsonl` before deleting anything.
- `docker logs` disappear once an `--rm` container exits. The manifest and staged files remain in the mounted workdir.

## Persistent Log Variant

If persistent stdout logs are needed, run foreground Docker under `nohup` and tee to a host file:

```bash
mkdir -p /root/.cache/face_processing/dataset-processing/logs

nohup sh -c '
  docker run --rm --init \
    --name dataset-processing-cpu \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v /root/.cache/face_processing/dataset-processing:/workspace-data \
    -v /root/.config/rclone:/root/.config/rclone \
    semyonvlasov/dataset-processing-cpu:latest \
    python3 batch/gdrive_processor.py /workspace/repo/configs/gdrive_container_cpu.yaml \
    2>&1 | tee -a /root/.cache/face_processing/dataset-processing/logs/raw_drive.log
' >/dev/null 2>&1 &
```

Then follow the persisted log:

```bash
tail -f /root/.cache/face_processing/dataset-processing/logs/raw_drive.log
```
