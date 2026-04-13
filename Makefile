REMOTE ?= root@127.0.0.1
PORT ?= 22
REMOTE_ROOT ?= /root/lipsync_test/face_processing
REMOTE_GIT_BRANCH ?= main
REMOTE_GIT_URL ?=

RCLONE_CONFIG_PATH ?= $(HOME)/.config/rclone/rclone.conf
MODEL_PATH ?= $(CURDIR)/assets/face_landmarker_v2_with_blendshapes.task

REMOTE_RCLONE_CONFIG ?= /root/.config/rclone/rclone.conf
REMOTE_MODEL_PATH ?= $(REMOTE_ROOT)/assets/face_landmarker_v2_with_blendshapes.task

DOCKER_IMAGE ?= face-processing-cpu:latest
DOCKER_CONTAINER ?= face-processing-cpu
DOCKER_DATA_ROOT ?= /root/.cache/face_processing/process-docker
DOCKER_CONFIG_PATH ?= configs/gdrive_container_cpu.yaml
DOCKER_LAUNCHER_LOG ?= $(DOCKER_DATA_ROOT)/launcher.log
DOCKER_LAUNCHER_PID ?= $(DOCKER_DATA_ROOT)/launcher.pid
SSH_OPTS = -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)"
SCP_OPTS = -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$(PORT)"

.PHONY: remote-sync-code remote-rclone-config remote-docker-host-setup remote-docker-model remote-docker-build remote-bootstrap-docker remote-docker-start remote-docker-stop remote-docker-tail

remote-sync-code:
	PORT="$(PORT)" REMOTE="$(REMOTE)" REMOTE_ROOT="$(REMOTE_ROOT)" REMOTE_GIT_BRANCH="$(REMOTE_GIT_BRANCH)" REMOTE_GIT_URL="$(REMOTE_GIT_URL)" \
		bash scripts/sync_remote_code.sh

remote-rclone-config:
	@test -f "$(RCLONE_CONFIG_PATH)" || (echo "missing local rclone config: $(RCLONE_CONFIG_PATH)" && exit 1)
	@ssh $(SSH_OPTS) "$(REMOTE)" "mkdir -p '$(dir $(REMOTE_RCLONE_CONFIG))'"
	@scp $(SCP_OPTS) "$(RCLONE_CONFIG_PATH)" "$(REMOTE):$(REMOTE_RCLONE_CONFIG)"
	@echo "remote-rclone-config complete: $(REMOTE):$(REMOTE_RCLONE_CONFIG)"

remote-docker-host-setup:
	@ssh $(SSH_OPTS) "$(REMOTE)" "\
		set -euo pipefail; \
		if ! command -v apt-get >/dev/null 2>&1; then \
			echo 'remote-docker-host-setup expects Ubuntu/Debian with apt-get'; \
			exit 1; \
		fi; \
		sudo_cmd=''; \
		if [ \"\$$(id -u)\" -ne 0 ]; then sudo_cmd='sudo'; fi; \
		if command -v docker >/dev/null 2>&1; then \
			echo 'docker already installed; skipping apt install'; \
		else \
			\$$sudo_cmd apt-get update; \
			DEBIAN_FRONTEND=noninteractive \$$sudo_cmd apt-get install -y ca-certificates docker.io git openssh-client; \
		fi; \
		\$$sudo_cmd systemctl enable --now docker >/dev/null 2>&1 || true; \
		docker --version; \
	"
	@echo "remote-docker-host-setup complete: $(REMOTE)"

remote-docker-model:
	@test -f "$(MODEL_PATH)" || (echo "missing model asset: $(MODEL_PATH)" && exit 1)
	@ssh $(SSH_OPTS) "$(REMOTE)" "mkdir -p '$(dir $(REMOTE_MODEL_PATH))'"
	@scp $(SCP_OPTS) "$(MODEL_PATH)" "$(REMOTE):$(REMOTE_MODEL_PATH)"
	@echo "remote-docker-model complete: $(REMOTE):$(REMOTE_MODEL_PATH)"

remote-docker-build:
	@ssh $(SSH_OPTS) "$(REMOTE)" "\
		set -euo pipefail; \
		cd '$(REMOTE_ROOT)'; \
		docker build -f docker/Dockerfile.cpu -t '$(DOCKER_IMAGE)' .; \
	"
	@echo "remote-docker-build complete: $(REMOTE):$(DOCKER_IMAGE)"

remote-bootstrap-docker: remote-sync-code remote-rclone-config remote-docker-host-setup remote-docker-model remote-docker-build
	@echo "remote-bootstrap-docker complete: $(REMOTE):$(REMOTE_ROOT)"

remote-docker-start:
	@ssh $(SSH_OPTS) "$(REMOTE)" "bash -lc '\
		set -euo pipefail; \
		mkdir -p \"$(DOCKER_DATA_ROOT)\" \"$(DOCKER_DATA_ROOT)/process/logs\"; \
		if [ -f \"$(DOCKER_LAUNCHER_PID)\" ]; then \
			existing_pid=\$$(cat \"$(DOCKER_LAUNCHER_PID)\" 2>/dev/null || true); \
			if [ -n \"\$$existing_pid\" ] && ps -p \"\$$existing_pid\" >/dev/null 2>&1; then \
				echo \"docker launcher already running pid=\$$existing_pid\"; \
				exit 1; \
			fi; \
			rm -f \"$(DOCKER_LAUNCHER_PID)\"; \
		fi; \
		docker rm -f \"$(DOCKER_CONTAINER)\" >/dev/null 2>&1 || true; \
		cd \"$(REMOTE_ROOT)\"; \
		nohup env IMAGE_TAG=\"$(DOCKER_IMAGE)\" CONTAINER_NAME=\"$(DOCKER_CONTAINER)\" DATA_ROOT=\"$(DOCKER_DATA_ROOT)\" RCLONE_CONFIG_PATH=\"$(REMOTE_RCLONE_CONFIG)\" MODEL_PATH=\"$(REMOTE_MODEL_PATH)\" CONFIG_PATH=\"$(DOCKER_CONFIG_PATH)\" SKIP_BUILD=1 bash docker/run_gdrive_cpu.sh > \"$(DOCKER_LAUNCHER_LOG)\" 2>&1 < /dev/null & \
		pid=\$$!; \
		echo \"\$$pid\" > \"$(DOCKER_LAUNCHER_PID)\"; \
		sleep 1; \
		ps -p \"\$$pid\" >/dev/null; \
		echo \"started pid=\$$pid\"; \
	'"
	@echo "remote-docker-start complete: $(REMOTE):$(PORT)"

remote-docker-stop:
	@ssh $(SSH_OPTS) "$(REMOTE)" "bash -lc '\
		set -euo pipefail; \
		if [ -f \"$(DOCKER_LAUNCHER_PID)\" ]; then \
			existing_pid=\$$(cat \"$(DOCKER_LAUNCHER_PID)\" 2>/dev/null || true); \
			if [ -n \"\$$existing_pid\" ] && ps -p \"\$$existing_pid\" >/dev/null 2>&1; then \
				kill \"\$$existing_pid\" 2>/dev/null || true; \
				sleep 1; \
			fi; \
			rm -f \"$(DOCKER_LAUNCHER_PID)\"; \
		fi; \
		docker rm -f \"$(DOCKER_CONTAINER)\" >/dev/null 2>&1 || true; \
		echo stopped; \
	'"
	@echo "remote-docker-stop complete: $(REMOTE):$(PORT)"

remote-docker-tail:
	@ssh $(SSH_OPTS) "$(REMOTE)" "bash -lc '\
		set -euo pipefail; \
		mkdir -p \"$(DOCKER_DATA_ROOT)/process/logs\"; \
		touch \"$(DOCKER_DATA_ROOT)/process/logs/raw_drive.log\"; \
		tail -n 100 -f \"$(DOCKER_DATA_ROOT)/process/logs/raw_drive.log\"; \
	'"
