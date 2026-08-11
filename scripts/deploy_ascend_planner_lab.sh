#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-snapmind-planner-lab}"
BASE_IMAGE="${BASE_IMAGE:?Set BASE_IMAGE to a validated Planner Lab or platform image}"
IMAGE_TAG="${IMAGE_TAG:-momentseek-29154-platform:snapmind-planner-lab-$(date +%Y%m%d-%H%M%S)}"
NPU_DEVICE="${NPU_DEVICE:-2}"
APP_PORT="${APP_PORT:-8010}"
ENV_FILE="${ENV_FILE:-${WORK_ROOT}/builds/planner-lab/container.env}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
BACKUP_NAME="${BACKUP_NAME:-${CONTAINER_NAME}-backup-$(date +%Y%m%d-%H%M%S)}"

for command_name in docker curl; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'Missing command: %s\n' "$command_name" >&2
    exit 1
  }
done

test -f "$SOURCE_DIR/docker/Dockerfile.planner-lab-overlay"
test -d "$RUNTIME_DIR"
test -d "$MODEL_DIR"
mkdir -p "$(dirname "$ENV_FILE")"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$SOURCE_DIR/docker/Dockerfile.planner-lab-overlay" \
  -t "$IMAGE_TAG" \
  "$SOURCE_DIR"

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  test -z "$(docker ps -a --filter "name=^/${BACKUP_NAME}$" --format '{{.Names}}')"
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME" >"$ENV_FILE"
  chmod 600 "$ENV_FILE"
  docker stop "$CONTAINER_NAME" >/dev/null
  docker rename "$CONTAINER_NAME" "$BACKUP_NAME"
elif [[ ! -f "$ENV_FILE" ]]; then
  printf 'No existing container and ENV_FILE does not exist: %s\n' "$ENV_FILE" >&2
  exit 1
fi

restore_previous() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if docker inspect "$BACKUP_NAME" >/dev/null 2>&1; then
    docker rename "$BACKUP_NAME" "$CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
  fi
}

if ! docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -e "APP_PORT=$APP_PORT" \
  --device "/dev/davinci${NPU_DEVICE}:/dev/davinci${NPU_DEVICE}" \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v "$RUNTIME_DIR:/app/runtime" \
  -v "$MODEL_DIR:/app/models" \
  "$IMAGE_TAG" >/dev/null; then
  restore_previous
  exit 1
fi

for attempt in $(seq 1 60); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER_NAME")"
  if [[ "$health" == "healthy" ]]; then
    curl -fsS --max-time 15 "http://127.0.0.1:${APP_PORT}/api/planner-lab/capabilities" >/dev/null
    printf 'Planner Lab deployed: image=%s container=%s backup=%s\n' \
      "$IMAGE_TAG" "$CONTAINER_NAME" "$BACKUP_NAME"
    exit 0
  fi
  if [[ "$health" == "unhealthy" || "$health" == "exited" || "$health" == "dead" ]]; then
    docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
    restore_previous
    exit 1
  fi
  sleep 2
done

docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
restore_previous
exit 1
