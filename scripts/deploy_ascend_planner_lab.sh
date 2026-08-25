#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-snapmind-planner-lab}"
BASE_IMAGE="${BASE_IMAGE:?Set BASE_IMAGE to a validated Planner Lab or platform image}"
IMAGE_TAG="${IMAGE_TAG:-momentseek-29154-platform:snapmind-planner-lab-$(date +%Y%m%d-%H%M%S)}"
NPU_DEVICE="${NPU_DEVICE:-2}"
APP_PORT="${APP_PORT:-8010}"
PLANNER_LAB_ORCHESTRATION_ENABLED="${PLANNER_LAB_ORCHESTRATION_ENABLED:-true}"
# Deployment-only input.  The application Settings key remains
# VISUAL_ANN_TOP_K; this prefixed variable prevents an experiment from
# accidentally changing the shared environment file.
PLANNER_LAB_VISUAL_ANN_TOP_K="${PLANNER_LAB_VISUAL_ANN_TOP_K:-${VISUAL_ANN_TOP_K:-2000}}"
ENV_FILE="${ENV_FILE:-${WORK_ROOT}/builds/planner-lab/container.env}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
BACKUP_NAME="${BACKUP_NAME:-${CONTAINER_NAME}-backup-$(date +%Y%m%d-%H%M%S)}"

while [[ "$PLANNER_LAB_VISUAL_ANN_TOP_K" == 0* \
  && "$PLANNER_LAB_VISUAL_ANN_TOP_K" != "0" ]]; do
  PLANNER_LAB_VISUAL_ANN_TOP_K="${PLANNER_LAB_VISUAL_ANN_TOP_K#0}"
done
if [[ ! "$PLANNER_LAB_VISUAL_ANN_TOP_K" =~ ^[0-9]+$ \
  || ${#PLANNER_LAB_VISUAL_ANN_TOP_K} -gt 5 ]] \
  || (( PLANNER_LAB_VISUAL_ANN_TOP_K < 1 \
    || PLANNER_LAB_VISUAL_ANN_TOP_K > 16383 )); then
  printf 'PLANNER_LAB_VISUAL_ANN_TOP_K must be an integer from 1 to 16383: %s\n' \
    "$PLANNER_LAB_VISUAL_ANN_TOP_K" >&2
  exit 1
fi

# Health exposes these exact Settings fields.  Explicit caller values win;
# source/image-derived fallbacks keep archive and git-worktree deployments
# traceable without trusting stale values captured in ENV_FILE.
PLANNER_LAB_GIT_COMMIT="${PLANNER_LAB_GIT_COMMIT:-${GIT_COMMIT:-}}"
if [[ -z "$PLANNER_LAB_GIT_COMMIT" ]] \
  && command -v git >/dev/null 2>&1 \
  && git -C "$SOURCE_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
  PLANNER_LAB_GIT_COMMIT="$(git -C "$SOURCE_DIR" rev-parse --verify HEAD)"
fi
PLANNER_LAB_GIT_COMMIT="${PLANNER_LAB_GIT_COMMIT:-unknown}"
PLANNER_LAB_RELEASE_ID="${PLANNER_LAB_RELEASE_ID:-${RELEASE_ID:-${IMAGE_TAG##*:}}}"

for command_name in docker curl grep python3; do
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
  if ! docker rename "$CONTAINER_NAME" "$BACKUP_NAME"; then
    printf 'Could not rename stopped container; restarting it: %s\n' \
      "$CONTAINER_NAME" >&2
    docker start "$CONTAINER_NAME" >/dev/null || true
    exit 1
  fi
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

verify_deployment() {
  local container_environment health_payload expected
  container_environment="$(
    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
      "$CONTAINER_NAME"
  )"
  for expected in \
    "VISUAL_ANN_TOP_K=$PLANNER_LAB_VISUAL_ANN_TOP_K" \
    "RELEASE_ID=$PLANNER_LAB_RELEASE_ID" \
    "GIT_COMMIT=$PLANNER_LAB_GIT_COMMIT" \
    "IMAGE_TAG=$IMAGE_TAG"; do
    if ! grep -Fqx -- "$expected" <<<"$container_environment"; then
      printf 'Container environment verification failed: %s\n' "$expected" >&2
      return 1
    fi
  done

  health_payload="$(
    curl -fsS --max-time 15 "http://127.0.0.1:${APP_PORT}/api/health"
  )"
  if ! printf '%s' "$health_payload" | python3 -c '
import json
import sys

body = json.load(sys.stdin)
expected = dict(zip(("release_id", "git_commit", "image_tag"), sys.argv[1:]))
if body.get("status") != "ok":
    raise SystemExit(f"health status is not ok: {body.get('"'"'status'"'"')!r}")
for key, value in expected.items():
    if body.get(key) != value:
        raise SystemExit(
            f"health metadata mismatch for {key}: "
            f"expected={value!r} actual={body.get(key)!r}"
        )
' "$PLANNER_LAB_RELEASE_ID" "$PLANNER_LAB_GIT_COMMIT" "$IMAGE_TAG"; then
    return 1
  fi
  curl -fsS --max-time 15 \
    "http://127.0.0.1:${APP_PORT}/api/planner-lab/capabilities" >/dev/null
}

if ! docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -e "APP_PORT=$APP_PORT" \
  -e "ORCHESTRATION_ENABLED=$PLANNER_LAB_ORCHESTRATION_ENABLED" \
  -e "VISUAL_ANN_TOP_K=$PLANNER_LAB_VISUAL_ANN_TOP_K" \
  -e "RELEASE_ID=$PLANNER_LAB_RELEASE_ID" \
  -e "GIT_COMMIT=$PLANNER_LAB_GIT_COMMIT" \
  -e "IMAGE_TAG=$IMAGE_TAG" \
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
    if ! verify_deployment; then
      docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
      restore_previous
      exit 1
    fi
    printf 'Planner Lab deployed: image=%s container=%s backup=%s visual_ann_top_k=%s release=%s git=%s\n' \
      "$IMAGE_TAG" "$CONTAINER_NAME" "$BACKUP_NAME" \
      "$PLANNER_LAB_VISUAL_ANN_TOP_K" "$PLANNER_LAB_RELEASE_ID" \
      "$PLANNER_LAB_GIT_COMMIT"
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
