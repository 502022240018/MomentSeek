#!/usr/bin/env bash
# MomentSeek 0829 Development Environment Deployment Script
# 基于 deploy_ascend_shared_server.sh 修改，使用独立的镜像和容器命名

set -Eeuo pipefail

# ==================== 0829 特定配置 ====================
WORK_ROOT="/home/momentseek_0829_develop/workplace/MomentSeek"
SOURCE_DIR="${WORK_ROOT}"
MODEL_DIR="/home/momentseek-29154/models/platform"
RUNTIME_DIR="${WORK_ROOT}/runtime"
LOG_DIR="${WORK_ROOT}/logs"

# 镜像和容器命名（0829专用）
BASE_IMAGE="${ASCEND_RUNTIME_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts}"
IMAGE_REPO="momentseek-0829-platform"
CONTAINER_NAME="momentseek-0829-platform"
ROLLBACK_NAME="${CONTAINER_NAME}-rollback"

# 从环境变量或默认值读取配置
# 优先从.env.0829读取配置
if [[ -f "${WORK_ROOT}/.env.0829" ]]; then
    source "${WORK_ROOT}/.env.0829"
fi
NPU_ID="${HOST_NPU_DEVICE_ID:-4}"
APP_PORT="${APP_PORT:-8100}"
CPU_THREAD_LIMIT="${CPU_THREAD_LIMIT:-8}"
CONTAINER_CPU_LIMIT="${CONTAINER_CPU_LIMIT:-24}"
CONTAINER_PID_LIMIT="${CONTAINER_PID_LIMIT:-2048}"

# 开发模式配置
DEV_MODE="${DEV_MODE:-false}"
DEV_SKIP_BUILD="${DEV_SKIP_BUILD:-false}"

# Milvus配置
MILVUS_ENABLED="${MILVUS_ENABLED:-true}"
MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19531}"
MILVUS_READ_ENABLED="${MILVUS_READ_ENABLED:-true}"
MILVUS_WRITE_ENABLED="${MILVUS_WRITE_ENABLED:-true}"
MILVUS_FALLBACK_ENABLED="${MILVUS_FALLBACK_ENABLED:-true}"
MILVUS_ROLLOUT_PERCENT="${MILVUS_ROLLOUT_PERCENT:-100}"
MILVUS_QUERY_TIMEOUT_SECONDS="${MILVUS_QUERY_TIMEOUT_SECONDS:-3}"
MILVUS_SEARCH_VIDEO_BATCH_SIZE="${MILVUS_SEARCH_VIDEO_BATCH_SIZE:-8}"

SEARCH_PREWARM_ENABLED="${SEARCH_PREWARM_ENABLED:-true}"
SEARCH_PREWARM_REQUIRED="${SEARCH_PREWARM_REQUIRED:-true}"

BUILD_DIR="${SOURCE_DIR}/.server-build"
INSIGHTFACE_WHEEL="${SOURCE_DIR}/vendor-wheels/insightface-1.0.1-py3-none-any.whl"
INSIGHTFACE_SHA256="5f373f6fedbdda5cbc59a34ca386a75a2995cdaf6899402590ae9eb4308fc2e8"

# ==================== 工具函数 ====================
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nDEPLOY_FAILED: %s\n' "$*" >&2; exit 1; }

rollback_on_error() {
  local rc=$?
  local line="${1:-unknown}"
  set +e
  printf '\nDEPLOY_FAILED_AT_LINE=%s\n' "$line" >&2
  if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rename "$ROLLBACK_NAME" "$CONTAINER_NAME" >/dev/null 2>&1
    docker start "$CONTAINER_NAME" >/dev/null 2>&1
    printf 'Previous platform container was restored automatically.\n' >&2
  fi
  exit "$rc"
}
trap 'rollback_on_error "$LINENO"' ERR

# ==================== 环境检查 ====================
log "Step 1: Pre-flight checks"

[[ -d "$SOURCE_DIR/.git" ]] || fail "Git source not found: $SOURCE_DIR"
[[ -f "$SOURCE_DIR/backend/requirements-ascend.txt" ]] || fail "Missing Ascend requirements"
[[ -f "$SOURCE_DIR/frontend/package-lock.json" ]] || fail "Missing frontend package-lock.json"
[[ "$CPU_THREAD_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CPU_THREAD_LIMIT must be a positive integer"
[[ "$CONTAINER_CPU_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_CPU_LIMIT must be a positive integer"
[[ "$CONTAINER_PID_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_PID_LIMIT must be a positive integer"

for command_name in docker git curl npu-smi flock ss sha256sum python3; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command: $command_name"
done

# 端口检查
[[ "$APP_PORT" =~ ^[0-9]+$ ]] && ((APP_PORT >= 1 && APP_PORT <= 65535)) \
  || fail "Invalid APP_PORT: $APP_PORT"
printf 'app_port=%s\n' "$APP_PORT"

# Milvus预检（如果启用）
if [[ "${MILVUS_ENABLED,,}" =~ ^(true|1|yes|on)$ ]]; then
  log "Checking Milvus connectivity: ${MILVUS_HOST}:${MILVUS_PORT}"
  python3 - "$MILVUS_HOST" "$MILVUS_PORT" "$MILVUS_QUERY_TIMEOUT_SECONDS" <<'PY'
import socket
import sys

host, port_text, timeout_text = sys.argv[1:]
timeout = float(timeout_text)
if timeout <= 0:
    raise SystemExit("MILVUS_QUERY_TIMEOUT_SECONDS must be greater than zero")
try:
    with socket.create_connection((host, int(port_text)), timeout=timeout):
        pass
    print(f"milvus_preflight=PASS endpoint={host}:{port_text}")
except Exception as e:
    print(f"WARNING: Cannot reach Milvus at {host}:{port_text}: {e}", file=sys.stderr)
    print("If you haven't started Milvus yet, start it with: docker compose -f compose.milvus.yml --env-file .env.0829 up -d", file=sys.stderr)
PY
fi

# 磁盘空间检查
log "Step 2: Disk space check"
available_gb=$(df -BG "$WORK_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
[[ "$available_gb" -ge 10 ]] || fail "Insufficient disk space: ${available_gb}GB (need at least 10GB)"
printf 'disk_available=%sGB\n' "$available_gb"

# NPU检查
log "Step 3: NPU availability check"
if ! npu-smi info -t proc-mem -i "$NPU_ID" -c 0 >/dev/null 2>&1; then
  fail "NPU $NPU_ID is not accessible or proc-mem query failed"
fi
npu_status=$(npu-smi info -i "$NPU_ID" -m 2>/dev/null | grep -oP '(?<=Health Status\s{2}: )\w+' || echo "unknown")
printf 'npu_id=%s status=%s\n' "$NPU_ID" "$npu_status"

# 端口占用检查
log "Step 4: Port availability check"
if ss -lntp | grep -qE ":${APP_PORT}\s"; then
  existing_proc=$(ss -lntp | grep -E ":${APP_PORT}\s" | head -1)
  printf 'WARNING: Port %s is already in use:\n%s\n' "$APP_PORT" "$existing_proc" >&2
  fail "Port $APP_PORT is already bound. Stop the service using it first."
fi

# InsightFace wheel检查
log "Step 5: InsightFace wheel checksum"
[[ -f "$INSIGHTFACE_WHEEL" ]] || fail "Missing InsightFace wheel: $INSIGHTFACE_WHEEL"
actual_sha=$(sha256sum "$INSIGHTFACE_WHEEL" | awk '{print $1}')
[[ "$actual_sha" == "$INSIGHTFACE_SHA256" ]] || fail "InsightFace wheel checksum mismatch"
printf 'insightface_wheel=OK\n'

# Git状态检查
log "Step 6: Git status check"
GIT_COMMIT=$(git -C "$SOURCE_DIR" rev-parse --short HEAD)
GIT_BRANCH=$(git -C "$SOURCE_DIR" rev-parse --abbrev-ref HEAD)
printf 'git_commit=%s branch=%s\n' "$GIT_COMMIT" "$GIT_BRANCH"

# ==================== 构建镜像 ====================
log "Step 7: Build Docker image"

mkdir -p "$BUILD_DIR" "$LOG_DIR" "$RUNTIME_DIR"

IMAGE_TAG="${IMAGE_REPO}:${GIT_COMMIT}"
CURRENT_TAG="${IMAGE_REPO}:current"

# 开发模式：检查是否需要构建
if [[ "${DEV_MODE,,}" =~ ^(true|1|yes|on)$ ]]; then
  if [[ "${DEV_SKIP_BUILD,,}" =~ ^(true|1|yes|on)$ ]]; then
    log "DEV_MODE: Skipping build (DEV_SKIP_BUILD=true)"
    # 检查镜像是否存在
    if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
      if docker image inspect "$CURRENT_TAG" >/dev/null 2>&1; then
        log "Using existing image: $CURRENT_TAG"
        docker tag "$CURRENT_TAG" "$IMAGE_TAG"
      else
        fail "No image found. Set DEV_SKIP_BUILD=false to build."
      fi
    else
      log "Using existing image: $IMAGE_TAG"
    fi
  else
    log "DEV_MODE: Building only if image doesn't exist"
    if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
      log "Image already exists: $IMAGE_TAG (skipping build)"
    else
      log "Building new image: $IMAGE_TAG"
      cp "$SOURCE_DIR/Dockerfile.ascend" "$BUILD_DIR/Dockerfile"
      docker build \
        --file "$BUILD_DIR/Dockerfile" \
        --build-arg ASCEND_RUNTIME_IMAGE="$BASE_IMAGE" \
        --tag "$IMAGE_TAG" \
        --tag "$CURRENT_TAG" \
        "$SOURCE_DIR" \
        2>&1 | tee "$LOG_DIR/image-build-$(date +%F-%H%M%S).log"
    fi
  fi
else
  # 生产模式：总是重新构建
  log "PRODUCTION MODE: Building image"
  cp "$SOURCE_DIR/Dockerfile.ascend" "$BUILD_DIR/Dockerfile"
  docker build \
    --file "$BUILD_DIR/Dockerfile" \
    --build-arg ASCEND_RUNTIME_IMAGE="$BASE_IMAGE" \
    --tag "$IMAGE_TAG" \
    --tag "$CURRENT_TAG" \
    "$SOURCE_DIR" \
    2>&1 | tee "$LOG_DIR/image-build-$(date +%F-%H%M%S).log"
fi

printf 'image_tag=%s\n' "$IMAGE_TAG"

# ==================== 模型校验 ====================
log "Step 8: Verify production models"

# 确保runtime目录存在
mkdir -p "$RUNTIME_DIR"

# 在临时容器中校验模型，将lock文件写入runtime目录
VERIFY_CONTAINER="momentseek-0829-verify-$(date +%s)"
docker run --rm \
  --name "$VERIFY_CONTAINER" \
  -v "$MODEL_DIR:/app/models:ro" \
  -v "$RUNTIME_DIR:/app/runtime" \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE_TAG" \
  python3 /app/scripts/verify_models.py \
  --manifest /app/deploy/models/ascend-prod.models.json \
  --lock /app/runtime/models.lock.json \
  || fail "Model verification failed. Check models in $MODEL_DIR"

printf 'models_verified=OK lock_file=%s/models.lock.json\n' "$RUNTIME_DIR"

# ==================== 准备部署 ====================
log "Step 9: Check for existing container and running jobs"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  # 检查是否有运行中的任务
  if job_response=$(curl -fsSL --max-time 5 "http://127.0.0.1:${APP_PORT}/api/jobs" 2>/dev/null); then
    active_jobs=$(echo "$job_response" | python3 -c "import sys,json; jobs=json.load(sys.stdin); print(len([j for j in jobs if j.get('status') in ('queued','running')]))")
    if [[ "$active_jobs" -gt 0 ]]; then
      fail "Found $active_jobs active jobs. Cancel them before deployment."
    fi
  fi

  # 重命名旧容器为rollback
  log "Renaming old container to rollback"
  docker rename "$CONTAINER_NAME" "$ROLLBACK_NAME" 2>/dev/null || true
  docker stop "$ROLLBACK_NAME" 2>/dev/null || true
else
  printf 'no_existing_container\n'
fi

# ==================== 启动新容器 ====================
log "Step 10: Start new container"

# 基础 docker run 参数
DOCKER_RUN_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  --network host
  --device "/dev/davinci${NPU_ID}:/dev/davinci${NPU_ID}"
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v "$MODEL_DIR:/app/models:ro"
  -v "$RUNTIME_DIR:/app/runtime"
  --cpus="$CONTAINER_CPU_LIMIT"
  --pids-limit="$CONTAINER_PID_LIMIT"
  -e ENV_PROFILE=prod.ascend
  -e APP_PORT="$APP_PORT"
  -e APP_DATA_DIR=/app/runtime
  -e APP_MODEL_DIR=/app/models
  -e NPU_ENABLED=true
  -e NPU_DEVICE_ID=0
  -e INDEXER_MODE=daemon
  -e NPU_WORKER_MODE=isolated
  -e INDEXER_IDLE_TIMEOUT_SECONDS=0
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
  -e OPENBLAS_NUM_THREADS="$CPU_THREAD_LIMIT"
  -e OMP_NUM_THREADS="$CPU_THREAD_LIMIT"
  -e VISUAL_MODEL=siglip2-so400m-384
  -e VISUAL_HF_CACHE_DIR=/app/models/hf-cache
  -e FACE_PROVIDER=cann
  -e ASR_ENGINE=funasr
  -e ASR_VAD_STRATEGY=silero_12s
  -e OCR_ENGINE=rapidocr_acl
  -e OCR_DEVICE=npu
  -e MODEL_MANIFEST=deploy/models/ascend-prod.models.json
  -e MILVUS_ENABLED="$MILVUS_ENABLED"
  -e MILVUS_HOST="$MILVUS_HOST"
  -e MILVUS_PORT="$MILVUS_PORT"
  -e MILVUS_READ_ENABLED="$MILVUS_READ_ENABLED"
  -e MILVUS_WRITE_ENABLED="$MILVUS_WRITE_ENABLED"
  -e MILVUS_FALLBACK_ENABLED="$MILVUS_FALLBACK_ENABLED"
  -e VISUAL_USE_DISKANN="${VISUAL_USE_DISKANN:-false}"
  -e VISUAL_ANN_TOP_K="${VISUAL_ANN_TOP_K:-500}"
  -e VISUAL_SAMPLE_SIZE="${VISUAL_SAMPLE_SIZE:-500}"
  -e VISUAL_SAMPLE_STRATEGY="${VISUAL_SAMPLE_STRATEGY:-systematic}"
  -e SEARCH_PREWARM_ENABLED="${SEARCH_PREWARM_ENABLED:-false}"
  -e SEARCH_PREWARM_REQUIRED="${SEARCH_PREWARM_REQUIRED:-false}"
  --restart unless-stopped
)

# 开发模式：添加代码挂载和热重载
if [[ "${DEV_MODE,,}" =~ ^(true|1|yes|on)$ ]]; then
  log "DEV_MODE: Enabling code mount and hot reload"
  DOCKER_RUN_ARGS+=(
    -v "$SOURCE_DIR/backend/app:/app/backend/app"
    -v "$SOURCE_DIR/frontend/dist:/app/backend/app/static:ro"
    -e WATCHFILES_FORCE_POLLING="true"
  )
  # 修改启动命令以启用热重载
  DOCKER_RUN_ARGS+=(
    "$IMAGE_TAG"
    sh -c "exec uvicorn app.main:app --host 0.0.0.0 --port \${APP_PORT:-8000} --workers 1 --reload"
  )
else
  # 生产模式：使用默认命令
  DOCKER_RUN_ARGS+=("$IMAGE_TAG")
fi

docker run "${DOCKER_RUN_ARGS[@]}"

printf 'container_started=%s mode=%s\n' "$CONTAINER_NAME" "${DEV_MODE:-production}"

# ==================== 健康检查 ====================
log "Step 11: Health check"

max_wait=60
waited=0
while [[ $waited -lt $max_wait ]]; do
  if health_response=$(curl -fsSL --max-time 3 "http://127.0.0.1:${APP_PORT}/api/health" 2>/dev/null); then
    printf '\nhealth_check=PASS\n'
    echo "$health_response" | python3 -m json.tool || echo "$health_response"
    break
  fi
  printf '.'
  sleep 2
  waited=$((waited + 2))
done

[[ $waited -lt $max_wait ]] || fail "Health check timeout after ${max_wait}s"

# ==================== 清理 ====================
log "Step 12: Cleanup"

if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
  docker rm "$ROLLBACK_NAME" >/dev/null 2>&1 || true
  printf 'rollback_container_removed\n'
fi

# ==================== 完成 ====================
log "✅ Deployment successful!"
printf '\n===========================================\n'
printf 'Container: %s\n' "$CONTAINER_NAME"
printf 'Image: %s\n' "$IMAGE_TAG"
printf 'Mode: %s\n' "${DEV_MODE:-production}"
if [[ "${DEV_MODE,,}" =~ ^(true|1|yes|on)$ ]]; then
  printf 'Hot Reload: ENABLED ✓\n'
  printf 'Code Mount: %s -> /app/backend/app\n' "$SOURCE_DIR/backend/app"
fi
printf 'Port: %s\n' "$APP_PORT"
printf 'NPU: %s\n' "$NPU_ID"
printf 'Runtime: %s\n' "$RUNTIME_DIR"
printf 'Models: %s\n' "$MODEL_DIR"
printf '\nAccess the service at: http://127.0.0.1:%s\n' "$APP_PORT"
printf 'API docs: http://127.0.0.1:%s/docs\n' "$APP_PORT"
if [[ "${DEV_MODE,,}" =~ ^(true|1|yes|on)$ ]]; then
  printf '\n💡 DEV MODE TIP: Code changes will auto-reload.\n'
  printf '   Watch logs: docker logs -f %s\n' "$CONTAINER_NAME"
fi
printf '===========================================\n'
