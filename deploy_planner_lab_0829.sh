#!/usr/bin/env bash
# MomentSeek 0829 - Planner Lab Overlay 部署脚本
# 基于同事的 deploy_ascend_planner_lab.sh 适配 0829 环境
# 使用 overlay 方式快速部署 Planner Lab 实验功能

set -Eeuo pipefail

# ==================== 0829 特定配置 ====================
WORK_ROOT="/home/momentseek_0829_develop/workplace/MomentSeek_planner"
SOURCE_DIR="${WORK_ROOT}"
RUNTIME_DIR="${WORK_ROOT}/runtime"
MODEL_DIR="/home/momentseek-29154/models/platform"

# 基础镜像：当前运行的 0829 主镜像
BASE_IMAGE="${BASE_IMAGE:-momentseek-0829-platform:current}"

# Overlay 镜像和容器命名
IMAGE_TAG="momentseek-0829-planner-lab:$(date +%Y%m%d-%H%M%S)"
CONTAINER_NAME="momentseek-0829-planner-lab"
BACKUP_NAME="${CONTAINER_NAME}-backup-$(date +%Y%m%d-%H%M%S)"

# 从 .env.0829 读取配置
if [[ -f "${WORK_ROOT}/.env.0829" ]]; then
    source "${WORK_ROOT}/.env.0829"
fi

# Planner Lab 使用独立端口（避开主容器 8100）
APP_PORT="${PLANNER_LAB_PORT:-8101}"
NPU_ID="${HOST_NPU_DEVICE_ID:-1}"

# ==================== 工具函数 ====================
log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }
error() { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

restore_previous() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if docker inspect "$BACKUP_NAME" >/dev/null 2>&1; then
    docker rename "$BACKUP_NAME" "$CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
    log "Rolled back to previous container: $BACKUP_NAME"
  fi
}

# ==================== 环境检查 ====================
log "Step 1: Pre-flight checks"

for cmd in docker curl; do
  command -v "$cmd" >/dev/null 2>&1 || error "Missing command: $cmd"
done

test -f "$SOURCE_DIR/docker/Dockerfile.planner-lab-overlay" || error "Missing Dockerfile.planner-lab-overlay"
test -d "$RUNTIME_DIR" || error "Missing runtime directory: $RUNTIME_DIR"
test -d "$MODEL_DIR" || error "Missing model directory: $MODEL_DIR"

# 检查基础镜像是否存在
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  error "Base image not found: $BASE_IMAGE"
fi

# 检查端口占用
if ss -lntp | grep -qE ":${APP_PORT}\s"; then
  log "WARNING: Port $APP_PORT is already in use"
  existing=$(ss -lntp | grep -E ":${APP_PORT}\s" | head -1)
  echo "$existing"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]] || error "Aborted by user"
fi

# 检查 vLLM 服务可达性
log "Step 2: Check vLLM connectivity"
VLLM_URL="${QWEN35_VLLM_BASE_URL:-http://127.0.0.1:18082/v1}"
if curl -fsSL --max-time 3 "${VLLM_URL%/v1}/health" >/dev/null 2>&1 || \
   curl -fsSL --max-time 3 "${VLLM_URL}/models" >/dev/null 2>&1; then
  log "✓ vLLM service reachable at $VLLM_URL"
else
  log "⚠ WARNING: Cannot reach vLLM at $VLLM_URL"
  log "  Planner Lab will fall back to heuristic mode"
  log "  To enable LLM mode, ensure vLLM service is running"
fi

# ==================== 构建 Overlay 镜像 ====================
log "Step 3: Build overlay image"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$SOURCE_DIR/docker/Dockerfile.planner-lab-overlay" \
  -t "$IMAGE_TAG" \
  "$SOURCE_DIR"

log "✓ Built image: $IMAGE_TAG"

# ==================== 备份旧容器 ====================
log "Step 4: Backup existing container (if any)"

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  # 检查备份名是否已存在
  if docker inspect "$BACKUP_NAME" >/dev/null 2>&1; then
    error "Backup name already exists: $BACKUP_NAME"
  fi

  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rename "$CONTAINER_NAME" "$BACKUP_NAME"
  log "✓ Backed up old container as: $BACKUP_NAME"
else
  log "No existing container to backup"
fi

# ==================== 启动新容器 ====================
log "Step 5: Start new Planner Lab container"

# 构建环境变量参数
ENV_ARGS=(
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
  -e VISUAL_MODEL=siglip2-so400m-384
  -e VISUAL_HF_CACHE_DIR=/app/models/hf-cache
  -e FACE_PROVIDER=cann
  -e ASR_ENGINE=funasr
  -e ASR_VAD_STRATEGY=silero_12s
  -e OCR_ENGINE=rapidocr_acl
  -e OCR_DEVICE=npu
  -e MODEL_MANIFEST=deploy/models/ascend.models.json
  -e MILVUS_ENABLED="${MILVUS_ENABLED:-true}"
  -e MILVUS_HOST="${MILVUS_HOST:-localhost}"
  -e MILVUS_PORT="${MILVUS_PORT:-19531}"
  -e VISUAL_USE_DISKANN="${VISUAL_USE_DISKANN:-true}"
  -e VISUAL_ANN_TOP_K="${VISUAL_ANN_TOP_K:-500}"
  # Planner Lab 专属配置
  -e PLANNER_LAB_ENABLED="${PLANNER_LAB_ENABLED:-true}"
  -e ORCHESTRATION_ENABLED="${ORCHESTRATION_ENABLED:-true}"
  -e ORCHESTRATION_CONFIG_PATH="${ORCHESTRATION_CONFIG_PATH:-deploy/orchestration/qwen35-vllm.json}"
  -e ORCHESTRATION_PROFILE="${ORCHESTRATION_PROFILE:-qwen35-unified}"
  -e ORCHESTRATION_FAIL_OPEN="${ORCHESTRATION_FAIL_OPEN:-true}"
  -e ORCHESTRATION_TRACE_ENABLED="${ORCHESTRATION_TRACE_ENABLED:-true}"
  -e ORCHESTRATION_TRACE_PATH="${ORCHESTRATION_TRACE_PATH:-runtime/orchestration-traces.jsonl}"
  -e QWEN35_VLLM_BASE_URL="${QWEN35_VLLM_BASE_URL:-http://127.0.0.1:18082/v1}"
  -e QWEN35_PLANNER_MODEL="${QWEN35_PLANNER_MODEL:-qwen3.5-4b}"
  -e QWEN35_RERANKER_MODEL="${QWEN35_RERANKER_MODEL:-qwen3.5-4b}"
)

if ! docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --restart unless-stopped \
  "${ENV_ARGS[@]}" \
  --device "/dev/davinci${NPU_ID}:/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$RUNTIME_DIR:/app/runtime" \
  -v "$MODEL_DIR:/app/models:ro" \
  "$IMAGE_TAG" >/dev/null; then
  error "Failed to start container"
  restore_previous
fi

log "✓ Container started: $CONTAINER_NAME"

# ==================== 健康检查 ====================
log "Step 6: Health check and capability verification"

max_wait=60
waited=0

while [[ $waited -lt $max_wait ]]; do
  # 检查容器状态
  container_status=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "gone")

  if [[ "$container_status" != "running" ]]; then
    log "Container status: $container_status"
    docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
    restore_previous
    error "Container failed to start"
  fi

  # 尝试健康检查
  if curl -fsSL --max-time 3 "http://127.0.0.1:${APP_PORT}/api/health" >/dev/null 2>&1; then
    log "✓ Health check passed"
    break
  fi

  printf '.'
  sleep 2
  waited=$((waited + 2))
done

if [[ $waited -ge $max_wait ]]; then
  docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
  restore_previous
  error "Health check timeout after ${max_wait}s"
fi

# 验证 Planner Lab 能力接口
log "Step 7: Verify Planner Lab capabilities"

if capabilities=$(curl -fsSL --max-time 5 "http://127.0.0.1:${APP_PORT}/api/planner-lab/capabilities" 2>&1); then
  log "✓ Planner Lab capabilities endpoint OK"

  # 检查编排状态
  if echo "$capabilities" | grep -q '"enabled".*true'; then
    if echo "$capabilities" | grep -q '"orchestration".*"enabled".*true'; then
      log "✓ LLM orchestration ENABLED (Qwen3.5 mode)"
    else
      log "⚠ LLM orchestration DISABLED (heuristic mode)"
    fi
  else
    log "⚠ WARNING: Planner Lab may not be fully enabled"
  fi
else
  log "⚠ WARNING: Cannot verify capabilities endpoint"
  log "  Response: $capabilities"
fi

# ==================== 清理备份 ====================
log "Step 8: Cleanup"

if docker inspect "$BACKUP_NAME" >/dev/null 2>&1; then
  read -p "Remove backup container? (Y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    docker rm "$BACKUP_NAME" >/dev/null 2>&1 || true
    log "✓ Removed backup container"
  else
    log "Backup retained: $BACKUP_NAME"
    log "  To restore: docker stop $CONTAINER_NAME && docker start $BACKUP_NAME"
  fi
fi

# ==================== 完成 ====================
echo ""
log "✅ Planner Lab deployment successful!"
echo ""
echo "=========================================="
echo "Container:     $CONTAINER_NAME"
echo "Image:         $IMAGE_TAG"
echo "Port:          $APP_PORT"
echo "NPU:           $NPU_ID"
echo "vLLM:          ${QWEN35_VLLM_BASE_URL:-http://127.0.0.1:18082/v1}"
echo ""
echo "Access Planner Lab:"
echo "  Web:         http://127.0.0.1:${APP_PORT}"
echo "  Capabilities: http://127.0.0.1:${APP_PORT}/api/planner-lab/capabilities"
echo "  API Docs:    http://127.0.0.1:${APP_PORT}/docs"
echo ""
echo "Logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Audit trace:"
echo "  tail -f ${RUNTIME_DIR}/orchestration-traces.jsonl"
echo ""
if docker inspect "$BACKUP_NAME" >/dev/null 2>&1; then
  echo "Backup available:"
  echo "  To rollback: docker stop $CONTAINER_NAME && docker start $BACKUP_NAME"
  echo ""
fi
echo "=========================================="
