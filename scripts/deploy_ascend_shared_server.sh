#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts}"
IMAGE_REPO="${IMAGE_REPO:-momentseek-29154-platform}"
IMAGE_NAME="${IMAGE_NAME:-}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-platform}"
ROLLBACK_NAME="${CONTAINER_NAME}-rollback"
NPU_ID="${NPU_ID:-5}"
APP_PORT="${APP_PORT:-}"
CPU_THREAD_LIMIT="${CPU_THREAD_LIMIT:-8}"
CONTAINER_CPU_LIMIT="${CONTAINER_CPU_LIMIT:-24}"
CONTAINER_PID_LIMIT="${CONTAINER_PID_LIMIT:-2048}"
MILVUS_ENABLED="${MILVUS_ENABLED:-false}"
MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_READ_ENABLED="${MILVUS_READ_ENABLED:-true}"
MILVUS_WRITE_ENABLED="${MILVUS_WRITE_ENABLED:-true}"
MILVUS_FALLBACK_ENABLED="${MILVUS_FALLBACK_ENABLED:-true}"
MILVUS_ROLLOUT_PERCENT="${MILVUS_ROLLOUT_PERCENT:-100}"
MILVUS_QUERY_TIMEOUT_SECONDS="${MILVUS_QUERY_TIMEOUT_SECONDS:-3}"
MILVUS_SEARCH_VIDEO_BATCH_SIZE="${MILVUS_SEARCH_VIDEO_BATCH_SIZE:-8}"
SEARCH_PREWARM_ENABLED="${SEARCH_PREWARM_ENABLED:-true}"
SEARCH_PREWARM_REQUIRED="${SEARCH_PREWARM_REQUIRED:-true}"
BUILD_DIR="${SOURCE_DIR}/.server-build"
INSIGHTFACE_WHEEL="${INSIGHTFACE_WHEEL:-${SOURCE_DIR}/vendor-wheels/insightface-1.0.1-py3-none-any.whl}"
INSIGHTFACE_SHA256="5f373f6fedbdda5cbc59a34ca386a75a2995cdaf6899402590ae9eb4308fc2e8"

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

[[ -d "$SOURCE_DIR/.git" ]] || fail "Git source not found: $SOURCE_DIR"
[[ -f "$SOURCE_DIR/backend/requirements-ascend.txt" ]] || fail "Missing Ascend requirements"
[[ -f "$SOURCE_DIR/frontend/package-lock.json" ]] || fail "Missing frontend package-lock.json"
[[ "$CPU_THREAD_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CPU_THREAD_LIMIT must be a positive integer"
[[ "$CONTAINER_CPU_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_CPU_LIMIT must be a positive integer"
[[ "$CONTAINER_PID_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_PID_LIMIT must be a positive integer"
for command_name in docker git curl npu-smi flock ss sha256sum python3; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command: $command_name"
done

APP_PORT_SOURCE="explicit"
if [[ -z "$APP_PORT" ]]; then
  APP_PORT_SOURCE="default"
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    APP_PORT="$(docker container inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
      "$CONTAINER_NAME" 2>/dev/null | sed -n 's/^APP_PORT=//p' | tail -1)"
    if [[ -n "$APP_PORT" ]]; then
      APP_PORT_SOURCE="existing-container"
    fi
  fi
  APP_PORT="${APP_PORT:-18500}"
fi
[[ "$APP_PORT" =~ ^[0-9]+$ ]] && ((APP_PORT >= 1 && APP_PORT <= 65535)) \
  || fail "Invalid APP_PORT: $APP_PORT"
printf 'app_port=%s source=%s\n' "$APP_PORT" "$APP_PORT_SOURCE"
[[ -n "$MILVUS_HOST" ]] || fail "MILVUS_HOST must not be empty"
[[ "$MILVUS_PORT" =~ ^[0-9]+$ ]] && ((MILVUS_PORT >= 1 && MILVUS_PORT <= 65535)) \
  || fail "Invalid MILVUS_PORT: $MILVUS_PORT"
for boolean_name in \
  MILVUS_ENABLED MILVUS_READ_ENABLED MILVUS_WRITE_ENABLED MILVUS_FALLBACK_ENABLED; do
  boolean_value="${!boolean_name}"
  boolean_value="${boolean_value,,}"
  case "$boolean_value" in
    true|false|1|0|yes|no|on|off) ;;
    *) fail "${boolean_name} must be a boolean, got: ${!boolean_name}" ;;
  esac
done
if [[ "${MILVUS_ENABLED,,}" =~ ^(true|1|yes|on)$ ]]; then
  python3 - "$MILVUS_HOST" "$MILVUS_PORT" "$MILVUS_QUERY_TIMEOUT_SECONDS" <<'PY'
import socket
import sys

host, port_text, timeout_text = sys.argv[1:]
timeout = float(timeout_text)
if timeout <= 0:
    raise SystemExit("MILVUS_QUERY_TIMEOUT_SECONDS must be greater than zero")
with socket.create_connection((host, int(port_text)), timeout=timeout):
    pass
print(f"milvus_preflight=PASS endpoint={host}:{port_text}")
PY
fi

exec 9>"$WORK_ROOT/.platform-deploy.lock"
flock -n 9 || fail "Another MomentSeek deployment is running"
SOURCE_SHA="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
SOURCE_SHORT="$(git -C "$SOURCE_DIR" rev-parse --short=12 HEAD)"
IMAGE_NAME="${IMAGE_NAME:-${IMAGE_REPO}:${SOURCE_SHORT}}"
grep -q '"vite": "6.4.3"' "$SOURCE_DIR/frontend/package.json" \
  || fail "Frontend dependencies are not pinned to the validated Vite 6.4.3 toolchain"
grep -q '"lockfileVersion": 3' "$SOURCE_DIR/frontend/package-lock.json" \
  || fail "Unsupported or missing npm lock file"

log "1/8 Resource and ownership checks"
mkdir -p "$MODEL_DIR" "$RUNTIME_DIR" "$LOG_DIR" "$BUILD_DIR/wheels"
OCR_OM_ROOT="$MODEL_DIR/rapidocr/ascend/910b4-cann9-profile"
[[ -d "$OCR_OM_ROOT/det" ]] || fail "Required OCR Det OM directory is missing: $OCR_OM_ROOT/det"
[[ -d "$OCR_OM_ROOT/cls" ]] || fail "Required OCR Cls OM directory is missing: $OCR_OM_ROOT/cls"
[[ -f "$OCR_OM_ROOT/rec-dynamic-width-b5/PP-OCRv6_rec_small-b5-dynamic-width-1600.om" ]] \
  || fail "Required OCR Rec dynamic OM is missing"
[[ -f "$OCR_OM_ROOT/rec-dynamic-width-b5/PP-OCRv6_rec_small-b5-dynamic-width-2048.om" ]] \
  || fail "Required OCR Rec wide dynamic OM is missing"
[[ -f "$INSIGHTFACE_WHEEL" ]] || fail "Required build artifact is missing: $INSIGHTFACE_WHEEL"
printf '%s  %s\n' "$INSIGHTFACE_SHA256" "$INSIGHTFACE_WHEEL" | sha256sum -c -
cp -f "$INSIGHTFACE_WHEEL" "$BUILD_DIR/wheels/"
df -h "$WORK_ROOT"
npu-smi info -t proc-mem -i "$NPU_ID" -c 0 2>&1 | tee "$LOG_DIR/npu-${NPU_ID}-before-deploy.log"
if ss -lnt 2>/dev/null | grep -Eq ":${APP_PORT}[[:space:]]"; then
  existing_running="$(docker container inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  [[ "$existing_running" == "true" ]] \
    || fail "Port ${APP_PORT} is occupied by something other than ${CONTAINER_NAME}"
fi
git -C "$SOURCE_DIR" status -sb
git -C "$SOURCE_DIR" log -1 --oneline
printf '.server-build/\n' >>"$SOURCE_DIR/.git/info/exclude"
sort -u "$SOURCE_DIR/.git/info/exclude" -o "$SOURCE_DIR/.git/info/exclude"

log "2/8 Stage the checked-in production build definition"
cp "$SOURCE_DIR/Dockerfile.ascend" "$BUILD_DIR/Dockerfile"

log "3/8 Check required external endpoints"
for url in \
  https://pypi.org/simple/ \
  https://registry.npmmirror.com/; do
  curl -ILsS --connect-timeout 8 --max-time 30 -o /dev/null \
    -w "url=${url} http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\\n" "$url" || true
done

log "4/8 Build derivative platform image"
docker build --network host \
  --build-arg "ASCEND_RUNTIME_IMAGE=$BASE_IMAGE" \
  --label "org.opencontainers.image.revision=$SOURCE_SHA" \
  --label "org.opencontainers.image.source=https://github.com/502022240018/MomentSeek" \
  -f "$BUILD_DIR/Dockerfile" \
  -t "$IMAGE_NAME" \
  "$SOURCE_DIR" 2>&1 | tee "$LOG_DIR/platform-image-build.log"

log "Verify required production models before quiescing the current platform"
docker run --rm \
  -v "$MODEL_DIR:/app/models:ro" \
  "$IMAGE_NAME" \
  python3 /app/scripts/verify_models.py \
    --manifest /app/deploy/models/ascend-prod.models.json \
    --lock /tmp/ascend-prod.models.lock.json

DEVICE_ARGS=(
  --device "/dev/davinci${NPU_ID}"
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64
)

log "5/8 Quiesce previous platform and run NPU smoke test"
if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
  fail "Stale rollback container exists: ${ROLLBACK_NAME}; inspect it before continuing"
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  existing_running="$(docker container inspect --format '{{.State.Running}}' "$CONTAINER_NAME")"
  if [[ "$existing_running" == "true" ]]; then
    jobs_json="$(curl -fsS --connect-timeout 2 --max-time 10 \
      "http://127.0.0.1:${APP_PORT}/api/jobs")" \
      || fail "Cannot audit active jobs before stopping the existing platform"
    active_jobs="$(python3 -c \
      'import json,sys; print(sum(j.get("status") in {"queued","running"} for j in json.load(sys.stdin)))' \
      <<<"$jobs_json")" \
      || fail "Cannot parse the existing platform job list"
    [[ "$active_jobs" == 0 ]] \
      || fail "Existing platform has ${active_jobs} queued/running job(s); cancel them before deployment"
  fi
  docker rename "$CONTAINER_NAME" "$ROLLBACK_NAME"
  if [[ "$existing_running" == "true" ]]; then
    docker stop --time 30 "$ROLLBACK_NAME" >/dev/null
  fi
fi

# Resident model pools keep the selected NPU allocated for the lifetime of the
# old container.  Stop it before starting the temporary smoke-test container;
# rollback_on_error restores it if the smoke test or replacement fails.
smoke_ok=0
for attempt in 1 2 3; do
  printf 'npu_smoke_attempt=%s/3\n' "$attempt"
  if docker run --rm "${DEVICE_ARGS[@]}" "$IMAGE_NAME" python3 -c \
    'import torch; import torch_npu; import torchaudio; import funasr; import open_clip; from transformers import Siglip2Model; from app.indexing.asr import _load_silero_onnx_vad; _load_silero_onnx_vad(); x=torch.arange(4,dtype=torch.float32,device="npu:0"); print("npu_result",(x*2).cpu().tolist()); print("device_imports_and_silero_onnx=PASS"); print("image_smoke=PASS")'; then
    smoke_ok=1
    break
  fi
  sleep $((attempt * 2))
done
if [[ "$smoke_ok" != 1 ]]; then
  # fail() exits explicitly and therefore is not guaranteed to fire ERR in all
  # Bash contexts. Restore here as well so a failed smoke test never leaves the
  # previously healthy service stopped.
  if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
    docker rename "$ROLLBACK_NAME" "$CONTAINER_NAME" >/dev/null
    docker start "$CONTAINER_NAME" >/dev/null
    printf 'Previous platform container was restored automatically.\n' >&2
  fi
  fail "NPU smoke test failed after 3 attempts"
fi

log "6/8 Start replacement platform container"
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  --cpus "$CONTAINER_CPU_LIMIT" \
  --pids-limit "$CONTAINER_PID_LIMIT" \
  "${DEVICE_ARGS[@]}" \
  -e ENV_PROFILE=prod.ascend \
  -e APP_PORT="$APP_PORT" \
  -e APP_DATA_DIR=/app/runtime \
  -e APP_MODEL_DIR=/app/models \
  -e NPU_ENABLED=true \
  -e NPU_DEVICE_ID=0 \
  -e INDEXER_MODE=daemon \
  -e NPU_WORKER_MODE=isolated \
  -e INDEXER_IDLE_TIMEOUT_SECONDS=0 \
  -e OPENBLAS_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e OPENBLAS_DEFAULT_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e OMP_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e MKL_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e NUMEXPR_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e BLIS_NUM_THREADS="$CPU_THREAD_LIMIT" \
  -e TOKENIZERS_PARALLELISM=false \
  -e VISUAL_MODEL=siglip2-so400m-384 \
  -e VISUAL_HF_CACHE_DIR=/app/models/hf-cache \
  -e FACE_PROVIDER=cann \
  -e FACE_ORT_INTRA_OP_THREADS="$CPU_THREAD_LIMIT" \
  -e FACE_ORT_INTER_OP_THREADS=1 \
  -e ASR_ENGINE=funasr \
  -e ASR_DEVICE=auto \
  -e ASR_VAD_STRATEGY=silero_12s \
  -e ASR_MODEL_LOCAL_FILES_ONLY=true \
  -e ASR_SEMANTIC_LOCAL_FILES_ONLY=true \
  -e SPEAKER_DEVICE=npu \
  -e SPEAKER_MODEL_REPO=/app/models/3D-Speaker \
  -e SPEAKER_MODEL_CACHE_DIR=/app/models/3dspeaker-cache \
  -e MILVUS_ENABLED="$MILVUS_ENABLED" \
  -e MILVUS_HOST="$MILVUS_HOST" \
  -e MILVUS_PORT="$MILVUS_PORT" \
  -e MILVUS_READ_ENABLED="$MILVUS_READ_ENABLED" \
  -e MILVUS_WRITE_ENABLED="$MILVUS_WRITE_ENABLED" \
  -e MILVUS_FALLBACK_ENABLED="$MILVUS_FALLBACK_ENABLED" \
  -e MILVUS_ROLLOUT_PERCENT="$MILVUS_ROLLOUT_PERCENT" \
  -e MILVUS_QUERY_TIMEOUT_SECONDS="$MILVUS_QUERY_TIMEOUT_SECONDS" \
  -e MILVUS_SEARCH_VIDEO_BATCH_SIZE="$MILVUS_SEARCH_VIDEO_BATCH_SIZE" \
  -e SEARCH_PREWARM_ENABLED="$SEARCH_PREWARM_ENABLED" \
  -e SEARCH_PREWARM_REQUIRED="$SEARCH_PREWARM_REQUIRED" \
  -e OCR_ENGINE=rapidocr_acl \
  -e OCR_DEVICE=npu \
  -e OCR_ACL_MODEL_DIR=rapidocr/ascend/910b4-cann9-profile \
  -e MODEL_MANIFEST=/app/deploy/models/ascend-prod.models.json \
  -v "$RUNTIME_DIR:/app/runtime" \
  -v "$MODEL_DIR:/app/models" \
  "$IMAGE_NAME"

log "7/8 API health check"
healthy=0
for attempt in $(seq 1 30); do
  if curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${APP_PORT}/api/health"; then
    printf '\n'
    healthy=1
    break
  fi
  if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if [[ "$healthy" != 1 ]]; then
  docker logs --tail 200 "$CONTAINER_NAME" || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
    docker rename "$ROLLBACK_NAME" "$CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
    printf 'Previous platform container was restored automatically.\n' >&2
  fi
  fail "Platform health check failed"
fi

log "Run production OCR ACL backend self-test"
docker exec -i "$CONTAINER_NAME" python3 - <<'PY'
from app.indexing.ocr import create_ocr_backend

backend = create_ocr_backend(
    "rapidocr_acl",
    device="npu",
    device_id=0,
    model_root="/app/models/rapidocr",
    acl_model_dir="/app/models/rapidocr/ascend/910b4-cann9-profile",
    ocr_version="PP-OCRv6",
    det_lang="ch",
    rec_lang="ch",
    model_type="small",
    npu_self_test=True,
)
print("ocr_acl_providers=", backend.providers)
backend.close()
print("OCR_ACL_BACKEND_SELF_TEST=PASS")
PY

log "8/8 Verify required model inventory in the replacement container"
docker exec "$CONTAINER_NAME" python3 /app/scripts/verify_models.py \
  --manifest /app/deploy/models/ascend-prod.models.json \
  --lock /app/runtime/ascend-prod.models.lock.json

if docker container inspect "$ROLLBACK_NAME" >/dev/null 2>&1; then
  docker rm "$ROLLBACK_NAME" >/dev/null
fi

docker image tag "$IMAGE_NAME" "${IMAGE_REPO}:current"

docker ps --filter "name=^/${CONTAINER_NAME}$" --format 'name={{.Names}} image={{.Image}} status={{.Status}}'
npu-smi info -t proc-mem -i "$NPU_ID" -c 0 2>&1 || true
printf '\nDEPLOY_OK=1\nPLATFORM_URL=http://127.0.0.1:%s\nCONTAINER=%s\nIMAGE=%s\n' \
  "$APP_PORT" "$CONTAINER_NAME" "$IMAGE_NAME"
