#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momentseek-ppocr-acl-compare}"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"
REPORT_HOST="${REPORT_HOST:-${LOG_DIR}/ppocr-det-onnx-om-compare.json}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nPPOCR_ACL_COMPARE_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nPPOCR_ACL_COMPARE_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

for command_name in docker npu-smi; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command: $command_name"
done
[[ -f "$SOURCE_DIR/scripts/compare_ppocr_onnx_om.py" ]] || fail "Comparison script is missing"
[[ -d "$MODEL_DIR" ]] || fail "Model directory is missing: $MODEL_DIR"
mkdir -p "$LOG_DIR"

log "1/4 Verify the approved experiment NPU and image"
printf 'image=%s\nphysical_npu=%s\n' "$IMAGE_NAME" "$PHYSICAL_NPU"
npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" \
  || fail "Physical NPU ${PHYSICAL_NPU} is not empty; refusing to interfere with another process"
if docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1; then
  fail "Experiment container already exists: $EXPERIMENT_NAME"
fi

log "2/4 Verify model artifacts"
ONNX_HOST="$MODEL_DIR/rapidocr/PP-OCRv6_det_small.onnx"
OM_HOST="$MODEL_DIR/rapidocr/ascend/910b4-cann9-profile/det/PP-OCRv6_det_small-1x3x736x1312.om"
[[ -f "$ONNX_HOST" ]] || fail "ONNX model is missing: $ONNX_HOST"
[[ -f "$OM_HOST" ]] || fail "OM model is missing: $OM_HOST"

log "3/4 Run isolated CPU ONNX versus NPU OM comparison"
docker run --rm \
  --name "$EXPERIMENT_NAME" \
  --device "/dev/davinci${PHYSICAL_NPU}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_DIR:/app/models:ro" \
  -v "$SOURCE_DIR/scripts:/work/scripts:ro" \
  -v "$LOG_DIR:/work/logs" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" sh -lc '
    set -eu
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    . /usr/local/Ascend/cann/set_env.sh
    python3 /work/scripts/compare_ppocr_onnx_om.py \
      --onnx /app/models/rapidocr/PP-OCRv6_det_small.onnx \
      --om /app/models/rapidocr/ascend/910b4-cann9-profile/det/PP-OCRv6_det_small-1x3x736x1312.om \
      --shape 1 3 736 1312 \
      --device-id 0 --warmup 2 --runs 5 \
      --output /work/logs/ppocr-det-onnx-om-compare.json
  '

log "4/4 Show report and released NPU resources"
python3 -m json.tool "$REPORT_HOST"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nPPOCR_ACL_COMPARE_OK=1\nREPORT=%s\n' "$REPORT_HOST"
