#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
VIDEO_HOST="${VIDEO_HOST:-${RUNTIME_DIR}/uploads/8e43cd0b84b74077b7f652b09374da9e.mp4}"
TIMESTAMPS="${TIMESTAMPS:-0 1 34 35 57 58}"
RUN_ID="${RUN_ID:-$(date '+%F-%H%M%S')}"
OUTPUT_HOST="${OUTPUT_HOST:-${LOG_DIR}/ocr-acl-diagnostic-${RUN_ID}}"
ARCHIVE_HOST="${OUTPUT_HOST}.tar.gz"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momentseek-ocr-acl-diagnostic}"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"

fail() { printf '\nOCR_ACL_DIAGNOSTIC_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nOCR_ACL_DIAGNOSTIC_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

[[ -f "$VIDEO_HOST" ]] || fail "video is missing: $VIDEO_HOST"
case "$VIDEO_HOST" in "$RUNTIME_DIR"/uploads/*) ;; *) fail "VIDEO_HOST must be under runtime/uploads" ;; esac
VIDEO_RELATIVE="${VIDEO_HOST#${RUNTIME_DIR}/}"
mkdir -p "$OUTPUT_HOST"

npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" || fail "NPU ${PHYSICAL_NPU} is not idle"
docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1 \
  && fail "experiment container already exists: $EXPERIMENT_NAME"

printf 'video=%s\ntimestamps=%s\nimage=%s\nphysical_npu=%s\noutput=%s\n' \
  "$VIDEO_HOST" "$TIMESTAMPS" "$IMAGE_NAME" "$PHYSICAL_NPU" "$OUTPUT_HOST"

docker run --rm --name "$EXPERIMENT_NAME" \
  --device "/dev/davinci${PHYSICAL_NPU}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_DIR:/app/models:ro" -v "$RUNTIME_DIR:/app/runtime:ro" \
  -v "$SOURCE_DIR/scripts:/work/scripts:ro" -v "$OUTPUT_HOST:/work/output" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" sh -lc '
    set -eu
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    . /usr/local/Ascend/cann/set_env.sh
    export PYTHONPATH="/app/backend:/work/scripts:${PYTHONPATH:-}"
    python3 /work/scripts/ocr_acl_diagnostic.py \
      --video "$1" --timestamps ${2} --decode-height 720 \
      --model-root /app/models/rapidocr \
      --om-root /app/models/rapidocr/ascend/910b4-cann9-profile \
      --device-id 0 --output-dir /work/output
  ' sh "/app/runtime/$VIDEO_RELATIVE" "$TIMESTAMPS"

tar -czf "$ARCHIVE_HOST" -C "$LOG_DIR" "$(basename "$OUTPUT_HOST")"
cat "$OUTPUT_HOST/summary.txt"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nOCR_ACL_DIAGNOSTIC_COMPLETE=1\nHTML=%s\nJSON=%s\nSUMMARY=%s\nARCHIVE=%s\n' \
  "$OUTPUT_HOST/report.html" "$OUTPUT_HOST/report.json" \
  "$OUTPUT_HOST/summary.txt" "$ARCHIVE_HOST"
