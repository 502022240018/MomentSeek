#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momentseek-ppocr-real-frame}"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"
VIDEO_HOST="${VIDEO_HOST:-$(find "$RUNTIME_DIR/uploads" -type f -name '*.mp4' -print -quit)}"
TIMESTAMP="${TIMESTAMP:-0}"
REPORT_HOST="${REPORT_HOST:-${LOG_DIR}/ppocr-real-frame-compare.json}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nPPOCR_REAL_FRAME_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nPPOCR_REAL_FRAME_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

[[ -f "$VIDEO_HOST" ]] || fail "Video is missing; set VIDEO_HOST explicitly: $VIDEO_HOST"
case "$VIDEO_HOST" in
  "$RUNTIME_DIR"/uploads/*) ;;
  *) fail "VIDEO_HOST must be under $RUNTIME_DIR/uploads" ;;
esac
VIDEO_RELATIVE="${VIDEO_HOST#${RUNTIME_DIR}/}"
mkdir -p "$LOG_DIR"

log "1/3 Require an idle experiment NPU"
npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" \
  || fail "Physical NPU ${PHYSICAL_NPU} is not empty"
docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1 \
  && fail "Experiment container already exists: $EXPERIMENT_NAME"

log "2/3 Run the real-frame RapidOCR comparison"
printf 'video=%s\ntimestamp=%s\nimage=%s\n' "$VIDEO_HOST" "$TIMESTAMP" "$IMAGE_NAME"
docker run --rm --name "$EXPERIMENT_NAME" \
  --device "/dev/davinci${PHYSICAL_NPU}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_DIR:/app/models:ro" \
  -v "$RUNTIME_DIR:/app/runtime:ro" \
  -v "$SOURCE_DIR/scripts:/work/scripts:ro" \
  -v "$LOG_DIR:/work/logs" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" sh -lc '
    set -eu
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    . /usr/local/Ascend/cann/set_env.sh
    export PYTHONPATH="/app/backend:/work/scripts:${PYTHONPATH:-}"
    python3 /work/scripts/compare_ppocr_real_frame.py \
      --video "$1" --timestamp "$2" --decode-height 720 \
      --model-root /app/models/rapidocr \
      --om-root /app/models/rapidocr/ascend/910b4-cann9-profile \
      --device-id 0 --output /work/logs/ppocr-real-frame-compare.json
  ' sh "/app/runtime/$VIDEO_RELATIVE" "$TIMESTAMP"

log "3/3 Show report and released resources"
python3 -m json.tool "$REPORT_HOST"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nPPOCR_REAL_FRAME_OK=1\nREPORT=%s\n' "$REPORT_HOST"
