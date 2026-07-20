#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momentseek-ppocr-rec-dynamic-width}"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"
ONNX_HOST="${MODEL_DIR}/rapidocr/PP-OCRv6_rec_small.onnx"
OUTPUT_DIR_HOST="${MODEL_DIR}/rapidocr/ascend/910b4-cann9-profile/rec-dynamic-width-b5"
OM_HOST="${OUTPUT_DIR_HOST}/PP-OCRv6_rec_small-b5-dynamic-width.om"
REPORT_HOST="${LOG_DIR}/ppocr-rec-dynamic-width.json"
WIDTHS=(320 384 448 512 576 640 704 768 812 832 896 960 1024)

fail() { printf '\nPPOCR_REC_DYNAMIC_WIDTH_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nPPOCR_REC_DYNAMIC_WIDTH_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

[[ -f "$ONNX_HOST" ]] || fail "missing Rec ONNX: $ONNX_HOST"
mkdir -p "$OUTPUT_DIR_HOST" "$LOG_DIR"
npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" || fail "NPU ${PHYSICAL_NPU} is not empty"
docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1 \
  && fail "experiment container already exists"

dynamic_dims="$(IFS=';'; echo "${WIDTHS[*]}")"
printf 'image=%s\nphysical_npu=%s\ndynamic_widths=%s\n' \
  "$IMAGE_NAME" "$PHYSICAL_NPU" "$dynamic_dims"

docker run --rm --name "$EXPERIMENT_NAME" \
  --device "/dev/davinci${PHYSICAL_NPU}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_DIR:/app/models" -v "$SOURCE_DIR/scripts:/work/scripts:ro" \
  -v "$LOG_DIR:/work/logs" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" sh -lc '
    set -eu
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    . /usr/local/Ascend/cann/set_env.sh
    om=/app/models/rapidocr/ascend/910b4-cann9-profile/rec-dynamic-width-b5/PP-OCRv6_rec_small-b5-dynamic-width.om
    mkdir -p "$(dirname "$om")"
    if [ ! -f "$om" ]; then
      atc \
        --model=/app/models/rapidocr/PP-OCRv6_rec_small.onnx \
        --framework=5 \
        --output="${om%.om}" \
        --input_format=ND \
        --input_shape="x:5,3,48,-1" \
        --dynamic_dims="$1" \
        --soc_version=Ascend910B4 \
        --precision_mode=must_keep_origin_dtype \
        --log=error
    fi
    python3 /work/scripts/compare_ppocr_onnx_om.py \
      --onnx /app/models/rapidocr/PP-OCRv6_rec_small.onnx \
      --om "$om" --shape 5 3 48 812 --dynamic-dims \
      --device-id 0 --warmup 1 --runs 3 \
      --output /work/logs/ppocr-rec-dynamic-width.json
  ' sh "$dynamic_dims"

python3 -m json.tool "$REPORT_HOST"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nPPOCR_REC_DYNAMIC_WIDTH_OK=1\nOM=%s\nREPORT=%s\n' "$OM_HOST" "$REPORT_HOST"
