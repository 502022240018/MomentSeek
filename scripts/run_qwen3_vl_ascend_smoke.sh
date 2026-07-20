#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORK_ROOT}/vlm-exp}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-2B-Instruct}"
MODEL_HOST="${MODEL_HOST:-${EXPERIMENT_ROOT}/models/${MODEL_NAME}}"
IMAGE_HOST="${IMAGE_HOST:-${EXPERIMENT_ROOT}/input/test.jpg}"
RUNS="${RUNS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
VENV_HOST="${VENV_HOST:-${EXPERIMENT_ROOT}/venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/output}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momentseek-qwen3-vl-smoke}"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"

fail() { printf '\nQWEN3_VL_SMOKE_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nQWEN3_VL_SMOKE_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

for command_name in docker npu-smi; do
  command -v "$command_name" >/dev/null || fail "missing command: $command_name"
done
[[ "$PHYSICAL_NPU" =~ ^[0-9]+$ ]] || fail "PHYSICAL_NPU must be numeric"
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || fail "RUNS must be positive"
[[ "$WARMUP_RUNS" =~ ^[0-9]+$ ]] || fail "WARMUP_RUNS must be non-negative"
[[ -d "$SOURCE_DIR" ]] || fail "source directory missing: $SOURCE_DIR"
[[ -f "$SOURCE_DIR/scripts/qwen3_vl_ascend_smoke.py" ]] || fail "smoke script missing; update the repository"
[[ -d "$MODEL_HOST" ]] || fail "model directory missing: $MODEL_HOST"
[[ -f "$MODEL_HOST/config.json" ]] || fail "model config missing: $MODEL_HOST/config.json"
[[ -f "$IMAGE_HOST" ]] || fail "test image missing: $IMAGE_HOST"
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || fail "image missing: $IMAGE_NAME"
docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1 \
  && fail "experiment container already exists: $EXPERIMENT_NAME"

npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" \
  || fail "NPU ${PHYSICAL_NPU} is not empty"

mkdir -p "$EXPERIMENT_ROOT" "$OUTPUT_DIR"
if [[ "$INSTALL_DEPS" == "true" || ( "$INSTALL_DEPS" == "auto" && ! -x "$VENV_HOST/bin/python" ) ]]; then
  docker run --rm --name "${EXPERIMENT_NAME}-env" \
    -v "$EXPERIMENT_ROOT:/work" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
    "$IMAGE_NAME" bash -lc '
      set -e
      source /usr/local/Ascend/cann/set_env.sh
      python3 -m venv --system-site-packages /work/venv
      /work/venv/bin/python -m pip install --upgrade pip
      /work/venv/bin/pip install \
        "transformers>=4.57,<5" "qwen-vl-utils==0.0.14" accelerate pillow
      /work/venv/bin/python -c \
        "import torch, torch_npu, transformers; print(torch.__version__, torch_npu.__version__, transformers.__version__)"
    '
fi
[[ -x "$VENV_HOST/bin/python" ]] || fail "venv missing; run with INSTALL_DEPS=true"

timestamp="$(date +%Y%m%d-%H%M%S)"
output_host="${OUTPUT_DIR}/${MODEL_NAME}-${timestamp}.json"
log_host="${OUTPUT_DIR}/${MODEL_NAME}-${timestamp}.log"

docker run --rm --name "$EXPERIMENT_NAME" \
  --device "/dev/davinci${PHYSICAL_NPU}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$EXPERIMENT_ROOT:/work" -v "$SOURCE_DIR/scripts:/work/repo-scripts:ro" \
  -v "$MODEL_HOST:/vlm/model:ro" -v "$IMAGE_HOST:/vlm/input:ro" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" bash -lc '
    set -e
    source /usr/local/Ascend/cann/set_env.sh
    /work/venv/bin/python /work/repo-scripts/qwen3_vl_ascend_smoke.py \
      --model "$1" --image "$2" --runs "$3" --warmup-runs "$4" \
      --max-new-tokens "$5" --output "$6"
  ' bash /vlm/model /vlm/input \
    "$RUNS" "$WARMUP_RUNS" "$MAX_NEW_TOKENS" "/work/output/$(basename "$output_host")" \
  2>&1 | tee "$log_host"

[[ -f "$output_host" ]] || fail "result was not written: $output_host"
python3 -m json.tool "$output_host"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nQWEN3_VL_SMOKE_OK=1\nRESULT=%s\nLOG=%s\n' "$output_host" "$log_host"
