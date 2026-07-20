#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORK_ROOT}/vlm-exp}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PHYSICAL_NPU="${PHYSICAL_NPU:-7}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-2B-Instruct}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_HOST="${MODEL_HOST:-${EXPERIMENT_ROOT}/models/${MODEL_NAME}}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-auto}"
IMAGE_HOST="${IMAGE_HOST:-${EXPERIMENT_ROOT}/input/test.jpg}"
VIDEO_HOST="${VIDEO_HOST:-}"
FRAME_TIMESTAMP="${FRAME_TIMESTAMP:-10}"
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
[[ "$DOWNLOAD_MODEL" =~ ^(auto|true|false)$ ]] \
  || fail "DOWNLOAD_MODEL must be auto, true, or false"
[[ -d "$SOURCE_DIR" ]] || fail "source directory missing: $SOURCE_DIR"
[[ -f "$SOURCE_DIR/scripts/qwen3_vl_ascend_smoke.py" ]] || fail "smoke script missing; update the repository"
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || fail "image missing: $IMAGE_NAME"
docker container inspect "$EXPERIMENT_NAME" >/dev/null 2>&1 \
  && fail "experiment container already exists: $EXPERIMENT_NAME"

npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" \
  || fail "NPU ${PHYSICAL_NPU} is not empty"

mkdir -p "$EXPERIMENT_ROOT/input" "$OUTPUT_DIR"
if [[ ! -f "$IMAGE_HOST" ]]; then
  if [[ -z "$VIDEO_HOST" ]]; then
    VIDEO_HOST="$(find "$RUNTIME_DIR/uploads" -type f \
      \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.webm' \) \
      -print -quit 2>/dev/null || true)"
  fi
  [[ -f "$VIDEO_HOST" ]] \
    || fail "test image is missing and no runtime video was found; set IMAGE_HOST or VIDEO_HOST"
  printf 'Extracting test frame: video=%s timestamp=%s image=%s\n' \
    "$VIDEO_HOST" "$FRAME_TIMESTAMP" "$IMAGE_HOST"
  mkdir -p "$(dirname "$IMAGE_HOST")"
  docker run --rm --name "${EXPERIMENT_NAME}-frame" \
    -v "$VIDEO_HOST:/source/video:ro" \
    -v "$(dirname "$IMAGE_HOST"):/output" \
    "$IMAGE_NAME" ffmpeg -hide_banner -loglevel error -y \
      -ss "$FRAME_TIMESTAMP" -i /source/video -frames:v 1 \
      "/output/$(basename "$IMAGE_HOST")"
fi
[[ -f "$IMAGE_HOST" ]] || fail "test image missing after frame extraction: $IMAGE_HOST"

if [[ "$INSTALL_DEPS" == "true" \
  || ( "$INSTALL_DEPS" == "auto" \
    && ( ! -x "$VENV_HOST/bin/python" || ! -x "$VENV_HOST/bin/modelscope" ) ) ]]; then
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
        "transformers>=4.57,<5" "qwen-vl-utils==0.0.14" accelerate pillow modelscope
      /work/venv/bin/python -c \
        "import torch, torch_npu, transformers; print(torch.__version__, torch_npu.__version__, transformers.__version__)"
    '
fi
[[ -x "$VENV_HOST/bin/python" ]] || fail "venv missing; run with INSTALL_DEPS=true"

model_complete=false
if [[ -f "$MODEL_HOST/config.json" ]] \
  && find "$MODEL_HOST" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
  model_complete=true
fi
if [[ "$DOWNLOAD_MODEL" == "true" || ( "$DOWNLOAD_MODEL" == "auto" && "$model_complete" != "true" ) ]]; then
  mkdir -p "$MODEL_HOST"
  printf 'Downloading model from ModelScope: id=%s target=%s\n' "$MODEL_ID" "$MODEL_HOST"
  docker run --rm --name "${EXPERIMENT_NAME}-model" \
    -v "$EXPERIMENT_ROOT:/work" \
    -v "$(dirname "$MODEL_HOST"):/download" \
    "$IMAGE_NAME" bash -lc '
      set -e
      /work/venv/bin/modelscope download --model "$1" --local_dir "$2"
    ' bash "$MODEL_ID" "/download/$(basename "$MODEL_HOST")"
fi
[[ -f "$MODEL_HOST/config.json" ]] || fail "model config missing after download: $MODEL_HOST/config.json"
find "$MODEL_HOST" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . \
  || fail "model weights missing after download: $MODEL_HOST"

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
