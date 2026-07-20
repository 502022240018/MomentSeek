#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORK_ROOT}/vlm-exp}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
PLATFORM_PORT="${PLATFORM_PORT:-8000}"
PHYSICAL_NPU="${PHYSICAL_NPU:-7}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-2B-Instruct}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_HOST="${MODEL_HOST:-${EXPERIMENT_ROOT}/models/${MODEL_NAME}}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-agent/ascend-qwen3-vl-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/output}"

fail() { printf '\nQWEN3_VL_EXPERIMENT_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nQWEN3_VL_EXPERIMENT_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

for command_name in curl docker git npu-smi python3; do
  command -v "$command_name" >/dev/null || fail "missing command: $command_name"
done

[[ -d "$SOURCE_DIR/.git" ]] || fail "git repository missing: $SOURCE_DIR"
cd "$SOURCE_DIR"
current_branch="$(git branch --show-current)"
printf 'repository=%s\nbranch=%s\ncommit=%s\n' \
  "$SOURCE_DIR" "$current_branch" "$(git rev-parse --short HEAD)"
if [[ "$current_branch" != "$EXPECTED_BRANCH" ]]; then
  fail "expected branch ${EXPECTED_BRANCH}; fetch and switch to the experiment branch first"
fi
[[ -x scripts/run_qwen3_vl_ascend_smoke.sh ]] \
  || fail "smoke runner missing or not executable"

tracked_changes="$(git status --porcelain --untracked-files=no)"
[[ -z "$tracked_changes" ]] || fail "tracked source files have local modifications"

container_status="$(docker container inspect --format '{{.State.Status}}' "$PLATFORM_CONTAINER" 2>/dev/null)" \
  || fail "platform container missing: $PLATFORM_CONTAINER"
[[ "$container_status" == "running" ]] || fail "platform container is not running"
health="$(curl -fsS --connect-timeout 2 --max-time 10 \
  "http://127.0.0.1:${PLATFORM_PORT}/api/health")" \
  || fail "platform health check failed"
printf 'platform_health=%s\n' "$health"

mkdir -p "$EXPERIMENT_ROOT/models" "$EXPERIMENT_ROOT/input" "$OUTPUT_DIR"
if [[ -f "$MODEL_HOST/config.json" ]] \
  && find "$MODEL_HOST" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
  printf 'model=%s\nmodel_size=%s\n' \
    "$MODEL_HOST" "$(du -sh "$MODEL_HOST" | awk '{print $1}')"
else
  printf 'model_missing_or_incomplete=%s\nmodelscope_id=%s\n' "$MODEL_HOST" "$MODEL_ID"
  printf 'The smoke runner will download the official model from ModelScope.\n'
fi

printf '\nAvailable runtime videos (first 10):\n'
find "$RUNTIME_DIR/uploads" -type f \
  \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.webm' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null | head -10 || true

printf '\nNPU preflight:\n'
npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" \
  || fail "NPU ${PHYSICAL_NPU} is not empty"

WORK_ROOT="$WORK_ROOT" SOURCE_DIR="$SOURCE_DIR" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
RUNTIME_DIR="$RUNTIME_DIR" PLATFORM_CONTAINER="$PLATFORM_CONTAINER" \
PHYSICAL_NPU="$PHYSICAL_NPU" MODEL_NAME="$MODEL_NAME" MODEL_HOST="$MODEL_HOST" \
MODEL_ID="$MODEL_ID" \
OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/run_qwen3_vl_ascend_smoke.sh

latest_result="$(find "$OUTPUT_DIR" -type f -name '*.json' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$latest_result" && -f "$latest_result" ]] \
  || fail "no JSON result found after smoke test"

printf '\nLatest result:\n'
python3 -m json.tool "$latest_result"
printf '\nNPU release check:\n'
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nQWEN3_VL_EXPERIMENT_OK=1\nRESULT=%s\n' "$latest_result"
