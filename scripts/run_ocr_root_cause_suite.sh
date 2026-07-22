#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
APP_PORT="${APP_PORT:-8000}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
VIDEO_HOST="${VIDEO_HOST:-${RUNTIME_DIR}/uploads/dd75f7ce5aa04f57b9a28a08d91f37ac.mkv}"
SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-300}"
SEMANTIC_TEXT_COUNT="${SEMANTIC_TEXT_COUNT:-2000}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-900}"
CPU_LIMIT="${CPU_LIMIT:-16}"
MEMORY_LIMIT="${MEMORY_LIMIT:-32g}"
PIDS_LIMIT="${PIDS_LIMIT:-512}"
THREAD_LIMIT="${THREAD_LIMIT:-}"
CASES_CSV="${CASES_CSV:-semantic_only,ocr_only,ocr_semantic,face_ocr_semantic}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-momentseek-ocr-root}"
RUN_ID="${RUN_ID:-$(date '+%F-%H%M%S')}"
SUITE_DIR="${SUITE_DIR:-${LOG_DIR}/ocr-root-cause-suite-${RUN_ID}}"
ARCHIVE="${SUITE_DIR}.tar.gz"
IMAGE_NAME="${IMAGE_NAME:-$(docker container inspect --format '{{.Config.Image}}' "$PLATFORM_CONTAINER")}"
IFS=',' read -r -a CASES <<<"$CASES_CSV"

fail() { printf '\nOCR_ROOT_CAUSE_SUITE_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nOCR_ROOT_CAUSE_SUITE_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

[[ -f "$VIDEO_HOST" ]] || fail "video is missing: $VIDEO_HOST"
case "$VIDEO_HOST" in "$RUNTIME_DIR"/uploads/*) ;; *) fail "VIDEO_HOST must be under runtime/uploads" ;; esac
(( ${#CASES[@]} > 0 )) || fail "CASES_CSV must contain at least one case"
for case_name in "${CASES[@]}"; do
  case "$case_name" in
    semantic_only|ocr_only|ocr_semantic|face_ocr_semantic) ;;
    *) fail "unsupported case in CASES_CSV: $case_name" ;;
  esac
done
THREAD_ENV_ARGS=()
if [[ -n "$THREAD_LIMIT" ]]; then
  [[ "$THREAD_LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "THREAD_LIMIT must be a positive integer"
  for variable in OPENBLAS_NUM_THREADS OPENBLAS_DEFAULT_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS; do
    THREAD_ENV_ARGS+=( -e "${variable}=${THREAD_LIMIT}" )
  done
  THREAD_ENV_ARGS+=( -e TOKENIZERS_PARALLELISM=false )
fi
VIDEO_RELATIVE="${VIDEO_HOST#${RUNTIME_DIR}/}"
command -v timeout >/dev/null || fail "host command is missing: timeout"
mkdir -p "$SUITE_DIR/cases"

active_jobs="$(curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${APP_PORT}/api/jobs" | python3 -c '
import json,sys
print(sum(j.get("status") in {"queued","running"} for j in json.load(sys.stdin)))
')" || fail "cannot audit active platform jobs"
[[ "$active_jobs" == "0" ]] || fail "platform has ${active_jobs} queued/running job(s); cancel or finish them first"

npu_processes="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1)"
printf '%s\n' "$npu_processes"
grep -q 'No process in device' <<<"$npu_processes" || fail "NPU ${PHYSICAL_NPU} is not idle"

cat >"$SUITE_DIR/config.txt" <<EOF
image=$IMAGE_NAME
physical_npu=$PHYSICAL_NPU
video=$VIDEO_HOST
sample_fps=$SAMPLE_FPS
max_frames=$MAX_FRAMES
semantic_text_count=$SEMANTIC_TEXT_COUNT
case_timeout_seconds=$CASE_TIMEOUT_SECONDS
cpu_limit=$CPU_LIMIT
memory_limit=$MEMORY_LIMIT
pids_limit=$PIDS_LIMIT
thread_limit=${THREAD_LIMIT:-unset}
cases=$CASES_CSV
container_prefix=$CONTAINER_PREFIX
EOF
cat "$SUITE_DIR/config.txt"

monitor_case() {
  local container="$1"
  local output="$2"
  local attempts=0
  while ! docker container inspect "$container" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    [[ "$attempts" -lt 30 ]] || return 0
    sleep 1
  done
  while [[ "$(docker container inspect --format '{{.State.Running}}' "$container" 2>/dev/null || true)" == "true" ]]; do
    {
      printf 'MONITOR_TIME=%s\n' "$(date '+%F %T')"
      docker stats --no-stream --format \
        'cpu={{.CPUPerc}} mem={{.MemUsage}} block={{.BlockIO}} pids={{.PIDs}}' "$container" 2>&1 || true
      docker exec "$container" sh -lc '
        pid="$(ps -eo pid,args | awk "/[o]cr_root_cause_probe.py/{print \$1; exit}")"
        echo "probe_pid=$pid"
        if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
          grep -E "^(State|Threads|VmRSS|VmSize):" "/proc/$pid/status" || true
          printf "wchan="; cat "/proc/$pid/wchan" 2>/dev/null || true; echo
        fi
        echo "processes=$(find /proc -maxdepth 1 -type d -name \"[0-9]*\" | wc -l)"
        ps -eo comm= | sort | uniq -c | sort -nr | head -12
      ' 2>&1 || true
      npu-smi info -t usages -i "$PHYSICAL_NPU" -c 0 2>&1 \
        | grep -E 'HBM Usage|Aicore Usage|Aivector Usage|Aicpu Usage|NPU Utilization' || true
      echo
    } >>"$output"
    sleep 5
  done
}

wait_npu_idle() {
  local attempt output
  for attempt in $(seq 1 30); do
    output="$(npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1 || true)"
    if grep -q 'No process in device' <<<"$output"; then
      return 0
    fi
    sleep 1
  done
  printf '%s\n' "$output" >&2
  fail "NPU ${PHYSICAL_NPU} was not released after an isolated case"
}

run_case() {
  local case_name="$1"
  local case_dir="$SUITE_DIR/cases/$case_name"
  local container="${CONTAINER_PREFIX}-${case_name//_/-}"
  mkdir -p "$case_dir"
  docker container inspect "$container" >/dev/null 2>&1 && fail "stale case container exists: $container"
  printf '\n[%s] case=%s\n' "$(date '+%F %T')" "$case_name"

  set +e
  timeout --signal=TERM --kill-after=30s "$CASE_TIMEOUT_SECONDS" \
    docker run --name "$container" \
      --cpus "$CPU_LIMIT" --memory "$MEMORY_LIMIT" --pids-limit "$PIDS_LIMIT" \
      "${THREAD_ENV_ARGS[@]}" \
      --device "/dev/davinci${PHYSICAL_NPU}" \
      --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
      -v "$MODEL_DIR:/app/models:ro" -v "$RUNTIME_DIR:/app/runtime:ro" \
      -v "$SOURCE_DIR/backend:/work/backend:ro" -v "$SOURCE_DIR/scripts:/work/scripts:ro" \
      -v "$case_dir:/work/output" \
      -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
      "$IMAGE_NAME" sh -lc '
        set -eu
        export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
        . /usr/local/Ascend/cann/set_env.sh
        export PYTHONPATH="/work/backend:/work/scripts:${PYTHONPATH:-}"
        exec python3 /work/scripts/ocr_root_cause_probe.py \
          --case "$1" --video "$2" --output-dir /work/output \
          --ocr-model-root /app/models/rapidocr \
          --ocr-om-root /app/models/rapidocr/ascend/910b4-cann9-profile \
          --face-root /app/models/insightface \
          --semantic-model-root /app/models/text-embeddings \
          --sample-fps "$3" --max-frames "$4" --semantic-text-count "$5" \
          --device-id 0
      ' sh "$case_name" "/app/runtime/$VIDEO_RELATIVE" "$SAMPLE_FPS" "$MAX_FRAMES" "$SEMANTIC_TEXT_COUNT" \
      >"$case_dir/case.log" 2>&1 &
  local run_pid=$!
  monitor_case "$container" "$case_dir/host-monitor.log" &
  local monitor_pid=$!
  local rc=0
  if wait "$run_pid"; then
    rc=0
  else
    rc=$?
  fi
  docker stop --time 5 "$container" >/dev/null 2>&1 || true
  wait "$monitor_pid" 2>/dev/null || true
  set -e

  local timed_out=0
  [[ "$rc" == "124" || "$rc" == "137" ]] && timed_out=1
  printf 'exit_code=%s\ntimed_out=%s\n' "$rc" "$timed_out" >"$case_dir/status.txt"
  printf 'case=%s exit_code=%s timed_out=%s\n' "$case_name" "$rc" "$timed_out"
  tail -n 30 "$case_dir/case.log" || true
  docker rm -f "$container" >/dev/null 2>&1 || true
  wait_npu_idle
}

for case_name in "${CASES[@]}"; do
  run_case "$case_name"
done

python3 "$SOURCE_DIR/scripts/summarize_ocr_root_cause_suite.py" \
  --suite-dir "$SUITE_DIR" --cases "$CASES_CSV"
tar -czf "$ARCHIVE" -C "$(dirname "$SUITE_DIR")" "$(basename "$SUITE_DIR")"
npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 || true
printf '\nOCR_ROOT_CAUSE_SUITE_COMPLETE=1\nSUMMARY=%s\nJSON=%s\nARCHIVE=%s\n' \
  "$SUITE_DIR/summary.txt" "$SUITE_DIR/summary.json" "$ARCHIVE"
