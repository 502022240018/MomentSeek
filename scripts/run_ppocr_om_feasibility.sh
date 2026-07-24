#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-platform}"
APP_PORT="${APP_PORT:-8000}"
MODEL_ROOT="${MODEL_ROOT:-/app/models/rapidocr}"
PROFILE="${PROFILE:-/app/runtime/ocr-shape-profile.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/models/rapidocr/ascend/910b4-cann9-profile}"
DECODE_HEIGHT="${DECODE_HEIGHT:-720}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-12}"
MAX_VIDEOS="${MAX_VIDEOS:-12}"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"
PRECISION_MODE="${PRECISION_MODE:-must_keep_origin_dtype}"
RUN_ID="$(date '+%F-%H%M%S')"

[[ "$PROFILE" == /app/runtime/* ]] \
  || { printf 'PROFILE must be under /app/runtime: %s\n' "$PROFILE" >&2; exit 1; }
[[ "$OUTPUT_DIR" == /app/models/* ]] \
  || { printf 'OUTPUT_DIR must be under /app/models: %s\n' "$OUTPUT_DIR" >&2; exit 1; }
PROFILE_HOST="$RUNTIME_DIR/${PROFILE#/app/runtime/}"
OUTPUT_RELATIVE="${OUTPUT_DIR#/app/models/}"
MANIFEST_HOST="$WORK_ROOT/models/platform/$OUTPUT_RELATIVE/build-manifest.json"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nOCR_OM_FEASIBILITY_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nOCR_OM_FEASIBILITY_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

for command_name in docker curl python3 tee; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing host command: $command_name"
done
[[ -f "$SOURCE_DIR/scripts/ocr_shape_profile.py" ]] || fail "Missing profiler in $SOURCE_DIR"
[[ -f "$SOURCE_DIR/scripts/build_ppocr_om_from_profile.py" ]] || fail "Missing OM builder in $SOURCE_DIR"
docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 \
  || fail "Container not found: $CONTAINER_NAME"
[[ "$(docker container inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]] \
  || fail "Container is not running: $CONTAINER_NAME"
mkdir -p "$LOG_DIR"

log "1/6 Audit platform health and active indexing jobs"
curl -fsS --connect-timeout 2 --max-time 10 \
  "http://127.0.0.1:${APP_PORT}/api/health" | python3 -m json.tool
jobs_json="$(curl -fsS --connect-timeout 2 --max-time 10 \
  "http://127.0.0.1:${APP_PORT}/api/jobs")"
active_jobs="$(python3 -c \
  'import json,sys; print(sum(j.get("status") in {"queued","running"} for j in json.load(sys.stdin)))' \
  <<<"$jobs_json")"
[[ "$active_jobs" == 0 ]] \
  || fail "Platform has ${active_jobs} queued/running job(s); finish or cancel them first"

log "2/6 Check container tools and model files"
docker exec "$CONTAINER_NAME" sh -c '
  set -eu
  command -v atc
  test -f /app/models/rapidocr/PP-OCRv6_det_small.onnx
  test -f /app/models/rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx
  test -f /app/models/rapidocr/PP-OCRv6_rec_small.onnx
  python3 -c "import cv2, numpy, onnxruntime, rapidocr; print(\"OCR experiment imports: PASS\")"
'
CANN_ENV_SCRIPT="$(docker exec "$CONTAINER_NAME" sh -c '
  for candidate in \
    /usr/local/Ascend/cann/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    if [ -f "$candidate" ]; then
      printf "%s\n" "$candidate"
      exit 0
    fi
  done
  exit 1
')" || fail "Cannot find the CANN set_env.sh inside the container"
printf 'cann_env_script=%s\n' "$CANN_ENV_SCRIPT"
docker exec -e CANN_ENV_SCRIPT="$CANN_ENV_SCRIPT" "$CONTAINER_NAME" sh -lc '
  . "$CANN_ENV_SCRIPT"
  python3 -c "import tbe; print(\"CANN TBE import: PASS\")"
' || fail "CANN set_env.sh did not make the tbe module importable"

log "3/6 Stage versioned experiment scripts in the container"
docker cp "$SOURCE_DIR/scripts/ocr_shape_profile.py" \
  "$CONTAINER_NAME:/tmp/ocr_shape_profile.py"
docker cp "$SOURCE_DIR/scripts/build_ppocr_om_from_profile.py" \
  "$CONTAINER_NAME:/tmp/build_ppocr_om_from_profile.py"

log "4/6 Profile real OCR tensor shapes"
PROFILE_LOG="$LOG_DIR/ppocr-shape-profile-${RUN_ID}.log"
docker exec -e PYTHONPATH=/app/backend "$CONTAINER_NAME" \
  python3 /tmp/ocr_shape_profile.py \
  --video-root /app/runtime/uploads \
  --model-root "$MODEL_ROOT" \
  --output "$PROFILE" \
  --decode-height "$DECODE_HEIGHT" \
  --frames-per-video "$FRAMES_PER_VIDEO" \
  --max-videos "$MAX_VIDEOS" \
  2>&1 | tee "$PROFILE_LOG"

log "5/6 Print profile coverage and compile missing exact-shape OM artifacts"
python3 - "$PROFILE_HOST" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print("profile=", path)
print("videos=", len(data["videos"]))
for video in data["videos"]:
    print("video", video["source_size"], "seconds=", video["duration"],
          "frames=", video["sampled_frames"], "path=", video["path"])
print(json.dumps(data["tensor_shapes"], ensure_ascii=False, indent=2))
PY

BUILD_LOG="$LOG_DIR/ppocr-om-build-${RUN_ID}.log"
log "Compile one detector artifact as the ATC gate"
docker exec -e CANN_ENV_SCRIPT="$CANN_ENV_SCRIPT" "$CONTAINER_NAME" sh -lc \
  '. "$CANN_ENV_SCRIPT"
   export PYTHONPATH="/app/backend:${PYTHONPATH:-}"
   python3 /tmp/build_ppocr_om_from_profile.py \
     --profile "$1" --model-root "$2" --output-dir "$3" \
     --soc-version "$4" --precision-mode "$5" \
     --stages det --max-artifacts 1' \
  sh "$PROFILE" "$MODEL_ROOT" "$OUTPUT_DIR" "$SOC_VERSION" "$PRECISION_MODE" \
  2>&1 | tee -a "$BUILD_LOG"

log "ATC gate passed; compile all remaining exact-shape artifacts"
docker exec -e CANN_ENV_SCRIPT="$CANN_ENV_SCRIPT" "$CONTAINER_NAME" sh -lc \
  '. "$CANN_ENV_SCRIPT"
   export PYTHONPATH="/app/backend:${PYTHONPATH:-}"
   python3 /tmp/build_ppocr_om_from_profile.py \
     --profile "$1" --model-root "$2" --output-dir "$3" \
     --soc-version "$4" --precision-mode "$5"' \
  sh "$PROFILE" "$MODEL_ROOT" "$OUTPUT_DIR" "$SOC_VERSION" "$PRECISION_MODE" \
  2>&1 | tee "$BUILD_LOG"

log "6/6 Verify build manifest"
python3 - "$MANIFEST_HOST" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
failed = [
    item for item in data["artifacts"]
    if item.get("returncode") != 0 or not item.get("om_exists")
]
print("manifest=", path)
print("success=", data.get("success"))
print("artifact_count=", len(data["artifacts"]))
print("failed_count=", len(failed))
for item in failed:
    print("FAILED", item["stage"], item["input_shape"], item["log_path"])
if failed or not data.get("success"):
    raise SystemExit(1)
PY

printf '\nOCR_OM_FEASIBILITY_OK=1\nPROFILE=%s\nMANIFEST=%s\nPROFILE_LOG=%s\nBUILD_LOG=%s\n' \
  "$PROFILE_HOST" "$MANIFEST_HOST" "$PROFILE_LOG" "$BUILD_LOG"
printf 'Exact-shape artifacts remain experimental and are not enabled in production.\n'
