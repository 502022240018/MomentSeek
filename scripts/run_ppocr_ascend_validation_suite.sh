#!/usr/bin/env bash
set -u -o pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
RUNTIME_DIR="${RUNTIME_DIR:-${WORK_ROOT}/runtime}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"
PHYSICAL_NPU="${PHYSICAL_NPU:-6}"
PLATFORM_CONTAINER="${PLATFORM_CONTAINER:-momentseek-29154-platform}"
MAX_VIDEOS="${MAX_VIDEOS:-3}"
RUN_ID="${RUN_ID:-$(date '+%F-%H%M%S')}"
SUITE_DIR="${LOG_ROOT}/ocr-ascend-suite-${RUN_ID}"
CASE_DIR="${SUITE_DIR}/cases"
STATUS_TSV="${SUITE_DIR}/status.tsv"
SUMMARY_JSON="${SUITE_DIR}/summary.json"
SUMMARY_TXT="${SUITE_DIR}/summary.txt"
ARCHIVE="${SUITE_DIR}.tar.gz"

mkdir -p "$CASE_DIR"
printf 'kind\tcase\tstatus\texit_code\treport\tlog\n' >"$STATUS_TSV"

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >>"$STATUS_TSV"
}

run_dynamic_rec() {
  local log="$CASE_DIR/dynamic-rec.log"
  if PHYSICAL_NPU="$PHYSICAL_NPU" \
      LOG_DIR="$CASE_DIR" \
      EXPERIMENT_NAME="momentseek-rec-dw-${RUN_ID//[^0-9]/}" \
      bash "$SOURCE_DIR/scripts/run_ppocr_rec_dynamic_width_test.sh" >"$log" 2>&1; then
    record dynamic_rec width812 PASS 0 "$CASE_DIR/ppocr-rec-dynamic-width.json" "$log"
  else
    local rc=$?
    record dynamic_rec width812 FAIL "$rc" "$CASE_DIR/ppocr-rec-dynamic-width.json" "$log"
  fi
}

run_real_case() {
  local video="$1" timestamp="$2" case_id="$3"
  local report="$CASE_DIR/${case_id}.json" log="$CASE_DIR/${case_id}.log"
  if PHYSICAL_NPU="$PHYSICAL_NPU" \
      VIDEO_HOST="$video" TIMESTAMP="$timestamp" \
      LOG_DIR="$CASE_DIR" REPORT_HOST="$report" \
      EXPERIMENT_NAME="momentseek-ocr-${case_id//[^a-zA-Z0-9]/}" \
      bash "$SOURCE_DIR/scripts/run_ppocr_all_stages_compare.sh" >"$log" 2>&1; then
    record real_frame "$case_id" PASS 0 "$report" "$log"
  else
    local rc=$?
    record real_frame "$case_id" FAIL "$rc" "$report" "$log"
  fi
}

printf '[%s] OCR Ascend validation suite\n' "$(date '+%F %T')"
printf 'suite_dir=%s\nphysical_npu=%s\nmax_videos=%s\n' \
  "$SUITE_DIR" "$PHYSICAL_NPU" "$MAX_VIDEOS"

if ! docker container inspect "$PLATFORM_CONTAINER" >/dev/null 2>&1; then
  printf 'Platform container is required: %s\n' "$PLATFORM_CONTAINER" >&2
  exit 1
fi
if ! docker exec "$PLATFORM_CONTAINER" command -v ffprobe >/dev/null 2>&1; then
  printf 'ffprobe is required inside platform container: %s\n' "$PLATFORM_CONTAINER" >&2
  exit 1
fi
if ! npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 2>&1 \
    | tee "$SUITE_DIR/npu-before.txt" | grep -q 'No process in device'; then
  printf 'NPU %s is not idle\n' "$PHYSICAL_NPU" >&2
  exit 1
fi

run_dynamic_rec

mapfile -t videos < <(find "$RUNTIME_DIR/uploads" -maxdepth 1 -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) -print | sort | head -n "$MAX_VIDEOS")
if ((${#videos[@]} == 0)); then
  record discovery no_videos FAIL 2 '' "$SUITE_DIR/discovery.log"
  printf 'No uploaded videos found\n' >"$SUITE_DIR/discovery.log"
fi

video_index=0
for video in "${videos[@]}"; do
  video_index=$((video_index + 1))
  video_relative="${video#${RUNTIME_DIR}/}"
  duration="$(docker exec "$PLATFORM_CONTAINER" ffprobe -v error \
    -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \
    "/app/runtime/$video_relative" 2>/dev/null || true)"
  midpoint="$(python3 - "$duration" <<'PY'
import sys
try:
    duration = float(sys.argv[1])
except (ValueError, IndexError):
    duration = 0.0
print(round(max(0.0, duration / 2.0), 3))
PY
)"
  first="$(python3 - "$duration" <<'PY'
import sys
try:
    duration = float(sys.argv[1])
except (ValueError, IndexError):
    duration = 0.0
print(round(min(10.0, max(0.0, duration * 0.1)), 3))
PY
)"
  run_real_case "$video" "$first" "v${video_index}-early"
  if [[ "$midpoint" != "$first" ]]; then
    run_real_case "$video" "$midpoint" "v${video_index}-middle"
  fi
done

python3 - "$STATUS_TSV" "$SUMMARY_JSON" "$SUMMARY_TXT" <<'PY'
import csv
import json
import sys
from pathlib import Path

status_path, json_path, text_path = map(Path, sys.argv[1:])
with status_path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

for row in rows:
    report_path = Path(row["report"]) if row["report"] else None
    if not report_path or not report_path.is_file():
        continue
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["report_error"] = repr(exc)
        continue
    final = report.get("final_result", {})
    row["texts_exact_match"] = final.get("texts_exact_match")
    row["cpu_pipeline_seconds"] = report.get("cpu_pipeline_seconds")
    row["om_pipeline_seconds"] = report.get("om_pipeline_seconds_with_per_stage_load")

passed = sum(row["status"] == "PASS" for row in rows)
failed = len(rows) - passed
summary = {
    "schema_version": 1,
    "completed": True,
    "passed": passed,
    "failed": failed,
    "cases": rows,
}
json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "MomentSeek PP-OCR Ascend validation suite",
    f"cases={len(rows)} passed={passed} failed={failed}",
    "",
]
for row in rows:
    extra = ""
    if "texts_exact_match" in row:
        extra = f" texts_exact_match={row['texts_exact_match']}"
    lines.append(
        f"[{row['status']}] {row['kind']} {row['case']} rc={row['exit_code']}{extra}"
    )
    if row["status"] != "PASS":
        lines.append(f"  log={row['log']}")
text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

npu-smi info -t proc-mem -i "$PHYSICAL_NPU" -c 0 >"$SUITE_DIR/npu-after.txt" 2>&1 || true
tar -czf "$ARCHIVE" -C "$LOG_ROOT" "$(basename "$SUITE_DIR")"
printf '\nOCR_ASCEND_SUITE_COMPLETE=1\nSUMMARY=%s\nARCHIVE=%s\n' "$SUMMARY_TXT" "$ARCHIVE"
