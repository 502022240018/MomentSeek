#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
PHYSICAL_NPU="${PHYSICAL_NPU:-7}"
THREAD_LIMITS_CSV="${THREAD_LIMITS_CSV:-4,8,16,24,24,16,8,4}"
BENCHMARK_CASE="${BENCHMARK_CASE:-semantic_only}"
CPU_LIMIT="${CPU_LIMIT:-24}"
SEMANTIC_TEXT_COUNT="${SEMANTIC_TEXT_COUNT:-2000}"
MAX_FRAMES="${MAX_FRAMES:-300}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}"
RUN_ID="${RUN_ID:-$(date '+%F-%H%M%S')}"
SWEEP_DIR="${SWEEP_DIR:-${LOG_DIR}/ocr-thread-sweep-${RUN_ID}}"

fail() { printf '\nOCR_THREAD_SWEEP_FAILED: %s\n' "$*" >&2; exit 1; }
[[ -x "$SOURCE_DIR/scripts/run_ocr_root_cause_suite.sh" ]] \
  || fail "root-cause suite is missing or not executable"
IFS=',' read -r -a THREAD_LIMITS <<<"$THREAD_LIMITS_CSV"
(( ${#THREAD_LIMITS[@]} > 0 )) || fail "THREAD_LIMITS_CSV must not be empty"
for value in "${THREAD_LIMITS[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "invalid thread limit: $value"
done
case "$BENCHMARK_CASE" in
  semantic_only|ocr_only|ocr_semantic|face_ocr_semantic) ;;
  *) fail "unsupported BENCHMARK_CASE: $BENCHMARK_CASE" ;;
esac

mkdir -p "$SWEEP_DIR"
printf 'thread_limits=%s\nbenchmark_case=%s\ncpu_limit=%s\nsemantic_text_count=%s\nmax_frames=%s\n' \
  "$THREAD_LIMITS_CSV" "$BENCHMARK_CASE" "$CPU_LIMIT" \
  "$SEMANTIC_TEXT_COUNT" "$MAX_FRAMES" >"$SWEEP_DIR/config.txt"

iteration=0
for thread_limit in "${THREAD_LIMITS[@]}"; do
  iteration=$((iteration + 1))
  run_name="$(printf 'run-%02d-thread-%02d' "$iteration" "$thread_limit")"
  printf '\n=== %s start %s ===\n' "$run_name" "$(date '+%F %T')"
  WORK_ROOT="$WORK_ROOT" SOURCE_DIR="$SOURCE_DIR" LOG_DIR="$LOG_DIR" \
  PHYSICAL_NPU="$PHYSICAL_NPU" CASES_CSV="$BENCHMARK_CASE" \
  THREAD_LIMIT="$thread_limit" CPU_LIMIT="$CPU_LIMIT" \
  CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
  SEMANTIC_TEXT_COUNT="$SEMANTIC_TEXT_COUNT" MAX_FRAMES="$MAX_FRAMES" \
  CONTAINER_PREFIX="momentseek-thread-sweep-${RUN_ID}-${iteration}-${thread_limit}" \
  SUITE_DIR="$SWEEP_DIR/$run_name" \
    bash "$SOURCE_DIR/scripts/run_ocr_root_cause_suite.sh"
done

python3 - "$SWEEP_DIR" "$BENCHMARK_CASE" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
benchmark_case = sys.argv[2]
rows = []
for run_dir in sorted(root.glob("run-*-thread-*")):
    if not run_dir.is_dir():
        continue
    report = json.loads((run_dir / "cases" / benchmark_case / "report.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())["cases"][0]
    phases = {phase["name"]: phase for phase in report.get("phases", [])}
    phase_seconds = {
        name: value.get("elapsed_seconds") for name, value in phases.items()
        if value.get("elapsed_seconds") is not None
    }
    thread_limit = int(report["thread_environment"]["OPENBLAS_NUM_THREADS"])
    rows.append({
        "run": run_dir.name,
        "thread_limit": thread_limit,
        "success": bool(summary["success"]),
        "total_seconds": round(sum(phase_seconds.values()), 3),
        "phase_seconds": json.dumps(phase_seconds, sort_keys=True),
        "max_threads": summary["max_threads"],
        "max_rss_mb": summary["max_rss_mb"],
        "openblas_warning": summary["openblas_warning"],
        "bad_memory_unallocation": summary["bad_memory_unallocation"],
    })

with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

grouped = defaultdict(list)
for row in rows:
    grouped[row["thread_limit"]].append(row)
aggregate = []
for limit, values in sorted(grouped.items()):
    aggregate.append({
        "thread_limit": limit,
        "runs": len(values),
        "all_success": all(row["success"] for row in values),
        "median_total_seconds": round(statistics.median(row["total_seconds"] for row in values), 3),
        "max_threads": max(row["max_threads"] for row in values),
        "max_rss_mb": max(row["max_rss_mb"] for row in values),
        "any_openblas_warning": any(row["openblas_warning"] for row in values),
        "any_bad_memory_unallocation": any(row["bad_memory_unallocation"] for row in values),
    })
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
print(json.dumps(aggregate, indent=2))
PY

printf '\nOCR_THREAD_SWEEP_COMPLETE=1\nRESULTS=%s\nAGGREGATE=%s\n' \
  "$SWEEP_DIR/results.csv" "$SWEEP_DIR/aggregate.json"
