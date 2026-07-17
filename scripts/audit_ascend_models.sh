#!/usr/bin/env bash
set -uo pipefail

# Read-only deployment/model audit for the shared Ascend server.
# It does not download models, restart services, or modify the model directory.

PROJECT_ROOT="${PROJECT_ROOT:-/home/momentseek-29154/platform}"
MODEL_ROOT="${MODEL_ROOT:-/home/momentseek-29154/models/platform}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-platform}"
APP_PORT="${APP_PORT:-8000}"
MANIFEST="${MANIFEST:-/app/deploy/models/ascend-prod.models.json}"
TMP_LOCK="/tmp/momentseek-model-audit-lock.json"
FAILURES=0

section() {
  printf '\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

run_check() {
  local title="$1"
  shift
  printf '\n--- %s ---\n' "$title"
  if "$@"; then
    printf 'RESULT: PASS - %s\n' "$title"
  else
    local rc=$?
    printf 'RESULT: FAIL(rc=%s) - %s\n' "$rc" "$title"
    FAILURES=$((FAILURES + 1))
  fi
}

section "MomentSeek Ascend model/deployment audit"
date -Is
printf 'project_root=%s\nmodel_root=%s\ncontainer=%s\napp_port=%s\n' \
  "$PROJECT_ROOT" "$MODEL_ROOT" "$CONTAINER_NAME" "$APP_PORT"

section "1/6 Host and checkout"
uname -a || true
df -h /home || true
if [[ -d "$PROJECT_ROOT/.git" ]]; then
  git -C "$PROJECT_ROOT" status --short --branch || true
  git -C "$PROJECT_ROOT" log -1 --format='commit=%H%nsubject=%s%nauthor=%an%ndate=%aI' || true
else
  printf 'ERROR: Git checkout not found: %s\n' "$PROJECT_ROOT"
  FAILURES=$((FAILURES + 1))
fi

section "2/6 Container and service"
docker ps --filter "name=^/${CONTAINER_NAME}$" \
  --format 'name={{.Names}} image={{.Image}} status={{.Status}}' || true
run_check "container is running" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME"
run_check "local health endpoint" curl -fsS --max-time 10 \
  "http://127.0.0.1:${APP_PORT}/api/health"
printf '\n'
docker inspect -f 'image={{.Config.Image}} network={{.HostConfig.NetworkMode}}' \
  "$CONTAINER_NAME" 2>/dev/null || true
docker inspect -f '{{range .Mounts}}mount={{.Source}} -> {{.Destination}} rw={{.RW}}{{println}}{{end}}' \
  "$CONTAINER_NAME" 2>/dev/null || true

section "3/6 Model manifest"
if docker exec "$CONTAINER_NAME" test -f "$MANIFEST"; then
  docker exec -i "$CONTAINER_NAME" python3 - "$MANIFEST" <<'PY' || true
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
print(f"profile={manifest.get('name')} allow_download={manifest.get('allow_download')}")
for item in manifest.get("models", []):
    print(
        f"required={str(item.get('required')).lower():5} "
        f"kind={item.get('kind'):12} name={item.get('name')} "
        f"target={item.get('target')}"
    )
PY
else
  printf 'ERROR: manifest missing in container: %s\n' "$MANIFEST"
  FAILURES=$((FAILURES + 1))
fi

section "4/6 Official model verification (no download)"
docker exec "$CONTAINER_NAME" rm -f "$TMP_LOCK" 2>/dev/null || true
if docker exec "$CONTAINER_NAME" \
  python3 /app/scripts/verify_models.py --manifest "$MANIFEST" --lock "$TMP_LOCK"; then
  printf 'RESULT: PASS - every required model verified\n'
  docker exec "$CONTAINER_NAME" cat "$TMP_LOCK" || true
else
  rc=$?
  printf 'RESULT: FAIL(rc=%s) - one or more required models are absent or incomplete\n' "$rc"
  FAILURES=$((FAILURES + 1))
fi
docker exec "$CONTAINER_NAME" rm -f "$TMP_LOCK" 2>/dev/null || true

section "5/6 Model storage inventory"
if [[ -d "$MODEL_ROOT" ]]; then
  du -sh "$MODEL_ROOT" || true
  printf '\nTop-level sizes:\n'
  du -h --max-depth=2 "$MODEL_ROOT" 2>/dev/null | sort -h | tail -60 || true
  printf '\nModel files (up to 300, symlinks included):\n'
  find "$MODEL_ROOT" -maxdepth 7 \( -type f -o -type l \) \
    -printf '%y %s %p -> %l\n' 2>/dev/null | sort | head -300 || true
else
  printf 'ERROR: model root missing: %s\n' "$MODEL_ROOT"
  FAILURES=$((FAILURES + 1))
fi

section "6/6 NPU and recent logs"
npu-smi info || true
printf '\nProcesses on physical NPU 5 (idle is acceptable):\n'
npu-smi info -t proc-mem -i 5 -c 0 || true
printf '\nRecent container logs:\n'
docker logs --tail 80 "$CONTAINER_NAME" 2>&1 || true

section "Audit summary"
if (( FAILURES == 0 )); then
  printf 'AUDIT_RESULT=PASS\n'
else
  printf 'AUDIT_RESULT=INCOMPLETE\n'
fi
printf 'failed_checks=%s\n' "$FAILURES"
printf 'NOTE: Missing models are expected before model preparation; send the full output back for the next step.\n'

# The report itself is useful even when models are missing, so leave with success.
exit 0
