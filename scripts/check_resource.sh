#!/usr/bin/env bash
set -euo pipefail

load_env_file() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0
  while IFS='=' read -r key value; do
    [[ -n "${key:-}" ]] || continue
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value%%#*}"
    value="${value%$'\r'}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    export "$key=$value"
  done < "$env_file"
}

load_env_file ".env"

echo "Host: $(hostname)  Time: $(date)"
npu-smi info
echo
echo "Port ${APP_PORT:-8000}:"
ss -ltn "sport = :${APP_PORT:-8000}" || true
echo
if [[ -n "${HOST_NPU_DEVICE_ID-}" ]]; then
  echo "Target host NPU ${HOST_NPU_DEVICE_ID}:"
  npu-smi info | sed -n "/| ${HOST_NPU_DEVICE_ID} /,+3p" || true
  echo
fi
echo "MomentSeek containers:"
docker ps --filter name=momentseek-mvp --format '{{.Names}}  {{.Status}}  {{.Ports}}'
