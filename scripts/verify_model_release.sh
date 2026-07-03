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

device="${HOST_NPU_DEVICE_ID-}"
if [[ -z "$device" ]]; then
  device="${ASCEND_VISIBLE_DEVICES-}"
fi
if [[ -z "$device" ]]; then
  echo "HOST_NPU_DEVICE_ID or ASCEND_VISIBLE_DEVICES must be set before checking host NPU release." >&2
  exit 2
fi

echo "Checking host NPU ${device}; no inference process should remain after a completed index stage."
npu-smi info | sed -n "/| ${device} /,+3p"
