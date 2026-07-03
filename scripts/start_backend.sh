#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
port="8000"

if [[ -f "$repo_root/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*APP_PORT[[:space:]]*=[[:space:]]*(.+)[[:space:]]*$ ]] || continue

    port="${BASH_REMATCH[1]}"
    port="${port%$'\r'}"
    port="${port#"${port%%[![:space:]]*}"}"
    port="${port%"${port##*[![:space:]]}"}"
    port="${port%\"}"
    port="${port#\"}"
    port="${port%\'}"
    port="${port#\'}"
    break
  done < "$repo_root/.env"
fi

cd "$repo_root"

python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port "$port" --reload
