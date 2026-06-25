#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)  Time: $(date)"
npu-smi info
echo
echo "Port ${APP_PORT:-8300}:"
ss -ltn "sport = :${APP_PORT:-8300}" || true
echo
echo "MomentSeek containers:"
docker ps --filter name=momentseek-mvp --format '{{.Names}}  {{.Status}}  {{.Ports}}'

