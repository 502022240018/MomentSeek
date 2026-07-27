#!/usr/bin/env bash
# MomentSeek 0829 - 停止所有服务（主应用 + Milvus）

set -e

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }

log "Stopping all MomentSeek 0829 services"
echo ""

# 停止主应用容器
if docker ps -q -f name=momentseek-0829-platform >/dev/null 2>&1; then
    log "Stopping main application..."
    docker stop momentseek-0829-platform
    log "✅ Main application stopped"
else
    log "Main application container not running"
fi

# 停止Milvus栈
if [[ -f .env.0829 ]] && [[ -f compose.milvus.yml ]]; then
    log "Stopping Milvus stack..."
    docker-compose -f compose.milvus.yml --env-file .env.0829 down
    log "✅ Milvus stack stopped"
else
    log "Milvus configuration not found, skipping"
fi

echo ""
log "✅ All services stopped"
echo ""
echo "To restart services, run: ./start_all_0829.sh"
