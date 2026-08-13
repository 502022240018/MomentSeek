#!/usr/bin/env bash
# MomentSeek 0829 - 停止Milvus服务

set -Eeuo pipefail

cd "$(dirname "$0")"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

# 检查.env.0829是否存在
if [[ ! -f .env.0829 ]]; then
    echo "Error: .env.0829 not found!"
    exit 1
fi

log "Stopping Milvus stack for momentseek-0829"

# 停止Milvus栈
docker-compose -f compose.milvus.yml --env-file .env.0829 down

log "✅ Milvus stack stopped"
