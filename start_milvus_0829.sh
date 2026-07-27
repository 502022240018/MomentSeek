#!/usr/bin/env bash
# MomentSeek 0829 - 启动Milvus服务
# 使用独立的端口和容器名避免与29154冲突

set -Eeuo pipefail

cd "$(dirname "$0")"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

# 检查.env.0829是否存在
if [[ ! -f .env.0829 ]]; then
    echo "Error: .env.0829 not found!"
    echo "Please create .env.0829 first."
    exit 1
fi

log "Starting Milvus stack for momentseek-0829"
log "Using configuration from .env.0829"

# 显示配置信息
source .env.0829
echo "Compose project: ${COMPOSE_PROJECT_NAME}"
echo "Network: ${MOMENTSEEK_NETWORK_NAME}"
echo "Milvus gRPC port: ${MILVUS_GRPC_PORT}"
echo "Milvus health port: ${MILVUS_HEALTH_PORT}"
echo "MinIO console port: ${MINIO_CONSOLE_PORT}"

# 启动Milvus栈
docker-compose -f compose.milvus.yml --env-file .env.0829 up -d

log "Waiting for Milvus to become healthy..."
max_wait=120
waited=0
while [[ $waited -lt $max_wait ]]; do
  if curl -fsSL --max-time 3 "http://127.0.0.1:${MILVUS_HEALTH_PORT}/healthz" >/dev/null 2>&1; then
    log "✅ Milvus is healthy!"
    break
  fi
  printf '.'
  sleep 3
  waited=$((waited + 3))
done

if [[ $waited -ge $max_wait ]]; then
    echo ""
    echo "WARNING: Milvus health check timeout after ${max_wait}s"
    echo "Check logs with: docker-compose -f compose.milvus.yml --env-file .env.0829 logs"
    exit 1
fi

echo ""
log "✅ Milvus stack started successfully!"
echo ""
echo "===========================================  "
echo "Service URLs:"
echo "  Milvus gRPC:    127.0.0.1:${MILVUS_GRPC_PORT}"
echo "  Milvus health:  http://127.0.0.1:${MILVUS_HEALTH_PORT}/healthz"
echo "  MinIO console:  http://127.0.0.1:${MINIO_CONSOLE_PORT}"
echo ""
echo "Containers:"
docker-compose -f compose.milvus.yml --env-file .env.0829 ps
echo "==========================================="
