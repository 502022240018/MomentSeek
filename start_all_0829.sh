#!/usr/bin/env bash
# MomentSeek 0829 - 一键启动所有服务
# 依次启动Milvus和主应用

set -e

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }
error() { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

log "MomentSeek 0829 一键部署开始"
echo ""

# 检查环境
if [[ ! -f .env.0829 ]]; then
    error ".env.0829 配置文件不存在"
fi

# 步骤1: 环境检查
log "步骤 1/3: 环境检查"
if [[ -f check_env_0829.sh ]]; then
    if ! ./check_env_0829.sh; then
        error "环境检查失败，请修复后重试"
    fi
else
    log "跳过环境检查（check_env_0829.sh 不存在）"
fi

# 步骤2: 启动Milvus（可选）
source .env.0829
if [[ "${MILVUS_ENABLED,,}" == "true" ]]; then
    log "步骤 2/3: 启动 Milvus"
    if [[ -f start_milvus_0829.sh ]]; then
        ./start_milvus_0829.sh
    else
        error "start_milvus_0829.sh 不存在"
    fi
else
    log "步骤 2/3: 跳过 Milvus（已禁用）"
fi

# 步骤3: 部署主应用
log "步骤 3/3: 部署主应用"
if [[ -f deploy_0829.sh ]]; then
    ./deploy_0829.sh
else
    error "deploy_0829.sh 不存在"
fi

# 完成
echo ""
log "✅ 所有服务启动完成！"
echo ""
echo "=========================================="
echo "访问地址："
echo "  Web界面:   http://127.0.0.1:${APP_PORT:-8100}"
echo "  API文档:   http://127.0.0.1:${APP_PORT:-8100}/docs"
echo "  健康检查:  http://127.0.0.1:${APP_PORT:-8100}/api/health"
echo ""
echo "查看日志："
echo "  docker logs -f momentseek-0829-platform"
echo ""
echo "快速参考："
echo "  cat QUICK_REFERENCE_0829.md"
echo "=========================================="
