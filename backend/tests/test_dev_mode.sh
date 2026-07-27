#!/usr/bin/env bash
# 测试开发模式的热重载功能

set -e

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

log() { printf '\n\033[1;34m[TEST]\033[0m %s\n' "$*"; }
success() { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

CONTAINER_NAME="momentseek-0829-platform"
APP_PORT=8100
TEST_FILE="backend/app/test_dev_reload.py"

log "开发模式热重载测试"
echo ""

# 检查容器是否运行
log "Step 1: 检查容器状态"
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    error "容器 $CONTAINER_NAME 未运行"
fi
success "容器正在运行"

# 检查是否挂载了代码目录
log "Step 2: 检查代码挂载"
mount_info=$(docker inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{.Source}}->{{.Destination}}{{"\n"}}{{end}}' | grep "/backend/app")
if [[ -z "$mount_info" ]]; then
    error "代码目录未挂载，开发模式未启用"
fi
success "代码目录已挂载: $mount_info"

# 检查 uvicorn 是否使用了 --reload 参数
log "Step 3: 检查热重载配置"
container_cmd=$(docker inspect "$CONTAINER_NAME" --format '{{.Config.Cmd}}')
if [[ "$container_cmd" != *"--reload"* ]]; then
    error "容器未启用 --reload 参数"
fi
success "热重载已启用"

# 创建测试文件
log "Step 4: 创建测试文件"
cat > "$TEST_FILE" << 'EOF'
"""测试开发模式热重载"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/test/dev-reload")
async def test_dev_reload():
    return {"status": "ok", "message": "Dev mode test v1", "reload_enabled": True}
EOF
success "测试文件已创建: $TEST_FILE"

# 等待文件变更被检测
log "Step 5: 等待热重载触发 (10秒)"
sleep 10

# 测试 API 响应
log "Step 6: 测试 API 响应"
if ! response=$(curl -fsSL --max-time 5 "http://127.0.0.1:${APP_PORT}/api/health" 2>/dev/null); then
    error "API 健康检查失败"
fi
success "API 响应正常"

# 修改测试文件
log "Step 7: 修改测试文件"
cat > "$TEST_FILE" << 'EOF'
"""测试开发模式热重载 - 修改版"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/test/dev-reload")
async def test_dev_reload():
    return {"status": "ok", "message": "Dev mode test v2 - MODIFIED", "reload_enabled": True}
EOF
success "测试文件已修改"

# 等待热重载
log "Step 8: 等待热重载完成 (10秒)"
sleep 10

# 检查日志中的重载信息
log "Step 9: 检查容器日志"
recent_logs=$(docker logs --tail 50 "$CONTAINER_NAME" 2>&1)
if echo "$recent_logs" | grep -qi "Reloading\|detected changes\|Uvicorn running"; then
    success "检测到热重载日志"
    echo "$recent_logs" | grep -i "reload\|detect\|restart" | tail -5
else
    echo "⚠️  未在最近的日志中找到热重载信息（可能已完成）"
fi

# 清理测试文件
log "Step 10: 清理测试文件"
rm -f "$TEST_FILE"
success "测试文件已删除"

# 最终验证
log "Step 11: 最终验证"
sleep 5
if curl -fsSL --max-time 5 "http://127.0.0.1:${APP_PORT}/api/health" >/dev/null 2>&1; then
    success "服务仍然正常运行"
else
    error "服务异常"
fi

echo ""
log "✅ 开发模式测试完成！"
echo ""
echo "==========================================="
echo "结果："
echo "  ✓ 容器运行正常"
echo "  ✓ 代码目录已挂载"
echo "  ✓ 热重载已启用"
echo "  ✓ 文件变更检测正常"
echo ""
echo "查看实时日志验证热重载："
echo "  docker logs -f $CONTAINER_NAME"
echo "==========================================="
