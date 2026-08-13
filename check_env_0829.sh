#!/usr/bin/env bash
# MomentSeek 0829 - 快速检查工具
# 用于部署前检查所有必要条件

set -e

WORK_ROOT="/home/momentseek_0829_develop/workplace/MomentSeek"
MODEL_DIR="/home/momentseek-29154/models/platform"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "=========================================="
echo "MomentSeek 0829 环境检查"
echo "=========================================="
echo ""

# 1. 检查必要命令
echo "1. 检查必要命令..."
all_commands_ok=true
for cmd in docker git curl npu-smi python3 ss; do
    if command -v "$cmd" >/dev/null 2>&1; then
        check_pass "$cmd 已安装"
    else
        check_fail "$cmd 未找到"
        all_commands_ok=false
    fi
done
echo ""

# 2. 检查配置文件
echo "2. 检查配置文件..."
if [[ -f "$WORK_ROOT/.env.0829" ]]; then
    check_pass ".env.0829 存在"
    source "$WORK_ROOT/.env.0829"
else
    check_fail ".env.0829 不存在"
    echo "   请先创建配置文件"
    exit 1
fi
echo ""

# 3. 检查NPU
echo "3. 检查NPU设备..."
NPU_ID="${HOST_NPU_DEVICE_ID:-6}"
if npu-smi info -i "$NPU_ID" >/dev/null 2>&1; then
    check_pass "NPU $NPU_ID 可访问"

    # 检查NPU是否被占用
    if npu_proc=$(npu-smi info -t proc-mem -i "$NPU_ID" -c 0 2>/dev/null | grep -v "Total Used Memory"); then
        if echo "$npu_proc" | grep -q "0 B"; then
            check_pass "NPU $NPU_ID 空闲"
        else
            check_warn "NPU $NPU_ID 可能被占用"
            echo "$npu_proc"
        fi
    fi
else
    check_fail "NPU $NPU_ID 不可访问"
    echo "   可用的NPU设备："
    npu-smi info | grep "NPU ID" || echo "   无法获取NPU列表"
fi
echo ""

# 4. 检查端口
echo "4. 检查端口占用..."
APP_PORT="${APP_PORT:-8100}"
MILVUS_PORT="${MILVUS_GRPC_PORT:-19531}"
MILVUS_HEALTH="${MILVUS_HEALTH_PORT:-9092}"
MINIO_PORT="${MINIO_CONSOLE_PORT:-9002}"

for port in "$APP_PORT" "$MILVUS_PORT" "$MILVUS_HEALTH" "$MINIO_PORT"; do
    if ss -lntp 2>/dev/null | grep -q ":${port} "; then
        check_warn "端口 $port 已被占用"
        ss -lntp 2>/dev/null | grep ":${port} " | head -1
    else
        check_pass "端口 $port 可用"
    fi
done
echo ""

# 5. 检查模型目录
echo "5. 检查模型目录..."
if [[ -d "$MODEL_DIR" ]]; then
    check_pass "模型目录存在: $MODEL_DIR"

    # 检查关键子目录
    for subdir in hf-cache insightface funasr text-embeddings rapidocr 3D-Speaker 3dspeaker-cache; do
        if [[ -d "$MODEL_DIR/$subdir" ]]; then
            check_pass "  $subdir/"
        else
            check_warn "  $subdir/ 缺失"
        fi
    done
else
    check_fail "模型目录不存在: $MODEL_DIR"
fi
echo ""

# 6. 检查磁盘空间
echo "6. 检查磁盘空间..."
available_gb=$(df -BG "$WORK_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
if [[ "$available_gb" -ge 20 ]]; then
    check_pass "可用空间: ${available_gb}GB"
elif [[ "$available_gb" -ge 10 ]]; then
    check_warn "可用空间: ${available_gb}GB (建议20GB以上)"
else
    check_fail "可用空间不足: ${available_gb}GB (需要至少10GB)"
fi
echo ""

# 7. 检查现有容器
echo "7. 检查现有容器..."
if docker ps -a --format '{{.Names}}' | grep -q "^momentseek-0829"; then
    check_warn "发现现有的 0829 容器"
    docker ps -a --filter name=momentseek-0829 --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    check_pass "无冲突的容器"
fi
echo ""

# 8. 检查基础镜像
echo "8. 检查Docker基础镜像..."
BASE_IMAGE="${ASCEND_RUNTIME_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts}"
if docker images "$BASE_IMAGE" --format '{{.Repository}}' | grep -q .; then
    check_pass "基础镜像已存在: $(echo "$BASE_IMAGE" | cut -d: -f2)"
else
    check_warn "基础镜像需要下载: $(echo "$BASE_IMAGE" | cut -d: -f2)"
    echo "   首次构建将会较慢"
fi
echo ""

# 9. 检查vendor-wheels
echo "9. 检查InsightFace wheel..."
WHEEL_PATH="$WORK_ROOT/vendor-wheels/insightface-1.0.1-py3-none-any.whl"
if [[ -f "$WHEEL_PATH" ]]; then
    check_pass "InsightFace wheel 存在"
    actual_sha=$(sha256sum "$WHEEL_PATH" | awk '{print $1}')
    expected_sha="5f373f6fedbdda5cbc59a34ca386a75a2995cdaf6899402590ae9eb4308fc2e8"
    if [[ "$actual_sha" == "$expected_sha" ]]; then
        check_pass "  checksum 正确"
    else
        check_fail "  checksum 不匹配"
    fi
else
    check_fail "InsightFace wheel 缺失: $WHEEL_PATH"
fi
echo ""

# 10. 总结
echo "=========================================="
echo "检查完成"
echo "=========================================="
echo ""

if [[ "$all_commands_ok" == false ]]; then
    echo -e "${RED}错误：缺少必要命令，无法继续${NC}"
    exit 1
fi

if [[ ! -f "$WHEEL_PATH" ]]; then
    echo -e "${RED}错误：缺少InsightFace wheel，无法构建镜像${NC}"
    echo "请联系同事或从备份复制该文件"
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo -e "${RED}错误：模型目录不存在，请确认路径${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过，可以开始部署${NC}"
echo ""
echo "部署步骤："
echo "  1. 启动Milvus:  ./start_milvus_0829.sh"
echo "  2. 部署应用:    ./deploy_0829.sh"
echo ""
echo "详细文档: DEPLOY_0829_GUIDE.md"
