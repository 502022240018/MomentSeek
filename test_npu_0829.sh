#!/usr/bin/env bash
# 测试NPU索引功能

echo "=== MomentSeek 0829 NPU索引测试 ==="
echo ""

# 1. 检查服务健康
echo "1. 检查服务健康..."
if curl -s http://127.0.0.1:8100/api/health | grep -q '"status": "ok"'; then
    echo "✅ 服务健康"
else
    echo "❌ 服务异常"
    exit 1
fi

# 2. 检查NPU设备
echo ""
echo "2. 检查容器NPU设备..."
docker exec momentseek-0829-platform ls -la /dev/davinci4 >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ NPU 4 设备已挂载"
else
    echo "❌ NPU 4 设备未挂载"
    exit 1
fi

# 3. 检查环境变量
echo ""
echo "3. 检查容器环境变量..."
docker exec momentseek-0829-platform env | grep -E "(NPU_ENABLED|NPU_DEVICE_ID|ASCEND_VISIBLE_DEVICES)"

# 4. 检查NPU宿主机占用
echo ""
echo "4. 检查NPU 4宿主机占用..."
npu_proc=$(npu-smi info -t proc-mem -i 4 -c 0 | grep "Process id" || echo "No process")
echo "$npu_proc"

# 5. 查看可用视频
echo ""
echo "5. 查看可用视频..."
video_count=$(curl -s http://127.0.0.1:8100/api/videos | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
echo "视频数量: $video_count"

if [ "$video_count" -eq "0" ]; then
    echo ""
    echo "⚠️  没有上传的视频，请先上传视频测试索引功能"
    echo "   访问: http://127.0.0.1:8100"
else
    echo ""
    echo "✅ 可以开始测试索引功能"
    echo "   在Web界面选择视频并构建Visual索引"
fi

echo ""
echo "=== 测试完成 ==="
echo ""
echo "下一步："
echo "  1. 访问 http://127.0.0.1:8100"
echo "  2. 上传测试视频"
echo "  3. 选择视频，勾选Visual通道"
echo "  4. 点击【构建索引】"
echo "  5. 观察NPU占用: npu-smi info -t proc-mem -i 4 -c 0"
