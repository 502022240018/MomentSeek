# MomentSeek 0829 Development Environment - 部署指南

## 📋 环境说明

这是一个独立的开发环境，与同事的 29154 环境完全隔离：

- **镜像命名**: `momentseek-0829-platform`
- **容器命名**: `momentseek-0829-platform`, `momentseek-0829-milvus` 等
- **网络端口**: 
  - 主应用: `8100` (避免8000冲突)
  - Milvus gRPC: `19531` (避免19530冲突)
  - Milvus health: `9092`
  - MinIO console: `9002`
- **共享模型**: `/home/momentseek-29154/models/platform` (只读，与同事共享)
- **独立runtime**: `/home/momentseek_0829_develop/workplace/MomentSeek/runtime` (你的专用数据)
- **NPU设备**: `6` (需要确认可用，避免与NPU 5冲突)

## ⚠️ 部署前必读

### 1. 确认NPU可用性

```bash
# 查看所有NPU状态
npu-smi info

# 检查NPU 6是否空闲（没有进程占用）
npu-smi info -t proc-mem -i 6 -c 0

# 如果NPU 6被占用，在.env.0829中修改以下三个变量为其他空闲NPU编号：
# HOST_NPU_DEVICE_ID=6
# ASCEND_VISIBLE_DEVICES=6
# ASCEND_RT_VISIBLE_DEVICES=6
```

### 2. 确认模型目录存在

```bash
# 检查共享模型目录
ls -la /home/momentseek-29154/models/platform

# 应该看到以下子目录：
# - hf-cache/
# - insightface/
# - funasr/
# - faster-whisper/
# - text-embeddings/
# - rapidocr/
# - 3D-Speaker/
# - 3dspeaker-cache/
```

### 3. 检查端口占用

```bash
# 确认8100端口未被占用
ss -lntp | grep :8100

# 确认Milvus端口未被占用
ss -lntp | grep :19531
ss -lntp | grep :9092
ss -lntp | grep :9002
```

## 🚀 快速部署流程

### 第一步：启动 Milvus（可选但推荐）

如果需要使用向量数据库功能：

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 启动Milvus栈（etcd + MinIO + Milvus）
./start_milvus_0829.sh
```

**验证Milvus启动**：
```bash
# 检查健康状态
curl http://127.0.0.1:9092/healthz

# 查看容器
docker compose -f compose.milvus.yml --env-file .env.0829 ps
```

如果不需要Milvus，修改 `.env.0829` 中的 `MILVUS_ENABLED=false`。

### 第二步：构建并启动 MomentSeek 主应用

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 确保runtime目录存在
mkdir -p runtime logs

# 执行部署脚本（会自动构建镜像、校验模型、启动容器）
./deploy_0829.sh
```

**部署脚本会执行以下步骤**：
1. ✅ 环境检查（命令、磁盘、NPU、端口）
2. ✅ Milvus连接测试（如果启用）
3. ✅ 构建Docker镜像 `momentseek-0829-platform:xxxxx`
4. ✅ 在临时容器中校验模型完整性
5. ✅ 停止旧容器（如果存在）
6. ✅ 启动新容器 `momentseek-0829-platform`
7. ✅ 健康检查
8. ✅ 清理

### 第三步：验证部署

```bash
# 检查容器状态
docker ps | grep momentseek-0829

# 检查健康接口
curl http://127.0.0.1:8100/api/health | python3 -m json.tool

# 检查NPU占用
npu-smi info -t proc-mem -i 6 -c 0

# 查看容器日志
docker logs -f momentseek-0829-platform
```

### 第四步：访问服务

- **Web界面**: http://127.0.0.1:8100
- **API文档**: http://127.0.0.1:8100/docs
- **健康检查**: http://127.0.0.1:8100/api/health

## 🔧 日常操作

### 查看日志
```bash
# 实时日志
docker logs -f momentseek-0829-platform

# 最近100行
docker logs --tail 100 momentseek-0829-platform

# 最近10分钟
docker logs --since 10m momentseek-0829-platform
```

### 重启服务
```bash
# 重启主容器
docker restart momentseek-0829-platform

# 重启Milvus栈
docker compose -f compose.milvus.yml --env-file .env.0829 restart
```

### 停止服务
```bash
# 停止主容器
docker stop momentseek-0829-platform

# 停止Milvus栈
docker compose -f compose.milvus.yml --env-file .env.0829 down
```

### 完全清理（谨慎！）
```bash
# 停止并删除容器
docker rm -f momentseek-0829-platform

# 停止并删除Milvus栈
docker compose -f compose.milvus.yml --env-file .env.0829 down -v

# 删除镜像
docker rmi $(docker images momentseek-0829-platform -q)

# 删除运行时数据（会丢失所有上传的视频和索引！）
rm -rf runtime/*
```

## 🐛 故障排查

### 问题1: NPU被占用
```bash
# 症状：部署失败，提示NPU不可用
# 解决：
npu-smi info -t proc-mem -i 6 -c 0  # 查看占用进程
# 如果有进程占用，修改.env.0829中的NPU编号为其他空闲设备
```

### 问题2: 端口被占用
```bash
# 症状：部署失败，提示端口已绑定
# 解决：
ss -lntp | grep :8100  # 找到占用进程
# 停止占用进程，或修改.env.0829中的APP_PORT
```

### 问题3: 模型校验失败
```bash
# 症状：部署失败，提示模型缺失或不匹配
# 解决：
ls -R /home/momentseek-29154/models/platform  # 检查模型目录
# 对比 deploy/models/ascend-prod.models.json 中要求的路径
# 联系同事确认模型位置，或从其他环境复制
```

### 问题4: Milvus连接失败
```bash
# 症状：应用启动但搜索功能异常
# 解决：
curl http://127.0.0.1:9092/healthz  # 检查Milvus健康
docker compose -f compose.milvus.yml --env-file .env.0829 ps  # 检查容器
docker compose -f compose.milvus.yml --env-file .env.0829 logs milvus  # 查看日志

# 如果不需要Milvus，修改.env.0829：
# MILVUS_ENABLED=false
```

### 问题5: 镜像构建慢
```bash
# 症状：docker build步骤耗时很长
# 原因：首次构建需要下载基础镜像和安装依赖
# 解决：耐心等待，后续构建会利用缓存加速
# 查看构建日志：
tail -f logs/image-build-*.log
```

### 问题6: 健康检查超时
```bash
# 症状：容器启动但健康检查一直失败
# 排查：
docker logs momentseek-0829-platform  # 查看启动日志
docker exec momentseek-0829-platform ps aux  # 查看容器内进程
docker exec momentseek-0829-platform curl http://127.0.0.1:8100/api/health  # 容器内测试
```

## 📁 目录结构

```
/home/momentseek_0829_develop/workplace/MomentSeek/
├── .env.0829                          # 你的环境配置
├── deploy_0829.sh                     # 部署脚本
├── start_milvus_0829.sh              # Milvus启动脚本
├── runtime/                           # 你的运行时数据（独立）
│   ├── catalog.sqlite3               # 数据库
│   ├── uploads/                      # 上传的视频
│   ├── indexes/                      # 索引文件
│   ├── etcd/                         # Milvus etcd数据
│   ├── minio/                        # Milvus对象存储
│   └── milvus/                       # Milvus向量数据
├── logs/                              # 构建和部署日志
└── .server-build/                     # 临时构建文件

/home/momentseek-29154/models/platform/  # 共享模型（只读）
├── hf-cache/                          # Hugging Face模型
├── insightface/                       # 人脸识别模型
├── funasr/                           # 语音识别模型
├── faster-whisper/                   # Whisper模型
├── text-embeddings/                  # 文本embedding
├── rapidocr/                         # OCR模型
├── 3D-Speaker/                       # 说话人识别
└── 3dspeaker-cache/                  # 说话人模型缓存
```

## 🔐 安全注意事项

1. **模型目录只读挂载**：`/home/momentseek-29154/models/platform` 在容器中以只读方式挂载，不会影响同事的模型
2. **独立runtime**：所有运行时数据写入你自己的 `runtime/` 目录
3. **网络隔离**：使用独立的网络名称和端口，不与29154环境冲突
4. **NPU独占**：确保使用未被占用的NPU设备编号

## 📞 需要帮助？

1. 查看完整日志：`docker logs momentseek-0829-platform > debug.log`
2. 检查系统资源：`docker stats momentseek-0829-platform`
3. 对比同事环境：`docker ps | grep momentseek-29154`


