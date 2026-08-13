# MomentSeek 0829 快速参考

## 📌 命名和端口配置总结

| 项目 | 29154环境 | 0829环境 (你的) |
|------|-----------|----------------|
| 镜像名 | momentseek-29154-platform | momentseek-0829-platform |
| 容器名 | momentseek-29154-platform | momentseek-0829-platform |
| 主应用端口 | 8000 | 8100 |
| Milvus gRPC | 19530 | 19531 |
| Milvus health | 9091 | 9092 |
| MinIO console | 9001 | 9002 |
| NPU设备 | 5 | 6 (可调整) |
| 模型目录 | /home/momentseek-29154/models/platform | /home/momentseek-29154/models/platform (共享) |
| Runtime目录 | /home/momentseek-29154/runtime | /home/momentseek_0829_develop/workplace/MomentSeek/runtime |

## 🚀 完整部署流程

### 步骤 0: 环境检查（推荐先执行）
```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek
./check_env_0829.sh
```

### 步骤 1: 启动 Milvus（可选）
```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek
./start_milvus_0829.sh

# 验证Milvus
curl http://127.0.0.1:9092/healthz
docker compose -f compose.milvus.yml --env-file .env.0829 ps
```

### 步骤 2: 部署主应用
```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek
./deploy_0829.sh
```

### 步骤 3: 验证部署
```bash
# 健康检查
curl http://127.0.0.1:8100/api/health | python3 -m json.tool

# 检查容器
docker ps | grep momentseek-0829

# 访问Web界面
http://127.0.0.1:8100
```
http://127.0.0.1:8100

## 📝 日常管理命令

### 查看日志
```bash
# 实时查看
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

# 重启Milvus
docker compose -f compose.milvus.yml --env-file .env.0829 restart
```

### 停止服务
```bash
# 停止主容器
docker stop momentseek-0829-platform

# 停止Milvus
docker compose -f compose.milvus.yml --env-file .env.0829 down
```

### 查看资源占用
```bash
# Docker资源
docker stats momentseek-0829-platform

# NPU占用
npu-smi info -t proc-mem -i 6 -c 0

# 磁盘空间
df -h /home/momentseek_0829_develop
```

## 🔍 故障排查命令

### 检查NPU
```bash
# 查看所有NPU
npu-smi info

# 检查NPU 6占用情况
npu-smi info -t proc-mem -i 6 -c 0

# 如果被占用，修改.env.0829中的NPU编号
```

### 检查端口
```bash
# 检查端口占用
ss -lntp | grep :8100
ss -lntp | grep :19531
ss -lntp | grep :9092
ss -lntp | grep :9002
```

### 检查容器
```bash
# 查看所有0829容器
docker ps -a | grep momentseek-0829

# 查看容器详细信息
docker inspect momentseek-0829-platform

# 进入容器
docker exec -it momentseek-0829-platform bash
```

### 检查模型
```bash
# 查看模型目录
ls -R /home/momentseek-29154/models/platform | head -50

# 在容器中验证模型
docker exec momentseek-0829-platform \
  python3 /app/scripts/verify_models.py \
  --manifest /app/deploy/models/ascend-prod.models.json \
  --lock /app/models/models.lock.json
```

### 检查Milvus
```bash
# Milvus健康检查
curl http://127.0.0.1:9092/healthz

# 查看Milvus日志
docker compose -f compose.milvus.yml --env-file .env.0829 logs milvus

# 查看所有Milvus栈容器
docker compose -f compose.milvus.yml --env-file .env.0829 ps
```

## 🔄 重新部署

### 代码更新后重新部署
```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 更新代码
git pull origin main

# 重新部署（会自动构建新镜像）
./deploy_0829.sh
```

### 强制重新构建
```bash
# 删除旧容器
docker rm -f momentseek-0829-platform

# 删除旧镜像
docker rmi $(docker images momentseek-0829-platform -q)

# 重新部署
./deploy_0829.sh
```

## 🗑️ 完全清理（谨慎！）

```bash
# 停止并删除主容器
docker stop momentseek-0829-platform
docker rm momentseek-0829-platform

# 停止并删除Milvus栈
docker compose -f compose.milvus.yml --env-file .env.0829 down -v

# 删除镜像
docker rmi $(docker images momentseek-0829-platform -q)

# 清理运行时数据（会丢失所有上传的视频和索引！）
rm -rf /home/momentseek_0829_develop/workplace/MomentSeek/runtime/*

# 清理日志
rm -rf /home/momentseek_0829_develop/workplace/MomentSeek/logs/*
```

## 📂 重要文件位置

```
/home/momentseek_0829_develop/workplace/MomentSeek/
├── .env.0829                    # 环境配置
├── check_env_0829.sh            # 环境检查脚本
├── start_milvus_0829.sh         # Milvus启动脚本
├── deploy_0829.sh               # 主应用部署脚本
├── DEPLOY_0829_GUIDE.md         # 详细部署指南
├── QUICK_REFERENCE_0829.md      # 本快速参考（当前文件）
├── runtime/                     # 运行时数据（你的专用）
│   ├── catalog.sqlite3         # 元数据数据库
│   ├── uploads/                # 上传视频
│   ├── indexes/                # 索引文件
│   └── milvus/                 # Milvus数据
└── logs/                        # 构建和部署日志
```

## 🔧 配置修改

### 修改NPU设备
编辑 `.env.0829`，修改以下三行为相同的空闲NPU编号：
```bash
HOST_NPU_DEVICE_ID=6
ASCEND_VISIBLE_DEVICES=6
ASCEND_RT_VISIBLE_DEVICES=6
```

### 修改端口
编辑 `.env.0829`：
```bash
APP_PORT=8100                    # 主应用端口
MILVUS_GRPC_PORT=19531          # Milvus gRPC
MILVUS_HEALTH_PORT=9092         # Milvus health
MINIO_CONSOLE_PORT=9002         # MinIO console
```

### 禁用Milvus
编辑 `.env.0829`：
```bash
MILVUS_ENABLED=false
```

## 🌐 访问地址

- **主应用Web**: http://127.0.0.1:8100
- **API文档**: http://127.0.0.1:8100/docs
- **健康检查**: http://127.0.0.1:8100/api/health
- **Milvus健康**: http://127.0.0.1:9092/healthz
- **MinIO控制台**: http://127.0.0.1:9002

## 🆘 获取帮助

1. 查看详细部署指南: `cat DEPLOY_0829_GUIDE.md`
2. 查看项目文档: `ls docs/*.md`
3. 查看容器日志: `docker logs momentseek-0829-platform > debug.log`
