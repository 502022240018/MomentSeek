# 🎉 MomentSeek 0829 环境已准备完毕

## ✅ 已完成的配置

### 1. 镜像和容器命名替换
- ✅ 镜像名: `momentseek-29154-platform` → `momentseek-0829-platform`
- ✅ 容器名: `momentseek-29154-platform` → `momentseek-0829-platform`
- ✅ Compose项目名: `momentseek-0829`
- ✅ 网络名: `momentseek-0829-net`

### 2. 端口配置（避免冲突）
- ✅ 主应用: `8000` → `8100`
- ✅ Milvus gRPC: `19530` → `19531`
- ✅ Milvus health: `9091` → `9092`
- ✅ MinIO console: `9001` → `9002`

### 3. 资源隔离配置
- ✅ NPU设备: `6` (可在`.env.0829`中调整)
- ✅ 模型目录: `/home/momentseek-29154/models/platform` (与同事共享，只读)
- ✅ Runtime目录: `/home/momentseek_0829_develop/workplace/MomentSeek/runtime` (独立)

### 4. 创建的文件清单
```
📁 /home/momentseek_0829_develop/workplace/MomentSeek/
├── 📄 .env.0829                      # 你的环境配置文件
├── 🔧 check_env_0829.sh              # 环境检查脚本（部署前运行）
├── 🚀 start_milvus_0829.sh           # Milvus启动脚本
├── 🚀 deploy_0829.sh                 # 主应用部署脚本
├── 🚀 start_all_0829.sh              # 一键启动所有服务
├── 📖 DEPLOY_0829_GUIDE.md           # 详细部署指南（推荐阅读）
├── 📖 QUICK_REFERENCE_0829.md        # 快速命令参考
└── 📖 SETUP_COMPLETE_0829.md         # 本文件
```

## 🚀 现在开始部署！

### 方式一：一键部署（推荐新手）

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 一条命令启动所有服务
./start_all_0829.sh
```

### 方式二：分步部署（推荐有经验用户）

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 步骤1: 环境检查（推荐）
./check_env_0829.sh

# 步骤2: 启动Milvus（可选，如果需要向量检索）
./start_milvus_0829.sh

# 步骤3: 部署主应用
./deploy_0829.sh
```

## ⚠️ 部署前请确认

### 必须检查的事项：

1. **NPU可用性**
   ```bash
   # 确认NPU 6是否空闲
   npu-smi info -t proc-mem -i 6 -c 0
   
   # 如果被占用，编辑.env.0829修改以下三行为其他空闲NPU编号：
   # HOST_NPU_DEVICE_ID=6
   # ASCEND_VISIBLE_DEVICES=6
   # ASCEND_RT_VISIBLE_DEVICES=6
   ```

2. **端口可用性**
   ```bash
   # 确认端口未被占用
   ss -lntp | grep -E ':(8100|19531|9092|9002)'
   ```

3. **模型目录存在**
   ```bash
   # 确认共享模型目录可访问
   ls -la /home/momentseek-29154/models/platform
   ```

4. **InsightFace wheel存在**
   ```bash
   # 确认wheel文件存在
   ls -lh /home/momentseek_0829_develop/workplace/MomentSeek/vendor-wheels/insightface-*.whl
   ```

### 一键检查（推荐）：
```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek
./check_env_0829.sh
```

## 📝 部署后验证

部署成功后运行以下命令验证：

```bash
# 1. 检查容器状态
docker ps | grep momentseek-0829

# 2. 健康检查
curl http://127.0.0.1:8100/api/health | python3 -m json.tool

# 3. 查看日志
docker logs --tail 50 momentseek-0829-platform

# 4. 检查NPU占用
npu-smi info -t proc-mem -i 6 -c 0

# 5. 访问Web界面
# 在浏览器打开: http://127.0.0.1:8100
```

## 🎯 预期结果

部署成功后你应该看到：

### 1. 容器运行中
```bash
$ docker ps | grep momentseek-0829
momentseek-0829-platform   Up 2 minutes   (healthy)
momentseek-0829-milvus     Up 5 minutes   (healthy)
momentseek-0829-etcd       Up 5 minutes
momentseek-0829-minio      Up 5 minutes
```

### 2. 健康检查返回OK
```json
{
  "status": "ok",
  "env_profile": "prod.ascend",
  "npu_enabled": true,
  "npu_device_id": 0,
  "visual_model": "siglip2-so400m-384",
  "milvus_enabled": true
}
```

### 3. 可以访问Web界面
- 主页: http://127.0.0.1:8100
- API文档: http://127.0.0.1:8100/docs

## 🔧 常用命令速查

### 查看日志
```bash
docker logs -f momentseek-0829-platform              # 实时日志
docker logs --tail 100 momentseek-0829-platform      # 最近100行
```

### 重启服务
```bash
docker restart momentseek-0829-platform              # 重启主应用
docker compose -f compose.milvus.yml --env-file .env.0829 restart  # 重启Milvus
```

### 停止服务
```bash
docker stop momentseek-0829-platform                 # 停止主应用
docker compose -f compose.milvus.yml --env-file .env.0829 down     # 停止Milvus
```

### 查看资源
```bash
docker stats momentseek-0829-platform                # Docker资源占用
npu-smi info -t proc-mem -i 6 -c 0                  # NPU占用
df -h /home/momentseek_0829_develop                 # 磁盘空间
```

## 📚 详细文档

- **详细部署指南**: `cat DEPLOY_0829_GUIDE.md`
- **快速命令参考**: `cat QUICK_REFERENCE_0829.md`
- **项目文档**: `ls docs/*.md`

## 🆘 遇到问题？

### 1. NPU被占用
```bash
# 查看占用情况
npu-smi info -t proc-mem -i 6 -c 0

# 解决方案：修改.env.0829中的NPU编号
vim .env.0829  # 修改 HOST_NPU_DEVICE_ID, ASCEND_VISIBLE_DEVICES, ASCEND_RT_VISIBLE_DEVICES
```

### 2. 端口被占用
```bash
# 查看占用进程
ss -lntp | grep :8100

# 解决方案：修改.env.0829中的端口
vim .env.0829  # 修改 APP_PORT
```

### 3. 模型校验失败
```bash
# 检查模型目录
ls -R /home/momentseek-29154/models/platform

# 联系同事确认模型位置
```

### 4. 部署失败自动回滚
```bash
# 查看部署日志
cat logs/image-build-*.log

# 查看详细错误
docker logs momentseek-0829-platform
```

### 5. 更多问题
```bash
# 查看详细故障排查指南
cat DEPLOY_0829_GUIDE.md | grep -A 20 "故障排查"
```

## 🔄 更新代码后重新部署

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek

# 更新代码
git pull origin main

# 重新部署（会自动构建新镜像）
./deploy_0829.sh
```

## 🎓 下一步

部署成功后，你可以：

1. **上传视频**: 访问 http://127.0.0.1:8100，点击"上传视频"
2. **建立索引**: 选择视频，勾选需要的通道（Visual/Face/ASR/OCR），点击"构建索引"
3. **搜索测试**: 使用文本或图片搜索已索引的视频片段
4. **查看API文档**: http://127.0.0.1:8100/docs

## 📞 获取更多帮助

- **项目README**: `cat README.md`
- **开发文档**: `cat docs/DEVELOPMENT.md`
- **部署文档**: `cat docs/DEPLOYMENT.md`
- **模型文档**: `cat docs/MODELS.md`

---

祝部署顺利！🚀
