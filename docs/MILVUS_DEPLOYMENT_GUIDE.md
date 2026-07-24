# Milvus 集成部署指南

本文档指导如何在开发环境中启动 Milvus 并验证集成。

## 第一步：启动 Milvus 服务

### 1.1 启动 Milvus（不包含应用）

首先单独启动 Milvus 及其依赖服务，确保它们正常运行：

```bash
docker compose -f compose.milvus.yml up -d
```

> **说明**：`compose.milvus.yml` 会自动创建共享网络 `momentseek-mvp-net`（如果尚不存在）。
> 后续运行 `compose.yml` 时会复用该网络，两个 Compose stack 可按任意顺序启动。

### 1.2 检查服务状态

等待约 60-90 秒让服务完全启动，然后检查：

```bash
docker ps
```

应该看到三个容器运行：
- `momentseek-etcd`
- `momentseek-minio`
- `momentseek-milvus`

### 1.3 验证 Milvus 健康状态

```bash
docker exec -it momentseek-milvus curl http://localhost:9091/healthz
```

应该返回空响应（HTTP 200），表示健康。

检查日志确认没有错误：

```bash
docker logs momentseek-milvus --tail 50
```

---

## 第二步：重新构建应用镜像（包含 pymilvus）

正式试用版使用 `pymilvus==2.6.16` 和
`milvusdb/milvus:v2.6.20`。2.4 数据目录不得直接挂载到 2.6；本项目从
空 collection 建库并通过保留的 NPZ 全量 backfill。详细版本选择见
`docs/DEPENDENCY_BASELINE.md`。

### 2.1 停止现有应用容器

```bash
docker compose -f compose.cuda.yml -f compose.dev.yml down
```

### 2.2 重新构建镜像

```bash
docker compose -f compose.cuda.yml build
```

这次构建会快很多，因为只需要安装新的 Python 包（pymilvus）。

---

## 第三步：启动完整环境

### 3.1 组合启动所有服务

```bash
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml up app 
# 前端
cd frontend
$env:VITE_API_PROXY_TARGET="http://127.0.0.1:8300"; npm run dev


```

这会启动：
- 应用容器（带 GPU 支持）
- Milvus + etcd + MinIO
- 开发模式（代码热重载）

### 3.2 检查应用日志

```bash
docker logs momentseek-mvp-app-cuda --tail 50
```

如果 Milvus 连接成功，应该会看到类似日志：
```
INFO - Connecting to Milvus at milvus:19530
INFO - Creating collection: visual_embeddings
INFO - Collection visual_embeddings created and loaded successfully
...
```

---

## 第四步：运行测试脚本验证集成

### 4.1 进入应用容器

```bash
docker exec -it momentseek-mvp-app-cuda bash
```

### 4.2 运行 Milvus 连接测试

```bash
cd /app/backend
python -m tests.test_milvus_connection
```

预期输出：
```
============================================================
Testing Milvus Connection
============================================================

1. Health Check
   Milvus health: ✓ OK

2. Collection Status
   visual_embeddings:
     - Loaded: True
     - Entities: 0
   asr_embeddings:
     - Loaded: True
     - Entities: 0
   ...

3. Embedding Dimensions
   visual: 1152d
   asr: 384d
   face: 512d
   ocr: 384d
   speaker:...

============================================================
✓ All tests passed!
============================================================
```

### 4.3 测试批量写入和幂等性

测试脚本会自动运行第二部分测试，验证：
- 批量写入是否正常
- Upsert 幂等性（重复插入不会产生重复数据）

---

## 第五步：验证数据持久化

### 5.1 检查数据目录

```bash
ls -lh runtime/milvus
ls -lh runtime/minio
ls -lh runtime/etcd
```

应该看到 Milvus 的数据文件已经创建。

### 5.2 重启服务验证持久化

```bash
docker compose -f compose.yml -f compose.milvus.yml -f compose.cuda.yml down
docker compose -f compose.yml -f compose.milvus.yml -f compose.cuda.yml up -d
```

等待服务启动后，再次运行测试脚本，之前插入的测试数据应该还在。

---

## 常见问题排查

### 问题 1：Milvus 无法启动

**症状**：`docker logs momentseek-milvus` 显示错误

**解决方案**：
1. 检查 etcd 和 minio 是否正常：
   ```bash
   docker logs momentseek-etcd
   docker logs momentseek-minio
   ```
2. 确保 runtime 目录有写权限
3. 清理数据重新启动：
   ```bash
   docker compose -f compose.milvus.yml down -v
   rm -rf runtime/milvus runtime/minio runtime/etcd
   docker compose -f compose.milvus.yml up -d
   ```

### 问题 2：应用无法连接 Milvus

**症状**：应用日志显示 "Connection refused" 或 timeout

**解决方案**：
1. 确认 Milvus 端口暴露：
   ```bash
   docker port momentseek-milvus
   ```
   应该显示 `19530/tcp -> 0.0.0.0:19530`

2. 检查网络连接：
   ```bash
   docker exec -it momentseek-mvp-app-cuda ping milvus
   ```

3. 确保两个服务在同一个 Docker 网络：
   ```bash
   docker network inspect momentseek-mvp-net
   ```

### 问题 3：Embedding 维度不匹配

**症状**：插入数据时报错 "dimension mismatch"

**解决方案**：
1. 检查你的模型实际输出维度：
   ```python
   import numpy as np
   from your_model import extract_embedding
   
   embedding = extract_embedding(sample_input)
   print(f"Actual dimension: {len(embedding)}")
   ```

2. 更新 `milvus_schema.py` 中的 `EMBEDDING_DIMS` 字典

3. 删除旧 collection 重新创建：
   ```python
   from pymilvus import utility
   utility.drop_collection("collection_name")
   # 重启应用，会自动重建
   ```

---

## 下一步

测试通过后，我们将：
1. 验证每个模型的实际 embedding 维度
2. 创建各模态的 Indexer 类（Visual, ASR, Face, OCR）
3. 集成到现有的索引流程中
4. 对比 npz 和 Milvus 的性能

---

## 环境变量参考

可以在 `.env` 文件中配置：

```env
# Milvus 配置
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Milvus 资源限制（开发环境）
MILVUS_CPUS=4
MILVUS_MEMORY_LIMIT=4g

# 批量写入配置
MILVUS_BATCH_SIZE=100
```

---

## NPU 服务器部署（Ascend）

本节说明如何在搭载昇腾 NPU 的服务器上部署应用及 Milvus。  
此前 `compose.ascend.yml` 和 `compose.server.yml` 均为引入 Milvus 之前的版本；
以下流程基于已更新的文件。

### 架构说明

```
┌─────────────────────────────────────────────────────┐
│  Docker 网络: momentseek-mvp-net                    │
│                                                     │
│  ┌──────────────────────┐   ┌─────────────────────┐ │
│  │  app (Ascend NPU)    │──▶│  milvus :19530      │ │
│  │  momentseek-mvp:     │   │  (CPU-only)         │ │
│  │    ascend            │   ├─────────────────────┤ │
│  │  NPU_ENABLED=true    │   │  etcd  :2379        │ │
│  │  OCR_DEVICE=auto     │   │  minio :9000        │ │
│  └──────────────────────┘   └─────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**关键点**：
- Milvus / etcd / MinIO 均为纯 CPU 服务，与底层加速卡无关，`compose.milvus.yml` 在 GPU 和 NPU 环境中完全通用。
- 应用容器挂载 NPU 设备文件并读取宿主机驱动，Milvus 通过 Docker 网络 DNS（`milvus:19530`）访问。
- `OCR_DEVICE=auto` 在 `NPU_ENABLED=true` 时自动选用 Ascend ACL 离线模型；
  `FACE_PROVIDER` 和 `ASR_SEMANTIC_DEVICE` 默认保持 CPU，NPU 路径尚未全量验证。

---

### Compose 文件组合矩阵

| 场景 | 命令（在项目根目录执行）|
|------|----------------------|
| **NPU 生产部署**（完整栈）| `compose.yml` + `compose.milvus.yml` + `compose.ascend.yml` |
| **服务器引导**（无 NPU 卡，先验证 Milvus）| `compose.yml` + `compose.milvus.yml` + `compose.server.yml` |
| GPU 本地开发 | `compose.yml` + `compose.milvus.yml` + `compose.cuda.yml` + `compose.dev.yml` |

---

### 前置条件

在服务器上确认以下各项：

```bash
# 1. 确认 NPU 驱动已安装
ls /usr/local/Ascend/driver

# 2. 确认可用的 davinci 设备（记下编号，例如 0）
ls /dev/davinci*

# 3. 确认 CANN 基础镜像可用（本地 image ID 或 registry 地址）
docker images | grep -i ascend

# 4. 确认 Docker Compose v2
docker compose version
```

所需环境变量（建议写入服务器上的 `.env` 文件）：

```env
# 宿主机 NPU 设备编号（必填）
HOST_NPU_DEVICE_ID=0

# CANN/torch_npu ARM64 基础镜像（使用可追溯 registry tag，不使用本机 image ID）
ASCEND_RUNTIME_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts

# 应用访问地址
APP_PUBLIC_URL=http://<服务器IP>:8000

# 可选资源限制
APP_CPUS=12
APP_MEMORY_LIMIT=16g
MILVUS_CPUS=4
MILVUS_MEMORY_LIMIT=4g
```

---

### 阶段一：服务器引导模式（无 NPU）

在正式挂载 NPU 卡之前，先用 `compose.server.yml` 验证镜像构建和 Milvus 连通性。
此模式 `NPU_ENABLED=false`，所有推理回退到 CPU。

#### 步骤 1：启动 Milvus 栈

```bash
docker compose -f compose.milvus.yml up -d
```

等待 60–90 秒后确认健康：

```bash
docker exec -it momentseek-milvus curl http://localhost:9091/healthz
```

#### 步骤 2：构建 Ascend 镜像

> Ascend 镜像使用离线安装方式（`--no-index`），需要先将 vendor wheels
> 放置在 `vendor-wheels/` 目录，并确认 `backend/requirements-ascend.txt` 存在。

```bash
docker compose -f compose.yml -f compose.server.yml build app
```

#### 步骤 3：启动应用（引导模式）

```bash
docker compose -f compose.yml -f compose.milvus.yml -f compose.server.yml up -d app
```

#### 步骤 4：验证 Milvus 连接

```bash
docker logs momentseek-mvp-app-server --tail 50
```

应看到 Milvus 连接成功及 collection 创建日志，无 `Connection refused`。

进入容器执行完整连接测试：

```bash
docker exec -it momentseek-mvp-app-server bash -c "cd /app/backend && python -m app.indexing.test_milvus_connection"
```

---

### 阶段二：NPU 生产部署

引导模式验证通过后，切换到 `compose.ascend.yml`。

#### 步骤 1：停止引导模式（如已运行）

```bash
docker compose -f compose.yml -f compose.server.yml down app
# Milvus 可保持运行，数据不丢失
```

#### 步骤 2：设置 NPU 设备编号

```bash
export HOST_NPU_DEVICE_ID=0   # 或写入 .env
```

#### 步骤 3：启动完整 NPU 栈

```bash
docker compose -f compose.yml -f compose.milvus.yml -f compose.ascend.yml up -d
```

#### 步骤 4：验证 NPU 使能

```bash
docker logs momentseek-mvp-app-ascend --tail 80
```

期望看到：
```
INFO - NPU enabled, device_id=0
INFO - OCR: loading Ascend ACL model from rapidocr/ascend/910b4-cann9-profile
INFO - Connecting to Milvus at milvus:19530
INFO - Collection visual_embeddings loaded, entities=...
```

验证 NPU 设备已挂载：

```bash
docker exec -it momentseek-mvp-app-ascend ls /dev/davinci*
```

#### 步骤 5：运行全量 Milvus 集成测试

```bash
docker exec -it momentseek-mvp-app-ascend bash -c \
  "cd /app/backend && python -m app.indexing.test_milvus_connection"
```

---

### NPU 环境特定配置说明

| 环境变量 | NPU 推荐值 | 说明 |
|---------|-----------|------|
| `NPU_ENABLED` | `true` | 启用 Ascend NPU 推理路径 |
| `OCR_DEVICE` | `auto` | NPU_ENABLED=true 时自动选用 ACL 离线模型 |
| `FACE_PROVIDER` | `cpu` | InsightFace NPU 路径未全量验证，保持 CPU |
| `ASR_SEMANTIC_DEVICE` | `cpu` | sentence-transformers NPU 支持待验证 |
| `ASCEND_VISIBLE_DEVICES` | `$HOST_NPU_DEVICE_ID` | 由 compose.ascend.yml 自动注入 |
| `ASCEND_RT_VISIBLE_DEVICES` | `$HOST_NPU_DEVICE_ID` | CANN 运行时设备路由 |
| `MODEL_IDLE_POLICY` | `process_exit` | 防止 NPU 显存驻留；如需热池改为 daemon 模式 |

---

### NPU 环境问题排查

#### 问题 A：`HOST_NPU_DEVICE_ID` 未设置导致启动失败

**症状**：`docker compose up` 报错 `variable is not set: HOST_NPU_DEVICE_ID`

**解决**：在 `.env` 中添加 `HOST_NPU_DEVICE_ID=0`（或对应的设备编号），或在命令前 export：

```bash
HOST_NPU_DEVICE_ID=0 docker compose -f compose.yml -f compose.milvus.yml -f compose.ascend.yml up -d
```

#### 问题 B：NPU 设备文件不存在

**症状**：容器启动失败，`/dev/davinci0: no such file`

**解决**：
```bash
# 检查宿主机设备
ls /dev/davinci*
# 确认驱动加载状态
npu-smi info
```

若驱动未加载，联系服务器管理员安装/重载 Ascend 驱动。

#### 问题 C：OCR NPU 自检失败

**症状**：日志出现 `OCR NPU self-test failed`，OCR 回退到 CPU

**这是预期行为**：`ocr_npu_self_test=true` 时，若 ACL 模型不兼容当前 CANN 版本，
会自动降级到 onnxruntime CPU 推理。若需强制使用 NPU，确认：
- `OCR_ACL_MODEL_DIR` 指向与当前 CANN 版本匹配的模型目录（默认 `rapidocr/ascend/910b4-cann9-profile`）
- 宿主机 CANN 版本与镜像内一致

#### 问题 D：应用连接 Milvus 超时（NPU 模式）

**症状**：`milvus:19530 connection timeout`

**检查网络**：
```bash
# 确认两个栈都在同一网络
docker network inspect momentseek-mvp-net | grep -A2 '"Name"'

# 从应用容器 ping milvus
docker exec -it momentseek-mvp-app-ascend ping milvus
```

若 NPU 应用容器不在网络中，检查是否三个 compose 文件都在同一 `docker compose` 命令中加载：
```bash
docker compose -f compose.yml -f compose.milvus.yml -f compose.ascend.yml ps
```

#### 问题 E：`ASCEND_RUNTIME_IMAGE` 为本地镜像 ID，换机器后失效

**症状**：`docker build` 失败，找不到基础镜像

**解决**：将 CANN 基础镜像推送到私有 registry，然后在 `.env` 中设置：
```env
ASCEND_RUNTIME_IMAGE=registry.example.com/cann-pytorch:8.0-torch2.1-aarch64
```

---

### 数据持久化（NPU 环境）

NPU 环境下数据持久化策略与 GPU 环境相同，Milvus 数据存储在宿主机 `runtime/` 目录：

```bash
# 检查 Milvus 数据目录
ls -lh ${HOST_RUNTIME_DIR:-./runtime}/milvus
ls -lh ${HOST_RUNTIME_DIR:-./runtime}/minio
ls -lh ${HOST_RUNTIME_DIR:-./runtime}/etcd
```

重启 NPU 应用不影响 Milvus 数据；完整重启流程：

```bash
# 只重启应用，保留 Milvus
docker compose -f compose.yml -f compose.ascend.yml restart app

# 全量重启（Milvus 数据保留在 runtime/ 目录）
docker compose -f compose.yml -f compose.milvus.yml -f compose.ascend.yml down
docker compose -f compose.yml -f compose.milvus.yml -f compose.ascend.yml up -d
```
