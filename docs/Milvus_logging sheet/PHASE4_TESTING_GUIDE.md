# Phase 4 测试指南：关闭 NPZ 写入，启用纯 Milvus 模式

**测试目标**: 验证在关闭 NPZ 写入后，系统完全依赖 Milvus 进行索引和检索

**当前环境状态**: 
- ✅ Docker 容器已运行（Milvus + 后端）
- ✅ Phase 4 代码已部署
- ✅ 需要配置环境变量并重启服务

---

## 一、前置条件确认

### 1.1 确认当前容器状态

根据您的环境，当前运行的容器：
```
momentseek-mvp-app-cuda   - 后端应用（CUDA 版本，端口 8300）
momentseek-milvus         - Milvus 向量数据库（端口 19530）
momentseek-etcd           - Milvus 元数据存储
momentseek-minio          - Milvus 对象存储
```

### 1.2 确认 Phase 4 功能就绪

✅ **您理解正确**：Phase 4 完成后，功能上**已经可以完全以 Milvus 为主导**

**关键配置项**:
- `NPZ_WRITE_ENABLED=false` - 停止写入 NPZ 文件（创建空占位符）
- `MILVUS_WRITE_ENABLED=true` - 启用 Milvus 写入
- `MILVUS_READ_ENABLED=true` - 启用 Milvus 读取
- `MILVUS_ROLLOUT_PERCENT=100` - 100% 流量走 Milvus

---

## 二、测试步骤

### 步骤 1：配置环境变量

编辑项目根目录的 `.env` 文件，添加或修改以下配置：

```bash
# Phase 4: 纯 Milvus 模式配置
NPZ_WRITE_ENABLED=false          # 关闭 NPZ 写入（核心配置）
MILVUS_WRITE_ENABLED=true        # 启用 Milvus 写入
MILVUS_READ_ENABLED=true         # 启用 Milvus 读取
MILVUS_ROLLOUT_PERCENT=100       # 100% 流量走 Milvus
MILVUS_SHADOW_COMPARE_ENABLED=false  # 关闭影子对比（不再需要）
MILVUS_FALLBACK_ENABLED=true     # 保留回退机制（可选）

# Milvus 连接配置（保持不变）
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

**配置说明**:
- `NPZ_WRITE_ENABLED=false` - **最关键**，启用 Phase 4 门控，索引时只创建空占位文件
- `MILVUS_ROLLOUT_PERCENT=100` - 确保所有检索请求都走 Milvus
- `MILVUS_SHADOW_COMPARE_ENABLED=false` - 已验证完毕，不需要再对比

---

### 步骤 2：重启后端容器

**是否需要重启？** ✅ **是的，必须重启后端容器**

**原因**:
- 环境变量更改需要重启才能生效
- `npz_write_enabled` 配置在应用启动时加载
- Docker 容器运行中的环境变量不会自动更新

**重启命令**:

```bash
# 方案 A：仅重启后端容器（推荐，速度快）
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml restart app

# 方案 B：完全重启（如果方案 A 有问题）
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml down
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml up -d app
```

**Milvus 容器是否需要重启？** ❌ **不需要**
- Milvus 配置没有变化
- 已经运行稳定，无需重启

---

### 步骤 3：验证配置生效

#### 3.1 检查容器日志

```bash
docker logs momentseek-mvp-app-cuda --tail 50
```

**预期日志（关键行）**:
```
INFO - NPZ write enabled: False          ← 确认 NPZ 写入已关闭
INFO - Milvus write enabled: True        ← 确认 Milvus 写入已启用
INFO - Milvus read enabled: True         ← 确认 Milvus 读取已启用
INFO - Milvus rollout percent: 100       ← 确认 100% 流量
```

#### 3.2 通过 API 验证配置

```bash
# 进入容器
docker exec -it momentseek-mvp-app-cuda bash

# 检查配置
python -c "
from app.settings import settings
print(f'NPZ Write Enabled: {settings.npz_write_enabled}')
print(f'Milvus Write Enabled: {settings.milvus_write_enabled}')
print(f'Milvus Read Enabled: {settings.milvus_read_enabled}')
print(f'Milvus Rollout: {settings.milvus_rollout_percent}%')
"
```

**预期输出**:
```
NPZ Write Enabled: False
Milvus Write Enabled: True
Milvus Read Enabled: True
Milvus Rollout: 100%
```

---

### 步骤 4：测试视频索引（核心测试）

#### 4.1 选择或上传测试视频

**选项 A：使用现有视频**
```bash
# 通过 API 获取视频列表
curl http://localhost:8300/api/videos | jq '.[] | {id, name}'

# 选择一个视频 ID
export TEST_VIDEO_ID="your-video-id-here"
```

**选项 B：上传新视频**
```bash
# 上传测试视频
curl -X POST http://localhost:8300/api/videos \
  -F "file=@/path/to/test-video.mp4" \
  -F "name=Phase4 Test Video"

# 从返回结果中获取 video_id
export TEST_VIDEO_ID="newly-created-video-id"
```

#### 4.2 触发索引（纯 Milvus 模式）

```bash
# 重建索引（5 个模态）
curl -X POST http://localhost:8300/api/videos/$TEST_VIDEO_ID/index \
  -H "Content-Type: application/json" \
  -d '{
    "modalities": ["visual", "face", "asr", "ocr", "speaker"]
  }'
```

**监控索引进度**:
```bash
# 查看索引状态
curl http://localhost:8300/api/videos/$TEST_VIDEO_ID | jq '.indexed_modalities'

# 实时查看后端日志
docker logs -f momentseek-mvp-app-cuda
```

#### 4.3 验证空占位文件创建

**关键验证点**：NPZ 文件应该是空的（0 字节）

```bash
# 方式 1：在宿主机检查
ls -lh runtime/indexes/$TEST_VIDEO_ID/*.npz

# 方式 2：在容器内检查
docker exec momentseek-mvp-app-cuda ls -lh /app/runtime/indexes/$TEST_VIDEO_ID/*.npz
```

**预期结果**:
```
-rw-r--r-- 1 root root    0 Jul 21 10:00 visual.npz      ← 0 字节
-rw-r--r-- 1 root root    0 Jul 21 10:00 face.npz        ← 0 字节
-rw-r--r-- 1 root root    0 Jul 21 10:01 asr.npz         ← 0 字节
-rw-r--r-- 1 root root    0 Jul 21 10:01 ocr.npz         ← 0 字节
-rw-r--r-- 1 root root    0 Jul 21 10:01 speaker.npz     ← 0 字节
```

**对比：Phase 4 之前的 NPZ 文件大小**:
```
-rw-r--r-- 1 root root  1.5M Jul 20 15:00 visual.npz     ← 有数据
-rw-r--r-- 1 root root  800K Jul 20 15:00 face.npz       ← 有数据
```

✅ **如果所有 NPZ 文件都是 0 字节，说明 Phase 4 门控工作正常！**

#### 4.4 验证 Milvus 数据写入

```bash
# 进入容器
docker exec -it momentseek-mvp-app-cuda bash

# 运行 Milvus 连接测试
cd /app/backend
python tests/test_milvus_connection.py
```

**预期输出**:
```
============================================================
Testing Milvus Connection
============================================================

1. Health Check
   Milvus health: ✓ OK

2. Collection Status
   visual_embeddings:
     - Loaded: True
     - Entities: 150+          ← 应该有数据
   asr_embeddings:
     - Loaded: True
     - Entities: 80+           ← 应该有数据
   face_embeddings:
     - Loaded: True
     - Entities: 20+           ← 应该有数据（如果视频中有人脸）
   ...

============================================================
✓ All tests passed!
============================================================
```

---

### 步骤 5：测试检索功能（前后端集成）

#### 5.1 启动前端（如果尚未启动）

```powershell
# 在新的 PowerShell 窗口中
cd frontend
$env:VITE_API_PROXY_TARGET="http://127.0.0.1:8300"
npm run dev
```

**前端是否需要重启？** ❌ **不需要**
- 前端只是 API 客户端
- 后端配置变化不影响前端代码
- 除非前端代码有修改

#### 5.2 通过 API 测试检索

**测试 Visual 检索**:
```bash
curl -X POST http://localhost:8300/api/search \
  -F "query_text=a person walking" \
  -F "modalities=visual" \
  -F "limit=5" | jq
```

**测试 ASR 检索**:
```bash
curl -X POST http://localhost:8300/api/search \
  -F "query_text=hello world" \
  -F "modalities=asr" \
  -F "limit=5" | jq
```

**测试 Face 检索**（需要先上传人脸参考图）:
```bash
curl -X POST http://localhost:8300/api/search \
  -F "face_image=@/path/to/face.jpg" \
  -F "modalities=face" \
  -F "limit=5" | jq
```

**预期结果**:
- 返回结果列表（非空，如果视频中有匹配内容）
- 每个结果包含：`video_id`, `timestamp_ms`, `score`, `snippet` 等
- 响应时间正常（< 1 秒）

#### 5.3 通过前端 UI 测试

1. 打开浏览器访问前端（通常是 `http://localhost:5173`）
2. 在搜索框输入查询文本（如 "person walking"）
3. 选择模态（Visual / ASR / OCR）
4. 点击搜索
5. **验证结果**：
   - ✅ 返回相关的视频片段
   - ✅ 时间戳正确
   - ✅ 可以播放视频
   - ✅ 无报错信息

---

### 步骤 6：验证空文件检测逻辑

**目的**：确认空 NPZ 文件被正确识别，检索自动切换到 Milvus

#### 6.1 查看后端日志（可选）

如果您想确认空文件检测逻辑是否被触发，可以临时添加日志：

```bash
docker exec -it momentseek-mvp-app-cuda bash

# 在容器内编辑 search.py（仅用于调试）
cd /app/backend
vi app/search.py

# 在 Line 966 附近（Visual 空文件检测处）添加：
# import logging
# _log = logging.getLogger(__name__)
# _log.info(f"Visual NPZ file size: {index_file.stat().st_size} bytes")
```

**更好的方式**：直接测试检索，如果返回结果正常，说明空文件检测已经工作。

---

## 三、预期结果总结

### 3.1 索引阶段

| 检查项 | 预期结果 | 验证方法 |
|--------|---------|---------|
| NPZ 文件大小 | 0 字节（所有模态） | `ls -lh runtime/indexes/$VIDEO_ID/*.npz` |
| Milvus 数据写入 | 有数据（实体数 > 0） | `python tests/test_milvus_connection.py` |
| 索引完成状态 | `indexed_modalities` 包含所有模态 | `curl /api/videos/$VIDEO_ID` |
| 后端日志 | 无错误，显示 Milvus 写入成功 | `docker logs momentseek-mvp-app-cuda` |

### 3.2 检索阶段

| 检查项 | 预期结果 | 验证方法 |
|--------|---------|---------|
| 检索返回结果 | 非空结果列表（如果有匹配） | API 测试或前端搜索 |
| 响应时间 | < 1 秒 | 观察 API 响应时间 |
| 结果准确性 | 时间戳和内容匹配 | 人工验证 |
| 前端展示 | 正常显示结果，可播放视频 | 浏览器测试 |

### 3.3 配置验证

| 配置项 | 预期值 | 验证方法 |
|--------|-------|---------|
| `npz_write_enabled` | `False` | Python 脚本检查 |
| `milvus_write_enabled` | `True` | Python 脚本检查 |
| `milvus_read_enabled` | `True` | Python 脚本检查 |
| `milvus_rollout_percent` | `100` | Python 脚本检查 |

---

## 四、常见问题排查

### 问题 1：NPZ 文件不是 0 字节

**症状**：`ls -lh runtime/indexes/*/visual.npz` 显示文件有数据（如 1.5M）

**原因**：
1. 配置未生效（容器未重启）
2. 使用了旧的索引数据

**解决方案**：
```bash
# 1. 确认配置
docker exec momentseek-mvp-app-cuda python -c "from app.settings import settings; print(settings.npz_write_enabled)"
# 应该输出 False

# 2. 如果输出 True，重启容器
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml restart app

# 3. 删除旧索引，重新索引
rm -rf runtime/indexes/$TEST_VIDEO_ID
curl -X POST http://localhost:8300/api/videos/$TEST_VIDEO_ID/index ...
```

---

### 问题 2：检索无结果

**症状**：API 返回空数组 `[]`

**可能原因**：
1. Milvus 中没有数据（索引未完成或失败）
2. 查询文本与视频内容不匹配
3. Milvus 读取未启用

**解决方案**：
```bash
# 1. 检查 Milvus 数据
docker exec momentseek-mvp-app-cuda python tests/test_milvus_connection.py

# 2. 检查索引状态
curl http://localhost:8300/api/videos/$TEST_VIDEO_ID | jq '.indexed_modalities'

# 3. 检查 Milvus 读取配置
docker exec momentseek-mvp-app-cuda python -c "
from app.settings import settings
print(f'Milvus Read: {settings.milvus_read_enabled}')
print(f'Rollout: {settings.milvus_rollout_percent}%')
"

# 4. 尝试更通用的查询
curl -X POST http://localhost:8300/api/search \
  -F "query_text=person" \
  -F "modalities=visual" | jq
```

---

### 问题 3：容器重启后配置丢失

**症状**：重启容器后，`npz_write_enabled` 又变回 `True`

**原因**：`.env` 文件未正确配置

**解决方案**：
```bash
# 1. 检查 .env 文件是否存在且包含配置
cat .env | grep NPZ_WRITE_ENABLED

# 2. 如果不存在，添加配置
echo "NPZ_WRITE_ENABLED=false" >> .env
echo "MILVUS_WRITE_ENABLED=true" >> .env
echo "MILVUS_READ_ENABLED=true" >> .env
echo "MILVUS_ROLLOUT_PERCENT=100" >> .env

# 3. 重启容器
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml restart app
```

---

### 问题 4：Milvus 连接失败

**症状**：后端日志显示 "Connection refused" 或 "timeout"

**解决方案**：
```bash
# 1. 检查 Milvus 容器状态
docker ps --filter "name=milvus"

# 2. 检查 Milvus 健康状态
docker exec momentseek-milvus curl http://localhost:9091/healthz

# 3. 检查网络连接
docker exec momentseek-mvp-app-cuda ping milvus

# 4. 如果 Milvus 未运行，启动它
docker compose -f compose.milvus.yml up -d

# 5. 等待 60-90 秒让 Milvus 完全启动
```

---

## 五、测试检查清单

**完成以下检查清单，确认 Phase 4 测试通过**：

- [ ] **配置检查**
  - [ ] `.env` 文件包含 `NPZ_WRITE_ENABLED=false`
  - [ ] 容器重启后配置生效（Python 脚本验证）
  
- [ ] **索引测试**
  - [ ] 视频索引完成（5 个模态）
  - [ ] NPZ 文件为空（0 字节）
  - [ ] Milvus 中有数据（实体数 > 0）
  - [ ] 后端日志无错误
  
- [ ] **检索测试**
  - [ ] Visual 检索返回结果
  - [ ] ASR 检索返回结果
  - [ ] Face 检索返回结果（如适用）
  - [ ] OCR 检索返回结果（如适用）
  
- [ ] **前端测试**
  - [ ] 前端搜索功能正常
  - [ ] 结果展示正确
  - [ ] 视频播放正常
  
- [ ] **性能测试**
  - [ ] 检索响应时间 < 1 秒
  - [ ] 无明显性能下降

---

## 六、回滚方案

**如果测试失败，需要回滚到双写模式**：

```bash
# 1. 修改 .env 文件
sed -i 's/NPZ_WRITE_ENABLED=false/NPZ_WRITE_ENABLED=true/' .env

# 或者直接编辑
vi .env
# 修改为：NPZ_WRITE_ENABLED=true

# 2. 重启容器
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml restart app

# 3. 验证配置恢复
docker exec momentseek-mvp-app-cuda python -c "from app.settings import settings; print(settings.npz_write_enabled)"
# 应该输出 True

# 4. 重建索引（恢复 NPZ 文件）
curl -X POST http://localhost:8300/api/videos/$TEST_VIDEO_ID/index \
  -H "Content-Type: application/json" \
  -d '{"modalities": ["visual", "face", "asr", "ocr", "speaker"]}'

# 5. 验证 NPZ 文件恢复
ls -lh runtime/indexes/$TEST_VIDEO_ID/*.npz
# 应该显示正常大小（非 0 字节）
```

---

## 七、下一步

**测试通过后**：
1. 在生产环境重复测试流程
2. 监控 1-2 周，确认 Milvus 稳定性
3. 准备 Phase 5：NPZ 读取路径清理
4. 准备 Phase 6：NPZ 文件物理清理

**测试失败后**：
1. 记录失败现象和错误日志
2. 执行回滚方案
3. 分析问题根因
4. 修复问题后重新测试

---

**测试联系人**: 开发团队  
**测试时间估计**: 30-60 分钟（首次测试）  
**建议测试时间**: 非高峰期或开发环境

**祝测试顺利！** 🚀
