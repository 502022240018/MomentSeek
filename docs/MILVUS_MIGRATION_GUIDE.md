# Milvus 向量库引入完整指南

本文档说明项目如何将索引存储从 NPZ 文件迁移到 Milvus 向量数据库。所有描述与当前代码实现完全一致。

---

## 目录

1. [引入背景](#引入背景)
2. [架构设计](#架构设计)
3. [文件变更清单](#文件变更清单)
4. [五个模态的实现](#五个模态的实现)
5. [检索策略说明](#检索策略说明)
6. [配置与部署](#配置与部署)
7. [后续优化方向](#后续优化方向)

---

## 引入背景

### 迁移目标

在**索引和检索效果完全不受影响**的前提下，将向量检索主存储从 NPZ
切换为 Milvus。

> 当前合并策略（优先于本文后续历史阶段记录）：Milvus 默认接管
> visual、face、ASR、OCR、speaker 的向量写入和检索；SQLite 继续保存
> 视频、人物、任务和人工覆盖等关系数据；NPZ 始终保留并作为 Milvus
> 连接或查询失败时的本地回退。上线前必须执行历史 NPZ backfill。

### 实施结果

✅ **已达到初步可用状态**

- 5 个模态（Visual / ASR / OCR / Face / Speaker）已全部完成 Milvus 迁移
- 索引写入：先原子写入 NPZ，再同步写入 Milvus，写入失败可由 NPZ backfill
- 检索读取：默认读取 Milvus；服务异常时由 `MILVUS_FALLBACK_ENABLED=true` 回退 NPZ
- 检索质量：与原 NPZ 路径功能等价

### 设计原则

1. **保持检索质量优先**：当前实现完全复现 NPZ 路径的评分逻辑，不追求 ANN 性能优化
2. **分阶段迁移**：通过 `MILVUS_WRITE_ENABLED` 控制，可在不重新索引的情况下暂停 Milvus 写入
3. **最小化改动**：indexer 层无感知 Milvus schema，统一由 `write_modality_to_milvus()` 层负责格式适配

---

## 架构设计

### 核心组件

#### 1. MilvusWriteContext

所有 `build_*_index()` 函数接受可选的 `milvus_ctx` 参数，由 `stage_runner.py` 统一构造并注入。

```python
# stage_runner.py 中的构造方式（_setup_milvus_context）
if settings.milvus_write_enabled:
    new_version = bump_asset_version(video_index_dir)  # 读取并递增 milvus_meta.json
    try:
        milvus_ctx = MilvusWriteContext(
            video_id=video["id"],
            asset_version=new_version,
            client=get_milvus_client(),
        )
    except Exception:
        milvus_ctx = None  # 连接失败时降级，不写 Milvus
```

**写入流程（P2 直写路径，当前热路径）**：
1. 如果 `milvus_ctx` 存在，调用 `write_modality_from_memory()` 将内存中的数组直接写入 Milvus，不产生中间 NPZ 文件
2. 写入失败时，若调用方提供了 `recovery_save_fn`，才将数据落到 NPZ 磁盘作为恢复文件
3. 写入失败按 `fail_policy` 处理（`raise` 或 `warn`）

> **传统恢复路径**：`write_modality_to_milvus(ctx, modality, npz_path)` 从已有 NPZ 文件写入 Milvus，
> 主要供 `reindex_from_file()` 手动恢复和离线批量补写使用，不是正常索引流程。

#### 2. 版本化主键

```
{video_id}#{asset_ver}#{model_ver}#{modality}#{segment_id}
```

**作用**：
- 不同 `asset_version` → PK 不重叠，旧数据保留到手动清理
- 不同 `model_version` → PK 同样不重叠，支持多模型共存
- 同一 `(video_id, asset_ver, model_ver)` 重复写入 → PK 相同，Milvus upsert 幂等覆盖

#### 3. Collection 列表

| Collection | 模型 | 维度 | 索引类型 | 度量 |
|---|---|---|---|---|
| `visual_embeddings` | SigLIP2-so400m-384 | 1152 | HNSW | COSINE |
| `asr_embeddings` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | HNSW | IP |
| `ocr_embeddings` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | HNSW | IP |
| `face_embeddings` | InsightFace buffalo_l | 512 | IVF_FLAT | L2 |

| `speaker_embeddings` | 3D-Speaker CAM++ | 192 | HNSW | COSINE |

> **注意**：`paraphrase-multilingual-MiniLM-L12-v2` 输出 384 维，不是 768 维。

#### 4. 失败处理策略

| `fail_policy` | 行为 | 适用场景 |
|---|---|---|
| `raise`（默认）| 抛出异常，中止整个索引任务 | 生产环境（推荐）|
| `warn` | 仅记录日志，索引继续 | 测试 / 开发 |

---

## 文件变更清单

### 新增文件（Milvus 专用模块）

| 文件 | 说明 |
|---|---|
| `backend/app/indexing/milvus_flags.py` | 功能开关辅助函数（`milvus_write_enabled` / `milvus_write_fail_policy`）|
| `backend/app/indexing/milvus_schema.py` | 5 个 collection schema + 版本化 PK 生成器 + 维度校验 |
| `backend/app/indexing/milvus_client.py` | 进程级单例客户端 + collection 自动初始化 + 删除操作 |
| `backend/app/indexing/milvus_indexer.py` | 5 个模态 Indexer + `write_modality_to_milvus()` + 失败策略分发 |
| `backend/app/indexing/milvus_search.py` | 5 个模态检索函数（`milvus_*_candidates`）|
| `backend/app/indexing/milvus_asset_version.py` | asset_version 管理（读取/递增 milvus_meta.json）|
| `backend/app/indexing/milvus_stage_lock.py` | 视频级索引锁（防止并发冲突）|
| `backend/app/indexing/batch_buffer.py` | 批量写入缓冲工具（当前仅测试使用）|
| `backend/scripts/backfill_milvus.py` | 为已有视频批量补写 Milvus 数据 |
| `backend/scripts/migrate_milvus_schema.py` | 迁移 Milvus schema 版本 |

### 修改文件（核心业务逻辑）

| 文件 | 改动说明 |
|---|---|
| `backend/app/settings.py` | 新增 4 个 Milvus 配置字段（`milvus_host` / `milvus_port` / `milvus_write_enabled` / `milvus_write_fail_policy`）|
| `backend/app/main.py` | `lifespan`：初始化 MilvusClient；`/api/videos`：读取 milvus_meta.json |
| `backend/app/stage_runner.py` | 构造 `MilvusWriteContext` 并注入全部 5 个 `build_*` |
| `backend/app/search.py` | **完全移除 NPZ 读取路径**，所有检索直接调用 `milvus_*_candidates()` |
| `backend/app/indexing/visual.py` | Milvus 双写钩子：NPZ 写入后调用 `write_modality_to_milvus()` 再删除 NPZ |
| `backend/app/indexing/faces.py` | Milvus 双写钩子：同上 |
| `backend/app/indexing/asr.py` | Milvus 双写钩子：同上 |
| `backend/app/indexing/ocr.py` | Milvus 双写钩子：同上；Phase 4 gate 顺序修复（先写 Milvus 再删 NPZ）|
| `backend/app/indexing/speaker.py` | Milvus 双写钩子：同上；无人声场景提前返回修复 |

### 测试文件

| 文件 | 说明 |
|---|---|
| `backend/tests/test_phase1_*.py` | Phase 1 元数据解耦验证 |
| `backend/tests/test_phase2_*.py` | Phase 2 Milvus 集成验证 + bug 修复验证 |
| `backend/tests/test_phase3_*.py` | Phase 3 NPZ 依赖清理验证 |

| `backend/tests/test_phase4_*.py` | Phase 4 NPZ 门控验证 |
| `backend/tests/test_milvus_search_metric.py` | 度量表一致性 + L2→cosine 转换验证 |

---

## 五个模态的实现

> **注意**：以下"索引实现"代码片段展示的是 **NPZ 中转写入路径**（`write_modality_to_milvus`），
> 该路径目前仅用于离线恢复和批量补写。  
> **当前热路径**是 `write_modality_from_memory()`，直接从内存写入 Milvus，不产生临时 NPZ 文件。  
> 各模态的实际行号可能已因 P2 改造而变化，以源文件为准。

### 1. Visual（视觉帧）

#### 索引实现

**位置**：`backend/app/indexing/visual.py:765-785`

```python
# 始终写入完整 NPZ
atomic_save_npz(output_path, **payload)

# Milvus 双写
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "visual", output_path)
    Path(output_path).unlink(missing_ok=True)  # 写入成功后删除 NPZ
```

**写入数据**：
- 每帧一条记录
- 包含 frame_idx、timestamp_ms、segment_id、segment_start_ms、segment_end_ms、embedding

#### 检索实现

**位置**：`backend/app/indexing/milvus_search.py:218` - `milvus_visual_candidates()`

**策略**：**query-all + Python 侧分布评分**

1. 通过 `_query_all()` 获取视频的所有帧 embedding
2. 在 Python 中计算 `dot-product` 得到原始分数
3. 按 segment 聚合帧分数
4. 应用 **robust z-score** 和 **百分位数** 归一化（与 NPZ 路径完全相同）
5. 从 Milvus 数据推导 `duration_ms` 和 `segment_ms`（不再依赖 manifest.json）

**不使用 ANN 原因**：robust z-score 和百分位分布需要视频内的**完整分布样本**，top-k ANN 会使分布失真。

---

### 2. Face（人脸轨迹）

#### 索引实现

**位置**：`backend/app/indexing/faces.py:169-191`

```python
# 始终写入完整 NPZ
atomic_save_npz(output_path, arrays)

# Milvus 双写
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "face", output_path)
    Path(output_path).unlink(missing_ok=True)
```

**写入数据**：
- 每个 track 一条记录
- 包含 track_idx、start_ms、end_ms、best_ms、embedding

#### 检索实现

**位置**：`backend/app/indexing/milvus_search.py:570` - `milvus_face_candidates()`

**策略**：**ANN 扩展召回 + 精确余弦重打分**

1. **ANN 搜索**：以 `limit * 2` 扩展召回（补偿近似误差）
2. **精确重打分**：取回 embedding 向量做精确余弦计算，消除 IVF/HNSW 量化误差
3. **L2→cosine 转换**：Milvus 返回 L2 距离，转换公式 `cosine = 1 - L2² / 2`（单位向量）
4. **阈值过滤**：应用固定阈值（默认 0.35）

**不使用纯 ANN 原因**：ANN 返回的距离值有量化误差，阈值判断需要精确余弦分数。

---

### 3. ASR（语音识别）

#### 索引实现

**位置**：`backend/app/indexing/asr.py:1060-1101`

```python
# 始终写入完整 NPZ
atomic_save_npz(output_path, arrays)

# Milvus 双写
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "asr", output_path)
    Path(output_path).unlink(missing_ok=True)
```

**写入数据**：
- 每个 ASR chunk 一条记录（无论是否有 embedding）
- 包含 segment_idx、start_ms、end_ms、text、has_embedding、embedding
- `has_embedding=False` 的记录：embedding 是零向量占位符，仅用于词法匹配

#### 检索实现

**位置**：`backend/app/indexing/milvus_search.py:417` - `milvus_asr_candidates()`

**策略**：**query-all + 词法+语义混合评分**

1. 通过 `_query_all()` 获取视频的所有 ASR chunks
2. **词法评分**：对每个 chunk 的 text 字段进行 n-gram 覆盖度计算
3. **语义评分**（如果有查询 embedding）：
   - 仅对 `has_embedding=True` 的 chunks 计算 dot-product
   - 对所有语义分数应用 robust z-score 归一化
4. **混合得分**：`score = α * semantic + (1-α) * lexical`（α=0.6）
5. 按时间窗口聚合多个 chunks

**不使用 ANN 原因**：
- 需要同时支持词法和语义两种评分
- robust z-score 需要完整的分布样本

---

### 4. OCR（文字识别）

#### 索引实现

**位置**：`backend/app/indexing/ocr.py:671-688`

**说明**：NPZ 先写入（完整数据），Milvus 读取后删除 NPZ。

```python
# 始终写入完整 NPZ（write_modality_to_milvus 需要读取）
_save_ocr_npz(output_path, chunks, embeddings, embedding_frame_indices)

# Milvus 双写
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "ocr", output_path)
    Path(output_path).unlink(missing_ok=True)  # 写入成功后删除
```

**写入数据**：
- 每帧一条记录（无论是否有 embedding）
- 包含 frame_idx、frame_ms、text（所有框拼接）、start_ms、end_ms、avg_box_score、has_embedding、embedding
- `has_embedding=False` 的记录：embedding 是零向量占位符，仅用于词法匹配

**已知限制**：单个文字框文本（`ocr_box_texts`）未写入 Milvus，检索结果展示的是整帧文本而非精确匹配框。

#### 检索实现

**位置**：`backend/app/indexing/milvus_search.py:489` - `milvus_ocr_candidates()`

**策略**：**query-all + 词法+语义混合评分**（与 ASR 类似）

1. 通过 `_query_all()` 获取视频的所有 OCR 帧
2. **词法评分**：对每帧的聚合 text 字段进行 n-gram 覆盖度计算
3. **语义评分**（如果有查询 embedding）：
   - 仅对 `has_embedding=True` 的帧计算 dot-product
   - 应用 robust z-score 归一化
4. **混合得分**：`score = α * semantic + (1-α) * lexical`（α=0.6）
5. 按显示窗口（start_ms/end_ms）聚合多个帧

**不使用 ANN 原因**：同 ASR。

---

### 5. Speaker（说话人聚类）

#### 索引实现

**位置**：`backend/app/indexing/speaker.py:309-316`

**关键修复**：无人声场景提前返回，不创建任何文件（已修复）

```python
# 提前检查：如果无有效人声片段，直接返回（不写任何文件）
if not len(eligible):
    return {
        "utterances": 0,
        "tracks": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

# 正常流程：保存 NPZ 索引
save_speaker_index(...)

# Milvus 双写
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "speaker", output_path)
    Path(output_path).unlink(missing_ok=True)
```

**写入数据**：
- 每个 utterance 一条记录
- 包含 utterance_idx、start_ms、end_ms、asr_chunk_idx、track_id、embedding

#### 检索实现

**位置**：`backend/app/indexing/milvus_search.py:644` - `milvus_speaker_candidates()`

**策略**：**ANN 扩展召回 + 精确余弦重打分**（与 Face 类似）

1. **ANN 搜索**：以 `limit * 2` 扩展召回
2. **精确重打分**：取回 embedding 向量做精确余弦计算
3. **阈值过滤**：应用固定阈值（默认 0.50）
4. 按 track_id 聚合多个 utterances

**当前状态**：
- 写入和检索函数已实现
- `SearchEngine.search()` **暂未集成** speaker 分支（与 NPZ 路径一致）
- speaker 搜索通过独立的 voice search 接口触发（`/api/voice-search` 等端点）

**不使用纯 ANN 原因**：同 Face，阈值判断需要精确余弦分数。

---

## 检索策略说明

### 当前策略总结

| 模态 | 策略 | 使用 ANN | 原因 |
|---|---|---|---|
| Visual | query-all + 分布评分 | ❌ | robust z-score 需要完整分布 |
| Face | ANN + 精确重打分 | ✅（辅助召回）| 阈值需要精确分数 |
| ASR | query-all + 混合评分 | ❌ | 词法+语义+分布归一化 |
| OCR | query-all + 混合评分 | ❌ | 词法+语义+分布归一化 |
| Speaker | ANN + 精确重打分 | ✅（辅助召回）| 阈值需要精确分数 |

### 为什么不使用 Milvus Disk ANN 加速

**核心原因**：当前实现的首要目标是**保持检索质量与 NPZ 路径完全一致**。

1. **Visual/ASR/OCR**：需要视频内的**完整分布样本**来计算 robust z-score 和百分位数
   - 如果用 top-k ANN，只能拿到高分样本，无法计算正确的分布参数
   - 分布失真会导致归一化错误，最终影响跨视频的分数可比性

2. **Face/Speaker**：虽然使用了 ANN，但仍需**精确重打分**
   - Milvus 的 IVF/HNSW 索引会引入量化误差
   - 阈值判断（如 0.35）对精度敏感，量化误差可能导致误判
   - 因此 ANN 仅用于召回候选，最终分数必须用原始 embedding 重新计算

### query-all 实现细节

**工具函数**：`_query_all()` 在 `milvus_search.py` 中实现

- 使用 `QueryIterator`（pymilvus ≥ 2.3）做游标遍历
- 旧版本自动回退到 limit/offset 分页
- 单次 query 最多 16384 条记录，自动分页处理大视频

**性能考虑**：
- 对于 10 分钟视频（~600 帧），query-all 延迟约 50-100ms
- 相比 ANN 的 10-20ms，增加了延迟，但保证了检索质量
- 实际场景中，多视频并发检索时，网络和模型编码才是主要瓶颈

---

## 配置与部署

### 环境变量配置

**实际有效的 Milvus 配置项**（对应 `backend/app/settings.py`）：

```env
# ========== Milvus 连接 ==========
MILVUS_HOST=milvus                   # 默认 "milvus"（Docker Compose 服务名）
MILVUS_PORT=19530                    # 默认 19530

# ========== 功能开关 ==========
MILVUS_WRITE_ENABLED=true            # 默认 true；生产环境必须为 true
                                     # 设为 false 仅用于本地开发（无 Milvus 实例）

# ========== 写入失败策略 ==========
MILVUS_WRITE_FAIL_POLICY=raise       # 默认 "raise"
                                     # "raise" — 索引失败立即抛出异常（推荐）
                                     # "warn"  — 仅记日志，索引继续（测试/开发用）
```

**不存在的配置项**（2026-7-22完成迁移后废弃）：

<details>
<summary>点击展开已废弃的环境变量</summary>

以下变量在迁移指南的早期版本中出现，但在目前版本 `settings.py` 中移除：

- `NPZ_WRITE_ENABLED` — 原控制是否写入 NPZ 文件。实际实现用 `milvus_write_enabled` 替代：当 Milvus 启用时，NPZ 在写入 Milvus 后自动删除；当 Milvus 禁用时，NPZ 保留。
- `MILVUS_READ_ENABLED` — 原控制检索时是否从 Milvus 读取。实际 `search.py` 已强制使用 Milvus，无此开关。
- `MILVUS_ROLLOUT_PERCENT` — 原灰度路由百分比。
- `MILVUS_SHADOW_COMPARE_ENABLED` — 原影子对比（NPZ vs Milvus）。Phase 4 后 NPZ 数据已清空。
- `MILVUS_FALLBACK_ENABLED` — 原 Milvus 失败时回退到 NPZ。Phase 4 后 NPZ 已无法回退。
- `MILVUS_WRITE_QUEUE_PATH` — 原失败重试队列路径。

</details>

### 部署步骤

#### 1. 启动 Milvus

使用项目提供的 Docker Compose 配置：

```bash
# 启动 Milvus standalone（包含 etcd + minio + milvus）
docker compose -f compose.milvus.yml up -d
```

参考 `MILVUS_DEPLOYMENT_GUIDE.md` 了解完整部署细节。

#### 2. 迁移现有视频

为所有已索引的视频创建 `milvus_meta.json`：

```bash
cd backend
python -m scripts.backfill_milvus
```

该脚本会在每个视频的索引目录下创建：
```json
{"asset_version": "1"}
```

#### 3. 重新索引所有视频

**重要**：必须在 `MILVUS_WRITE_ENABLED=true` 下重新索引，才能将数据写入 Milvus。

```bash
# 方法 1：通过前端 UI 批量重新索引
# 方法 2：通过 API 批量触发
# 方法 3：编写脚本调用 stage_runner
```

#### 4. 确认纯 Milvus 模式

确认所有视频索引完成后，核对环境变量：

```env
MILVUS_WRITE_ENABLED=true    # 默认值，确认未被覆盖
MILVUS_HOST=milvus           # 确认指向正确的 Milvus 实例
```

当前实现已是纯 Milvus 模式——`MILVUS_WRITE_ENABLED=true` 时，NPZ 在写入 Milvus 后自动删除，无需额外配置。

#### 5. 清理旧 NPZ 文件（可选）

纯 Milvus 模式运行稳定后，可以删除旧的 NPZ 文件释放磁盘空间：

```bash
# 删除所有 .npz 文件（保留 .npz 空占位文件）
find runtime/indexes -name "*.npz" -size +0 -delete
```

**警告**：删除前请确保：
- Milvus 数据完整且可用
- 已备份重要视频的索引数据

### 数据管理

#### 查看 Milvus 数据

```python
from app.indexing.milvus_client import get_milvus_client
from pymilvus import utility

client = get_milvus_client()

# 查看所有 collections
print(utility.list_collections())

# 查看某个 collection 的记录数和加载状态
print(client.stats("visual_embeddings"))

# 查询某个视频的数据（通过 pymilvus Collection 直接查询）
col = client.collection_for("visual")
results = col.query(
    expr='video_id == "870c06c0cfba41609ee6ae6c5ebca32f"',
    output_fields=["video_id", "asset_version", "segment_id"],
    limit=10,
)
```

#### 删除视频数据

```python
from app.indexing.milvus_client import get_milvus_client

client = get_milvus_client()
video_id = "870c06c0cfba41609ee6ae6c5ebca32f"

# 删除该视频在所有 collections 中的数据
client.delete_video(video_id)
```

#### 版本化数据清理

同一视频的多次索引会创建不同的 `asset_version`，旧版本数据需要手动清理：

```python
# 删除特定版本（旧版本 asset_version="1"）
client.delete_video_version("xxx", "1")

# 保留最新版本，删除所有旧版本
# 先读取 milvus_meta.json 获取当前版本号，再删除旧版本
# current_version = get_current_asset_version(video_id)
# client.delete_video_version(video_id, old_version)
```

---

## 后续优化方向

### 1. 检索性能优化

**目标**：在保持检索质量的前提下，利用 Milvus 的 ANN 能力提升性能。

#### Visual/ASR/OCR 模态

**当前瓶颈**：query-all 获取完整数据集，单视频延迟 50-100ms。

**优化方案（需要研究）**：
1. **分桶归一化**：将视频分成多个时间段，每段独立计算分布参数
   - ANN 只召回少数高分段，降低 query-all 的数据量
   - 需要验证分桶对检索质量的影响
2. **预计算分布参数**：索引时计算并存储视频级的分布统计量（mean/std/percentiles）
   - 检索时用 ANN 召回 top-k，再用预存参数做归一化
   - 需要验证参数的稳定性和准确性

#### Face/Speaker 模态

**当前状态**：已使用 ANN + 精确重打分，性能较好。

**可能优化**：
1. **量化调优**：调整 IVF/HNSW 参数，减少量化误差，降低重打分开销
2. **阈值自适应**：根据 ANN 返回的分数分布，动态调整召回倍数

### 2. 写入性能优化

#### 2.1 背景：为什么现在要经过 NPZ 中间步骤

**当前写入链路**（5 个模态一致）：

```
build_X_index()
  │
  ├── 推理/提取（帧解码、模型推理）
  │     ↓ 全量结果累积在内存（chunks[] / embeddings[]）
  ├── _save_X_npz()          → 写磁盘（ocr.npz / visual.npz / …）
  ├── write_modality_to_milvus(ctx, "X", npz_path)
  │       └─ np.load(npz_path)  ← 从磁盘回读
  │       └─ 逐批 insert 到 Milvus
  └── Path(npz_path).unlink()   → 删除临时 NPZ
```

NPZ 是**真实存在的磁盘临时文件**，不是逻辑抽象。每次索引都会产生一次完整的写入（磁盘）→ 读取（磁盘）→ 写入（Milvus）的往返。

**为什么最初这么设计**：

Milvus 作为第二阶段引入时，indexer 已经以"产出 NPZ"为终态。为了最小化改动，`write_modality_to_milvus()` 选择从 NPZ 路径读取，相当于在原有流程末尾接了一个数据转运层。好处是：

- indexer 对 Milvus schema 无感知，关注点分离
- Milvus 写失败时 NPZ 仍在磁盘，支持离线重试（`retry_pending_writes()`）
- 可以对已有视频的 NPZ 批量重新 ingest，无需重新跑推理

**实际磁盘开销**（10 分钟视频）：

| 模态 | NPZ 临时文件大小 | 影响评估 |
|---|---|---|
| Visual | ~2.7 MB（600帧 × 1152维 × float16） | **主要瓶颈** |
| ASR | ~150 KB | 低 |
| OCR | ~150 KB | 低 |
| Face | ~20 KB | 可忽略 |
| Speaker | ~38 KB | 可忽略 |

对于视频密集索引场景（多任务并发、磁盘 I/O 竞争），Visual 的 NPZ 中间文件是主要写入开销。

---

#### 2.2 ✅ 已实现：直写 Milvus（P2 直写路径）

**状态**：已实现。`milvus_indexer.py` 提供 `write_modality_from_memory()` 作为热路径，
推理结果直接从内存写入 Milvus，不再产生临时 NPZ 文件。

**当前链路**：

```
build_X_index()
  │
  ├── 推理（帧解码、模型推理）
  │     ↓ 结果累积在内存（arrays dict）
  │     └─ write_modality_from_memory(ctx, modality, arrays)
  │           └─ indexer.upsert_from_memory(ctx, **arrays)   ← 直接写 Milvus
  │                 └─ _upsert_batched()    ← 自适应批量 upsert + 重试
  │
  └── （失败时）recovery_save_fn() → 写入 NPZ 用于手动恢复
```

数据在内存中处理后直接 upsert，NPZ 仅在 Milvus 写入失败时作为恢复文件写出。

**关键接口**（`milvus_indexer.py`）：

| 函数 | 用途 |
|------|------|
| `write_modality_from_memory(ctx, modality, arrays, recovery_save_fn=None)` | 热路径：内存直写 |
| `write_modality_to_milvus(ctx, modality, npz_path)` | 遗留路径：从 NPZ 文件写入（用于 `reindex_from_file` 和离线补写）|
| `reindex_from_file(*, client, modality, video_id, asset_version, model_version, npz_path)` | 手动恢复入口 |

---

#### 2.3 与现有批量写入优化的关系

各层优化的当前状态：

- **直写 Milvus**（已完成，见 §2.2）：`write_modality_from_memory()` 消除了 NPZ 中间文件；NPZ 仅在 Milvus 写入失败时按需写出用于恢复。
- **批量缓冲**（已完成）：`_upsert_batched()` 在 `milvus_indexer.py` 中根据模态自动计算批量大小（Visual ~55 行/批、Speaker ~290 行/批），降低每次 upsert RPC 调用次数。
- **异步写入**（未实现）：如需进一步解耦推理延迟与写入延迟，可将 `write_modality_from_memory()` 改为非阻塞（推理线程不等待 Milvus 响应），通过后台线程或 asyncio 队列分离两者。

### 3. OCR 精确匹配

**当前限制**：单个文字框文本（`ocr_box_texts`）未写入 Milvus。

**优化方案**：
1. 修改 OCR schema，为每个文字框创建独立记录（一帧多条）
2. 或者将 `ocr_box_texts` 存储为 JSON 字段，检索后在 Python 侧做精确匹配

### 4. Speaker Service 集成

**✅ 已完成**（2026-07-22）

`speaker_service.py` 所有数据读取路径均已迁移至 Milvus，NPZ 依赖已清除：

| 路径 | 状态 |
|---|---|
| speaker 主数据（utterance embeddings / times / refs） | `_speaker_data_from_milvus()` 直接从 Milvus speaker collection 重建 |
| ASR 文本（用于标注时间线和声搜结果） | `_texts_from_milvus()` 直接从 Milvus ASR collection 查询 |
| voice sample embeddings（实体识别） | 从数据库 BLOB 读取；Pre-Phase 3 遗留文件保留兼容路径 |

**已删除的 NPZ 读取**：
- `_texts()` 中的 `asr.npz` fallback（`speaker_service.py:67-70`）
- `from app.indexing.speaker import load_speaker_index` 的无用导入

**遗留说明**：`_load_voice_embeddings_for_entity()` 中保留了对 pre-Phase 3 embedding 文件的兼容读取，
仅对 `voice_embedding` BLOB 为空的历史样本生效。待确认所有历史样本已迁移后可删除该分支。

### 5. 多模型共存

**当前架构支持**：PK 包含 `model_ver`，可以存储多个模型版本的 embedding。

**应用场景**：
- A/B 测试新模型
- 同时支持多个模型的检索（如中文模型 + 英文模型）

**需要开发**：
1. 配置系统指定当前使用的模型版本
2. 检索时根据配置选择对应的 `model_ver` 过滤条件

### 6. 分布式部署

**当前状态**：单机 Milvus standalone。

**优化方案**：
1. 切换到 Milvus cluster 模式（需要 3+ 节点）
2. 利用 data node 分片和 query node 副本提升吞吐量
3. 配置 S3/OSS 作为持久化存储

---

## 常见问题

### Q1: 索引过程中 Milvus 写入失败怎么办？

**A1**：默认的 `fail_policy=raise` 会让索引任务立即抛出异常终止。

**排查步骤**：
1. 检查 Milvus 服务是否正常运行（`docker ps`）
2. 查看 backend 日志中的 Milvus 错误信息
3. 确认网络连接和 `MILVUS_HOST` / `MILVUS_PORT` 配置

**策略选择**：
- `MILVUS_WRITE_FAIL_POLICY=raise` — 索引失败立即停止（推荐，确保数据完整性）
- `MILVUS_WRITE_FAIL_POLICY=warn` — 仅记录日志，索引继续（该模态在 Milvus 中缺失数据）

**手动重试**：
```python
from app.indexing.milvus_indexer import reindex_from_file
from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_schema import MODEL_VERSIONS

# 为单个模态的 NPZ 文件手动重新写入 Milvus（恢复用）
reindex_from_file(
    client=get_milvus_client(),
    modality="visual",
    video_id="<video_id>",
    asset_version="1",
    model_version=MODEL_VERSIONS["visual"],
    npz_path="path/to/visual.npz",
)
```

### Q2: 检索结果与之前 NPZ 路径不一致？

**A2**：检查以下可能原因：

1. **asset_version 不匹配**：Milvus 中可能存在旧版本数据
   - 解决：删除旧版本数据或重新索引
2. **写入未完成**：索引过程被中断，部分数据未写入 Milvus
   - 解决：重新索引该视频
3. **度量表不一致**：Face 使用 L2，但代码期望 cosine
   - 解决：已在 `milvus_search.py` 中实现 L2→cosine 转换

### Q3: 如何回退到 NPZ 模式？

**A3**：当前架构下，NPZ 在 Milvus 写入后被立即删除，**无法直接回退到仅读 NPZ 的旧模式**。

**回退步骤**（需完整重新索引）：
1. 设置 `MILVUS_WRITE_ENABLED=false` — 新索引只写 NPZ，不写 Milvus
2. 重新索引所有视频（重新生成完整的 NPZ 文件）
3. 修改 `search.py` 恢复 NPZ 读取路径（当前代码已强制使用 Milvus）

### Q4: Milvus 数据占用多少磁盘空间？

**A4**：与原 NPZ 文件大小接近，略有增加（索引开销）。

**参考值**（10 分钟视频）：
- Visual: ~600 帧 × 1152 维 × 4 字节 ≈ 2.7 MB
- Face: ~10 tracks × 512 维 × 4 字节 ≈ 20 KB
- ASR: ~100 chunks × 384 维 × 4 字节 ≈ 150 KB
- OCR: ~100 帧 × 384 维 × 4 字节 ≈ 150 KB
- Speaker: ~50 utterances × 192 维 × 4 字节 ≈ 38 KB

**总计**：约 3 MB/视频（主要是 visual）

### Q5: 为什么 speaker 显示"未索引"？

**A5**：已在本次修复中解决。历史原因包括：

1. **Bug（已修复）**：`main.py` 的 `speaker_indexed` 判断逻辑有误，任何 Milvus 异常都被吞掉
2. **Bug（已修复）**：speaker 写入失败被 `warn` 策略静默处理，但 NPZ 已删除
3. **Bug（已修复）**：无人声视频仍然写入空的 speaker.npz

**当前状态**：所有 bug 已修复，speaker 索引和检索功能正常。

---

## 总结

Milvus 向量库已成功引入项目，**当前状态为初步可用**：

✅ **已完成**：
- 5 个模态全部支持 Milvus 写入和读取
- 纯 Milvus 模式（Phase 4）下索引和检索功能正常
- 检索质量与 NPZ 路径完全等价
- 版本化数据管理（支持重复索引和旧版本清理）

⚠️ **已知限制**：
- Visual/ASR/OCR 使用 query-all，未利用 ANN 加速
- Face/Speaker 虽然使用 ANN，但需要精确重打分
- OCR 单个文字框文本未存储

🚀 **后续方向**：
- 研究分桶归一化或预计算分布参数，优化 Visual/ASR/OCR 性能
- 支持多模型共存和分布式部署

---

*文档最后更新：2026-07-22*
