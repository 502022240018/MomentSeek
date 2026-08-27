# Milvus 向量存储详细交接文档

## 文档概述

本文档详细介绍 MomentSeek 平台中 Milvus 向量存储的完整实现，包括架构设计、模态实现、数据存储分工以及扩展指南。

**目标读者**：需要理解或维护 Milvus 向量存储层的开发者。

**相关文档**：
- `../README.md` - 项目整体介绍（位于项目根目录）
- `ARCHITECTURE.md` - 平台架构
- `MILVUS_ONLY_MIGRATION.md` - 从旧索引迁移到 Milvus-only 模式的操作手册

---

## 目录

1. [整体架构](#整体架构)
2. [文件职责](#文件职责)
3. [Collection 设计](#collection-设计)
4. [数据存储分工](#数据存储分工)
5. [索引构建流程](#索引构建流程)
6. [检索流程](#检索流程)
7. [新增模态指南](#新增模态指南)
8. [常见问题](#常见问题)

---

## 整体架构

### 设计原则

MomentSeek 平台采用 **Milvus-only** 架构：
- **唯一在线检索库**：所有向量检索通过 Milvus 完成，不依赖本地 NPZ 文件
- **失败即中断**：索引写入失败时直接中断，不发布该 `asset_version`
- **版本化发布**：每次重建使用新 `asset_version`，验证通过后发布，旧版本可延迟清理
- **强一致性**：所有 Collection 使用 `consistency_level="Strong"` 保证读写一致性

### 模块分层

```
backend/app/vector_store/milvus/
├── __init__.py                      # 空文件，标记 Python 包
├── milvus_client.py                 # 客户端生命周期、Collection 初始化、索引管理
├── milvus_schema.py                 # Schema 定义、主键构建、维度常量
├── milvus_indexer.py                # 所有模态的索引写入器（直接写入 Milvus）
├── milvus_search.py                 # 所有模态的检索实现（ANN、混合检索）
├── milvus_search_visual_v2.py       # Visual 模态的优化检索实现
├── milvus_stage_lock.py             # 跨平台文件锁（防止并发重建冲突）
└── row_contract.py                  # 读取行的严格标量校验器
```

---

## 文件职责

### 1. `milvus_client.py` (719 行)

**核心职责**：管理 Milvus 连接生命周期、Collection 初始化和全局配置。

#### 关键类

**`MilvusClient`** - 进程级单例客户端
```python
# 获取单例
from app.vector_store.milvus import get_milvus_client
client = get_milvus_client()

# 主要方法
client.collection(name: str) -> Collection                 # 按名称获取 Collection
client.collection_for(modality: str) -> Collection         # 按模态获取 Collection
client.delete_video(video_id: str) -> dict[str, int]       # 删除视频所有记录
client.delete_video_modality(video_id, modality) -> int    # 删除特定模态
client.count_video_modality_version(video_id, modality, asset_version) -> int
```

**`ExistingMilvusCollectionsClient`** - 维护专用客户端
- 仅连接明确指定的 Collection（不创建新 Collection）
- 使用独立的连接别名，不影响主客户端
- 用于离线迁移脚本，避免触发 schema 验证

#### 全局配置

**索引配置映射** (`_COLLECTION_CONFIGS`)：
```python
{
    "visual_embeddings": {
        "schema": create_visual_schema,
        "index": None,  # 运行时从 settings.visual_use_diskann 动态解析
    },
    "asr_embeddings": {
        "schema": create_asr_schema,
        "indexes": {
            "embedding": {...},         # DiskANN 密集向量索引
            "sparse_embedding": {...},  # BM25 稀疏索引
        },
    },
    # ... 其他模态
}
```

**索引类型** (`_STATIC_INDEX_CONFIGS`)：
- **Visual**: HNSW 或 DiskANN（可配置切换）
- **ASR/OCR**: DiskANN + BM25 混合索引
- **Face**: DiskANN (COSINE)
- **Speaker**: DiskANN (COSINE)

#### 启动时行为

1. **连接 Milvus**：读取 `settings.milvus_host` 和 `settings.milvus_port`
2. **初始化 Collections**：
   - 不存在 → 创建 Collection + 创建索引 + 加载到内存
   - 已存在 → 加载（如果未加载）+ Schema 验证
3. **Schema 验证**（启动即失败）：
   - ASR/OCR Collection 必须包含 `sparse_embedding` 字段和 `bm25_*` 函数
   - 不支持运行时 Schema 迁移，必须手动重建

---

### 2. `milvus_schema.py` (318 行)

**核心职责**：定义所有 Collection 的 Schema、主键生成规则和维度常量。

#### 嵌入维度常量

```python
EMBEDDING_DIMS = {
    "visual":  1152,  # SigLIP2-so400m-384
    "asr":     384,   # paraphrase-multilingual-MiniLM-L12-v2
    "ocr":     384,   # 同 ASR 模型
    "face":    512,   # InsightFace buffalo_l
    "speaker": 192,   # 3D-Speaker CAM++
}
```

**重要**：更换模型后，必须：
1. 更新 `EMBEDDING_DIMS`
2. 更新 `MODEL_VERSIONS`
3. 删除并重建对应 Collection（Milvus 不支持修改向量维度）

#### 主键格式

所有模态采用统一的确定性主键格式（支持幂等 upsert）：

```
{video_id}#{asset_ver}#{model_ver}#{modality}#{segment_id}
```

- `video_id`: 视频唯一标识符
- `asset_ver`: 资产版本号（重建递增）
- `model_ver`: 模型版本（模型变更时递增）
- `modality`: 模态名称
- `segment_id`: 段内序号（格式因模态而异）

**主键构建函数**：
```python
visual_pk(video_id, asset_ver, frame_idx, model_ver)          # f{frame_idx:08d}
asr_pk(video_id, asset_ver, segment_idx, model_ver)           # s{segment_idx:08d}
ocr_pk(video_id, asset_ver, frame_idx, region_idx, model_ver) # f{frame_idx:08d}r{region_idx:04d}
face_pk(video_id, asset_ver, track_idx, model_ver)            # t{track_idx:08d}
speaker_pk(video_id, asset_ver, utterance_idx, model_ver)     # u{utterance_idx:08d}
```

#### Schema 创建函数

每个模态一个 `create_*_schema()` 函数，返回 `CollectionSchema` 对象：

- `create_visual_schema()` - 视觉帧嵌入
- `create_asr_schema()` - ASR 文本嵌入 + BM25
- `create_ocr_schema()` - OCR 文本嵌入 + BM25
- `create_face_schema()` - 人脸轨迹嵌入
- `create_face_group_schema()` - 人脸分组（视频内身份聚类）
- `create_entity_face_sample_schema()` - 命名实体人脸样本（跨视频）
- `create_speaker_schema()` - 说话人话语嵌入

#### 文本截断

```python
truncate_text_for_milvus(text: str) -> str
```
- 产品限制：2000 字符
- Milvus VARCHAR 限制：5000 UTF-8 字节
- 截断后保证 UTF-8 完整性（不产生半字符）

---

### 3. `milvus_indexer.py` (694 行)

**核心职责**：将内存数组写入 Milvus（无文件恢复机制）。

#### 设计哲学

- **直接写入**：接受内存数组，直接 upsert 到 Milvus
- **失败即中断**：写入失败抛出异常，调用方不发布该 `asset_version`
- **批量 upsert**：自动计算每个模态的最优批次大小（目标 256 KB/batch）
- **指数退避重试**：瞬时错误（RateLimit、UnexpectedError）自动重试 3 次

#### 核心上下文

```python
@dataclass
class MilvusWriteContext:
    video_id: str
    asset_version: str
    client: MilvusClient
    model_versions: dict[str, str] = field(default_factory=dict)
```

传递给所有 `build_*` 函数，避免在构建函数内部创建客户端。

#### 索引器类

每个模态一个索引器类，提供 `upsert_from_memory()` 方法：

**`VisualMilvusIndexer`**
```python
def upsert_from_memory(
    ctx: MilvusWriteContext,
    *,
    embeddings: np.ndarray,              # [N_frames, 1152]
    frame_times_ms: np.ndarray,          # [N_frames]
    segment_frame_offsets: np.ndarray,   # [N_segments+1], 段边界
    segment_times_ms: np.ndarray,        # [N_segments, 2], (start, end)
    duration_ms: int,
) -> int
```
- 验证段边界完整性（不重叠、不遗漏）
- 为每一帧分配 `segment_id` 和显式时间边界

**`AsrMilvusIndexer`**
```python
def upsert_from_memory(
    ctx: MilvusWriteContext,
    *,
    chunk_times_ms: np.ndarray,           # [N_chunks, 2]
    texts: list[str],                     # [N_chunks]
    embeddings: np.ndarray | None = None, # [N_semantic, 384]
    embedding_chunk_indices: np.ndarray | None = None,  # [N_semantic]
) -> int
```
- 支持稀疏语义嵌入（某些 chunk 仅有文本，无嵌入）
- `has_embedding` 字段标记是否有真实嵌入（False 时填充零向量）
- BM25 索引由 Milvus Function Field 自动计算

**`OcrMilvusIndexer`**
```python
def upsert_from_memory(
    ctx: MilvusWriteContext,
    *,
    frame_times_ms: np.ndarray,
    frame_windows_ms: np.ndarray,        # [N_frames, 2]
    embeddings: np.ndarray | None,
    embedding_frame_indices: np.ndarray | None,
    box_frame_indices: np.ndarray | None,
    box_texts: list[str] | None,
    box_scores: np.ndarray | None,
) -> int
```
- 同一帧多个文本框 → 聚合为单行（文本拼接、置信度平均）
- 支持稀疏语义嵌入

**`FaceMilvusIndexer`**
```python
def upsert_from_memory(
    ctx: MilvusWriteContext,
    *,
    embeddings: np.ndarray,              # [N_tracks, 512]
    track_times_ms: np.ndarray,          # [N_tracks, 3]: (start, end, best)
    group_model_version: str,
    group_embeddings: np.ndarray | None, # [N_groups, 512]
    group_track_indices: np.ndarray | None,
    group_times_ms: np.ndarray | None,
    # ... 其他 group 字段
) -> int
```
- 同时写入 `face_embeddings` (轨迹) 和 `face_groups` (分组)
- 分组嵌入是代表性轨迹的嵌入（用于跨视频身份匹配）

**`SpeakerMilvusIndexer`**
```python
def upsert_from_memory(
    ctx: MilvusWriteContext,
    *,
    utterance_embeddings: np.ndarray,    # [N_utterances, 192]
    utterance_times_ms: np.ndarray,      # [N_utterances, 2]
    utterance_refs: np.ndarray,          # [N_utterances, 2]: (asr_chunk_idx, track_id)
) -> int
```
- `asr_chunk_idx` 关联到 ASR Collection 的 `segment_idx`（稳定引用）

#### 批量大小优化

**自适应批次大小**（目标 256 KB/RPC）：
```python
_MODALITY_BATCH = {
    "visual":  ~55 rows   (1152*4 + 256 ≈ 4864 B/row)
    "asr":     ~115 rows  (384*4 + 512 ≈ 2048 B/row)
    "ocr":     ~115 rows
    "face":    ~120 rows  (512*4 + 128 ≈ 2176 B/row)
    "speaker": ~290 rows  (192*4 + 128 ≈ 896 B/row)
}
```

#### 重试机制

```python
_RETRYABLE_CODES = frozenset({1, 9999})  # UnexpectedError, RateLimit
_RETRY_MAX = 3
_RETRY_BASE_DELAY = 1.0  # 指数退避：1s → 2s → 4s
```

---

### 4. `milvus_search.py` (898 行)

**核心职责**：实现所有模态的向量检索，返回统一的 `Candidate` 对象。

#### 设计原则

- **只读操作**：不修改 Milvus 数据
- **明确失败**：连接/超时失败抛出 `MilvusServiceError`，空结果是有效答案
- **索引验证**：首次检索时验证索引类型与配置匹配（防止配置漂移）
- **严格行校验**：使用 `row_contract.py` 验证时间元数据（缺失/非法 → 丢弃）

#### 检索函数

**Visual - ANN 分段聚合**
```python
def milvus_visual_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,           # [D] 或 [N_queries, D]
    duration_ms: int | None,
    segment_ms: int | None,
    profile: str = "balanced",   # "precision" | "balanced" | "recall"
    limit: int = 72,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]
```
- 调用 `milvus_search_visual_v2.py` 中的 ANN 实现
- 多查询聚合：`0.65 * mean + 0.35 * min`
- 分段聚合：Top-N 帧平均分（可配置）
- 结果传给 VLM 重排，不做分布采样

**ASR - DiskANN + BM25 混合检索**
```python
def milvus_asr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
    query_text: str,
    query_embedding: np.ndarray | None,
    limit: int,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]
```
- 三路回退：
  - `hybrid`：query_embedding + query_text 都存在
  - `dense-only`：query_text 为空
  - `bm25-only`：query_embedding 为 None
- 使用 Milvus `WeightedRanker` 融合（语义权重 > 词汇权重）
- `hybrid_score` 不在 [0,1]，全局动态阈值在 `search.py` 中应用

**OCR - DiskANN + BM25 混合检索**
```python
def milvus_ocr_candidates_hybrid(...)  # 参数同 ASR
```
- 词汇权重 > 语义权重（OCR 文本短、词汇匹配更重要）
- 其他逻辑同 ASR

**Face - ANN 绝对阈值**
```python
def milvus_face_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    asset_version: str,
    limit: int,
    threshold: float | None = None,  # 默认 0.35
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]
```
- 单位归一化嵌入 + COSINE 度量 → `_distance` 就是余弦相似度（无需重算）
- `threshold` 仅影响 `above_threshold` 标志

**Speaker - ANN 绝对阈值**
```python
def milvus_speaker_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    asset_version: str,
    limit: int,
    threshold: float | None = None,  # 默认 0.50
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]
```
- 单位归一化 + COSINE → 信任 Milvus 返回的 `_distance`
- 不再重排（优化移除了二阶段重算）

**注意**：文档曾提到的 `milvus_speaker_candidates_scoped()` 跨视频作用域检索函数目前代码中不存在。如需跨视频声纹匹配，需在调用层实现多视频循环检索。

#### 索引验证缓存

```python
_verified_index_modalities: set[str] = set()  # 每个进程生命周期验证一次
```

首次检索时验证：
1. 获取 Collection 的 `index_type` 和 `metric_type`
2. 与配置期望对比（例如 Visual 期望 DISKANN/COSINE）
3. 不匹配 → 抛出 `MilvusServiceError`（要求重建索引）
4. 匹配 → 缓存模态名，后续检索跳过验证

---

### 5. `milvus_search_visual_v2.py` (524 行)

**核心职责**：Visual 模态的优化 ANN 检索实现（相比全查询扫描减少 60-80% 延迟）。

#### 主入口

```python
def milvus_visual_candidates_ann(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
    query_texts: list[np.ndarray],  # 多个子查询
    limit: int = 20,
    profile: str = "balanced",
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]
```

#### 流程

1. **索引验证**：验证 Collection 索引类型（HNSW 或 DiskANN）与配置匹配
2. **ANN 召回**：批量 ANN 检索（所有子查询一次 RPC）
3. **多查询聚合**：
   - 单查询：直接使用分数
   - 多查询：`0.65 * mean + 0.35 * min`（保留旧版语义）
4. **分段聚合**：
   - 每段取 Top-N 帧（默认 N=3）的平均分作为段分数
   - 选择最高分帧作为 `best_ms`
5. **排序和截断**：按分数降序，应用 profile cap（recall=500，其他=limit）

#### 参数自适应

```python
# HNSW: ef 参数
search_params = {"metric_type": "COSINE", "params": {"ef": max(top_k, 128)}}

# DiskANN: search_list 参数
search_params = {"metric_type": "COSINE", "params": {"search_list": max(top_k, 100)}}
```

#### 时间元数据验证

```python
def _valid_visual_results(...) -> list[dict]:
```
- 拒绝缺失时间字段的行
- 拒绝 `timestamp_ms` 不在 `[segment_start_ms, segment_end_ms]` 内的行
- 拒绝同一 `segment_id` 有多个不同时间边界的行（数据不一致）

---

### 6. `milvus_stage_lock.py` (104 行)

**核心职责**：防止同一 `(video_id, stage)` 并发重建导致数据交错。

#### 使用方式

```python
from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

with video_stage_lock(index_dir, video_id="vid123", stage="visual"):
    # 安全执行 delete + re-index
    client.delete_video_modality(video_id, "visual")
    # ... 写入新数据
```

#### 平台行为

- **Linux/macOS**：`fcntl.flock()` POSIX 咨询锁
- **Windows**：`msvcrt.locking()` 1 字节锁
- **非阻塞**：已被占用立即抛出 `StageLockError`
- **自动释放**：进程崩溃时 OS 自动释放文件描述符锁

#### 锁文件位置

```
{index_dir}/.{stage}.lock
```
例如：`/data/videos/vid123/.visual.lock`

---

### 7. `row_contract.py` (42 行)

**核心职责**：严格验证从 Milvus 读取的标量字段（时间、索引）。

#### 设计原则

- **失败即中断**：缺失或非法字段抛出 `ValueError`，不制造兼容性默认值
- **类型强制**：处理 NumPy 标量、Python `int/float`、布尔陷阱

#### 函数

```python
def required_int_field(entity: Any, field: str) -> int
```
- 拒绝 `None` 和布尔值
- 接受整数和可转整数的有限浮点数
- 示例：`frame_idx = required_int_field(entity, "frame_idx")`

```python
def required_nonnegative_int_field(entity: Any, field: str) -> int
```
- 同上，额外要求 `>= 0`

```python
def required_time_window(entity: Any) -> tuple[int, int]
```
- 读取 `start_ms` 和 `end_ms`
- 验证 `0 <= start_ms < end_ms`
- 示例：`start_ms, end_ms = required_time_window(entity)`

---

## Collection 设计

### 模态到 Collection 的映射

```python
_COLLECTION_FOR_MODALITY = {
    "visual":  "visual_embeddings",
    "asr":     "asr_embeddings",       # 可通过 settings 重定向
    "ocr":     "ocr_embeddings",
    "face":    "face_embeddings",
    "speaker": "speaker_embeddings",   # 可通过 settings 重定向
}
```

**动态重定向**（用于 AB 测试）：
- `MILVUS_ASR_COLLECTION` 环境变量可指定自定义 Collection 名
- `MILVUS_SPEAKER_COLLECTION` 同理
- 冲突检测：自定义名称不能与其他保留 Collection 冲突

### Collection Schema 详解

#### 1. `visual_embeddings` - 视觉帧

**字段**：
```python
pk                VARCHAR(512)      PRIMARY KEY
video_id          VARCHAR(255)
asset_version     VARCHAR(64)
model_version     VARCHAR(64)
frame_idx         INT64             # 帧序号
timestamp_ms      INT64             # 帧时间戳
segment_id        INT64             # 所属段 ID
segment_start_ms  INT64             # 段起始时间
segment_end_ms    INT64             # 段结束时间
embedding         FLOAT_VECTOR(1152)
```

**索引**：
- HNSW 或 DiskANN (COSINE)，运行时可配置切换
- HNSW: `{"M": 16, "efConstruction": 200}`
- DiskANN: `{"max_degree": 56, "search_list_size": 128, "pq_code_budget_gb": 0.125}`

**设计要点**：
- `segment_id` 支持两种分段策略：
  - 固定窗口：`segment_id = timestamp_ms // segment_ms`，边界可从 `segment_id * segment_ms` 计算
  - 镜头分割：显式边界存储在 `segment_start_ms` / `segment_end_ms`
- 每帧显式存储段边界，避免检索时重算

**数据规模**：约 1 FPS 采样 → 100 万帧/小时视频

---

#### 2. `asr_embeddings` - ASR 文本

**字段**：
```python
pk                VARCHAR(512)      PRIMARY KEY
video_id          VARCHAR(255)
asset_version     VARCHAR(64)
model_version     VARCHAR(64)
segment_idx       INT64             # ASR 块序号（Speaker 引用此字段）
start_ms          INT64
end_ms            INT64
text              VARCHAR(5000)     # 启用中文分词分析器
has_embedding     BOOL              # True=有语义嵌入，False=仅词汇
embedding         FLOAT_VECTOR(384) # 语义嵌入
sparse_embedding  SPARSE_FLOAT_VECTOR  # BM25 自动生成
```

**函数**：
```python
Function: bm25_asr
  input: text
  output: sparse_embedding
  type: BM25
```

**索引**：
- `embedding`: DiskANN (IP) - 语义检索
- `sparse_embedding`: SPARSE_INVERTED_INDEX (BM25) - 词汇检索

**设计要点**：
- **混合检索**：语义 + 词汇，Milvus `WeightedRanker` 融合
- **稀疏嵌入支持**：某些 chunk（如标点、语气词）可能只有文本无嵌入
- `has_embedding=False` 时填充零向量占位，检索时过滤
- `segment_idx` 稳定性：Speaker Collection 的 `asr_chunk_idx` 引用此字段

**数据规模**：约 5-10 秒/chunk → 360-720 行/小时视频

---

#### 3. `ocr_embeddings` - OCR 文本

**字段**：
```python
pk                VARCHAR(512)      PRIMARY KEY
video_id          VARCHAR(255)
asset_version     VARCHAR(64)
model_version     VARCHAR(64)
frame_idx         INT64             # 帧序号
region_idx        INT64             # 区域序号（当前实现固定为 0）
frame_ms          INT64             # 帧精确时间
start_ms          INT64             # 窗口起始（帧前后 ±N 秒）
end_ms            INT64             # 窗口结束
text              VARCHAR(5000)     # 聚合该帧所有文本框
avg_box_score     FLOAT             # 文本框置信度平均
has_embedding     BOOL
embedding         FLOAT_VECTOR(384)
sparse_embedding  SPARSE_FLOAT_VECTOR
```

**函数**：
```python
Function: bm25_ocr
  input: text
  output: sparse_embedding
  type: BM25
```

**索引**：同 ASR（DiskANN + BM25）

**设计要点**：
- **帧级聚合**：同一帧多个文本框 → 拼接为一行
- **时间窗口**：`[start_ms, end_ms]` 是帧可见的时间范围（避免闪现）
- **词汇优先**：OCR 文本短，BM25 权重 > 语义权重

**数据规模**：约 1-2 FPS OCR 采样 → 50-100 万帧/小时视频

---

#### 4. `face_embeddings` - 人脸轨迹

**字段**：
```python
pk              VARCHAR(512)      PRIMARY KEY
video_id        VARCHAR(255)
asset_version   VARCHAR(64)
model_version   VARCHAR(64)
track_idx       INT64             # 轨迹序号
start_ms        INT64             # 轨迹起始
end_ms          INT64             # 轨迹结束
best_ms         INT64             # 最佳质量帧时间戳
embedding       FLOAT_VECTOR(512) # 最佳帧人脸嵌入（单位归一化）
```

**索引**：
- DiskANN (COSINE) - 当前生产配置

**设计要点**：
- 一个轨迹 = 一个连续出现的人脸
- `embedding` 是轨迹中质量最高帧的人脸嵌入
- 单位归一化（ArcFace 模型输出）

**数据规模**：约 50-500 轨迹/小时视频

---

#### 5. `face_groups` - 人脸分组（视频内身份）

**字段**：
```python
pk                         VARCHAR(512)      PRIMARY KEY
video_id                   VARCHAR(255)
asset_version              VARCHAR(64)
model_version              VARCHAR(64)       # 分组算法版本
group_idx                  INT64
representative_track_idx   INT64             # 代表性轨迹
start_ms                   INT64             # 首次出现
end_ms                     INT64             # 最后出现
best_ms                    INT64             # 最佳质量时刻
bbox_x1, bbox_y1, bbox_x2, bbox_y2  FLOAT   # 代表性边界框（归一化 [0,1]）
representative_quality     FLOAT             # 代表性帧质量
duration_ms                INT64             # 总出现时长
occurrence_count           INT64             # 出现次数
importance_score           FLOAT             # 重要性分数
embedding                  FLOAT_VECTOR(512) # 代表性嵌入
```

**索引**：IVF_FLAT (L2)，`{"nlist": 256}`

**设计要点**：
- **聚类结果**：将视频内所有 `face_embeddings` 轨迹聚类为身份组
- **`model_version` = 分组算法版本**（非人脸模型版本）
- 不同 `group_model_version` 可共存（重新分组不删除旧版本）
- 用于：
  - 视频内"同一个人"查询
  - 跨视频身份匹配（通过 `embedding` ANN 检索）

**数据规模**：约 5-50 组/小时视频（取决于人物数量）

---

#### 6. `entity_face_samples` - 命名实体人脸样本

**字段**：
```python
pk                     VARCHAR(512)      PRIMARY KEY
entity_id              VARCHAR(255)      # 实体 ID（对应 SQLite entities 表）
sample_id              VARCHAR(255)      # 样本 ID
source_video_id        VARCHAR(255)      # 来源视频
source_asset_version   VARCHAR(64)
source_group_idx       INT64             # 来源分组
quality                FLOAT
embedding              FLOAT_VECTOR(512)
```

**索引**：IVF_FLAT (L2)，`{"nlist": 256}`

**设计要点**：
- **跨视频身份**：用户标注某个 `face_group` 为命名实体（如"张三"）
- 样本来自用户选择的视频中的分组
- 检索时：query 向量 vs. 该实体的所有样本 → 判断是否匹配

**`video_scoped=False`**：
- 不按 `video_id` 删除（跨视频共享）
- 删除实体时从 SQLite `entities` 表级联删除

**数据规模**：每个实体 1-10 个样本

---

#### 7. `speaker_embeddings` - 说话人话语

**字段**：
```python
pk              VARCHAR(512)      PRIMARY KEY
video_id        VARCHAR(255)
asset_version   VARCHAR(64)
model_version   VARCHAR(64)
utterance_idx   INT64             # 话语序号
start_ms        INT64
end_ms          INT64
asr_chunk_idx   INT64             # 关联 asr_embeddings.segment_idx
track_id        INT64             # 说话人轨迹 ID（视频内唯一）
embedding       FLOAT_VECTOR(192) # 声纹嵌入（单位归一化）
```

**索引**：DiskANN (COSINE)

**设计要点**：
- **话语级粒度**：一个连续说话片段 = 一个话语
- **关联 ASR**：`asr_chunk_idx` 指向对应文本块（稳定引用）
- **track_id 唯一性**：同一 `track_id` 的所有话语来自同一说话人
- **声纹身份匹配**：跨视频检索相似声纹

**数据规模**：约 50-200 话语/小时视频

---

### video_scoped 属性

```python
"video_scoped": True   # 默认，delete_video() 会删除
"video_scoped": False  # entity_face_samples 不随视频删除
```

---

## 数据存储分工

### Milvus 存储的数据

✅ **向量嵌入**
- 所有模态的嵌入向量（Visual、ASR、OCR、Face、Speaker）
- 稀疏向量（ASR/OCR 的 BM25 索引）

✅ **向量关联的元数据**
- 时间戳、时间窗口（`start_ms`, `end_ms`, `best_ms`）
- 文本内容（ASR/OCR 的 `text` 字段）
- 索引和引用（`frame_idx`, `track_idx`, `asr_chunk_idx`）
- 版本标识（`asset_version`, `model_version`）
- 质量分数（`avg_box_score`, `representative_quality`）
- 几何信息（Face 的 `bbox_*` 字段）

✅ **向量检索专用数据**
- 所有需要 ANN 检索的数据
- 混合检索的文本字段（启用分词器）

---

### SQLite 存储的数据

✅ **资产元数据和状态**
- 视频基本信息（`videos` 表）：名称、路径、时长、分辨率、状态
- 模态发布记录（`video_modality_publications` 表）：
  - `asset_version`：当前发布版本
  - `row_count`：行数
  - `status`：就绪状态
  - `metadata_json`：模态特定元数据（如分段策略）

✅ **任务和作业状态**
- 索引作业（`jobs` 表）：状态、进度、错误、worker PID
- 仿色任务（`color_grading_tasks` 表）

✅ **身份管理**
- 命名实体（`entities` 表）：ID、名称、参考路径、嵌入路径
- 人脸身份绑定（`face_identity_bindings` 表）
- 说话人身份绑定（`speaker_identity_bindings` 表）
- 声纹样本（`voice_samples` 表）：路径、嵌入路径、空间标识

✅ **用户管理的组织结构**
- 文件夹（`folders` 表）
- 视频-文件夹关系（`video_folders` 表）
- 说话人显示名（`video_speakers` 表）
- 话语覆盖（`utterance_overrides` 表）

✅ **清理队列**
- Milvus 清理队列（`milvus_cleanup_queue` 表）：失败重试

---

### 为什么需要 SQLite？

Milvus 是**向量数据库**，不是关系数据库：

❌ **Milvus 不支持**：
- 复杂关系查询（JOIN、子查询）
- 事务语义（跨 Collection 的 ACID）
- 灵活的元数据查询（如"所有未完成的作业"）
- 级联删除（外键约束）
- 用户定义的约束（UNIQUE、CHECK）

✅ **SQLite 适合**：
- 关系数据建模
- 轻量级本地存储（单文件数据库）
- 复杂查询和聚合
- 事务保证

---

### 能否移除 SQLite 依赖？

**短期内不推荐**，原因：

1. **Milvus 定位明确**：向量检索，不是关系数据库
2. **SQLite 开销小**：单进程访问，几乎零运维成本
3. **职责清晰**：
   - SQLite = 元数据、状态、关系
   - Milvus = 向量、检索
4. **替换成本高**：迁移到 PostgreSQL/MySQL 需要额外部署

**长期考虑**（大规模部署）：
- 可以用 **PostgreSQL** 替换 SQLite（支持并发、网络访问）
- **不推荐**用 Milvus 存储关系元数据（会导致数据建模混乱）

---

## 索引构建流程

### 整体流程

```
1. 用户上传视频 → Catalog.create_video()
2. 提交索引作业 → Catalog.create_job()
3. Worker 拉取作业 → 按模态顺序执行
4. 每个模态：
   4.1 获取文件锁（milvus_stage_lock）
   4.2 删除旧版本（delete_video_modality）
   4.3 构建内存数组（build_* 函数）
   4.4 写入 Milvus（MilvusIndexer.upsert_from_memory）
   4.5 验证行数（count_video_modality_version）
   4.6 发布版本（Catalog.publish_modality）
   4.7 释放文件锁
5. 所有模态完成 → 作业标记为 completed
```

### 关键代码路径

**入口**：`backend/app/execution/index_worker.py`

**模态构建函数**（每个模态一个）：
- `backend/app/indexing/modalities/visual/build.py::build_visual_index()`
- `backend/app/indexing/modalities/asr/build.py::build_asr_index()`
- `backend/app/indexing/modalities/ocr/build.py::build_ocr_index()`
- `backend/app/indexing/modalities/faces/build.py::build_face_index()`
- `backend/app/indexing/modalities/speaker/build.py::build_speaker_index()`

**写入流程示例**（Visual）：

```python
from app.vector_store.milvus import get_milvus_client
from app.vector_store.milvus.milvus_indexer import (
    MilvusWriteContext,
    write_modality_from_memory,
)

# 1. 提取嵌入（内存数组）
embeddings = extract_visual_embeddings(video_path)  # [N, 1152]
frame_times_ms = ...
segment_frame_offsets = ...
segment_times_ms = ...

# 2. 准备写入上下文
ctx = MilvusWriteContext(
    video_id=video_id,
    asset_version=asset_version,
    client=get_milvus_client(),
)

# 3. 获取锁 + 删除旧版本
with video_stage_lock(index_dir, video_id, "visual"):
    ctx.client.delete_video_modality(video_id, "visual")
    
    # 4. 写入 Milvus
    row_count = write_modality_from_memory(
        ctx,
        modality="visual",
        arrays={
            "embeddings": embeddings,
            "frame_times_ms": frame_times_ms,
            "segment_frame_offsets": segment_frame_offsets,
            "segment_times_ms": segment_times_ms,
            "duration_ms": duration_ms,
        },
    )
    
    # 5. 验证行数
    persisted = ctx.client.count_video_modality_version(
        video_id, "visual", asset_version
    )
    assert persisted == row_count, "行数不匹配"
    
    # 6. 发布版本
    catalog.publish_modality(
        video_id=video_id,
        modality="visual",
        asset_version=asset_version,
        row_count=row_count,
        metadata={...},
    )
```

### 版本化发布策略

**asset_version 生成**：
```python
asset_version = f"v{int(time.time())}"  # 基于时间戳，单调递增
```

**版本切换流程**：
1. 写入新 `asset_version` → Milvus
2. 验证行数
3. 更新 SQLite `video_modality_publications.asset_version`
4. 旧 `asset_version` 数据保留（延迟清理）

**优点**：
- 原子切换（SQLite 单行更新）
- 失败不影响旧版本（用户仍能检索）
- 可回滚（修改 SQLite 指针即可）

---

## 检索流程

### 整体流程

```
1. 用户发起查询 → /api/search
2. 查询规划器（planner）分解为子查询 + 模态选择
3. 每个模态并行检索：
   3.1 获取视频的 asset_version（从 SQLite）
   3.2 调用 milvus_*_candidates() 函数
   3.3 返回 Candidate 列表
4. 跨模态融合（RRF）
5. VLM 重排（Visual）
6. 返回用户
```

### 关键代码路径

**入口**：`backend/app/api/search_routes.py::search()`

**检索引擎**：`backend/app/retrieval/search.py::SearchEngine`

**模态检索调度**：
```python
# backend/app/retrieval/search.py

def _retrieve_candidates_for_video(
    self,
    video: dict,
    modalities: list[str],
    query_dict: dict,
) -> list[Candidate]:
    candidates = []
    
    # 获取发布版本
    publications = catalog.get_video_modality_publications(video["id"])
    
    for modality in modalities:
        pub = publications.get(modality)
        if not pub or pub["status"] != "ready":
            continue
        
        asset_version = pub["asset_version"]
        
        # 调用模态检索函数
        if modality == "visual":
            cands = milvus_visual_candidates(
                client=self.milvus_client,
                video_id=video["id"],
                query=query_dict["visual_embedding"],
                duration_ms=video["duration_ms"],
                ...
            )
        elif modality == "asr":
            cands = milvus_asr_candidates_hybrid(
                client=self.milvus_client,
                video_id=video["id"],
                asset_version=asset_version,
                query_text=query_dict["text"],
                query_embedding=query_dict.get("text_embedding"),
                ...
            )
        # ... 其他模态
        
        candidates.extend(cands)
    
    return candidates
```

### Candidate 对象

所有模态检索函数返回统一的 `Candidate` 对象：

```python
@dataclass
class Candidate:
    video_id: str
    start_time: float        # 秒
    end_time: float          # 秒
    score: float             # 融合分数
    modality: str
    evidence: str            # 显示给用户的证据文本
    raw_score: float         # 原始相似度分数
    above_threshold: bool    # 是否高于阈值
    best_time: float         # 最佳时刻（秒）
    unit_type: str           # "segment" | "chunk" | "frame" | "track" | "utterance"
    unit_id: int             # 单元 ID
    best_ms: int             # 最佳时刻（毫秒）
    text: str | None = None  # ASR/OCR 文本
    features: dict = field(default_factory=dict)  # 模态特定特征
```

---

## 新增模态指南

### 概述

新增模态需要实现以下组件：

1. **Schema 定义**（`milvus_schema.py`）
2. **索引器**（`milvus_indexer.py`）
3. **检索函数**（`milvus_search.py`）
4. **Collection 配置**（`milvus_client.py`）
5. **模态构建函数**（`backend/app/indexing/modalities/<modality>/build.py`）

---

### 步骤 1：定义 Schema

**文件**：`backend/app/vector_store/milvus/milvus_schema.py`

```python
# 1. 添加维度常量
EMBEDDING_DIMS["new_modality"] = 768

# 2. 添加模型版本
MODEL_VERSIONS["new_modality"] = "model-name-v1"

# 3. 定义主键构建函数
def new_modality_pk(
    video_id: str,
    asset_ver: str,
    segment_idx: int,
    model_ver: str = MODEL_VERSIONS["new_modality"]
) -> str:
    return make_pk(video_id, asset_ver, model_ver, "new_modality", f"s{segment_idx:08d}")

# 4. 定义 Schema
def create_new_modality_schema() -> CollectionSchema:
    fields = _common_fields() + [
        FieldSchema("segment_idx", DataType.INT64),
        FieldSchema("start_ms", DataType.INT64),
        FieldSchema("end_ms", DataType.INT64),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["new_modality"]),
        # ... 其他字段
    ]
    return CollectionSchema(fields, description="New modality embeddings")
```

**注意事项**：
- 主键必须包含 `video_id`, `asset_version`, `model_version`
- 时间字段命名惯例：`*_ms` 表示毫秒
- 向量字段命名：`embedding`（密集）或 `sparse_embedding`（稀疏）

---

### 步骤 2：实现索引器

**文件**：`backend/app/vector_store/milvus/milvus_indexer.py`

```python
class NewModalityMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        embeddings: np.ndarray,           # [N, 768]
        segment_times_ms: np.ndarray,     # [N, 2]
        # ... 其他输入
    ) -> int:
        """写入新模态数据到 Milvus。"""
        # 1. 验证输入
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        times_arr = np.asarray(segment_times_ms, dtype=np.int32)
        if emb_arr.shape[1] != EMBEDDING_DIMS["new_modality"]:
            raise ValueError(f"维度不匹配: 期望 {EMBEDDING_DIMS['new_modality']}")
        
        # 2. 构建行
        model_ver = ctx.model_ver("new_modality")
        col = ctx.client.collection_for("new_modality")
        rows = [
            {
                "pk": new_modality_pk(ctx.video_id, ctx.asset_version, idx, model_ver),
                "video_id": ctx.video_id,
                "asset_version": ctx.asset_version,
                "model_version": model_ver,
                "segment_idx": idx,
                "start_ms": int(times_arr[idx, 0]),
                "end_ms": int(times_arr[idx, 1]),
                "embedding": emb_arr[idx].tolist(),
            }
            for idx in range(len(emb_arr))
        ]
        
        # 3. 批量 upsert
        return _upsert_batched(col, rows, "new_modality")

# 4. 注册到调度表
_INDEXERS["new_modality"] = NewModalityMilvusIndexer()
```

**注意事项**：
- 所有 NumPy 数组转为 Python list（`.tolist()`）
- 时间字段必须是 Python `int`（不能是 NumPy int64）
- 失败时抛出异常（不要返回错误码）

---

### 步骤 3：实现检索函数

**文件**：`backend/app/vector_store/milvus/milvus_search.py`

```python
def milvus_new_modality_candidates(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
    query: np.ndarray,
    limit: int,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """新模态检索。"""
    # 1. 归一化查询向量
    query_norm = normalize(np.asarray(query, dtype=np.float32))
    
    # 2. 配置 ANN 参数
    search_params = {
        "metric_type": "COSINE",  # 或 "IP" / "L2"
        "params": {"search_list": max(limit, 100)},  # DiskANN
    }
    
    # 3. 执行检索
    col = client.collection_for("new_modality")
    try:
        results = col.search(
            data=[query_norm.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=f'video_id == "{video_id}" and asset_version == "{asset_version}"',
            output_fields=["segment_idx", "start_ms", "end_ms"],
            timeout=get_settings().milvus_query_timeout_seconds,
        )
    except Exception as exc:
        raise MilvusServiceError(f"新模态检索失败: {exc}") from exc
    
    # 4. 转换为 Candidate
    candidates = []
    invalid_rows = 0
    for hit in results[0]:
        score = float(hit.distance)
        try:
            start_ms, end_ms = _required_time_window(hit.entity)
            segment_idx = _required_int_field(hit.entity, "segment_idx")
        except (TypeError, ValueError, OverflowError):
            invalid_rows += 1
            continue
        
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=score,
            modality="new_modality",
            evidence=f"[new_modality] score={score:.3f}",
            raw_score=score,
            above_threshold=True,
            best_time=_seconds(start_ms),
            unit_type="segment",
            unit_id=segment_idx,
            best_ms=start_ms,
        ))
    
    _log_dropped_time_rows("new_modality", video_id, invalid_rows)
    return candidates
```

**注意事项**：
- 使用 `row_contract.py` 验证时间字段
- 空结果是有效答案（返回空列表），不抛出异常
- 连接/超时失败抛出 `MilvusServiceError`

---

### 步骤 4：注册 Collection

**文件**：`backend/app/vector_store/milvus/milvus_client.py`

```python
# 1. 定义索引配置
_STATIC_INDEX_CONFIGS["new_modality_embeddings"] = {
    "index_type": "DISKANN",
    "metric_type": "COSINE",
    "params": {
        "max_degree": 56,
        "search_list_size": 128,
        "pq_code_budget_gb": 0.125,
        "build_dram_budget_gb": 32.0,
    },
}

# 2. 注册 Collection
_COLLECTION_CONFIGS["new_modality_embeddings"] = {
    "schema": create_new_modality_schema,
    "index": _STATIC_INDEX_CONFIGS["new_modality_embeddings"],
    "video_scoped": True,  # 或 False（不随视频删除）
}

# 3. 添加模态映射
_COLLECTION_FOR_MODALITY["new_modality"] = "new_modality_embeddings"
```

---

### 步骤 5：实现构建函数

**文件**：`backend/app/indexing/modalities/new_modality/build.py`

```python
from app.vector_store.milvus import get_milvus_client
from app.vector_store.milvus.milvus_indexer import (
    MilvusWriteContext,
    write_modality_from_memory,
)
from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

def build_new_modality_index(
    video_id: str,
    video_path: Path,
    index_dir: Path,
    asset_version: str,
) -> dict:
    """构建新模态索引。"""
    # 1. 提取特征（内存数组）
    embeddings, segment_times_ms = extract_embeddings(video_path)
    
    # 2. 准备写入上下文
    ctx = MilvusWriteContext(
        video_id=video_id,
        asset_version=asset_version,
        client=get_milvus_client(),
    )
    
    # 3. 获取锁 + 删除旧版本 + 写入
    with video_stage_lock(index_dir, video_id, "new_modality"):
        ctx.client.delete_video_modality(video_id, "new_modality")
        
        row_count = write_modality_from_memory(
            ctx,
            modality="new_modality",
            arrays={
                "embeddings": embeddings,
                "segment_times_ms": segment_times_ms,
            },
        )
        
        # 4. 验证行数
        persisted = ctx.client.count_video_modality_version(
            video_id, "new_modality", asset_version
        )
        assert persisted == row_count
    
    # 5. 返回元数据
    return {
        "row_count": row_count,
        "metadata": {"duration_ms": int(segment_times_ms[-1, 1])},
    }
```

---

### 步骤 6：集成到检索引擎

**文件**：`backend/app/retrieval/search.py`

在 `SearchEngine._retrieve_candidates_for_video()` 中添加分支：

```python
elif modality == "new_modality":
    cands = milvus_new_modality_candidates(
        client=self.milvus_client,
        video_id=video["id"],
        asset_version=asset_version,
        query=query_dict["new_modality_embedding"],
        limit=self.settings.new_modality_limit,
        profiler=profiler,
    )
```

---

### 步骤 7：测试

**单元测试**：`backend/tests/test_new_modality.py`

```python
def test_new_modality_upsert():
    ctx = MilvusWriteContext(...)
    indexer = NewModalityMilvusIndexer()
    count = indexer.upsert_from_memory(
        ctx,
        embeddings=np.random.randn(10, 768).astype(np.float32),
        segment_times_ms=np.array([[0, 1000], [1000, 2000], ...]),
    )
    assert count == 10

def test_new_modality_search():
    candidates = milvus_new_modality_candidates(
        client=get_milvus_client(),
        video_id="test_video",
        asset_version="v1",
        query=np.random.randn(768),
        limit=10,
    )
    assert isinstance(candidates, list)
```

**集成测试**：启动 Milvus + 写入 + 检索

---

### 常见陷阱

1. **忘记删除旧版本** → 行数累积（重复结果）
   - 解决：`delete_video_modality()` 必须在写入前调用

2. **维度不匹配** → Milvus upsert 失败
   - 解决：`probe_embedding_dim()` 验证维度

3. **主键冲突** → 重复行但不更新
   - 解决：确保主键包含 `asset_version`

4. **时间字段类型错误** → NumPy int64 vs. Python int
   - 解决：显式转换 `int()`

5. **忘记获取锁** → 并发重建数据交错
   - 解决：用 `video_stage_lock()` 包裹写入

---

## 常见问题

### Q1: 如何切换 Visual 索引类型（HNSW ↔ DiskANN）？

**步骤**：
1. 修改环境变量：`VISUAL_USE_DISKANN=true`（或 `false`）
2. 删除 Visual Collection：
   ```python
   from pymilvus import utility
   utility.drop_collection("visual_embeddings")
   ```
3. 重启服务（自动创建新索引类型）
4. 重建所有视频的 Visual 索引

**注意**：不能热切换（必须重建 Collection）

---

### Q2: 如何查看 Milvus 中的数据？

**方法 1：Python 脚本**
```python
from app.vector_store.milvus import get_milvus_client

client = get_milvus_client()
col = client.collection("visual_embeddings")

# 查询特定视频
rows = col.query(
    expr='video_id == "vid123" and asset_version == "v1234567890"',
    output_fields=["frame_idx", "timestamp_ms"],
    limit=10,
)
for row in rows:
    print(row)
```

**方法 2：Milvus CLI**
```bash
# 连接 Milvus
milvus_cli

# 列出 Collection
list collections

# 查看 Schema
describe collection -c visual_embeddings

# 查询数据
query -c visual_embeddings -e 'video_id == "vid123"' -o frame_idx,timestamp_ms -l 10
```

---

### Q3: 如何清理已删除视频的 Milvus 数据？

**自动清理**（推荐）：
```python
# 删除视频时自动清理 Milvus
catalog.delete_video(video_id)  # 内部调用 client.delete_video()
```

**手动清理**：
```python
from app.vector_store.milvus import get_milvus_client

client = get_milvus_client()
result = client.delete_video("vid123")
print(result)  # {"visual_embeddings": 1000, "asr_embeddings": 500, ...}
```

**批量清理**（维护脚本）：
```bash
python backend/scripts/cleanup_orphaned_milvus_data.py
```

---

### Q4: 如何迁移到新模型版本？

**场景**：从 SigLIP-v1 (1152d) 升级到 SigLIP-v2 (2048d)

**步骤**：
1. 更新 `EMBEDDING_DIMS["visual"] = 2048`
2. 更新 `MODEL_VERSIONS["visual"] = "siglip2-v2"`
3. 删除旧 Collection：`utility.drop_collection("visual_embeddings")`
4. 重启服务（创建新 Collection）
5. 重建所有视频：
   ```bash
   python backend/scripts/reindex_all_videos.py --modality visual
   ```

**注意**：不同维度的 Collection 无法共存（必须删除重建）

---

### Q5: 如何诊断检索速度慢？

**步骤 1：启用 Profiler**
```python
from app.retrieval.retrieval_metrics import RetrievalProfiler

profiler = RetrievalProfiler()
candidates = milvus_visual_candidates(..., profiler=profiler)
print(profiler.get_report())
```

**步骤 2：检查索引类型**
- HNSW：适合 < 100 万向量
- DiskANN：适合 > 100 万向量（降低内存）

**步骤 3：调整 ANN 参数**
- DiskANN：增大 `search_list`（精度 ↑，速度 ↓）
- HNSW：增大 `ef`（精度 ↑，速度 ↓）

**步骤 4：检查 Milvus 负载**
```bash
# Milvus 日志
docker logs milvus-standalone

# 资源使用
docker stats milvus-standalone
```

---

### Q6: 如何备份和恢复 Milvus 数据？

**备份**（官方方法）：
```bash
# 使用 Milvus Backup 工具
milvus-backup create -n backup_20260825
```

**手动导出**（仅适合小数据量）：
```python
# 导出为 JSON（不推荐生产环境）
col = client.collection("visual_embeddings")
rows = col.query(expr="", output_fields=["*"], limit=1000000)
import json
with open("backup.json", "w") as f:
    json.dump(rows, f)
```

**恢复**：
```bash
milvus-backup restore -n backup_20260825
```

**推荐**：
- 使用 Milvus 官方备份工具
- 定期备份 SQLite 文件（`catalog.db`）
- 保留源视频文件（可随时重建索引）

---

### Q7: 如何处理 Schema 变更？

**不兼容变更**（需重建）：
- 修改向量维度
- 修改主键定义
- 删除必需字段

**兼容变更**（可在线添加）：
- 添加新字段（with default value）

**推荐做法**：
1. 创建新 Collection（如 `visual_embeddings_v2`）
2. 灰度迁移部分视频
3. 验证无误后切换所有流量
4. 删除旧 Collection

---

### Q8: Milvus 占用内存太大怎么办？

**原因**：默认所有数据加载到内存

**解决方案**：
1. **DiskANN 索引**（推荐）：
   - 向量存储在磁盘
   - PQ 压缩后的索引加载到内存
   - 内存占用 ≈ 原始数据的 10-20%

2. **Release Collection**：
   ```python
   col.release()  # 卸载内存
   col.load()     # 重新加载
   ```

3. **分片存储**（大规模）：
   - 使用 Milvus 集群模式
   - 数据分片到多个节点

---

## 总结

### 关键设计原则

1. **Milvus-only**：唯一在线检索源，无本地文件回退
2. **版本化发布**：新版本写入 → 验证 → 发布 → 清理旧版本
3. **失败即中断**：写入失败不发布，保证数据完整性
4. **模态隔离**：每个模态独立 Collection，互不干扰
5. **严格校验**：读取行时验证时间元数据，拒绝非法数据

### 分层职责

| 层次 | 文件 | 职责 |
|------|------|------|
| **连接管理** | `milvus_client.py` | 生命周期、Collection 初始化、全局配置 |
| **Schema** | `milvus_schema.py` | Collection 定义、主键生成、维度常量 |
| **写入** | `milvus_indexer.py` | 内存数组 → Milvus，批量 upsert，重试 |
| **检索** | `milvus_search.py` + `milvus_search_visual_v2.py` | ANN/混合检索 → Candidate |
| **并发控制** | `milvus_stage_lock.py` | 防止并发重建冲突 |
| **行校验** | `row_contract.py` | 严格验证时间/索引字段 |

### 扩展路径

- **新增模态**：按照"新增模态指南"7 步完成
- **切换索引类型**：删除 Collection + 重建
- **升级模型**：增加 `model_version` + 灰度迁移
- **水平扩展**：Milvus 集群模式（分片 + 副本）

---

**文档维护者**：请在 Schema、索引配置、检索逻辑变更时及时更新本文档。

**最后更新**：2026-08-27（行数修正、移除不存在的 scoped 函数、更新索引配置描述）
**创建日期**：2026-08-25
