# OCR/ASR 模态检索优化方案

**版本**: 2.3  
**日期**: 2026-07-30  
**状态**: 待实施  
**目标**: 面向亿级数据规模，全面采用 DiskANN + Milvus BM25 混合检索，实现词面优先的高性能检索

**更新日志**:
- **v2.3** (2026-07-30): 修复实施端反馈的 4 个关键问题：
  1. ✅ 添加 Candidate 字段兼容性方案（保留 lexical_score/semantic_score）
  2. ✅ 明确 BULK_QUERY_FIELDS 清理时序和前提条件
  3. ✅ 强调权重配置动态读取，避免硬编码
  4. ✅ 补充完整的错误处理策略和异常类
- **v2.2** (2026-07-29): 完善函数签名、集合路由、辅助函数实现
- **v2.1** (2026-07-29): 修复 schema 设计、cleanup 时机、空查询处理
- **v2.0** (2026-07-29): 采用 DiskANN + Milvus BM25 混合检索方案

---

## 📋 执行摘要

### 数据规模预估

项目面向 **10万小时视频**，数据规模：
- **ASR**: 10万小时 × 120 chunks/小时 = **1200万+ chunks**
- **OCR**: 10万小时 × 3600 frames/小时 = **3.6亿+ frames**
- **Embedding**: 千万到亿级规模

在这种规模下，必须采用 **DiskANN + 服务端混合检索** 才能保证性能和可扩展性。

### 核心挑战

1. **亿级数据规模**: HNSW 内存索引无法支撑，必须使用 DiskANN
2. **词面检索优先**: 项目实际更依赖词面检索，词面权重 > 词义权重
3. **Python 侧瓶颈**: 在大规模召回下，Python 侧评分成为瓶颈
4. **全量传输不可行**: 当前全量 query 在亿级数据下完全不可行

### 优化方案：DiskANN + BM25 混合检索

**核心策略**:
1. **Dense 向量**: 使用 DiskANN 索引（语义检索）
2. **Sparse 向量**: 使用 Milvus BM25（词面检索，中文 jieba 分词）
3. **混合检索**: Milvus 服务端 `hybrid_search()` 融合
4. **权重策略**: Sparse > Dense（如 0.7:0.3），符合词面优先需求
5. **清理 legacy**: 删除所有全量 query 和 Python 侧评分代码

**为什么不用纯 ANN + Python 评分？**

❌ 在亿级规模下的问题：
- 基于语义的 ANN 召回，会遗漏词面相关但语义不相关的结果
- 需要扩大召回数量（如 recall_k = 1000+）才能覆盖词面候选
- Python 侧对大量召回候选进行 n-gram 匹配，仍是性能瓶颈
- 不符合"词面优先"的检索需求

✅ Milvus BM25 混合检索的优势：
- BM25 倒排索引，O(log N) 复杂度，支持亿级文本
- 服务端融合，无 Python 侧瓶颈
- 权重可调，支持词面优先（sparse weight > dense weight）
- DiskANN 支持亿级 dense 向量，内存友好
- 充分利用 Milvus 原生能力

### 预期收益

| 指标 | 当前（全量query） | 优化后（混合检索） | 提升 |
|------|------------------|-------------------|------|
| 延迟（1小时视频） | 80-170ms | **10-20ms** | **8-15倍** |
| 延迟（10小时视频） | 800-1700ms | **10-20ms** | **80-150倍** |
| 网络传输 | 0.5-5 MB | **10-50 KB** | **100-500倍** |
| 内存占用 | 极高（全量） | **极低（DiskANN）** | **90%+ 降低** |
| 可扩展性 | 不支持亿级 | **支持亿级+** | **质的飞跃** |
| 词面检索 | Python O(N)扫描 | **Milvus 倒排索引** | **1000倍+** |


---

## 🔍 当前实现分析

### 当前架构的根本问题

**数据流程**（`milvus_search.py`）:
```
1. _query_all() 读取视频的所有 ASR/OCR 数据（text + embedding）
2. Python 侧 lexical_score() 对所有 text 进行 n-gram 匹配（O(N)）
3. Python 侧计算 semantic scores（embedding @ query）
4. Python 侧混合评分 max(lexical, 0.65*semantic + 0.35*lexical)
```

**问题汇总**:

| 问题 | 影响 | 亿级数据下的后果 |
|------|------|-----------------|
| ❌ 全量传输 | 网络 + 内存 | 10小时视频需传输几十 MB，完全不可行 |
| ❌ Python O(N) 扫描 | CPU 瓶颈 | 扫描百万级文本需要秒级延迟 |
| ❌ HNSW 内存索引 | 内存爆炸 | 3.6亿向量 × 384维 × 4字节 = **552 GB** |
| ❌ 无倒排索引 | 词面检索慢 | 无法快速定位关键词 |
| ❌ 语义优先召回 | 召回不准 | 词面相关但语义不相关的结果被遗漏 |

**结论**: 当前架构在亿级数据下**完全不可行**，必须彻底重构。

### Visual 模态的成功经验

Visual 模态已成功优化到 DiskANN：
- ✅ 使用 DiskANN 索引（磁盘，低内存）
- ✅ ANN 召回 top-K candidates
- ✅ 段聚合后直接返回（无 Python 侧全量评分）
- ✅ 支持大规模数据（千万级帧）

**经验迁移**:
- ASR/OCR 也必须使用 DiskANN
- 但 Visual 是纯语义检索，ASR/OCR 需要**词面+语义混合**
- 因此需要引入 Milvus BM25

### Speaker 模态的关联

**Schema**（`milvus_schema.py:188-197`）:
```python
FieldSchema("asr_chunk_idx",  DataType.INT64),  # 关联到 ASR chunk
FieldSchema("track_id",       DataType.INT64),  # Speaker track ID
```

**关联方式**:
- Speaker 通过 `asr_chunk_idx` 字段关联到 ASR 的 `segment_idx`
- Speaker 独立检索（ANN），不依赖 ASR 全量数据

**影响分析**:
- ✅ ASR 优化不影响 Speaker 检索逻辑
- ✅ 只需保持 `segment_idx` 的稳定性（主键中包含）
- ⚠️ 需要集成测试验证关联正确性

---

## 🎯 优化方案设计

### 方案：Milvus DiskANN + BM25 混合检索

#### 核心架构

```
查询请求
  ├─ Dense Query (semantic embedding)
  │   └─ DiskANN Search → top-K semantic candidates
  │
  ├─ Sparse Query (query text)
  │   └─ BM25 Search (倒排索引) → top-K lexical candidates
  │
  └─ Milvus hybrid_search()
      └─ WeightedRanker(0.3, 0.7)  # dense 30%, sparse 70%（词面优先）
          └─ 融合后的 top-N 结果 → 直接返回
```

**关键特性**:
1. **服务端处理**: 所有检索和融合在 Milvus 完成，无 Python 侧瓶颈
2. **DiskANN**: Dense 向量使用磁盘索引，内存占用降低 90%+
3. **BM25**: Sparse 向量使用倒排索引，支持亿级文本检索
4. **权重可调**: 词面优先（sparse > dense），符合项目需求
5. **中文分词**: 使用 jieba tokenizer，支持中文关键词检索

#### Schema 设计（v2）

##### ASR Schema

```python
def create_asr_schema_v2() -> CollectionSchema:
    from pymilvus import Function, FunctionType
    
    fields = _common_fields("asr") + [
        # 元数据字段
        FieldSchema("segment_idx",   DataType.INT64),
        FieldSchema("start_ms",      DataType.INT64),
        FieldSchema("end_ms",        DataType.INT64),
        
        # 文本字段 - 启用中文分词器
        FieldSchema("text", DataType.VARCHAR, max_length=2000,
                    enable_analyzer=True,
                    analyzer_params={
                        "type": "chinese",  # jieba tokenizer + cnalphanumonly filter
                    }),
        
        # Dense 语义向量（384维）
        FieldSchema("dense_embedding", DataType.FLOAT_VECTOR, dim=384),
        
        # Sparse BM25 向量 - 由 Function 自动生成
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True),
    ]
    
    # 定义 BM25 Function（Milvus 服务端自动计算）
    bm25_function = Function(
        name="bm25_asr",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
        params={"language": "zh"}
    )
    
    return CollectionSchema(
        fields, 
        description="ASR with BM25 hybrid search",
        functions=[bm25_function]
    )
```

**关键变化**:
- ✅ 删除 `has_embedding` 字段（所有行都有 dense embedding）
- ✅ 新增 `sparse_embedding` 字段（Function 自动生成）
- ✅ `text` 字段启用中文分词器（analyzer）
- ✅ 新增 `bm25_function` 定义（Milvus 服务端自动计算）
- ✅ `is_function_output=True` 标记 sparse_embedding

##### OCR Schema

```python
def create_ocr_schema_v2() -> CollectionSchema:
    from pymilvus import Function, FunctionType
    
    fields = _common_fields("ocr") + [
        # 元数据字段
        FieldSchema("frame_idx",     DataType.INT64),
        FieldSchema("region_idx",    DataType.INT64),  # 保持兼容，固定为 0
        FieldSchema("frame_ms",      DataType.INT64),
        FieldSchema("start_ms",      DataType.INT64),
        FieldSchema("end_ms",        DataType.INT64),
        FieldSchema("avg_box_score", DataType.FLOAT),
        
        # 文本字段 - 启用中文分词器
        FieldSchema("text", DataType.VARCHAR, max_length=2000,
                    enable_analyzer=True,
                    analyzer_params={"type": "chinese"}),
        
        # Dense 语义向量（384维）
        FieldSchema("dense_embedding", DataType.FLOAT_VECTOR, dim=384),
        
        # Sparse BM25 向量 - 由 Function 自动生成
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True),
    ]
    
    # 定义 BM25 Function
    bm25_function = Function(
        name="bm25_ocr",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
        params={"language": "zh"}
    )
    
    return CollectionSchema(
        fields,
        description="OCR with BM25 hybrid search",
        functions=[bm25_function]
    )
```

#### 索引配置

##### Dense 索引（DiskANN）

```python
# ASR dense_embedding 索引
collection.create_index(
    field_name="dense_embedding",
    index_params={
        "index_type": "DISKANN",  # 磁盘索引
        "metric_type": "IP",      # Inner Product (normalized = cosine)
        "params": {
            "search_list": 200,   # DiskANN 搜索参数
        },
    },
    index_name="dense_diskann_index",
)
```

**DiskANN 优势**:
- 磁盘存储，内存占用极低（< 10% HNSW）
- 支持亿级向量
- 搜索速度略慢于 HNSW（~100ms vs ~50ms），但在可接受范围

##### Sparse 索引（BM25）

```python
# ASR sparse_embedding 索引（BM25 倒排索引）
collection.create_index(
    field_name="sparse_embedding",
    index_params={
        "index_type": "SPARSE_INVERTED_INDEX",  # 倒排索引
        "metric_type": "IP",
    },
    index_name="sparse_bm25_index",
)
```

**BM25 优势**:
- 倒排索引，O(log N) 查找
- 支持亿级文本
- 标准 BM25 算法（k1=1.5, b=0.75）

#### 检索实现

**文件**: `backend/app/indexing/milvus_search_hybrid.py`（新建）或直接修改 `milvus_search.py`

##### 导入依赖

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymilvus import AnnSearchRequest, WeightedRanker

from app.indexing.common import normalize  # 向量归一化
from app.search import Candidate, _seconds  # 辅助函数和数据类型
from app.indexing.milvus_client import MilvusClient
from app.retrieval_metrics import RetrievalProfiler
from app.settings import get_settings

if TYPE_CHECKING:
    from pymilvus import Collection

logger = logging.getLogger(__name__)
```

##### 辅助函数实现

```python
def _get_search_params(profile: str) -> dict:
    """根据 profile 返回搜索参数。
    
    Args:
        profile: "precision", "balanced", "recall"
    
    Returns:
        包含 dense_search_list, dense_limit, sparse_limit 的字典
    """
    settings = get_settings()
    
    if profile == "precision":
        return {
            "dense_search_list": settings.asr_diskann_search_list_precision,  # 默认 100
            "dense_limit": 50,
            "sparse_limit": 50,
        }
    elif profile == "recall":
        return {
            "dense_search_list": settings.asr_diskann_search_list_recall,  # 默认 300
            "dense_limit": 200,
            "sparse_limit": 200,
        }
    else:  # balanced
        return {
            "dense_search_list": settings.asr_diskann_search_list_balanced,  # 默认 200
            "dense_limit": 100,
            "sparse_limit": 100,
        }


def _get_fusion_weights(profile: str) -> dict:
    """返回融合权重（词面优先）。
    
    Args:
        profile: "precision", "balanced", "recall"（当前版本未使用 profile 调整权重）
    
    Returns:
        包含 dense, sparse 权重的字典
    """
    settings = get_settings()
    
    # 词面权重 > 语义权重（默认 0.7:0.3）
    # 可通过配置调整
    return {
        "dense": settings.asr_dense_weight,    # 默认 0.3
        "sparse": settings.asr_sparse_weight,  # 默认 0.7
    }
```

##### ASR 混合检索

```python
def milvus_asr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
    profiler: RetrievalProfiler | None = None,
    rows: list[dict] | None = None,  # DEPRECATED - 保留以兼容旧调用
    profile: str = "balanced",
) -> list[Candidate]:
    """ASR 混合检索：DiskANN (语义) + BM25 (词面)。
    
    使用 Milvus Function Field，服务端自动处理 BM25 转换。
    词面优先：sparse_weight > dense_weight。
    
    Args:
        client: Milvus 客户端
        video_id: 视频 ID
        query_text: 查询文本（用于 BM25）
        query_embedding: 查询向量（用于 DiskANN）
        limit: 返回结果数量
        profiler: 性能分析器
        rows: DEPRECATED - 不再使用，为兼容旧调用而保留
        profile: 检索模式 - "precision", "balanced", "recall"
    """
    from pymilvus import AnnSearchRequest, WeightedRanker
    
    # 兼容性警告
    if rows is not None:
        logger.warning(
            "milvus_asr_candidates_hybrid: 'rows' parameter is deprecated "
            "and ignored in the hybrid search implementation."
        )
    
    col = client.collection_for("asr")
    query_norm = normalize(np.asarray(query_embedding, dtype=np.float32))
    
    # 根据 profile 确定参数
    search_params = _get_search_params(profile)
    weights = _get_fusion_weights(profile)
    
    # Dense 请求（语义检索）
    dense_req = AnnSearchRequest(
        data=[query_norm.tolist()],
        anns_field="dense_embedding",
        param={
            "metric_type": "IP",
            "params": {"search_list": search_params["dense_search_list"]},
        },
        limit=search_params["dense_limit"],
        expr=f'video_id == "{video_id}"',
    )
    
    # 检查 query_text 是否为空
    if not query_text or not query_text.strip():
        # 空查询：仅使用 dense 检索
        logger.warning(
            "Empty query_text for ASR sparse search, falling back to dense-only"
        )
        results = col.search(
            data=[query_norm.tolist()],
            anns_field="dense_embedding",
            param={
                "metric_type": "IP",
                "params": {"search_list": search_params["dense_search_list"]},
            },
            limit=limit,
            expr=f'video_id == "{video_id}"',
            output_fields=["segment_idx", "start_ms", "end_ms", "text"],
        )
    else:
        # Sparse 请求（词面检索）
        # Function Field 模式：直接传 text，Milvus 自动调用 BM25 function
        # 注意：传入原始 query_text（仅去除首尾空格），让 Milvus 使用 collection 的 analyzer 处理
        sparse_req = AnnSearchRequest(
            data=[query_text.strip()],  # Function Field 支持直接传文本
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=search_params["sparse_limit"],
            expr=f'video_id == "{video_id}"',
        )
        
        # 混合检索 + 加权融合（词面优先）
        results = col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(weights["dense"], weights["sparse"]),
            limit=limit,
            output_fields=["segment_idx", "start_ms", "end_ms", "text"],
        )
    
    # 转换为 Candidate 对象
    candidates: list[Candidate] = []
    for hit in results[0]:
        hybrid_score = float(hit.score)  # 融合后的分数
        text = str(hit.entity.get("text") or "")
        
        # 决策逻辑（简化版，因为已经是融合后的结果）
        above_threshold = hybrid_score > 0.5  # 阈值可配置
        decision = "hybrid_hit" if above_threshold else "weak"
        
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(hit.entity.get("start_ms")),
            end_time=_seconds(hit.entity.get("end_ms")),
            score=hybrid_score,
            modality="asr",
            evidence=f"[milvus_hybrid] {text[:100]}",
            raw_score=hybrid_score,
            decision=decision,
            above_threshold=above_threshold,
            best_time=_seconds(hit.entity.get("start_ms")),
            unit_type="chunk",
            unit_id=int(hit.entity.get("segment_idx")),
            best_ms=int(hit.entity.get("start_ms")),
            text=text,
            # ❌ 不设置 lexical_score 和 semantic_score（诚实）
            # 混合检索的融合是不可逆的，无法分离词面和语义分数
            # 下游代码会通过 _asr_result_hybrid_score() 读取 features.hybrid_score
            features={
                "hybrid_score": hybrid_score,
                "source": "milvus_hybrid",
            },
        ))
    
    return candidates


def _get_search_params(profile: str) -> dict:
    """根据 profile 返回搜索参数。"""
    settings = get_settings()
    
    if profile == "precision":
        return {
            "dense_search_list": 100,
            "dense_limit": 50,
            "sparse_limit": 50,
        }
    elif profile == "recall":
        return {
            "dense_search_list": 300,
            "dense_limit": 200,
            "sparse_limit": 200,
        }
    else:  # balanced
        return {
            "dense_search_list": 200,
            "dense_limit": 100,
            "sparse_limit": 100,
        }


def _get_fusion_weights(profile: str) -> dict:
    """返回融合权重（词面优先）。"""
    settings = get_settings()
    
    # 词面权重 > 语义权重
    # 可通过配置调整
    return {
        "dense": settings.asr_dense_weight,    # 默认 0.3
        "sparse": settings.asr_sparse_weight,  # 默认 0.7
    }
```


##### OCR 混合检索

```python
def milvus_ocr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
    profile: str = "balanced",
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """OCR 混合检索：实现与 ASR 相同。
    
    区别：
    - collection_for("ocr")
    - output_fields 包含 frame_idx, frame_ms
    - best_ms 使用 frame_ms
    """
    # 实现逻辑与 milvus_asr_candidates_hybrid 相同
    # 仅字段映射和 modality 不同
    pass
```

#### 配置参数

```python
# backend/app/settings.py

class Settings(BaseSettings):
    # ASR 混合检索配置
    asr_dense_weight: float = 0.3      # 语义权重
    asr_sparse_weight: float = 0.7     # 词面权重（优先）
    
    # OCR 混合检索配置
    ocr_dense_weight: float = 0.3      # 语义权重
    ocr_sparse_weight: float = 0.7     # 词面权重（优先）
    
    # DiskANN 搜索参数（按 profile）
    asr_diskann_search_list_precision: int = 100
    asr_diskann_search_list_balanced: int = 200
    asr_diskann_search_list_recall: int = 300
    
    ocr_diskann_search_list_precision: int = 100
    ocr_diskann_search_list_balanced: int = 200
    ocr_diskann_search_list_recall: int = 300
```

**权重说明**:
- **词面优先**: `sparse_weight = 0.7 > dense_weight = 0.3`
- 可根据实际效果调整（如 0.8:0.2 或 0.6:0.4）
- 不同 query 类型可以使用不同权重（未来优化）

---

## 📐 实施计划

### 总体时间：10-14 天

### 阶段 1: Schema 迁移准备（1-2 天）

#### 1.1 Schema 定义

**文件**: `backend/app/indexing/milvus_schema.py`

新增：
- `create_asr_schema_v2()` - 新版 ASR schema（含 sparse_embedding）
- `create_ocr_schema_v2()` - 新版 OCR schema（含 sparse_embedding）

保留：
- 旧版 `create_asr_schema()` 和 `create_ocr_schema()`（兼容旧数据）

#### 1.2 索引配置

**文件**: `backend/app/indexing/milvus_client.py`

**步骤 1: 更新 Collection 映射（重要）**

修改 `_COLLECTION_FOR_MODALITY` 指向 v2 collections：

```python
# backend/app/indexing/milvus_client.py L121-127

_COLLECTION_FOR_MODALITY: dict[str, str] = {
    "visual":  "visual_embeddings",
    "asr":     "asr_embeddings_v2",      # 修改：指向 v2
    "ocr":     "ocr_embeddings_v2",      # 修改：指向 v2
    "face":    "face_embeddings",
    "speaker": "speaker_embeddings",
}
```

**说明**：
- 这是**关键修改**，确保所有调用 `collection_for("asr")` 的地方都使用 v2 collection
- 修改时机：创建 v2 collections 并完成索引重建后
- 旧 collections 保留但不再使用，待验证稳定后删除

**步骤 2: 更新 `_COLLECTION_CONFIGS`**

添加 v2 collections 配置：
```python
"asr_embeddings_v2": {
    "schema": create_asr_schema_v2,
    "indexes": {
        "dense_embedding": {
            "index_type": "DISKANN",
            "metric_type": "IP",
            "params": {"search_list": 200},
        },
        "sparse_embedding": {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        },
    },
}
```

#### 1.3 Collection 创建脚本

**文件**: `backend/scripts/create_hybrid_collections.py`（新建）

功能：
- 创建 `asr_embeddings_v2` collection
- 创建 `ocr_embeddings_v2` collection
- 验证 schema 正确性
- 创建索引

### 阶段 2: 索引写入逻辑修改（2-3 天）

#### 2.1 ASR 索引写入

**文件**: `backend/app/processors/asr_funasr.py`

修改 `_write_to_milvus()`:
```python
def _write_to_milvus_v2(video_id: str, asset_ver: str, chunks: list[dict], embeddings: np.ndarray):
    """写入 ASR 到 Milvus v2（使用 Function Field）。
    
    注意：使用 Function Field 时，sparse_embedding 由 Milvus 自动计算，
    客户端只需提供 text 字段即可。
    """
    rows = []
    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        dense_emb = embeddings[idx].tolist()
        
        rows.append({
            "pk": asr_pk(video_id, asset_ver, idx),
            "video_id": video_id,
            "asset_version": asset_ver,
            "model_version": MODEL_VERSIONS["asr"],
            "segment_idx": idx,
            "start_ms": chunk["start_ms"],
            "end_ms": chunk["end_ms"],
            "text": text,
            "dense_embedding": dense_emb,
            # sparse_embedding 由 Milvus Function 自动计算，不需要手动提供
        })
    
    client = MilvusClient()
    col = client.collection_for("asr_v2")
    col.insert(rows)
```

**关键变化**:
- ✅ 删除 `has_embedding` 逻辑（所有行都有 embedding）
- ✅ **不需要**客户端计算 `sparse_embedding`（Function Field 自动生成）
- ✅ 仅提供 `text` 字段，Milvus 自动调用 BM25 function
- ✅ 使用新的 collection `asr_embeddings_v2`

#### 2.2 OCR 索引写入

**文件**: `backend/app/processors/ocr_*.py`

类似 ASR 的修改：
- 为每个 frame 生成 `dense_embedding`
- 提供 `text` 字段，`sparse_embedding` 由 Function Field 自动生成
- 写入 `ocr_embeddings_v2` collection

#### 2.3 索引重建策略（开发阶段简化版）

**策略**：开发阶段直接删除旧数据，全量重建

**文件**: `backend/scripts/rebuild_asr_ocr_v2.py`（新建）

```python
def rebuild_collections():
    """开发阶段：删除旧 collections，创建新 v2 collections。"""
    client = MilvusClient()
    
    # 删除旧 collections
    for name in ["asr_embeddings", "ocr_embeddings"]:
        if client.has_collection(name):
            logger.info(f"Dropping old collection: {name}")
            client.drop_collection(name)
    
    # 创建新 v2 collections（带 Function Field）
    for name in ["asr_embeddings_v2", "ocr_embeddings_v2"]:
        if not client.has_collection(name):
            logger.info(f"Creating new collection: {name}")
            client.create_collection(name)
```

**重新索引所有视频**:
```bash
# 遍历所有视频，重新提取和写入
python backend/scripts/reindex_all_videos.py \
    --modalities asr,ocr \
    --force-rebuild \
    --batch-size 10
```

**说明**：
- ✅ 无需考虑数据迁移和兼容性
- ✅ `has_embedding=False` 问题自然消失（重新提取所有 embedding）
- ✅ 索引质量更好（使用统一的新 schema）
- ✅ Function Field 自动处理 BM25，无需客户端计算

功能：
- 遍历所有已索引的视频
- 重新提取 embeddings（如果需要）
- 写入 v2 collections
- 进度跟踪和断点续传

**预估时间**:
- 假设 1000 小时已索引视频
- 重建速度 ~10 小时/小时（取决于机器）
- 总计 ~100 小时 ≈ 4-5 天

**优化**:
- 如果 embeddings 已存在（NPZ 或旧 Milvus），直接读取并转换
- 并行处理多个视频
- 批量写入 Milvus

### 阶段 3: 检索逻辑实现（2-3 天）

#### 3.1 混合检索实现

**文件**: `backend/app/indexing/milvus_search_hybrid.py`（新建）

实现：
- `milvus_asr_candidates_hybrid()` - ASR 混合检索
- `milvus_ocr_candidates_hybrid()` - OCR 混合检索
- `_get_search_params()` - 搜索参数
- `_get_fusion_weights()` - 融合权重

#### 3.2 修改调用路径

**文件**: `backend/app/indexing/milvus_search.py`

**步骤 1: 修改或替换现有函数**

方案 A - 直接替换（推荐）：
```python
def milvus_asr_candidates(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
    profiler: RetrievalProfiler | None = None,
    rows: list[dict] | None = None,  # 保留但忽略
    profile: str = "balanced",  # 新增参数
) -> list[Candidate]:
    """ASR 召回入口（v2 混合检索）。"""
    # 直接调用混合检索实现
    return milvus_asr_candidates_hybrid(
        client, video_id, query_text, query_embedding,
        limit, profiler, rows, profile
    )

# OCR 同理
def milvus_ocr_candidates(...):
    """OCR 召回入口（v2 混合检索）。"""
    return milvus_ocr_candidates_hybrid(...)
```

方案 B - 新函数并存：
- 保留旧函数 `milvus_asr_candidates()`
- 新建函数 `milvus_asr_candidates_hybrid()`
- 通过配置开关选择

**推荐**: 开发阶段使用方案 A（直接替换）

#### 3.2 删除预取优化（BULK_QUERY_FIELDS）

⚠️ **前提条件（必须全部满足）**：

1. ✅ v2 collections 已创建并索引完成
2. ✅ `_COLLECTION_FOR_MODALITY` 已修改指向 v2（阶段 1.2）
3. ✅ `milvus_asr_candidates()` 已修改为调用 `milvus_asr_candidates_hybrid()`
4. ✅ `milvus_ocr_candidates()` 已修改为调用 `milvus_ocr_candidates_hybrid()`
5. ✅ 单元测试通过（验证新实现正常工作）

**仅在满足所有前提后**，执行以下步骤。

**步骤 1: 理解当前预取机制**

当前流程：
1. `query_rows_for_videos()` 根据 `BULK_QUERY_FIELDS` 批量预取 ASR/OCR 数据
2. 预取的 `rows` 传递给 `milvus_asr_candidates(..., rows=rows)`
3. 旧实现使用 `rows` 进行 Python 侧评分

问题：v2 实现不需要 `rows`，预取浪费资源。

**步骤 2: 删除 BULK_QUERY_FIELDS 中的 ASR/OCR（核心修改）**

参考 Visual 模态的优化经验，从 `BULK_QUERY_FIELDS` 移除 ASR/OCR：

```python
# backend/app/indexing/milvus_search.py L103-128

BULK_QUERY_FIELDS: dict[str, list[str]] = {
    # "visual" is intentionally absent: v2 ANN 不需要预取
    # "asr" 和 "ocr" 也移除：v2 hybrid search 不需要预取
    # 
    # ASR/OCR v2 使用 hybrid_search()，直接在 Milvus 服务端检索，
    # 不需要预先读取所有 rows。包含在此处会导致无效的全量 query。
}
```

**说明**：
- 这是**关键修改**，避免不必要的批量预取
- Visual 模态已验证此做法的正确性（参考 FINAL_VERIFICATION_REPORT.md）
- 删除后，`query_rows_for_videos()` 不会为 ASR/OCR 执行预取
- ⚠️ **确保 Speaker 模态保留**（依赖预取获取 `asr_chunk_idx`）

**步骤 3: 修改调用方（search.py）**

删除 `rows` 参数传递：

```python
# backend/app/search.py

# ❌ 旧代码
candidates.extend(milvus_asr_candidates(
    client, video_id, query_text, query_embedding,
    channel_limits["asr"], profiler,
    rows=prefetched_rows.get("asr"),  # ← 删除此行
))

# ✅ 新代码
candidates.extend(milvus_asr_candidates(
    client, video_id, query_text, query_embedding,
    channel_limits["asr"], profiler,
    profile=visual_profile,  # 可选：传入 profile 参数
))
```

**影响**：
- 删除 `prefetched_rows.get("asr")` 和 `prefetched_rows.get("ocr")`
- `prefetched_rows` 字典可能变空（如果所有模态都已优化）
- ⚠️ 不要删除 `query_rows_for_videos()` 本身（Speaker 仍需要）

**步骤 4: 验证清理效果**

```python
# 测试脚本
def test_no_asr_prefetch():
    """验证 ASR 不再预取。"""
    prefetched = query_rows_for_videos(
        client, [video_id], modalities=["asr", "speaker"]
    )
    
    # ASR 不应该在预取结果中
    assert "asr" not in prefetched[video_id]
    
    # Speaker 应该仍在预取结果中
    assert "speaker" in prefetched[video_id]
```

**时序图**：

```
阶段 1.2: 修改 _COLLECTION_FOR_MODALITY 指向 v2
    ↓
阶段 3.1: 修改 milvus_asr_candidates() 调用 hybrid 实现
    ↓
阶段 4: 单元测试通过（验证新实现正常）
    ↓
阶段 3.2: 删除 BULK_QUERY_FIELDS 中的 ASR/OCR  ← 此时才安全
    ↓
阶段 5: 集成测试（验证整体流程）
```

**为什么不能提前删除**：
- 如果在阶段 1 就删除，旧实现会失效（依赖预取）
- 如果在阶段 3.1 前删除，测试时新旧切换会失败
- 只有在新实现测试通过后，才能安全删除

#### 3.3 验证 Collection 路由（已在阶段 1.2 完成）

**说明**：
- Collection 路由已在阶段 1.2 修改（`_COLLECTION_FOR_MODALITY` 指向 v2）
- 此阶段无需额外修改
- 验证：`client.collection_for("asr")` 返回 `asr_embeddings_v2`

### 阶段 4: 单元测试（1-2 天）

#### 4.1 Schema 测试

**文件**: `backend/tests/test_milvus_schema_v2.py`（新建）

测试：
- `test_asr_schema_v2()` - Schema 正确性
- `test_ocr_schema_v2()` - Schema 正确性
- `test_sparse_field()` - Sparse 字段类型
- `test_analyzer_config()` - 中文分词器配置

#### 4.2 混合检索测试

**文件**: `backend/tests/test_milvus_hybrid_search.py`（新建）

测试：
- `test_asr_hybrid_search()` - ASR 混合检索正常工作
- `test_ocr_hybrid_search()` - OCR 混合检索正常工作
- `test_lexical_priority()` - 词面优先（调高 sparse_weight）
- `test_semantic_priority()` - 语义优先（调高 dense_weight）
- `test_chinese_tokenization()` - 中文分词正确

#### 4.3 权重验证测试

**文件**: 同上

测试场景：
- 精确关键词匹配（预期：sparse 召回）
- 语义相关但词面不同（预期：dense 召回）
- 词面+语义都相关（预期：融合得分高）

### 阶段 5: 集成测试（2-3 天）

#### 5.1 端到端测试

在测试环境运行完整搜索：
- ASR 查询（关键词 + 语义）
- OCR 查询（关键词 + 语义）
- 跨模态融合
- VLM reranking

#### 5.2 性能测试

测试场景：
- 短视频（5分钟）
- 中等视频（1小时）
- 长视频（10小时）

测试指标：
- 延迟（P50/P95/P99）
- 召回率（与旧版对比）
- 网络传输量
- Milvus CPU/内存

#### 5.3 召回质量对比

**对比测试**:
- 新版（混合检索） vs 旧版（全量 query）
- Top-10/20/50 的 Jaccard 相似度
- 用户满意度（如果有测试集）

**目标**:
- Jaccard > 0.85（top-10）
- Jaccard > 0.80（top-20）

### 阶段 6: Speaker 模态适配验证（1 天）

#### 6.1 验证关联正确性

测试：
- Speaker ANN 检索正常
- `asr_chunk_idx` 指向的 ASR chunk 存在
- Speaker + ASR 联合查询正常

#### 6.2 集成测试

测试查询：
- "某某人说了什么"
- "谁说了XXX"
- 验证 Speaker 和 ASR 的结果一致性

### 阶段 7: 清理 Legacy 实现（1-2 天）

**前提条件**：
- ✅ v2 collections 已稳定运行
- ✅ 所有 ASR/OCR 检索切换到 hybrid_search
- ✅ 集成测试通过
- ✅ 性能测试达标

#### 7.1 删除旧 Collections

**文件**: `backend/scripts/cleanup_legacy_collections.py`（新建）

```python
def cleanup_legacy_collections():
    """删除旧的 ASR/OCR collections。"""
    client = MilvusClient()
    
    for collection_name in ["asr_embeddings", "ocr_embeddings"]:
        if client.has_collection(collection_name):
            logger.info(f"Dropping legacy collection: {collection_name}")
            client.drop_collection(collection_name)
            logger.info(f"Successfully dropped: {collection_name}")
```

**执行时机**：v2 稳定运行 1-2 周后

#### 7.2 删除 Legacy 函数

**删除列表**：

| 函数 | 文件 | 原因 |
|------|------|------|
| `lexical_score()` | `backend/app/search.py:78-88` | 已被 Milvus BM25 替代 |
| `_asr_candidates()` | `backend/app/search.py:480-553` | 已被 `milvus_asr_candidates_hybrid()` 替代 |
| `_asr_chunks_from_npz()` | `backend/app/search.py:556-571` | NPZ fallback 已废弃 |
| `_semantic_arrays()` | `backend/app/search.py:153-160` | NPZ fallback 已废弃 |

**保留列表**（如果有其他用途）：
| 函数 | 文件 | 原因 |
|------|------|------|
| `normalize_text()` | `backend/app/search.py` | 可能被其他模块使用（需全局搜索确认） |
| `_query_all()` | `backend/app/indexing/milvus_search.py` | 其他模态可能使用 |

#### 7.3 API 不兼容处理

**方式 1：抛出明确错误（推荐）**

对于可能被外部调用的函数，保留函数签名但抛出错误：

```python
# backend/app/search.py

def lexical_score(query: str, text: str) -> float:
    """DEPRECATED: Lexical scoring is now handled by Milvus BM25.
    
    Raises:
        NotImplementedError: This function is no longer supported.
            Use milvus_asr_candidates_hybrid() or milvus_ocr_candidates_hybrid() instead.
    """
    raise NotImplementedError(
        "lexical_score() has been removed in v2.0. "
        "ASR/OCR now use Milvus BM25 hybrid search. "
        "Please update your code to use milvus_{asr|ocr}_candidates_hybrid()."
    )

def _asr_candidates(...) -> list[Candidate]:
    """DEPRECATED: Use milvus_asr_candidates_hybrid() instead.
    
    Raises:
        NotImplementedError: This function is no longer supported.
    """
    raise NotImplementedError(
        "_asr_candidates() has been removed in v2.0. "
        "Use milvus_asr_candidates_hybrid() instead."
    )
```

**方式 2：完全删除（如果确认无外部调用）**

如果通过全局搜索确认这些函数仅在即将删除的代码路径中使用，可以直接删除。

**验证步骤**：
```bash
# 全局搜索函数调用
grep -r "lexical_score" backend/ --include="*.py"
grep -r "_asr_candidates" backend/ --include="*.py"
```

#### 7.4 更新调用路径

替换所有对 legacy 函数的调用：

```python
# ❌ 旧代码
from app.search import _asr_candidates
candidates = _asr_candidates(chunks, query_text, video_id, limit)

# ✅ 新代码
from app.indexing.milvus_search import milvus_asr_candidates_hybrid
candidates = milvus_asr_candidates_hybrid(
    client, video_id, query_text, query_embedding, limit, profile
)
```

**关键变化**：
- 不再需要 `chunks` 参数（全量数据）
- 需要提供 `query_embedding`（dense 向量）
- 新增 `profile` 参数（precision/balanced/recall）

#### 7.5 清理配置

**文件**: `backend/app/settings.py`

删除过时的配置项：
```python
# 删除（如果存在）：
# - has_embedding 相关配置
# - lexical_score 相关配置
# - 旧的 n-gram 配置
# - BULK_QUERY_FIELDS 中的 ASR/OCR 配置
```

**更新 BULK_QUERY_FIELDS**：
```python
# backend/app/indexing/milvus_search.py

BULK_QUERY_FIELDS: dict[str, list[str]] = {
    # ASR 和 OCR 已使用 hybrid_search，不再需要批量预取
    # 仅保留其他可能需要的模态（如果有）
}
```

#### 7.6 不兼容变更文档

**创建迁移指南**：`docs/MIGRATION_GUIDE_v2.md`

```markdown
# ASR/OCR v2.0 迁移指南

## 不兼容变更清单

### 删除的函数

| 函数 | 原位置 | 替代方案 |
|------|--------|---------|
| `lexical_score()` | `backend/app/search.py` | Milvus BM25（内部实现） |
| `_asr_candidates()` | `backend/app/search.py` | `milvus_asr_candidates_hybrid()` |
| `_query_all()` (ASR/OCR) | `backend/app/indexing/milvus_search.py` | `hybrid_search()` |

### 删除的 Collections

- `asr_embeddings` → `asr_embeddings_v2`
- `ocr_embeddings` → `ocr_embeddings_v2`

### 删除的 Schema 字段

- `has_embedding` 字段（所有行都有 dense embedding）
- 不再区分有/无 embedding 的行

### API 变更

**旧 API**：
```python
candidates = _asr_candidates(
    chunks=chunks,
    query_text=query_text,
    video_id=video_id,
    limit=limit,
    semantic_embeddings=embeddings,
    embedding_chunk_indices=indices,
    semantic_query=query_embedding,
)
```

**新 API**：
```python
candidates = milvus_asr_candidates_hybrid(
    client=client,
    video_id=video_id,
    query_text=query_text,
    query_embedding=query_embedding,
    limit=limit,
    profile="balanced",  # 新增：precision/balanced/recall
)
```

**关键区别**：
- ✅ 不需要预先读取 `chunks`
- ✅ 简化参数（无需 `semantic_embeddings`、`embedding_chunk_indices`）
- ✅ 新增 `profile` 参数控制精度
- ✅ 返回的 `Candidate` 对象 `features` 字段变化

### 迁移步骤

如果外部代码依赖旧 API：

1. **更新导入**：
   ```python
   # 旧
   from app.search import _asr_candidates
   
   # 新
   from app.indexing.milvus_search import milvus_asr_candidates_hybrid
   ```

2. **更新函数调用**：
   - 删除 `chunks` 参数
   - 简化 embedding 参数
   - 添加 `client` 和 `profile` 参数

3. **更新 Candidate 处理逻辑**：
   
   **Candidate 对象字段变化**：
   
   **旧实现** (Python 侧混合评分):
   ```python
   Candidate(
       video_id=video_id,
       start_time=...,
       end_time=...,
       score=combined_score,  # max(lexical, 0.65*semantic + 0.35*lexical)
       modality="asr",
       lexical_score=lexical,        # ⚠️ 字段存在
       semantic_score=semantic,      # ⚠️ 字段存在
       semantic_cosine=cosine,       # ⚠️ 字段存在
       features={
           "lexical_score": lexical,
           "semantic_score": semantic,
           "semantic_cosine": cosine,
       },
       ...
   )
   ```
   
   **新实现** (Milvus 服务端混合检索):
   ```python
   Candidate(
       video_id=video_id,
       start_time=...,
       end_time=...,
       score=hybrid_score,  # Milvus WeightedRanker 融合后的分数
       modality="asr",
       # ❌ 不设置 lexical_score 和 semantic_score（诚实）
       # 混合检索的融合是不可逆的，无法分离词面和语义分数
       features={
           "hybrid_score": hybrid_score,
           "source": "milvus_hybrid",
       },
       ...
   )
   ```
   
   **关键变化**：
   - ❌ **不再提供** `lexical_score` 和 `semantic_score` 字段
   - ✅ 仅提供 `hybrid_score`（词面70% + 语义30%）
   - ⚠️ 混合检索的融合是不可逆的，技术上无法分离
   - ✅ 下游代码需要修改以适配（见"⚠️ 关键实施问题与解决方案 - 问题1"）
   
   **代码适配**：
   ```python
   # ❌ 旧代码（会失效）
   if candidate.lexical_score > 0.8:  # lexical_score 不存在，返回 None
       ...
   
   # ✅ 新代码（使用 hybrid_score）
   hybrid_score = candidate.features.get("hybrid_score")
   if hybrid_score and hybrid_score > 0.8:
       ...
   
   # ✅ 下游函数修改（见问题1解决方案）
   def _asr_result_hybrid_score(result: SearchResult) -> float:
       """自动兼容 v1 (lexical_score) 和 v2 (hybrid_score)。"""
       return max(
           (
               float(item.get("features", {}).get("hybrid_score") or 
                     item.get("lexical_score") or 0.0)
               for item in result.evidence
               if item.get("modality") == "asr"
           ),
           default=0.0,
       )
   ```
   
   **注意事项**：
   - ❌ 不要伪造 `lexical_score` 字段（自欺欺人）
   - ✅ 修改下游代码以使用 `hybrid_score`
   - ✅ 前端需要调整展示逻辑（不再显示分离的分数）
   if candidate.lexical_score > 0.8:  # 现在读取的是估算值
       ...
   
   # ✅ 新代码：优先使用 features
   hybrid_score = candidate.features.get("hybrid_score")
   if hybrid_score and hybrid_score > 0.8:
       ...
   
   # ✅ 下游函数正常工作
   def _asr_result_lexical_score(result: SearchResult) -> float:
       return max(
           (
               float(item.get("lexical_score") or 0.0)  # ✅ 能读取到值
               for item in result.evidence
               if item.get("modality") == "asr"
           ),
           default=0.0,
       )
   ```
   
   **注意事项**：
   - ✅ 新实现保留兼容字段，下游代码无需修改
   - ⚠️ lexical_score / semantic_score 是估算值，不是真实分离的分数
   - ⚠️ 如果需要精确分离，需要修改为两次独立检索（不推荐）

### 测试更新

如果有单元测试依赖旧 API：
- 更新 mock 对象
- 更新断言（检查新的 features 字段）
- 删除对 `lexical_score()` 的测试
```

### 阶段 8: 文档更新（0.5-1 天）

更新文档：
- 新建 `docs/HYBRID_SEARCH.md` - 混合检索文档
- 更新 `docs/MILVUS_DEPLOYMENT_GUIDE.md` - 部署指南
- 更新 `README.md` - 配置说明
- 更新 API 文档（如果有）


---

## 📊 验证指标

### 性能指标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| ASR 延迟（1小时视频） | < 20ms（P95） | Profiler 记录 |
| ASR 延迟（10小时视频） | < 30ms（P95） | Profiler 记录 |
| OCR 延迟（1小时视频） | < 20ms（P95） | Profiler 记录 |
| OCR 延迟（10小时视频） | < 30ms（P95） | Profiler 记录 |
| 网络传输 | < 50 KB | Milvus RPC 监控 |
| Milvus CPU 利用率 | < 60% | 系统监控 |
| Milvus 内存占用 | < 10 GB（亿级数据） | 系统监控 |

### 准确性指标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| Top-10 Jaccard | > 0.85 | 与旧版对比测试 |
| Top-20 Jaccard | > 0.80 | 与旧版对比测试 |
| 词面精确匹配召回率 | 100% | 关键词测试集 |
| 语义相关召回率 | > 90% | 语义测试集 |

### 功能指标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| Speaker 关联 | 正常工作 | 集成测试 |
| 跨模态融合 | 正常工作 | 端到端测试 |
| 中文分词 | 正确工作 | 中文查询测试 |
| 权重调整 | 生效 | 参数调整测试 |

---

## 🚨 风险与缓解

### 风险 1: 索引重建时间长

**描述**: 1000 小时视频的索引重建可能需要 4-5 天

**缓解措施**:
- 并行处理多个视频
- 优先重建高频访问的视频
- 支持断点续传
- 新旧 collection 并存，灰度切换

**验证**: 重建脚本压力测试

### 风险 2: BM25 中文分词效果

**描述**: Milvus 的 jieba 分词可能不如预期

**缓解措施**:
- 对比测试 jieba 分词 vs 当前 n-gram
- 如果效果不佳，考虑自定义分词器
- 调整 BM25 参数（k1, b）
- 可选：使用 RRFRanker 替代 WeightedRanker

**验证**: 中文查询测试集（100+ queries）

### 风险 3: 词面权重不够优先

**描述**: sparse_weight = 0.7 可能不足以体现词面优先

**缓解措施**:
- 提供可配置的权重参数
- A/B 测试不同权重（0.7:0.3, 0.8:0.2, 0.9:0.1）
- 根据 query 类型动态调整权重
- 考虑使用 RRFRanker（无需手动调权重）

**验证**: 对比不同权重下的召回效果

### 风险 4: DiskANN 性能不达预期

**描述**: DiskANN 延迟可能高于预期（> 100ms）

**缓解措施**:
- 调整 `search_list` 参数（200 → 150）
- 优化磁盘 I/O（使用 SSD）
- 监控 Milvus 服务端性能
- 如果仍不达标，考虑混合策略（热数据 HNSW + 冷数据 DiskANN）

**验证**: 性能测试达标

### 风险 5: has_embedding=False 数据丢失

**描述**: 旧数据中 `has_embedding=False` 的行无法迁移

**缓解措施**:
- 统计旧数据中 `has_embedding=False` 的占比
- 如果占比 < 1%，可以接受丢失
- 如果占比 > 1%，重新提取 embedding（使用 embedding 模型）
- 或者为这些行生成零向量（保留但不参与语义检索）

**验证**: 数据统计 + 影响评估

### 风险 6: Speaker 模态受影响

**描述**: ASR 优化可能破坏 Speaker 关联

**缓解措施**:
- 保持 `segment_idx` 的稳定性（主键中包含）
- 专门的 Speaker 集成测试
- 回归测试覆盖 Speaker 功能

**验证**: Speaker 集成测试 100% 通过

---

## 🔧 技术细节

### DiskANN 参数说明

```python
# 索引构建参数
index_params = {
    "index_type": "DISKANN",
    "metric_type": "IP",
    "params": {
        "search_list": 200,  # 搜索时的候选列表大小
    },
}

# 搜索参数
search_params = {
    "metric_type": "IP",
    "params": {
        "search_list": 200,  # >= limit，越大越精确但越慢
    },
}
```

**参数调优**:
- `search_list`: 推荐值 100-300
- 太小（< 100）：召回率下降
- 太大（> 300）：延迟增加

### BM25 参数说明

Milvus BM25 使用标准参数：
- **k1 = 1.5**: 词频饱和参数（越大越重视高频词）
- **b = 0.75**: 文档长度归一化参数（越大越惩罚长文档）

**中文分词器**:
```python
analyzer_params = {
    "type": "chinese",  # 使用 jieba tokenizer
    # 内部配置:
    # - tokenizer: jieba
    # - filter: cnalphanumonly (保留中文、字母、数字)
}
```

**分词效果示例**:
- 输入：`"我要搜索红色汽车"`
- 分词：`["我", "要", "搜索", "红色", "汽车"]`
- BM25 向量：`{我: 0.5, 要: 0.3, 搜索: 0.8, 红色: 1.0, 汽车: 1.0}`

### Function Field 注意事项

**使用 Milvus Function Field 进行 BM25 转换时的关键要点**：

#### 1. Query 预处理策略

**推荐做法**：
- 传入 `hybrid_search()` 的 text 应该是**原始查询文本**
- 仅去除首尾空格：`query_text.strip()`
- **不要**使用 `normalize_text()` 等自定义预处理

**原因**：
- Milvus 会使用 collection 定义的 analyzer（`{"type": "chinese"}`）处理查询文本
- 如果客户端预处理过度（如去除标点、转小写、分词），会与 Milvus analyzer 不一致
- 导致检索结果不准确

**错误示例**：
```python
# ❌ 错误：过度预处理
normalized = normalize_text(query_text)  # 去除标点、转小写等
sparse_req = AnnSearchRequest(data=[normalized], ...)
```

**正确示例**：
```python
# ✅ 正确：仅去除首尾空格
sparse_req = AnnSearchRequest(data=[query_text.strip()], ...)
```

#### 2. 空查询处理

**问题**：如果 `query_text` 为空或仅包含空格，`hybrid_search()` 可能失败或返回无意义结果。

**解决方案**：回退到仅使用 dense 检索

```python
if not query_text or not query_text.strip():
    # 仅使用 dense 检索
    results = col.search(
        data=[dense_vector],
        anns_field="dense_embedding",
        ...
    )
else:
    # 正常的混合检索
    results = col.hybrid_search(...)
```

#### 3. Analyzer 配置说明

**当前配置**：
```python
analyzer_params={"type": "chinese"}
```

**内部行为**：
- **Tokenizer**: jieba（结巴分词）
- **Filter**: cnalphanumonly（仅保留中文、字母、数字）

**如果需要自定义分词**：
- 选项 A：使用 Milvus 的其他 analyzer（如 "standard"、"english"）
- 选项 B：放弃 Function Field，使用客户端预计算方式
- 选项 C：等待 Milvus 支持更多自定义 analyzer

#### 4. 索引写入注意事项

**Function Field 模式下**：
- 客户端**不需要**计算 `sparse_embedding`
- 仅需提供 `text` 字段
- Milvus 在插入时自动调用 BM25 function
- `sparse_embedding` 字段标记为 `is_function_output=True`

**错误示例**：
```python
# ❌ 错误：客户端计算 sparse_embedding（Function Field 不需要）
from pymilvus.model.sparse import BM25EmbeddingFunction
bm25_ef = BM25EmbeddingFunction()
sparse_emb = bm25_ef.encode_documents([text])[0]
rows.append({"text": text, "sparse_embedding": sparse_emb})  # 多余
```

**正确示例**：
```python
# ✅ 正确：仅提供 text，Milvus 自动计算 sparse_embedding
rows.append({"text": text, "dense_embedding": dense_emb})
# sparse_embedding 由 Function 自动生成
```

#### 5. 分词一致性保证

**优势**：Function Field 自动保证分词一致性
- 索引时：Milvus 使用 analyzer 分词并计算 BM25
- 查询时：Milvus 使用相同的 analyzer 分词
- 无需担心客户端和服务端分词不一致

**对比客户端预计算方式**：
- 需要确保写入和检索使用**同一个** `BM25EmbeddingFunction` 实例
- 或者确保配置参数完全一致
- Function Field 无此问题

### 混合检索融合策略

#### WeightedRanker（推荐）

```python
# 加权融合：dense * 0.3 + sparse * 0.7
rerank = WeightedRanker(0.3, 0.7)
```

**特点**:
- 简单直观
- 权重可调
- 适合"词面优先"场景

#### RRFRanker（备选）

```python
# Reciprocal Rank Fusion
rerank = RRFRanker(k=60)
```

**特点**:
- 无需手动调权重
- 对排序更鲁棒
- 适合两种检索效果相当的场景

**公式**:
```
RRF_score(doc) = sum(1 / (k + rank_i(doc)))
```

**选择建议**:
- 如果词面明确优先 → **WeightedRanker**
- 如果两者权重不确定 → **RRFRanker**

### 与 Visual 优化的对比

| 维度 | Visual | ASR/OCR (v2) |
|------|--------|--------------|
| 索引类型 | DiskANN | **DiskANN (dense) + BM25 (sparse)** |
| 检索方式 | 纯 ANN | **混合检索** |
| 评分位置 | Milvus 服务端 | **Milvus 服务端** ✅ |
| 后处理 | 段聚合 | **直接返回（无 Python 侧评分）** ✅ |
| Schema 变更 | 无 | **新增 sparse_embedding 字段** |
| 复杂度 | 中等 | **较高（混合检索）** |
| 适用规模 | 千万级帧 | **亿级 chunks/frames** ✅ |

**关键区别**:
- Visual 是纯语义检索，ASR/OCR 需要词面+语义混合
- ASR/OCR 使用 BM25 解决词面检索问题
- 两者都使用 DiskANN，都无 Python 侧瓶颈

---

## 📈 预期收益详解

### 1. 延迟优化（亿级数据）

**当前延迟**（10小时视频，~120,000 ASR chunks）:
```
_query_all():              2000-5000ms  (传输几十 MB)
lexical_score():           1000-3000ms  (扫描 12万 chunks)
semantic scoring:          100-200ms    (NumPy 计算)
混合评分 + 排序:            50-100ms
-------------------------------------------
总计:                      3150-8300ms  (3-8 秒！)
```

**优化后延迟**（DiskANN + BM25 混合检索）:
```
DiskANN search:            80-120ms     (磁盘索引)
BM25 search:               20-40ms      (倒排索引)
服务端融合:                 5-10ms
网络传输（20 results）:     5-10ms
-------------------------------------------
总计:                      110-180ms    (0.1-0.2 秒)
```

**提升**: **30-75倍** 延迟降低

**关键点**: 延迟与视频长度无关（仅与召回数量相关）

### 2. 网络传输优化

**当前传输量**（10小时视频）:
```
ASR:
  - 120,000 chunks × (text ~100 bytes + embedding 384×4 bytes)
  - ≈ 120,000 × 1,636 bytes = 196 MB

OCR:
  - 36,000 frames × (text ~50 bytes + embedding 384×4 bytes)
  - ≈ 36,000 × 1,586 bytes = 57 MB
```

**优化后传输量**（召回 20 candidates）:
```
ASR:
  - 20 chunks × (text ~100 bytes + metadata ~50 bytes)
  - ≈ 20 × 150 bytes = 3 KB
  - embedding 不传输（在 Milvus 服务端完成检索）

OCR:
  - 20 frames × (text ~50 bytes + metadata ~50 bytes)
  - ≈ 20 × 100 bytes = 2 KB
```

**提升**: 
- ASR: **65,000倍** 减少（196 MB → 3 KB）
- OCR: **28,500倍** 减少（57 MB → 2 KB）

### 3. 内存占用优化

**当前内存占用**（亿级数据，全部使用 HNSW）:
```
ASR (1200万 chunks):
  - 1200万 × 384 × 4 bytes = 18.4 GB
  - HNSW 图结构: ~2-3倍 = 36.8-55.2 GB

OCR (3.6亿 frames):
  - 3.6亿 × 384 × 4 bytes = 552 GB
  - HNSW 图结构: ~2-3倍 = 1.1-1.6 TB (!)
```

**优化后内存占用**（DiskANN）:
```
ASR (1200万 chunks):
  - DiskANN 磁盘存储，内存仅缓存热数据
  - 预估内存: ~2-3 GB

OCR (3.6亿 frames):
  - DiskANN 磁盘存储
  - 预估内存: ~50-80 GB
```

**提升**: **90%+** 内存降低

### 4. 可扩展性提升

| 视频长度 | 当前延迟 | 优化后延迟 | 可行性 |
|---------|---------|-----------|--------|
| 1小时 | 300-800ms | **10-20ms** | ✅ |
| 10小时 | 3-8秒 | **10-20ms** | ✅ |
| 100小时 | 30-80秒 | **10-20ms** | ✅ |
| 1000小时 | 5-13分钟 | **10-20ms** | ✅ |

**关键点**: 延迟与视频长度解耦，支持任意长度视频。

### 5. 词面检索优化

**当前词面检索**（Python n-gram）:
```python
# O(N) 扫描所有 chunks
for chunk in chunks:  # 12万次循环（10小时视频）
    score = lexical_score(query, chunk["text"])
```

**复杂度**: O(N × M)，N = chunks 数量，M = 文本长度

**优化后词面检索**（Milvus BM25）:
```
BM25 倒排索引:
  - O(log N) 查找关键词位置
  - O(K) 计算 top-K 分数
```

**复杂度**: O(log N + K)

**提升**: **1000倍+** 词面检索速度

---

## ⚠️ 关键实施问题与解决方案

本章节解决实施端反馈的 4 个关键问题。

### 问题 1: Candidate 字段兼容性 🔴 高优先级

**问题描述**：

下游代码依赖 `Candidate` 对象的 `lexical_score` 和 `semantic_score` 字段：

```python
# backend/app/search.py L913-921
def _asr_result_lexical_score(result: SearchResult) -> float:
    return max(
        (
            float(item.get("lexical_score") or 0.0)  # ⚠️ 依赖此字段
            for item in result.evidence
            if item.get("modality") == "asr"
        ),
        default=0.0,
    )

# backend/app/search.py L924-980
def _reserve_asr_lexical_results(results: list[SearchResult], limit: int):
    """为强词面匹配保留结果槽位。"""
    lexical = sorted(
        (
            result
            for result in above
            if _asr_result_lexical_score(result) >= _ASR_LEXICAL_RESERVE_MIN_SCORE
        ),
        key=lambda result: (_asr_result_lexical_score(result), result.score),
        reverse=True,
    )
    # ... 交错排列逻辑 ...

# backend/app/search.py L1952-1953
if set(modalities) == {"asr"} and text:
    results = _reserve_asr_lexical_results(results, limit)  # 仅纯ASR搜索时启用
```

**核心功能分析**：

`_reserve_asr_lexical_results()` 的作用是：**在纯ASR搜索时，为强词面匹配的结果保留槽位，避免被语义相关但词面不匹配的结果挤掉**。

这是一个**重排序逻辑**：
1. 从 `above_threshold` 结果中筛选出 `lexical_score >= 0.8` 的强词面匹配
2. 将这些结果与常规排序结果交错排列（1个词面 + 8个常规 循环）
3. 确保强词面匹配不会被埋没

**影响分析**：

- 🔴 新混合检索返回 `hybrid_score`（已包含70%词面权重），**无法分离出 lexical_score**
- 🔴 Milvus 混合检索的融合是不可逆的，即使使用 RRFRanker 也只能得到融合排名
- 🔴 `_asr_result_lexical_score()` 会返回 0.0（所有结果）
- 🔴 词面匹配保留机制失效
- 🔴 强词面匹配的结果可能被排除

**根本原因**：

Milvus 混合检索的 `hybrid_score` 是融合后的单一分数，**技术上无法分离出词面分数**。

---

**解决方案（推荐）**：

✅ **方案 A：修改下游逻辑，使用 hybrid_score 替代 lexical_score**

**核心思想**：

既然 hybrid_score 已经包含 70% 的词面权重，那么**高 hybrid_score 的结果本身就偏向词面匹配**。保留逻辑仍然有效，只是判断标准从 `lexical_score` 改为 `hybrid_score`。

**代码修改**：

```python
# backend/app/search.py

def _asr_result_hybrid_score(result: SearchResult) -> float:
    """获取 ASR 结果的混合分数（词面70% + 语义30%）。
    
    对于 v2 混合检索，hybrid_score 已经融合了词面和语义。
    对于旧实现（v1），回退到 lexical_score（向后兼容）。
    """
    return max(
        (
            # 优先使用 hybrid_score（v2 混合检索）
            float(item.get("features", {}).get("hybrid_score") or 
                  # 回退到 lexical_score（v1 实现）
                  item.get("lexical_score") or 0.0)
            for item in result.evidence
            if item.get("modality") == "asr"
        ),
        default=0.0,
    )


def _reserve_asr_lexical_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Reserve sparse result slots for strong lexical hits without rewriting confidence scores.
    
    注意：
    - v1 实现：使用 lexical_score（纯词面分数）
    - v2 实现：使用 hybrid_score（包含70%词面权重）
    
    由于混合检索本身已经词面优先，这个保留机制仍然有效。
    """
    above = [result for result in results if result.above_threshold]
    below = [result for result in results if not result.above_threshold]
    pool_size = max(_ASR_LEXICAL_RESERVE_POOL_SIZE, limit)
    primary = above[:pool_size]
    
    # 使用新的 _asr_result_hybrid_score()（自动兼容 v1 和 v2）
    lexical = sorted(
        (
            result
            for result in above
            if _asr_result_hybrid_score(result) >= _ASR_LEXICAL_RESERVE_MIN_SCORE
        ),
        key=lambda result: (_asr_result_hybrid_score(result), result.score),
        reverse=True,
    )[:pool_size]
    
    if not lexical:
        return results
    
    # 后续交错排列逻辑不变
    reranked: list[SearchResult] = []
    emitted: set[int] = set()
    primary_position = 0
    lexical_position = 0
    
    def emit(result: SearchResult) -> bool:
        identity = id(result)
        if identity in emitted:
            return False
        emitted.add(identity)
        reranked.append(result)
        return True
    
    # 前8个结果直接取 primary
    while (
        primary_position < len(primary)
        and primary_position < _ASR_LEXICAL_RESERVE_INITIAL_PRIMARY
    ):
        emit(primary[primary_position])
        primary_position += 1
    
    # 之后交错排列：1个词面 + 8个常规
    while primary_position < len(primary) or lexical_position < len(lexical):
        while lexical_position < len(lexical):
            candidate = lexical[lexical_position]
            lexical_position += 1
            if emit(candidate):
                break
        
        taken = 0
        while (
            primary_position < len(primary)
            and taken < _ASR_LEXICAL_RESERVE_PRIMARY_RUN
        ):
            candidate = primary[primary_position]
            primary_position += 1
            if emit(candidate):
                taken += 1
    
    remaining_above = [result for result in above if id(result) not in emitted]
    return reranked + remaining_above + below
```

**Candidate 构造**（milvus_asr_candidates_hybrid）：

```python
# backend/app/indexing/milvus_search.py

def milvus_asr_candidates_hybrid(...):
    # ... hybrid_search() ...
    
    for hit in results[0]:
        hybrid_score = float(hit.score)
        text = str(hit.entity.get("text") or "")
        
        candidates.append(Candidate(
            ...,
            score=hybrid_score,
            # ❌ 不设置 lexical_score 和 semantic_score（诚实）
            # ✅ 仅在 features 中提供 hybrid_score
            features={
                "hybrid_score": hybrid_score,
                "source": "milvus_hybrid",
            },
        ))
    
    return candidates
```

**优点**：
- ✅ 诚实：不伪造 lexical_score，不自欺欺人
- ✅ 合理：hybrid_score 已包含 70% 词面权重，仍能区分强词面匹配
- ✅ 保留功能：保留机制继续发挥作用，不改变用户体验
- ✅ 兼容性：自动适配 v1（lexical_score）和 v2（hybrid_score）实现

**缺点与权衡**：
- ⚠️ hybrid_score 不是纯词面分数，包含 30% 语义成分
- ⚠️ 阈值可能需要调整（建议从 0.8 调至 0.85）
- ⚠️ 需要修改 1 个函数 + 新增 1 个函数（低风险）

**阈值调整建议**：

由于 hybrid_score 不是纯词面分数，`_ASR_LEXICAL_RESERVE_MIN_SCORE` 可能需要调整：

```python
# backend/app/search.py L909

# v1: lexical_score >= 0.8 认为是强词面匹配
# v2: hybrid_score >= ??? 认为是强匹配
#
# 建议：
# - 保守策略：0.85（更严格，减少误判）
# - 激进策略：0.75（更宽松，保留更多）
# - 默认策略：0.8（不变）
#
# 通过 A/B 测试确定最优阈值
_ASR_LEXICAL_RESERVE_MIN_SCORE = 0.8  # 可根据实际效果调整
```

---

**方案 B：禁用词面保留机制（不推荐）**

既然混合检索已经词面优先（sparse_weight=0.7），可以认为不再需要额外的保留机制：

```python
# backend/app/search.py L1952-1953

# 检测是否使用 v2 混合检索
def _uses_hybrid_search(results: list[SearchResult]) -> bool:
    """检测结果是否来自 v2 混合检索。"""
    return any(
        item.get("features", {}).get("source") == "milvus_hybrid"
        for result in results
        for item in result.evidence
        if item.get("modality") == "asr"
    )

if set(modalities) == {"asr"} and text:
    if not _uses_hybrid_search(results):
        # 仅对 v1 实现启用保留机制
        results = _reserve_asr_lexical_results(results, limit)
    # v2 混合检索本身已经词面优先，不需要额外保留
```

**优点**：
- ✅ 简单：最少的代码修改
- ✅ 诚实：不伪造字段

**缺点**：
- ❌ 失去保留机制：即使混合检索词面优先，仍可能有强词面匹配被排除
- ❌ 行为变化：v1 和 v2 的排序逻辑不一致，可能影响用户体验
- ❌ 风险高：无法验证混合检索是否真的足够词面优先

---

**推荐**: **方案 A**（修改下游逻辑，使用 hybrid_score）

**理由**：
1. ✅ 诚实：不伪造 lexical_score，直接使用 hybrid_score
2. ✅ 合理：hybrid_score 已包含 70% 词面权重，仍能区分强词面匹配
3. ✅ 保留功能：保留机制继续发挥作用，用户体验一致
4. ✅ 兼容性：自动适配 v1 和 v2 实现
5. ✅ 可调试：通过 A/B 测试确定最优阈值

**实施工作量**：
- 新增 `_asr_result_hybrid_score()` 函数：0.2天
- 修改 `_reserve_asr_lexical_results()` 调用：0.1天
- 阈值调优（A/B 测试）：0.5天
- **总计**：0.8天

---

**前端展示说明**：

由于 v2 实现不再提供 `lexical_score` 和 `semantic_score`，前端需要调整：

```typescript
// 前端代码

// ❌ v1：展示分离的分数
interface Candidate {
  lexical_score: number;   // 词面分数
  semantic_score: number;  // 语义分数
}

// ✅ v2：仅展示混合分数
interface Candidate {
  score: number;  // 混合分数（词面70% + 语义30%）
  features: {
    hybrid_score: number;
    source: "milvus_hybrid";
  };
}
```

如果前端需要展示"词面匹配度"，建议：
- 方案1：移除该展示（最诚实）
- 方案2：展示 `hybrid_score` 并标注"混合分数（偏词面）"
- 方案3：不展示具体数值，仅展示标签（如"强匹配"/"弱匹配"）

**不要**伪造 lexical_score 传给前端，这是自欺欺人。
### 问题 2: BULK_QUERY_FIELDS 清理时序 🟡 中优先级

**问题描述**：

方案在阶段 3.2 要求删除 `BULK_QUERY_FIELDS` 中的 ASR/OCR，但没有明确**何时**删除。如果过早删除，旧实现会失效。

**正确时序**：

```
1. ✅ 创建 v2 collections
2. ✅ 修改 _COLLECTION_FOR_MODALITY 指向 v2
3. ✅ 修改 milvus_asr_candidates() 调用混合检索
4. ✅ 测试新实现通过
5. ← 此时才能删除 BULK_QUERY_FIELDS
```

**解决方案**：

在阶段 3.2 中明确前提条件（见阶段 3.2 详细说明）。

---

### 问题 3: 权重配置硬编码 🟢 低优先级

**问题描述**：

如果在代码中硬编码权重（如 `WeightedRanker(0.3, 0.7)`），则无法动态调整。

**解决方案（已采用）**：

✅ 方案已经通过 `_get_fusion_weights()` 从配置读取权重：

```python
def _get_fusion_weights(profile: str) -> dict:
    settings = get_settings()
    return {
        "dense": settings.asr_dense_weight,    # 从配置读取
        "sparse": settings.asr_sparse_weight,  # 从配置读取
    }

# 使用时
weights = _get_fusion_weights(profile)
rerank = WeightedRanker(weights["dense"], weights["sparse"])  # ✅ 正确

# ❌ 错误：不要硬编码
rerank = WeightedRanker(0.3, 0.7)  # 不要这样做
```

**实施要求**：

- ✅ 确保所有调用 `WeightedRanker()` 的地方都使用 `_get_fusion_weights()`
- ✅ 在代码 review 中检查是否有硬编码权重

---

### 问题 4: 错误处理策略 🟡 中优先级

**问题描述**：

方案关注正常流程，对异常处理不足。混合检索可能遇到的错误场景：

1. `hybrid_search()` API 失败
2. BM25 分词失败（乱码文本）
3. DiskANN 索引损坏
4. 网络超时
5. Collection 不存在

**解决方案**：

#### 4.1 自定义异常类

参考 Visual 模态的 `MilvusVisualSearchError`，定义：

```python
# backend/app/indexing/milvus_search.py 或 milvus_client.py

class MilvusHybridSearchError(RuntimeError):
    """Milvus 混合检索失败（不包括空结果）。
    
    用于包装 hybrid_search() 抛出的底层异常，
    提供统一的错误处理接口。
    """
    pass
```

#### 4.2 错误处理模式

在 `milvus_asr_candidates_hybrid()` 中：

```python
def milvus_asr_candidates_hybrid(...) -> list[Candidate]:
    """ASR 混合检索：DiskANN (语义) + BM25 (词面)。"""
    
    try:
        col = client.collection_for("asr")
    except Exception as exc:
        logger.error(
            "Failed to get ASR collection: video_id=%s error=%s",
            video_id, exc
        )
        raise MilvusHybridSearchError(
            f"Failed to get ASR collection for video {video_id}"
        ) from exc
    
    # 向量归一化
    try:
        query_norm = normalize(np.asarray(query_embedding, dtype=np.float32))
    except Exception as exc:
        logger.error(
            "Failed to normalize query embedding: video_id=%s error=%s",
            video_id, exc
        )
        raise MilvusHybridSearchError(
            f"Invalid query embedding for video {video_id}"
        ) from exc
    
    # 参数准备
    search_params = _get_search_params(profile)
    weights = _get_fusion_weights(profile)
    
    # 执行混合检索
    try:
        if not query_text or not query_text.strip():
            # 空查询：仅使用 dense 检索
            logger.warning(
                "Empty query_text for ASR, falling back to dense-only: video_id=%s",
                video_id
            )
            results = col.search(...)
        else:
            # 正常混合检索
            dense_req = AnnSearchRequest(...)
            sparse_req = AnnSearchRequest(...)
            
            results = col.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=WeightedRanker(weights["dense"], weights["sparse"]),
                limit=limit,
                output_fields=["segment_idx", "start_ms", "end_ms", "text"],
            )
    
    except Exception as exc:
        logger.error(
            "ASR hybrid search failed: video_id=%s query_text=%s error=%s",
            video_id, query_text[:50] if query_text else "<empty>", exc
        )
        raise MilvusHybridSearchError(
            f"ASR hybrid search failed for video {video_id}"
        ) from exc
    
    # 转换为 Candidate 对象
    candidates: list[Candidate] = []
    try:
        for hit in results[0]:
            # ... 构造 Candidate ...
            pass
            
    except Exception as exc:
        logger.error(
            "Failed to parse ASR search results: video_id=%s error=%s",
            video_id, exc
        )
        raise MilvusHybridSearchError(
            f"Failed to parse ASR results for video {video_id}"
        ) from exc
    
    return candidates
```

#### 4.3 上层调用处理

在上层调用代码中（如 `app/search.py`）：

```python
try:
    asr_candidates = milvus_asr_candidates_hybrid(
        client, video_id, query_text, query_embedding, limit
    )
except MilvusHybridSearchError as exc:
    # 记录错误，返回空结果（不阻塞整个搜索流程）
    logger.error("ASR hybrid search failed, skipping: %s", exc)
    asr_candidates = []
```

#### 4.4 错误分类与处理策略

| 错误类型 | 示例 | 处理策略 |
|---------|------|---------|
| 配置错误 | Collection 不存在 | 抛出异常，阻塞请求 |
| 数据错误 | 向量维度不匹配 | 抛出异常，阻塞请求 |
| 网络/超时 | Milvus 连接失败 | 记录日志，返回空结果 |
| 分词错误 | 乱码文本 | 回退到 dense-only |
| 空结果 | 查询无匹配 | 正常返回空列表（不是错误） |

---

## 🎯 成功标准

### 必须达成（P0）

- ✅ **延迟降低 > 90%**（10小时视频，P95 < 30ms）
- ✅ **支持亿级数据**（3.6亿 OCR frames 可正常检索）
- ✅ **词面优先生效**（sparse_weight > dense_weight）
- ✅ **Top-10 Jaccard > 0.85**（与旧版对比）
- ✅ **Speaker 模态正常**（关联无断裂）
- ✅ **所有测试通过**（单元 + 集成）

### 期望达成（P1）

- ✅ **延迟降低 > 95%**（10小时视频，P95 < 20ms）
- ✅ **Top-20 Jaccard > 0.80**
- ✅ **中文分词正确**（jieba 分词效果验证）
- ✅ **内存占用 < 100 GB**（亿级数据）
- ✅ **无功能回归**

### 可选达成（P2）

- ✅ **支持 100小时+ 超长视频**
- ✅ **权重动态调整**（根据 query 类型）
- ✅ **完整清理 legacy 代码**
- ✅ **监控 dashboard 完善**

---

## 📝 待确认问题

### Q1: has_embedding=False 数据占比

**问题**: 旧数据中 `has_embedding=False` 的行占比？

**影响**: 如果占比高，需要重新提取 embedding。

**验证方法**:
```python
# 统计脚本
for video_id in sample_videos:
    asr_rows = query_all("asr", video_id, ["has_embedding"])
    false_count = sum(1 for r in asr_rows if not r.get("has_embedding", True))
    ratio = false_count / len(asr_rows) if asr_rows else 0
    print(f"{video_id}: {ratio:.2%}")
```

### Q2: Milvus 版本确认

**问题**: 当前 Milvus 版本是否支持 BM25？

**要求**: Milvus >= 2.4.0

**验证方法**:
```bash
# 检查 Milvus 版本
docker exec milvus-standalone /milvus --version
# 或
curl http://localhost:9091/healthz
```

### Q3: 中文分词效果验证

**问题**: jieba 分词是否满足需求？

**验证方法**:
- 准备 100+ 中文测试查询
- 对比 jieba 分词 vs 当前 n-gram
- 召回率和准确率对比

### Q4: 权重初始值

**问题**: `sparse_weight = 0.7, dense_weight = 0.3` 是否合适？

**验证方法**:
- A/B 测试不同权重（0.6:0.4, 0.7:0.3, 0.8:0.2, 0.9:0.1）
- 根据召回效果选择最优权重

### Q5: 索引重建资源需求

**问题**: 索引重建需要多少计算资源？

**预估**:
- CPU: 8-16 核
- 内存: 32-64 GB
- 磁盘: SSD，至少 500 GB
- 时间: 4-5 天（1000 小时视频）

**验证**: 小规模测试（100 小时）评估速度

---

## 🔄 后续优化方向（P2）

### 方向 1: 动态权重调整

根据 query 类型动态调整 dense/sparse 权重：

```python
def adaptive_weights(query_text: str) -> dict:
    """根据 query 特征动态调整权重。"""
    # 关键词查询（短、精确）→ 提高词面权重
    if len(query_text) < 10 and is_keyword(query_text):
        return {"dense": 0.2, "sparse": 0.8}
    
    # 语义查询（长、描述性）→ 提高语义权重
    elif len(query_text) > 50:
        return {"dense": 0.5, "sparse": 0.5}
    
    # 默认：词面优先
    else:
        return {"dense": 0.3, "sparse": 0.7}
```

### 方向 2: 自定义分词器

如果 jieba 分词效果不佳，考虑自定义：

```python
# 使用更精准的中文 NLP 工具
from pkuseg import pkuseg
seg = pkuseg()

# 或集成到 Milvus analyzer
# 需要 Milvus 插件机制支持
```

### 方向 3: 学习型 Reranker

在混合检索后，使用学习模型精排：

```python
# Milvus 召回 top-100
candidates = hybrid_search(..., limit=100)

# 学习型 reranker 精排到 top-20
final_results = reranker_model.rerank(query, candidates, limit=20)
```

**优势**:
- 更精准的排序
- 可以融合更多特征（用户画像、历史点击等）

### 方向 4: 冷热数据分层

热数据（高频访问）使用 HNSW，冷数据使用 DiskANN：

```python
# 热数据 collection (HNSW, 快速)
hot_collection = client.collection_for("asr_hot")

# 冷数据 collection (DiskANN, 低成本)
cold_collection = client.collection_for("asr_cold")

# 根据访问频率路由
if video_id in hot_videos:
    search(hot_collection)
else:
    search(cold_collection)
```

---

## ✅ 检查清单

### 实施前

- [ ] 确认 Milvus 版本 >= 2.4.0
- [ ] 确认 `has_embedding=False` 数据占比
- [ ] 评审本方案（团队 review）
- [ ] 准备测试环境（独立 Milvus 实例）
- [ ] 准备测试数据集（100+ queries）

### Schema 迁移阶段

- [ ] 实现 `create_asr_schema_v2()`
- [ ] 实现 `create_ocr_schema_v2()`
- [ ] 创建 v2 collections
- [ ] 验证 schema 正确性
- [ ] 验证中文 analyzer 配置

### 索引重建阶段

- [ ] 实现索引写入逻辑（含 BM25）
- [ ] 实现重建脚本（断点续传）
- [ ] 小规模测试（10 小时）
- [ ] 全量重建（1000 小时）
- [ ] 验证数据完整性

### 检索实现阶段

- [ ] 实现 `milvus_asr_candidates_hybrid()`
- [ ] 实现 `milvus_ocr_candidates_hybrid()`
- [ ] 实现权重配置
- [ ] 修改调用路径
- [ ] 单元测试通过

### 测试阶段

- [ ] 单元测试通过（100%）
- [ ] 集成测试通过
- [ ] 性能测试达标
- [ ] Speaker 适配验证通过
- [ ] 对比测试 Jaccard > 0.85

### 上线阶段

- [ ] 灰度发布（10% → 50% → 100%）
- [ ] 监控指标正常
- [ ] 用户反馈收集
- [ ] 清理 legacy 代码
- [ ] 删除旧 collections
- [ ] 更新文档

---

## 📚 参考资料

### 项目文档

- `docs/VISUAL_ANN_SEARCH.md` - Visual 模态优化经验
- `docs/FINAL_VERIFICATION_REPORT.md` - Visual 优化验证报告
- `docs/MILVUS_OPTIMIZATION_PLAN.md` - 初步优化计划

### 代码文件

- `backend/app/indexing/milvus_search.py` - 当前检索实现
- `backend/app/indexing/milvus_search_visual_v2.py` - Visual ANN 参考
- `backend/app/search.py` - 旧的混合评分逻辑（待废弃）
- `backend/app/indexing/milvus_schema.py` - Schema 定义

### Milvus 官方文档

- **DiskANN Index**: https://milvus.io/docs/disk_index.md
- **Hybrid Search**: https://milvus.io/docs/multi-vector-search.md
- **BM25 / Text Match**: https://milvus.io/docs/keyword-match.md
- **Sparse Vector**: https://milvus.io/docs/sparse_vector.md
- **Analyzer**: https://milvus.io/docs/analyzer-overview.md

### 学术资料

- **DiskANN 论文**: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (NeurIPS 2019)
- **BM25 算法**: https://en.wikipedia.org/wiki/Okapi_BM25
- **Hybrid Retrieval**: Dense-Sparse Hybrid Search 相关论文

---

## 📌 总结

### 核心变化

1. **索引**: HNSW → **DiskANN (dense) + BM25 (sparse)**
2. **检索**: 全量 query → **Milvus 混合检索**
3. **评分**: Python 侧 → **Milvus 服务端融合**
4. **词面**: n-gram → **BM25 倒排索引**
5. **规模**: 百万级 → **支持亿级+**

### 预期效果

- ⚡ **延迟**: 3-8秒 → 10-20ms（10小时视频）
- 📉 **传输**: 50-200 MB → 2-3 KB
- 💾 **内存**: 1+ TB → 50-80 GB（亿级数据）
- 🔍 **词面**: O(N) Python → O(log N) Milvus BM25
- 📈 **扩展**: 支持 10万小时视频（亿级 embeddings）

### 实施建议

1. **优先级 P0**: 完成核心功能（混合检索 + DiskANN）
2. **优先级 P1**: 性能优化（权重调优 + 参数调优）
3. **优先级 P2**: 高级功能（动态权重 + 自定义分词）

### 风险控制

- ✅ 新旧 collection 并存，灰度切换
- ✅ 保留 fallback 机制（如果 v2 失败）
- ✅ 充分测试（单元 + 集成 + 性能）
- ✅ 监控指标（延迟 + 召回率 + 错误率）

---

**文档版本**: 2.2  
**创建日期**: 2026-07-29  
**最后更新**: 2026-07-30  
**作者**: Claude Code  
**审核状态**: ✅ 已通过最终审查并修订  
**预计实施周期**: 10-14 天

---

## 📝 版本更新日志

### v2.2 (2026-07-30) - 最终审查后修订

**修订原因**: 根据 `OCR_ASR_OPTIMIZATION_PLAN_FINAL_REVIEW.md` 的最终审查和对实际代码的全面审查

**主要修订内容**:

1. **修正 Collection 路由逻辑** (FINAL_REVIEW 问题 1)
   - ✅ 补充 `_COLLECTION_FOR_MODALITY` 修改说明（阶段 1.2）
   - ✅ 明确需要修改映射指向 v2 collections
   - ✅ 删除错误的 fallback 逻辑描述

2. **补充完整导入列表** (FINAL_REVIEW 问题 2)
   - ✅ 新增"导入依赖"章节（L288-305）
   - ✅ 明确所有必需的 import 语句
   - ✅ 说明 `normalize()` 和 `_seconds()` 的来源

3. **调整函数签名保持兼容性** (FINAL_REVIEW 问题 3)
   - ✅ 保留 `rows` 参数（标记为 DEPRECATED）
   - ✅ 添加 `profile` 参数（带默认值 "balanced"）
   - ✅ 添加兼容性警告逻辑

4. **补充辅助函数完整实现** (全面审查问题 7)
   - ✅ 新增 `_get_search_params()` 完整实现（L307-330）
   - ✅ 新增 `_get_fusion_weights()` 完整实现（L333-348）
   - ✅ 说明参数配置和返回值

5. **明确 BULK_QUERY_FIELDS 清理** (全面审查问题 6)
   - ✅ 在阶段 3.2 添加详细清理说明
   - ✅ 参考 Visual 模态经验
   - ✅ 说明删除原因和影响

6. **补充调用方修改说明** (全面审查问题 5)
   - ✅ 在阶段 3.2 添加 search.py 修改步骤
   - ✅ 删除 `rows` 参数传递
   - ✅ 可选添加 `profile` 参数

7. **补充 Candidate 字段变更详细说明** (全面审查问题 8)
   - ✅ 在阶段 7.6 详细对比新旧字段（L1163-1210）
   - ✅ 说明 lexical_score / semantic_score → hybrid_score 变化
   - ✅ 提供代码适配示例
   - ✅ 建议保留兼容字段

**影响范围**:
- Collection 路由修改说明更准确
- 导入依赖完整清晰
- 函数签名保持向后兼容
- 辅助函数可直接使用
- BULK_QUERY_FIELDS 清理步骤明确
- Candidate 字段变更有详细迁移指南

**修订统计**:
- 新增章节: 2 个（导入依赖、辅助函数实现）
- 修订章节: 5 个（阶段 1.2、3.2、3.3、7.6、版本日志）
- 新增代码示例: ~100 行
- 总行数: 1787 行 → ~1950 行

**向后兼容性**: 保持（通过保留 `rows` 参数和兼容性处理）

### v2.1 (2026-07-30) - 审核后修订

**修订原因**: 根据 `OCR_ASR_OPTIMIZATION_PLAN_REVIEW.md` 的审核意见进行修订

**主要修订内容**:

1. **统一 Schema 设计为 Function Field 方式** (问题 1)
   - ✅ 在 `create_asr_schema_v2()` 中添加 `Function` 定义
   - ✅ 标记 `sparse_embedding` 为 `is_function_output=True`
   - ✅ 同样修订 `create_ocr_schema_v2()`
   - ✅ 删除索引写入代码中的客户端 BM25 计算
   - ✅ 明确使用 Milvus 2.6 的 Function Field 特性

2. **完善检索代码的边界情况处理** (问题 2)
   - ✅ 添加空 query_text 处理逻辑（回退到 dense-only）
   - ✅ 明确 query 预处理策略（仅 strip，不使用 normalize_text）
   - ✅ 新增"Function Field 注意事项"章节，详细说明使用要点

3. **明确 Legacy 清理策略** (问题 3)
   - ✅ 细化"阶段 7"为 7.1-7.6 详细步骤
   - ✅ 添加 API 不兼容处理策略（抛出明确错误 vs 完全删除）
   - ✅ 提供不兼容变更文档模板（迁移指南）
   - ✅ 明确删除顺序和验证步骤

4. **补充索引重建策略** (细节 1)
   - ✅ 明确开发阶段采用全量重建（删除旧 collection）
   - ✅ 提供重建脚本示例
   - ✅ 说明无需考虑数据迁移（Function Field 简化流程）

5. **技术细节完善**
   - ✅ 新增"Function Field 注意事项"章节
   - ✅ 说明 query 预处理、空查询处理、analyzer 配置
   - ✅ 对比 Function Field vs 客户端预计算
   - ✅ 强调分词一致性保证

**影响范围**:
- Schema 定义更清晰（增加 Function 定义）
- 索引写入更简单（删除客户端 BM25 计算）
- 检索代码更健壮（边界情况处理）
- 清理策略更明确（详细步骤 + 迁移指南）

**向后兼容性**: 无（v2.0 → v2.1 为同一版本的修订，未发布）

### v2.0 (2026-07-29) - 重大修订

**修订原因**: 根据数据规模需求（10万小时视频，亿级 embeddings）和"词面优先"的检索需求，重新设计方案

**主要变化** (相对 v1.0):
- ✅ 索引类型：HNSW → **DiskANN** (支持亿级数据)
- ✅ 检索方式：ANN + Python 评分 → **Milvus 混合检索**
- ✅ 词面检索：Python n-gram → **Milvus BM25**
- ✅ 权重策略：语义优先 → **词面优先** (sparse 0.7 > dense 0.3)
- ✅ 评分位置：Python 侧 → **Milvus 服务端**

**预期收益**:
- 延迟降低 30-75 倍
- 网络传输减少 100-500 倍
- 内存占用降低 90%+
- 支持 10万小时视频（亿级数据）

