# ASR模态优化实施计划

**制定日期**: 2026-08-03  
**版本**: v1.2（二轮代码审核修订）  
**状态**: 待实施

> **v1.2 修订说明**：本版基于对 `backend/` 的第二轮逐行核对，修正/补齐了 v1.1 的以下问题：
> 1. **纠正错误理由**：v1.1 以"OCR `_ocr_display_text` 仍依赖"为由保留 `lexical_score()`，此理由不成立——`_ocr_display_text()` 的唯一调用者就是待删的 `_asr_candidates()`，OCR hybrid 路径根本不经过它。删除 `_asr_candidates()` 后 `lexical_score()` / `_ocr_display_text()` 等会成组变孤儿（见 2.4 节）。
> 2. **补齐易漏挂钩**：新增 1.3.1 节，明确必须在 `_init_collections`（`milvus_client.py:281` 一带）显式调用 `_validate_existing_asr_collection`，否则校验形同虚设。
> 3. **补齐静态配置**：新增 1.3.2 节，说明 `_COLLECTION_CONFIGS["asr_embeddings"]` 改用 `indexes` 字典、以及 `_STATIC_INDEX_TYPES["asr"]` 由 `HNSW` 改 `DISKANN`（对齐 OCR）。
> 4. **显式声明权衡**：删除 ASR 的 NPZ fallback 意味着 ASR 与 OCR 一样彻底失去 Milvus 故障兜底（见 2.4 节新增说明）。
>
> **v1.1 修订说明**（保留）：基于对 `backend/` 真实代码（尤其 OCR 现网实现）的逐行核对，修正了 v1.0 中的以下实质错误：
> 1. 去掉硬编码 `hybrid_score` 阈值（0.8/0.6），改用 OCR 同款**全局动态阈值**。
> 2. 舍弃 ASR-only 词面保底逻辑 `_reserve_asr_lexical_results`（hybrid BM25 已在服务端完成词面召回，Python 侧保底冗余）。
> 3. 索引写入路径更正为真实的 `AsrMilvusIndexer.upsert_from_memory()`（`milvus_indexer.py`），而非虚构的 `_save_asr_to_milvus()`。
> 4. 新增 ASR 全局阈值处理段（对称 OCR）。
> 5. Schema **原地替换** `create_asr_schema`（不使用 `_v2` 后缀函数），与 OCR 做法一致。
> 6. 删除灰度百分比放量方案，**直接原地替换、清理 legacy**（无回滚开关，代码库无流量分流机制）。
> 7. 新增 settings `@field_validator`（对称 OCR）。
> 8. 修正验证脚本预期输出（主键为 `pk`，含 `model_version`）、`BULK_QUERY_FIELDS` 前后对比、NPZ 遗留清理清单。

---

## 📋 执行摘要

基于Visual模态和OCR模态的成功优化经验，本计划旨在将ASR模态从全量query优化为DiskANN + BM25混合检索架构。

### 核心目标

1. **解决扩展性瓶颈**: 当前ASR使用全量query，随视频数和长度线性增长
2. **引入DiskANN**: 支持亿级向量规模，降低内存占用90%+
3. **实现混合检索**: Dense (语义) + Sparse (BM25词面) 服务端融合
4. **保持Speaker模态兼容**: ASR和Speaker共享时间轴，需确保优化不破坏Speaker功能

### 预期收益

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 延迟（1小时视频） | 500-1000ms | **< 50ms** | **20倍** |
| 网络传输 | 20-50MB | **< 10KB** | **5000倍** |
| 内存（亿级） | 300+GB | **< 80GB** | **70%降低** |
| 词面检索 | Python扫描 | **BM25倒排** | **100倍** |

---

## 🎯 当前架构分析

### 1. ASR模态现状

#### Schema结构
```python
# backend/app/vector_store/milvus/milvus_schema.py:138-151
def create_asr_schema() -> CollectionSchema:
    # _common_fields("asr") 提供 pk / video_id / asset_version / model_version
    fields = _common_fields("asr") + [
        FieldSchema("segment_idx",   DataType.INT64),                     # chunk序号
        FieldSchema("start_ms",      DataType.INT64),                     # 开始时间
        FieldSchema("end_ms",        DataType.INT64),                     # 结束时间
        FieldSchema("text",          DataType.VARCHAR, max_length=_TEXT_LEN),  # 文本内容（_TEXT_LEN=5000）
        FieldSchema("has_embedding", DataType.BOOL, default_value=True),  # 是否有语义向量
        FieldSchema("embedding",     DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["asr"]),  # 语义向量（384维）
    ]
    return CollectionSchema(fields, description="ASR chunk embeddings ...")
```

> ⚠️ **v1.0 勘误**：主键字段是 **`pk`**（由 `_common_fields` 生成，格式 `{video_id}#{asset_ver}#{model_ver}#asr#{segment_id}`），并非 `id`；且 `_common_fields` 还带 `model_version` 字段。`text` 上限为 `_TEXT_LEN=5000`，非 2000。

**关键特点**:
- 所有chunk都存储在Milvus（包括lexical-only的chunk）
- `has_embedding=False`标记词面专用chunk（embedding为零向量占位符）
- `has_embedding=True`标记有真实语义向量的chunk
- 按`segment_idx`排序，保持时间序列

#### 索引配置
```python
# backend/app/vector_store/milvus/milvus_client.py:58-62
"asr_embeddings": {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 16, "efConstruction": 200},
}
```

**问题**:
- 仅使用HNSW索引，无DiskANN（不支持大规模数据）
- 无BM25稀疏向量索引（词面检索在Python侧）
- 无文本analyzer配置（无倒排索引）

#### 检索实现
```python
# backend/app/vector_store/milvus/milvus_search.py:412-481
def milvus_asr_candidates(client, video_id, query_text, query_embedding, limit, ...):
    # 全量query - 拉取该视频所有ASR chunk
    rows = _query_all(
        client, "asr", video_id,
        ["segment_idx", "start_ms", "end_ms", "text", "has_embedding", "embedding"],
        profiler,
    )
    
    # 重建chunks列表（所有chunk，用于词面打分）
    chunks = [{"chunk_id": ..., "start_ms": ..., "end_ms": ..., "text": ...}]
    
    # 提取语义向量（仅has_embedding=True的chunk）
    semantic_embeddings = [rows[i]["embedding"] for i in semantic_local_indices]
    
    # 调用_asr_candidates进行Python侧打分
    return _asr_candidates(chunks, query_text, video_id, limit, ...)
```

**瓶颈**:
1. **全量query不可扩展**: 1小时视频约3600个chunk，需传输所有文本和embedding
2. **Python侧词面检索**: `lexical_score()`对所有chunk进行n-gram匹配，O(N)复杂度
3. **无倒排索引**: 无法快速定位包含关键词的chunk
4. **内存占用高**: HNSW索引全部在内存中

#### 打分逻辑
```python
# backend/app/retrieval/search.py:480-553
def _asr_candidates(chunks, query_text, video_id, limit, ...):
    # 词面打分：对所有chunk计算n-gram相似度
    lexical_scores = [lexical_score(query_text, chunk["text"]) for chunk in chunks]
    
    # 语义打分：仅对has_embedding=True的chunk计算余弦相似度
    semantic_scores = compute_semantic_scores(semantic_embeddings, semantic_query)
    
    # 混合打分：max(lexical, 0.65*semantic + 0.35*lexical)
    combined_scores = np.maximum(lexical_scores, 0.65 * semantic_scores + 0.35 * lexical_scores)
    
    # 过滤：保留词面命中或语义top-K
    candidate_indices = [i for i in argsort(combined_scores) 
                        if lexical_scores[i] > 0 or i in semantic_top_indices][:limit]
```

**特点**:
- **双路召回**: 词面命中 OR 语义top-K
- **融合权重**: 语义65% + 词面35%（当两者都有时）
- **决策逻辑**: 根据lexical/semantic分数判断为strong/semantic_hit/weak

---

### 2. Speaker模态现状

#### Schema结构
```python
# backend/app/vector_store/milvus/milvus_schema.py:214-223
def create_speaker_schema() -> CollectionSchema:
    fields = [
        FieldSchema("utterance_idx",  DataType.INT64),    # 发言段索引
        FieldSchema("start_ms",       DataType.INT64),    # 开始时间
        FieldSchema("end_ms",         DataType.INT64),    # 结束时间
        FieldSchema("asr_chunk_idx",  DataType.INT64),    # 关联的ASR chunk索引
        FieldSchema("track_id",       DataType.INT64),    # 说话人track ID
        FieldSchema("embedding",      DataType.FLOAT_VECTOR, dim=192),  # 声纹向量
    ]
```

**关键字段**:
- `asr_chunk_idx`: **关联ASR chunk的索引**，用于跨模态关联
- `track_id`: 说话人track ID
- `utterance_idx`: 发言段在视频中的序号

#### 索引配置
```python
"speaker_embeddings": {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}
```

#### 检索实现
```python
# backend/app/vector_store/milvus/milvus_search.py:736-803
def milvus_speaker_candidates(client, video_id, query, limit, threshold=0.50, ...):
    query_norm = normalize(query)
    ann_limit = min(limit * 2, 16_384)
    
    # ANN搜索（HNSW COSINE）
    hits = _ann_search(
        client, "speaker", video_id, query_norm.tolist(),
        ann_limit,
        ["utterance_idx", "start_ms", "end_ms", "track_id", "asr_chunk_idx", "embedding"],
        profiler,
    )
    
    # 重打分：使用精确余弦距离
    for hit in hits:
        if hit.get("embedding") is None:
            cosine = hit["_distance"]  # COSINE metric直接返回余弦值
        else:
            utt_vec = normalize(hit["embedding"])
            cosine = np.dot(query_norm, utt_vec)  # 重新计算精确值
    
    # 阈值过滤：threshold=0.50
    candidates = [c for c in scored if c.cosine >= threshold][:limit]
```

**特点**:
- 使用ANN + 重打分策略（类似Face模态）
- 阈值=0.50（CAM++模型校准值）
- **依赖asr_chunk_idx字段进行跨模态关联**

---

### 3. ASR与Speaker的关系

#### 数据关联
```
ASR Chunk (segment_idx=5)
  ├─ text: "今天天气不错"
  ├─ start_ms: 10000, end_ms: 12000
  └─ embedding: [384维语义向量]

Speaker Utterance (utterance_idx=2)
  ├─ asr_chunk_idx: 5  ← 关联到ASR chunk
  ├─ track_id: 1
  ├─ start_ms: 10000, end_ms: 12000
  └─ embedding: [192维声纹向量]
```

**关键依赖**:
1. **时间对齐**: Speaker的时间范围来自ASR分割结果
2. **索引关联**: `asr_chunk_idx`字段关联到ASR的`segment_idx`
3. **数据一致性**: ASR chunk删除/重建会影响Speaker数据的有效性

#### 潜在风险
如果ASR优化改变了：
- ✅ **Schema字段名**: `segment_idx` → 需保持或映射
- ✅ **Chunk切分逻辑**: 重建后`segment_idx`可能不连续
- ✅ **时间戳精度**: Speaker依赖ASR的时间边界
- ❌ **检索逻辑**: 不影响Speaker，因为Speaker是独立ANN检索

---

## 🚀 优化方案设计

### 方案概述

借鉴OCR模态的成功经验，将ASR升级为**DiskANN + BM25混合检索**架构。

### 核心技术栈

1. **Dense向量**: DiskANN索引（语义检索，磁盘存储）
2. **Sparse向量**: Milvus BM25 Function（词面检索，服务端计算）
3. **混合检索**: WeightedRanker融合（Milvus服务端）
4. **Analyzer**: `chinese`（内置中文分词器，支持中英文）

---

### 1. Schema升级

> **⚠️ 关键约定（对齐 OCR 现网做法）**：**原地替换** `create_asr_schema()`，**不**新增 `create_asr_schema_v2()` 后缀函数。OCR 的 `create_ocr_schema()` 就是原地升级为 hybrid 版本（`milvus_schema.py:154-200`），`milvus_client.py` 的 import 与 `_COLLECTION_CONFIGS` 均引用同名函数。若新增 `_v2` 而不改 import，新 schema 不会生效。

#### 新Schema设计
```python
# backend/app/vector_store/milvus/milvus_schema.py
# 原地替换现有 create_asr_schema()（保持函数名，与 OCR 一致）
def create_asr_schema() -> CollectionSchema:
    """ASR: DiskANN + BM25 hybrid search."""
    from pymilvus import Function, FunctionType

    # _common_fields("asr") 提供 pk / video_id / asset_version / model_version
    fields = _common_fields("asr") + [
        # 元数据字段（保持不变）
        FieldSchema("segment_idx",   DataType.INT64),      # 保留：Speaker依赖此字段
        FieldSchema("start_ms",      DataType.INT64),
        FieldSchema("end_ms",        DataType.INT64),

        # 文本字段 - 启用analyzer支持BM25（_TEXT_LEN=5000，与现有一致）
        FieldSchema("text", DataType.VARCHAR, max_length=_TEXT_LEN,
                    enable_analyzer=True,
                    analyzer_params={"type": "chinese"}),  # 中文分词器

        # 标记行类型（保持不变）
        FieldSchema("has_embedding", DataType.BOOL, default_value=True),

        # Dense语义向量（384维，取自 EMBEDDING_DIMS["asr"]）
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["asr"]),

        # Sparse BM25向量（Milvus自动计算）
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True),
    ]

    # BM25 Function定义
    bm25_function = Function(
        name="bm25_asr",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
        # 注意：Milvus 2.6不接受params参数
    )

    return CollectionSchema(
        fields,
        description="ASR: DiskANN + BM25 hybrid search",
        functions=[bm25_function]
    )
```

**关键变更**:
- ✅ **原地替换** `create_asr_schema`（不用 `_v2` 后缀），与 OCR 一致
- ✅ 保留`segment_idx`字段 - Speaker模态依赖
- ✅ 保留`has_embedding`字段 - 兼容词面专用chunk
- ✅ `text` 用 `_TEXT_LEN`(5000) 而非 2000；`embedding` 用 `EMBEDDING_DIMS["asr"]`(384)
- ✅ 启用`text`字段的analyzer - 支持BM25
- ➕ 新增`sparse_embedding`字段 - BM25向量
- ➕ 新增`bm25_asr` Function - 自动计算BM25向量

#### 向后兼容性
- `segment_idx`保持不变 → Speaker的`asr_chunk_idx`引用不受影响
- 时间戳字段不变 → 时间对齐逻辑不受影响
- 只要重建时保持chunk切分逻辑，`segment_idx`值保持稳定

---

### 2. 索引配置

```python
# backend/app/vector_store/milvus/milvus_client.py
# 从单 index 改为 indexes 字典（对齐 ocr_embeddings 的写法）
"asr_embeddings": {
    "schema": create_asr_schema,   # 原地替换后的同名函数
    "indexes": {
        # Dense索引：DiskANN（替换HNSW）
        "embedding": {
            "index_type": "DISKANN",
            "metric_type": "IP",
            "params": {
                "max_degree": 56,
                "search_list_size": 128,
                "pq_code_budget_gb": 0.125,
                "build_dram_budget_gb": 32.0,
            },
        },
        # Sparse索引：BM25倒排索引
        "sparse_embedding": {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",  # 必须使用BM25
            "params": {"drop_ratio_build": 0.2},
        },
    },
},
```

**关键点**:
- DiskANN替换HNSW：支持亿级数据，内存占用降低90%
- `metric_type="BM25"`：sparse索引必须使用BM25（不是IP）
- `drop_ratio_build=0.2`：构建时丢弃20%低频词，减少索引大小

---

### 3. 混合检索实现

```python
# backend/app/vector_store/milvus/milvus_search.py
def milvus_asr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray | None,
    limit: int,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """ASR hybrid search: DiskANN (semantic) + BM25 (lexical).
    
    Uses Milvus Function Field for server-side BM25 computation and WeightedRanker
    for result fusion. Semantic-first strategy (dense_weight > sparse_weight).
    
    When query_embedding is None, falls back to BM25-only search.
    """
    from pymilvus import AnnSearchRequest, WeightedRanker
    
    settings = get_settings()
    col = client.collection_for("asr")
    
    # 配置参数
    recall_size = settings.asr_hybrid_recall_size  # 默认200
    semantic_weight = settings.asr_semantic_weight  # 默认0.65
    lexical_weight = 1.0 - semantic_weight          # 默认0.35
    search_list = settings.asr_diskann_search_list  # 默认200
    
    # 处理None embedding - BM25-only fallback
    if query_embedding is None:
        if not query_text or not query_text.strip():
            return []
        
        with (profiler.span("milvus_rpc", "asr_bm25_only") if profiler else nullcontext()):
            results = col.search(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=f'video_id == "{video_id}"',
                output_fields=["segment_idx", "start_ms", "end_ms", 
                              "text", "has_embedding"],
            )
    else:
        # 归一化query embedding
        query_norm = normalize(np.asarray(query_embedding, dtype=np.float32))
        
        # 空查询文本：dense-only fallback
        if not query_text or not query_text.strip():
            with (profiler.span("milvus_rpc", "asr_dense_only") if profiler else nullcontext()):
                results = col.search(
                    data=[query_norm.tolist()],
                    anns_field="embedding",
                    param={"metric_type": "IP", "params": {"search_list": search_list}},
                    limit=limit,
                    expr=f'video_id == "{video_id}" AND has_embedding == True',
                    output_fields=["segment_idx", "start_ms", "end_ms", 
                                  "text", "has_embedding"],
                )
        else:
            # 混合检索：Dense + Sparse
            with (profiler.span("milvus_rpc", "asr_hybrid") if profiler else nullcontext()):
                # Dense检索请求
                dense_req = AnnSearchRequest(
                    data=[query_norm.tolist()],
                    anns_field="embedding",
                    param={"metric_type": "IP", "params": {"search_list": search_list}},
                    limit=recall_size,
                    expr=f'video_id == "{video_id}" AND has_embedding == True',
                )
                
                # Sparse检索请求
                sparse_req = AnnSearchRequest(
                    data=[query_text.strip()],
                    anns_field="sparse_embedding",
                    param={"metric_type": "BM25"},
                    limit=recall_size,
                    expr=f'video_id == "{video_id}"',
                )
                
                # 混合检索 + 加权融合
                results = col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=WeightedRanker(semantic_weight, lexical_weight),
                    limit=limit,
                    output_fields=["segment_idx", "start_ms", "end_ms", 
                                  "text", "has_embedding"],
                )
    
    # 转换为Candidate对象（不在此处硬编码阈值！）
    # ⚠️ 关键：hybrid_search + WeightedRanker 的融合分是 dense IP 分（约 [-1,1]）
    #    与 BM25 分（无界、可 >1）的加权和，不落在 [0,1]，因此 0.8/0.6 之类固定阈值
    #    没有意义。照搬 OCR 现网做法（milvus_search.py:606-612）：此处一律先置
    #    above_threshold=True，真正的判定交由 search.py 的全局动态阈值统一完成。
    candidates = []
    for hit in results[0]:
        hybrid_score = float(hit.score)
        text = str(hit.entity.get("text") or "")

        start_ms = int(hit.entity.get("start_ms") or 0)
        end_ms = int(hit.entity.get("end_ms") or 0)
        segment_idx = int(hit.entity.get("segment_idx") or 0)

        # above_threshold 初始置 True，稍后在 search.py 用全局动态阈值统一更新；
        # "· 低于阈值" 后缀也在全局阈值计算后再补。
        evidence = f"[asr_hybrid] {text[:100]} · hybrid={hybrid_score:.3f}"

        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=hybrid_score,
            modality="asr",
            evidence=evidence,
            raw_score=hybrid_score,
            above_threshold=True,
            best_time=_seconds(start_ms),
            unit_type="chunk",
            unit_id=segment_idx,  # 保留segment_idx作为unit_id
            best_ms=start_ms,
            text=text,
            features={
                "hybrid_score": hybrid_score,
                "source": "milvus_hybrid",
                "has_embedding": bool(hit.entity.get("has_embedding", True)),
            },
        ))

    return candidates
```

**关键特性**:
1. **三路fallback**: hybrid → dense-only → bm25-only
2. **服务端融合**: WeightedRanker在Milvus内部计算，不传输中间结果
3. **语义优先**: 默认权重65%语义 + 35%词面
4. **保留segment_idx**: 作为unit_id返回，保持与Speaker的兼容性
5. **无硬编码阈值**: `above_threshold=True` 占位，全局动态阈值在 `search.py` 统一判定（见下节）
6. **不再设置 `decision` 字段**: 混合分数无法映射到旧的 strong/semantic_hit 语义；`_fuse_candidate_groups`（`search.py:1069-1079`）在缺失匹配时回退为 `"hit"`，无需在此伪造。

---

### 3.1 全局动态阈值（对称 OCR，必须新增）

OCR 在收集完所有视频候选后，对 OCR 候选统一施加**全局动态阈值**（`search.py:1970-1980`）：

```python
# backend/app/retrieval/search.py —— 现有 OCR 段
ocr_candidates = [c for c in candidates if c.modality == "ocr"]
if ocr_candidates:
    global_top_score = max(float(c.score) for c in ocr_candidates)
    global_threshold = max(0.10, global_top_score * 0.3)
    for candidate in ocr_candidates:
        candidate.above_threshold = float(candidate.score) >= global_threshold
        if not candidate.above_threshold and " · 低于阈值" not in candidate.evidence:
            candidate.evidence += " · 低于阈值"
```

**ASR 需新增对称的一段**（紧随 OCR 段之后）：

```python
# backend/app/retrieval/search.py —— 新增 ASR 段
asr_candidates = [c for c in candidates if c.modality == "asr"]
if asr_candidates:
    global_top_score = max(float(c.score) for c in asr_candidates)
    global_threshold = max(0.10, global_top_score * 0.3)  # 可按分数分布调参
    for candidate in asr_candidates:
        candidate.above_threshold = float(candidate.score) >= global_threshold
        if not candidate.above_threshold and " · 低于阈值" not in candidate.evidence:
            candidate.evidence += " · 低于阈值"
```

**关键点**:
- 阈值基于**本次查询所有 ASR 候选的最高分**动态计算，天然适配 BM25 无界分数
- `above_threshold` 只影响展示/排序层，不影响 `_fuse_candidate_groups` 的分组
- 阈值系数（`0.3`）后续可依据实际分数分布调优，或提取到配置

---

### 4. 环境变量配置

```python
# backend/app/core/settings.py
class Settings(BaseSettings):
    # ASR混合检索配置（默认值对齐 OCR 现网：recall/search_list=100）
    asr_hybrid_recall_size: int = 100      # 每个子搜索召回数（recommended: 50-200）
    asr_semantic_weight: float = 0.65      # 语义权重；词面权重 = 1.0 - this（recommended: 0.55-0.75）
    asr_diskann_search_list: int = 100     # DiskANN 查询期 search_list（非构建期；recommended: 100-200）

    # ⚠️ 新增对称 OCR 的 field_validator（OCR 见 settings.py:211-222）
    @field_validator("asr_hybrid_recall_size", "asr_diskann_search_list")
    @classmethod
    def validate_asr_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ASR hybrid search parameters must be greater than 0")
        return value

    @field_validator("asr_semantic_weight")
    @classmethod
    def validate_asr_semantic_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("asr_semantic_weight must be between 0.0 and 1.0")
        return value
```

```bash
# .env 文件
ASR_HYBRID_RECALL_SIZE=100
ASR_SEMANTIC_WEIGHT=0.65
ASR_DISKANN_SEARCH_LIST=100
```

> **v1.0 勘误**：OCR 现网默认 `ocr_hybrid_recall_size=100`、`ocr_diskann_search_list=100`（`settings.py:173-175`），并非 200。ASR 默认值与之对齐；如需更高召回可按“参数调优”章节上调。`field_validator` 是 OCR 已有的模式，必须一并添加，否则非法配置无法 fail-fast。

**权重策略**:
- ASR: 语义65% + 词面35%（语义优先）
- OCR: 词面70% + 语义30%（词面优先，`ocr_lexical_weight=0.7`）

**原因**:
- ASR文本通常较长，语义信息丰富
- OCR文本较短（单帧），词面匹配更可靠

> **命名说明**：OCR 存的是 `ocr_lexical_weight`（词面优先），ASR 存 `asr_semantic_weight`（语义优先），两者是互补关系（`weight + (1-weight)`），语义不冲突。混合检索里 `WeightedRanker(semantic_weight, lexical_weight)` 的参数顺序须与 `reqs=[dense_req, sparse_req]` 一致。

---

### 4.1 舍弃 ASR-only 词面保底逻辑 `_reserve_asr_lexical_results`

**现状**（`search.py:2000-2001`）：当 `set(modalities) == {"asr"}` 且有 `text` 时，会调用 `_reserve_asr_lexical_results` 为“强词面命中”预留结果槽位。该函数依赖每个候选 evidence 里的 `lexical_score` 字段（`_asr_result_lexical_score` 读 `item.get("lexical_score")`，`search.py:967-975`）。

**为什么它在旧架构里存在**：旧的 Python 侧 `_asr_candidates` 把语义和词面**融合成单一 `combined_scores` 后排序**，纯词面命中容易被语义分数淹没，于是需要一个后处理步骤把强词面命中重新提到前面。

**为什么 hybrid 架构下不再需要**：
- BM25 词面召回已作为独立通道（`sparse_req`）参与 `hybrid_search`，并通过 `WeightedRanker` 的 `lexical_weight=0.35` 在**服务端**直接贡献到融合分。强词面命中天然获得高 BM25 分、自然排到前面，无需 Python 侧二次预留。
- hybrid 候选**不再填充** `Candidate.lexical_score`（默认 `None` → 序列化为 0.0），`_reserve_asr_lexical_results` 会对所有候选取到 0.0 而静默失效——继续保留它只是“看似生效实则死代码”。

**决定：直接删除，不保留。** 具体：
- 删除 `search.py:2000-2001` 的 `if set(modalities) == {"asr"} and text: results = _reserve_asr_lexical_results(...)` 调用。
- 删除函数 `_reserve_asr_lexical_results`（`search.py:978-1034`）、`_asr_result_lexical_score`（`search.py:967-975`）及常量 `_ASR_LEXICAL_RESERVE_*`（`search.py:961-964`）。
- 词面优先级完全交由 BM25 + `WeightedRanker` 在服务端保证；若上线后发现纯关键词查询召回不足，优先下调 `asr_semantic_weight`（提高词面权重），而非重新引入 Python 侧保底。

---

### 5. 索引生成逻辑（写入路径更正）

> **v1.0 勘误（重要）**：v1.0 让你去改 `backend/app/indexing/modalities/asr/asr.py` 的 `_save_asr_to_milvus()` / `_parse_funasr_chunks()` 并 `import BM25EmbeddingFunction`——**这些函数在代码库中并不存在**。ASR 真实写入路径是 `backend/app/vector_store/milvus/milvus_indexer.py` 的 `AsrMilvusIndexer.upsert_from_memory()`（第 254-311 行）。

#### 真实写入实现（现状，几乎无需改动）
```python
# backend/app/vector_store/milvus/milvus_indexer.py  (AsrMilvusIndexer.upsert_from_memory)
rows = []
for chunk_idx in range(n_chunks):
    emb     = chunk_to_embedding.get(chunk_idx)
    has_emb = emb is not None
    row = {
        "pk":            asr_pk(ctx.video_id, ctx.asset_version, chunk_idx, model_ver),  # 主键是 pk，不是 id
        "video_id":      ctx.video_id,
        "asset_version": ctx.asset_version,
        "model_version": model_ver,
        "segment_idx":   chunk_idx,                       # 枚举序，Speaker 依赖此稳定性
        "start_ms":      int(times[chunk_idx, 0]),
        "end_ms":        int(times[chunk_idx, 1]),
        "text":          texts[chunk_idx][:5000] if chunk_idx < len(texts) else "",
        "embedding":     emb.tolist() if has_emb else zero_vec,
        # 注意：绝不手工写 sparse_embedding —— 由 BM25 Function 在 insert 时自动生成
    }
    if write_has_embedding:                               # 通过 schema_fields 探测字段存在性
        row["has_embedding"] = has_emb
    rows.append(row)
return _upsert_batched(col, rows, "asr")
```

**关键结论**:
- ✅ 该方法**已经**满足 hybrid 需求：不写 `sparse_embedding`（BM25 Function 自动生成）、已带 `text`、`segment_idx`、`has_embedding`、零向量占位。
- ✅ **写入侧代码基本无需改动**——升级 schema + 索引配置后，重建即自动生成 `sparse_embedding`。
- ⚠️ 与 OCR 的 `OcrMilvusIndexer.upsert_from_memory()`（`milvus_indexer.py:339-418`）写法完全一致，可直接对照。
- 唯一需确认：`col.schema.fields` 探测逻辑在新 schema 下会正确识别 `has_embedding` 存在（已具备）。

---

## 📊 Speaker模态影响评估

### 1. Schema兼容性

#### ASR字段变更
| 字段 | v1 (当前) | v2 (优化后) | Speaker影响 |
|------|-----------|-------------|------------|
| `segment_idx` | INT64 | INT64 | ✅ 无影响 |
| `start_ms` | INT64 | INT64 | ✅ 无影响 |
| `end_ms` | INT64 | INT64 | ✅ 无影响 |
| `text` | VARCHAR | VARCHAR+analyzer | ✅ 无影响 |
| `embedding` | FLOAT_VECTOR | FLOAT_VECTOR | ✅ 无影响 |
| `has_embedding` | BOOL | BOOL | ✅ 无影响 |
| - | - | **sparse_embedding** | ✅ 新增字段，Speaker不使用 |

**结论**: Schema变更对Speaker完全透明。

### 2. 数据一致性

#### Chunk索引稳定性
```python
# ASR重建时需保证segment_idx稳定
# 实际位置：AsrMilvusIndexer.upsert_from_memory() (milvus_indexer.py:294-310)
for chunk_idx in range(n_chunks):
    row = {
        "segment_idx": chunk_idx,  # 枚举序，从0开始 → 与 chunk 切分顺序一一对应
        ...
    }
```

**风险点**:
- 如果ASR重新索引，`segment_idx`值保持稳定（基于枚举顺序）
- Speaker的`asr_chunk_idx`引用依然有效

**缓解措施**:
- 保持ASR chunk切分逻辑不变
- 重建ASR索引时，同时检查Speaker数据的有效性
- 提供迁移工具：验证`asr_chunk_idx`引用的有效性

### 3. 时间戳对齐

#### ASR时间戳来源
```python
# FunASR返回的时间戳
chunk = {
    "start_ms": 10000,  # FunASR VAD分割结果
    "end_ms": 12000,
}
```

#### Speaker时间戳来源
```python
# Speaker diarization结果
utterance = {
    "start_ms": 10000,  # 与ASR chunk对齐
    "end_ms": 12000,
    "asr_chunk_idx": 5,  # 引用ASR chunk
}
```

**关键依赖**:
- Speaker的时间范围基于ASR的chunk边界
- ASR优化不改变chunk切分逻辑 → 时间戳不变 → Speaker不受影响

**结论**: 时间戳对齐逻辑不受影响。

### 4. 检索逻辑独立性

#### ASR检索流程
```
用户查询 → ASR混合检索 → 返回相关chunk
```

#### Speaker检索流程
```
声纹查询 → Speaker ANN检索 → 返回相关utterance
```

**关键点**:
- ASR和Speaker是**独立的检索通道**
- Speaker不调用ASR的检索接口
- ASR优化不影响Speaker的ANN检索逻辑

**结论**: 检索逻辑完全独立，互不影响。

---

### 5. 跨模态关联场景

#### 场景1: 根据ASR结果查找说话人
```python
# 1. 用户搜索"天气"，命中ASR chunk_id=5
asr_candidate = Candidate(unit_id=5, text="今天天气不错")

# 2. 查询该chunk对应的说话人
speaker_utterances = query_speaker_by_asr_chunk(video_id, asr_chunk_idx=5)
# → 返回：track_id=1, start_ms=10000, end_ms=12000
```

**影响分析**:
- ✅ `unit_id`（即`segment_idx`）保持不变
- ✅ 跨模态查询逻辑不需要修改

#### 场景2: 根据说话人查找ASR文本
```python
# 1. 用户搜索声纹，命中Speaker utterance
speaker_candidate = Candidate(
    modality="speaker",
    features={"asr_chunk_idx": 5}
)

# 2. 查询该utterance对应的ASR文本
asr_chunk = query_asr_by_segment_idx(video_id, segment_idx=5)
# → 返回：text="今天天气不错"
```

**影响分析**:
- ✅ `asr_chunk_idx`引用的`segment_idx`保持不变
- ✅ 跨模态查询逻辑不需要修改

---

### 总结：Speaker模态影响

| 维度 | 影响程度 | 说明 |
|------|---------|------|
| Schema兼容性 | ✅ 无影响 | 关键字段保持不变 |
| 数据一致性 | ⚠️ 需注意 | 重建时保持segment_idx稳定性 |
| 时间戳对齐 | ✅ 无影响 | Chunk切分逻辑不变 |
| 检索逻辑 | ✅ 无影响 | 独立检索通道 |
| 跨模态关联 | ✅ 无影响 | segment_idx引用保持有效 |

**结论**: ASR优化对Speaker模态**几乎无影响**，只需在重建时保证segment_idx的稳定性。

---

## 🛠️ 实施步骤

### 阶段1: Schema和索引准备（第1天）

#### 1.1 更新Schema定义
```bash
# 修改文件：backend/app/vector_store/milvus/milvus_schema.py
# 【原地替换】create_asr_schema()（保持函数名，与 OCR 的 create_ocr_schema() 一致）
# 不新增 _v1/_v2 后缀函数
```

**关键代码**:
- 用 `_common_fields("asr")` 作为基础字段（pk / video_id / asset_version / model_version）
- 添加`text`字段的analyzer配置（`_TEXT_LEN=5000`）
- 添加`sparse_embedding`字段（`is_function_output=True`）
- 添加`bm25_asr` Function定义
- `embedding` 使用 `EMBEDDING_DIMS["asr"]`（384）

#### 1.2 更新索引配置
```bash
# 修改文件：backend/app/vector_store/milvus/milvus_client.py
# 更新_COLLECTION_CONFIGS["asr_embeddings"]
```

**关键代码**:
- 将单一`index`改为`indexes`字典
- 配置`embedding`的DiskANN索引
- 配置`sparse_embedding`的BM25索引

#### 1.3 Schema校验函数
```python
# backend/app/vector_store/milvus/milvus_client.py
_ASR_V2_REQUIRED_FIELDS = frozenset({
    "text",
    "embedding",
    "sparse_embedding",
    "has_embedding",
})
_ASR_V2_REQUIRED_INDEX_FIELDS = frozenset({"embedding", "sparse_embedding"})

def _validate_existing_asr_collection(col: Collection) -> None:
    """Fail fast when an existing ASR collection predates hybrid search."""
    schema = col.schema
    fields = {field.name for field in schema.fields}
    function_names = {
        function.name
        for function in (getattr(schema, "functions", None) or [])
    }
    index_fields = {
        index.field_name
        for index in col.indexes
        if getattr(index, "field_name", None)
    }
    
    missing_fields = sorted(_ASR_V2_REQUIRED_FIELDS - fields)
    missing_functions = sorted({"bm25_asr"} - function_names)
    missing_indexes = sorted(_ASR_V2_REQUIRED_INDEX_FIELDS - index_fields)
    
    if not (missing_fields or missing_functions or missing_indexes):
        return
    
    details: list[str] = []
    if missing_fields:
        details.append(f"fields={missing_fields}")
    if missing_functions:
        details.append(f"functions={missing_functions}")
    if missing_indexes:
        details.append(f"indexes={missing_indexes}")
    
    raise RuntimeError(
        "Milvus collection 'asr_embeddings' uses the legacy ASR schema "
        f"({', '.join(details)}). Drop and rebuild the ASR Milvus index "
        "before deploying hybrid ASR search."
    )
```

#### 1.3.1 【必须】把校验挂进 `_init_collections`（易漏）

> ⚠️ **v1.2 补充**：仅定义 `_validate_existing_asr_collection` 不会自动生效。OCR 是在 `_init_collections` 遍历已存在 collection 的分支里显式调用的（`milvus_client.py:281-282`）。ASR 必须在**同一位置**加一行对称调用，否则 legacy schema 的 fail-fast 形同虚设。

```python
# backend/app/vector_store/milvus/milvus_client.py  (_init_collections, else 分支)
else:
    col = Collection(name)
    if name == "ocr_embeddings":
        _validate_existing_ocr_collection(col)
    if name == "asr_embeddings":            # ← v1.2 新增
        _validate_existing_asr_collection(col)
    load_state = utility.load_state(name)
    ...
```

#### 1.3.2 【必须】同步更新两处静态索引配置（易漏）

> ⚠️ **v1.2 补充**：schema 从单 index 改为 `indexes` 字典后，还有两处静态配置需与 OCR 对齐，计划 v1.1 未提及：

1. **`milvus_client.py` 的 `_COLLECTION_CONFIGS["asr_embeddings"]`**：从 `{"schema": ..., "index": _STATIC_INDEX_CONFIGS["asr_embeddings"]}` 改为 `{"schema": create_asr_schema, "indexes": {...}}`（见 2 节的 DiskANN + BM25 配置）。删除对 `_STATIC_INDEX_CONFIGS["asr_embeddings"]` 的 `index` 引用。`_STATIC_INDEX_CONFIGS` 里的 `asr_embeddings` 条目本身可参照 OCR 做法保留（无害，`_init_collections` 走 `indexes` 分支后不再读取它）。

2. **`milvus_search.py:69` 的 `_STATIC_INDEX_TYPES["asr"]`**：从 `"HNSW"` 改为 `"DISKANN"`（对齐 `"ocr": "DISKANN"`）。
   - 功能上 hybrid 路径不经过 `_ann_search`（`get_modality_index_type("asr")` 不再被 ASR 检索调用），改不改都不影响运行；
   - 但保留 `"HNSW"` 会误导后续维护者，且可能触发断言/测试里 “asr → HNSW” 的旧预期。**建议同步改为 `"DISKANN"` 并同步更新相关测试断言**。

#### 1.4 验证
```bash
# 单元测试
python -m pytest backend/tests/test_asr_schema_v2.py -v

# 预期结果：schema字段、function、索引配置正确
```

---

### 阶段2: 混合检索实现（第2-3天）

#### 2.1 实现混合检索函数
```bash
# 修改文件：backend/app/vector_store/milvus/milvus_search.py
# 新增函数：milvus_asr_candidates_hybrid()
```

**关键逻辑**:
- 三路fallback：hybrid → dense-only → bm25-only
- 使用`AnnSearchRequest` + `WeightedRanker`
- 返回Candidate对象，保持与现有接口一致

#### 2.2 更新调用侧（`_milvus_candidates_for_video`，`search.py:1589`）
```bash
# 修改文件：backend/app/retrieval/search.py
# import 从 milvus_asr_candidates 改为 milvus_asr_candidates_hybrid（search.py:1540）
```

**修改前**（`search.py:1589-1597`）:
```python
candidates.extend(milvus_asr_candidates(
    client,
    video_id,
    text,
    semantic_query,
    channel_limits["asr"],
    profiler,
    rows=prefetched_rows.get("asr"),   # ← hybrid 版本不再需要
))
```

**修改后**:
```python
candidates.extend(milvus_asr_candidates_hybrid(
    client,
    video_id,
    text,
    semantic_query,
    channel_limits["asr"],
    profiler,
))
```

#### 2.3 移除bulk query逻辑
```bash
# 修改文件：backend/app/vector_store/milvus/milvus_search.py
# 从BULK_QUERY_FIELDS中移除"asr"
```

> ⚠️ **v1.0 勘误**：现网 `BULK_QUERY_FIELDS` **只有 `"asr"` 一项**，`"ocr"` 已在 OCR 优化时移除（见 `milvus_search.py:103-116` 的注释）。因此移除 `"asr"` 后该 dict 变为**空**。

**修改前**（`milvus_search.py:103-116`，实际现状）:
```python
BULK_QUERY_FIELDS: dict[str, list[str]] = {
    # "visual" and "ocr" are intentionally absent ...
    "asr": [
        "segment_idx", "start_ms", "end_ms",
        "text", "has_embedding", "embedding",
    ],
}
```

**修改后**:
```python
# ASR 也改用 hybrid_search()，与 visual/ocr 一样自带 search 调用、
# 不消费预取行；该 dict 现在为空。
BULK_QUERY_FIELDS: dict[str, list[str]] = {}
```

> 说明：`query_rows_for_videos` 只对 `BULK_QUERY_FIELDS` 中存在的 modality 预取，空 dict 即表示不再对任何 modality 做批量预取。

#### 2.4 清理Legacy代码（直接删除，无回滚开关）
按用户决策：**不保留回滚开关，直接清理 legacy**。需删除：
- `milvus_search.py`：`milvus_asr_candidates()`（全量 query + Python 打分版本）
- `search.py`：NPZ 路径 `_asr_for_video()`（`search.py:1464-1488`）及其在 `_candidates_for_video()` 中的 ASR 分支（`search.py:1515-1521`）
  > ⚠️ **v1.2 显式声明的权衡**：`milvus_fallback_enabled` 默认 `True`（`settings.py:165`）。当前 ASR 在 Milvus 查询失败或零召回时会走 NPZ 恢复通道（`search.py:1930-1939`）。删除 `_asr_for_video()` 后，**ASR 与 OCR 一样彻底失去 NPZ 兜底**——Milvus 异常时该模态直接返回空结果，只能靠修复代码 + 重新部署恢复。这是有意为之（与 OCR 现网一致，`_candidates_for_video` 中 OCR 分支早已不存在），但必须知情接受，而非当作单纯"清理"。
- `search.py`：ASR-only 词面保底 `_reserve_asr_lexical_results()` 及相关常量、`_asr_result_lexical_score()`、`set(modalities)=={"asr"}` 调用点（`search.py:2000-2001`）——详见 4.6 节
- `_asr_candidates()` / `_semantic_chunk_scores()`：`grep` 确认仅被 ASR 旧路径引用后一并删除

> **⚠️ v1.2 勘误（重要，纠正 v1.1 的错误理由）**：v1.1 声称"**保留** `lexical_score()`，因为 OCR 的 `_ocr_display_text()` 仍在使用（`search.py:620`）"——**这个理由不成立**。逐行核对调用链：
> - `_ocr_display_text()` 的**唯一**调用点是 `_asr_candidates()` 内部（`search.py:517`，def 在 `search.py:581`）。
> - OCR 的现网 hybrid 路径 `milvus_ocr_candidates_hybrid()` **根本不经过** `_ocr_display_text()`（它直接从 `hit.entity` 取 `text`，见 `milvus_search.py:606-645`）。
> - `lexical_score()` 的调用者只有两处：`_asr_candidates:493` 和 `_ocr_display_text:620`（`grep` 确认）。
>
> **结论**：一旦按本节删除 `_asr_candidates()`，`_ocr_display_text()` 立即成为死代码，`lexical_score()` 也随之失去**全部**调用者。因此下列函数会**成组变成孤儿**，应一并评估删除（或明确接受其成为孤儿，但**不得**用"OCR 仍依赖"作为保留理由）：
> - `_ocr_display_text()`（`search.py:581`）
> - `_text_candidate_decision()`（`search.py:459`）
> - `_text_candidate_evidence()`（`search.py:471`）
> - `_semantic_chunk_scores()`（`search.py:428`）
> - `_asr_chunks_from_npz()`（`search.py:556`）、`_semantic_arrays()`（`search.py:153`）——仅被已删除的 `_asr_for_video()` 引用
> - `lexical_score()`（`search.py:78`）
>
> 实施时：`grep` 每个函数确认无其它调用者后删除。保留本身无害，但**若保留，理由必须是"暂缓清理"，而非"OCR 依赖"**。

---

### 阶段3: 索引重建（第4-5天）

#### 3.1 删除旧Collection
```bash
# 方式1：手工删除（开发环境）
docker exec momentseek-0829-platform python3 -c "
from pymilvus import connections, utility
connections.connect(host='localhost', port=19531)
utility.drop_collection('asr_embeddings')
print('✓ 已删除旧ASR collection')
"

# 方式2：维护脚本（生产环境）
# backend/scripts/drop_asr_collection.py
```

**注意事项**:
- ⚠️ 删除前备份关键数据
- ⚠️ 确认Speaker数据不会受影响（独立collection）
- ⚠️ 在低峰时段执行

#### 3.2 重启服务
```bash
# 停止并删除容器
docker stop momentseek-0829-platform
docker rm momentseek-0829-platform

# 重新部署（加载新schema）
DEV_MODE=true DEV_SKIP_BUILD=true ./deploy_0829.sh
```

#### 3.3 验证新Schema
```bash
docker exec momentseek-0829-platform python3 -c "
from pymilvus import connections, Collection
connections.connect(host='localhost', port=19531)
col = Collection('asr_embeddings')

print('=== Schema Fields ===')
for field in col.schema.fields:
    print(f'  {field.name}: {field.dtype}')

print('\\n=== Functions ===')
functions = getattr(col.schema, 'functions', [])
for func in functions:
    print(f'  {func.name}: {func.function_type}')

print('\\n=== Indexes ===')
for idx in col.indexes:
    print(f'  {idx.field_name}: {idx._index_params}')
"
```

**预期输出**（注意：主键是 `pk`，并含 `model_version`；字段顺序取决于 `_common_fields` + 追加字段）:
```
=== Schema Fields ===
  pk: VARCHAR
  video_id: VARCHAR
  asset_version: VARCHAR
  model_version: VARCHAR
  segment_idx: INT64
  start_ms: INT64
  end_ms: INT64
  text: VARCHAR
  has_embedding: BOOL
  embedding: FLOAT_VECTOR
  sparse_embedding: SPARSE_FLOAT_VECTOR

=== Functions ===
  bm25_asr: BM25

=== Indexes ===
  embedding: {'index_type': 'DISKANN', 'metric_type': 'IP', ...}
  sparse_embedding: {'index_type': 'SPARSE_INVERTED_INDEX', 'metric_type': 'BM25', ...}
```

> ⚠️ **v1.0 勘误**：主键字段是 **`pk`**（非 `id`），且 `_common_fields` 会额外带 `model_version`。若脚本打印出 `id` 或缺 `model_version`，说明抄错了 schema。

#### 3.4 重建索引
```bash
# 方式1：通过前端UI逐个视频重建

# 方式2：批量重建脚本
# backend/scripts/rebuild_asr_batch.py
python backend/scripts/rebuild_asr_batch.py \
  --video-ids video1,video2,video3 \
  --concurrency 4
```

**关键点**:
- 索引生成逻辑自动处理`sparse_embedding`（Milvus Function）
- 保持`segment_idx`的顺序性
- 监控索引构建进度和错误

---

### 阶段4: 测试验证（第6-7天）

#### 4.1 单元测试
```bash
# 测试混合检索逻辑
python -m pytest backend/tests/test_asr_hybrid_search.py -v

# 测试fallback逻辑
python -m pytest backend/tests/test_asr_fallback.py -v
```

**测试用例**:
- ✅ 正常混合检索（有embedding + 有文本）
- ✅ Dense-only fallback（有embedding + 空文本）
- ✅ BM25-only fallback（无embedding + 有文本）
- ✅ 空结果处理（无embedding + 空文本）
- ✅ Candidate对象字段完整性

#### 4.2 集成测试
```bash
# 端到端测试
python backend/scripts/test_asr_e2e.py \
  --video-id a06335ada8a448998e5ba85231c86d3e \
  --query "天气" \
  --expected-min-candidates 5
```

**验证点**:
- ✅ 检索延迟 < 50ms
- ✅ 召回数量合理（5-20条）
- ✅ Evidence格式正确
- ✅ Hybrid score分布合理

#### 4.3 性能测试
```python
# backend/scripts/benchmark_asr_search.py
import time
from app.vector_store.milvus.milvus_client import get_milvus_client

client = get_milvus_client()
video_id = "test_video_1hour"
query_text = "人工智能"
query_embedding = get_embedding(query_text)

# 测试100次
latencies = []
for _ in range(100):
    start = time.time()
    candidates = milvus_asr_candidates_hybrid(
        client, video_id, query_text, query_embedding, limit=20
    )
    latencies.append(time.time() - start)

print(f"P50: {np.percentile(latencies, 50):.3f}s")
print(f"P95: {np.percentile(latencies, 95):.3f}s")
print(f"P99: {np.percentile(latencies, 99):.3f}s")
```

**目标**:
- P50 < 30ms
- P95 < 50ms
- P99 < 100ms

#### 4.4 中文分词验证
```bash
# 测试chinese analyzer对中英文的支持
python backend/scripts/test_chinese_analyzer_asr.py
```

**测试用例**:
- ✅ 纯中文："人工智能"
- ✅ 纯英文："machine learning"
- ✅ 中英混合："AI人工智能"
- ✅ 长句子："今天我们讨论人工智能的未来发展"

#### 4.5 Speaker兼容性验证
```python
# backend/scripts/verify_speaker_asr_consistency.py
# 验证ASR重建后，Speaker的asr_chunk_idx引用是否仍然有效

def verify_speaker_asr_links(video_id: str):
    # 1. 查询所有Speaker utterance
    speaker_rows = query_all_speakers(video_id)
    
    # 2. 对每个utterance，检查其asr_chunk_idx是否有效
    for utt in speaker_rows:
        asr_chunk_idx = utt["asr_chunk_idx"]
        
        # 查询对应的ASR chunk
        asr_chunk = query_asr_by_segment_idx(video_id, asr_chunk_idx)
        
        if asr_chunk is None:
            print(f"❌ 无效引用: utterance {utt['utterance_idx']} "
                  f"引用不存在的 asr_chunk_idx={asr_chunk_idx}")
        else:
            # 检查时间戳是否对齐
            if (utt["start_ms"] == asr_chunk["start_ms"] and 
                utt["end_ms"] == asr_chunk["end_ms"]):
                print(f"✅ 有效引用: utterance {utt['utterance_idx']} → "
                      f"asr_chunk {asr_chunk_idx}")
            else:
                print(f"⚠️ 时间戳不匹配: utterance {utt['utterance_idx']}")
```

---

### 阶段5: 直接切换上线（第8天）

> **v1.0 勘误（重要）**：v1.0 的“5%→25%→50%→100% 灰度放量”**无法实现**——`grep` 全仓库确认**没有任何流量分流机制**，也不存在 `ASR_HYBRID_TRAFFIC_RATIO` 这个设置。OCR 当时就是**直接原地替换、无 flag、无灰度**上线的。按用户决策：ASR 同样**直接切换、清理 legacy、不保留回滚开关**。

#### 5.1 切换方式（原地替换，无开关）
调用侧 `_milvus_candidates_for_video`（`search.py:1589`）直接改为调用 `milvus_asr_candidates_hybrid`，import 同步更新（`search.py:1540`）。不引入 `asr_use_hybrid` 布尔开关，也不保留旧函数——与 2.4 节的 legacy 清理一致。

```python
# backend/app/retrieval/search.py （_milvus_candidates_for_video 内）
candidates.extend(milvus_asr_candidates_hybrid(
    client, video_id, text, semantic_query, channel_limits["asr"], profiler,
))
```

#### 5.2 上线后监控（无回滚开关，异常时靠前向修复 / 重新部署）
- 错误率 < 0.1%
- P95 延迟 < 50ms
- 召回数量稳定（中位数 ≥ 3）
- hybrid_score 分布合理（用于校准 3.1 节的全局阈值系数）

> **无回滚开关**：一旦发现问题，通过修复代码 + 重新部署解决（旧实现已删除，git 历史可查）。这与用户“直接清理 legacy、不需要回滚”的决策一致，也与 OCR 上线方式一致。

---

## 📊 参数调优指南

### 1. 召回量调整

**默认配置**（对齐 OCR 现网）:
```bash
ASR_HYBRID_RECALL_SIZE=100
```

**调优方向**:
- 召回太少 → 增加到200-300
- 延迟太高 → 减少到50-80

### 2. 权重调整

**默认配置**:
```bash
ASR_SEMANTIC_WEIGHT=0.65  # 语义65% + 词面35%
```

**调优场景**:
- **需要更强的语义匹配**（用户查询抽象概念）:
  ```bash
  ASR_SEMANTIC_WEIGHT=0.75  # 语义75% + 词面25%
  ```

- **需要更强的词面匹配**（用户查询关键词）:
  ```bash
  ASR_SEMANTIC_WEIGHT=0.55  # 语义55% + 词面45%
  ```

### 3. DiskANN搜索列表

**默认配置**（对齐 OCR 现网）:
```bash
ASR_DISKANN_SEARCH_LIST=100
```

**调优方向**:
- 召回精度不足 → 增加到200-300
- 延迟太高 → 减少到50-80

### 4. 阈值调整（全局动态阈值，非硬编码）

> ⚠️ **v1.0 勘误**：v1.0 在 `milvus_asr_candidates_hybrid` 里硬编码 `hybrid_score >= 0.8 / 0.6` 判定 `above_threshold`/`decision`，这是**错误**的——`WeightedRanker` 的融合分是 dense IP 分（约 [-1,1]）与 **BM25 无界分**的加权和，不落在 [0,1]，固定阈值无意义。已改为 OCR 同款**全局动态阈值**。

**当前机制**（对齐 OCR，见 3.1 节 / `search.py` OCR 段 `1970-1980`）:
```python
# hybrid 函数内：一律 above_threshold=True，decision="hit"，不硬编码阈值
# search.py fusion 前：对全体 ASR 候选统一算全局阈值
asr_candidates = [c for c in candidates if c.modality == "asr"]
if asr_candidates:
    global_top_score = max(float(c.score) for c in asr_candidates)
    global_threshold = max(0.10, global_top_score * 0.3)   # 系数与 OCR 一致
    for c in asr_candidates:
        c.above_threshold = float(c.score) >= global_threshold
```

**建议**:
- 阈值系数 `0.3` 与 OCR 保持一致；收集实际分数分布后可单独调优
- 如需固定的 strong/weak 分级，应基于**同批次归一化后的分数**，而非原始融合分

---

## 🔍 监控与告警

### 1. 关键指标

#### 性能指标
```python
# backend/app/monitoring/asr_metrics.py
from prometheus_client import Histogram, Counter

asr_search_latency = Histogram(
    "asr_search_duration_seconds",
    "ASR search latency",
    ["search_type"],  # hybrid/dense_only/bm25_only
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

asr_search_results = Histogram(
    "asr_search_result_count",
    "ASR search result count",
    ["search_type"],
    buckets=[0, 5, 10, 20, 50, 100],
)

asr_search_errors = Counter(
    "asr_search_errors_total",
    "ASR search errors",
    ["error_type"],
)
```

#### 质量指标
```python
asr_hybrid_score_distribution = Histogram(
    "asr_hybrid_score",
    "ASR hybrid score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

asr_has_embedding_ratio = Histogram(
    "asr_has_embedding_ratio",
    "Ratio of chunks with embeddings",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
```

### 2. 告警规则

```yaml
# prometheus/alerts.yml
groups:
  - name: asr_search
    rules:
      # P95延迟超过100ms
      - alert: ASRSearchLatencyHigh
        expr: histogram_quantile(0.95, asr_search_latency_bucket) > 0.1
        for: 5m
        annotations:
          summary: "ASR search P95 latency > 100ms"
      
      # 错误率超过1%
      - alert: ASRSearchErrorRateHigh
        expr: rate(asr_search_errors_total[5m]) / rate(asr_search_total[5m]) > 0.01
        for: 5m
        annotations:
          summary: "ASR search error rate > 1%"
      
      # 召回数量异常低
      - alert: ASRSearchLowRecall
        expr: histogram_quantile(0.5, asr_search_result_count_bucket) < 3
        for: 10m
        annotations:
          summary: "ASR search median recall < 3 results"
```

### 3. Dashboard

**Grafana面板**:
1. **ASR Search Latency** (P50/P95/P99)
2. **ASR Search Result Count** (直方图)
3. **ASR Hybrid Score Distribution**
4. **ASR Search Type Breakdown** (hybrid/dense/bm25比例)
5. **ASR has_embedding Ratio**

---

## 🐛 已知问题与限制

### 1. Milvus 2.6 Analyzer限制

**问题**: 只支持`standard`和`chinese`分词器，不支持`jieba`。

**影响**: 中文分词质量依赖内置`chinese` analyzer。

**缓解措施**:
- 使用较高的`semantic_weight`以补偿分词质量
- 考虑升级到Milvus更高版本（如支持更多analyzer）

### 2. BM25 Function不支持参数

**问题**: Milvus 2.6的BM25 Function不接受任何params（k1, b等）。

**影响**: 无法自定义BM25参数。

**缓解措施**:
- 使用Milvus默认的BM25参数
- 通过权重调整（`asr_semantic_weight`）来平衡语义和词面

### 3. segment_idx稳定性依赖

**问题**: Speaker模态依赖ASR的`segment_idx`字段。

**影响**: ASR重建时必须保持`segment_idx`的稳定性。

**缓解措施**:
- 保持ASR chunk切分逻辑不变
- 提供验证工具检查Speaker-ASR引用的有效性
- 考虑将来使用更稳定的ID方案（如UUID）

### 4. 混合分数不可分解（并导致舍弃词面保底）

**问题**: `WeightedRanker`只返回融合后的`hybrid_score`，无法分别获取dense和sparse分数，因此 hybrid 候选不再填充 `Candidate.lexical_score` / `semantic_score`。

**影响**:
- Evidence展示无法分别显示语义分数和词面分数。
- 旧的 ASR-only 词面保底 `_reserve_asr_lexical_results` 依赖 `lexical_score`，在 hybrid 下会恒取 0.0 而失效——故**直接删除**该逻辑（见 4.1 节），词面优先级改由 BM25 + `WeightedRanker` 在服务端保证。

**缓解措施**:
- 在Evidence中只展示`hybrid_score`
- 若纯关键词查询召回不足，下调 `asr_semantic_weight`（提高词面权重），而非重新引入 Python 侧保底
- 未来如确需分数分解，可改用两次独立 `search()` + Python 侧融合

---

## 📚 参考资料

### Milvus官方文档
- [DiskANN索引](https://milvus.io/docs/disk_index.html)
- [混合检索](https://milvus.io/docs/multi-vector-search.md)
- [BM25 Function](https://milvus.io/docs/keyword-match.md)
- [Full Text Search](https://milvus.io/docs/full-text-search.md)

### 内部文档
- [Milvus优化总体计划](./Milvus_optimization_plan.md)
- [Visual模态优化记录](./Visual_record.md)
- [OCR模态优化记录](./OCR_record.md)

### 代码位置
- Schema定义: `backend/app/vector_store/milvus/milvus_schema.py`
- 索引配置: `backend/app/vector_store/milvus/milvus_client.py`
- 检索实现: `backend/app/vector_store/milvus/milvus_search.py`
- 候选生成: `backend/app/retrieval/search.py`
- 设置配置: `backend/app/core/settings.py`

---

## ✅ 验收标准

### 功能验收
- ✅ 混合检索正常工作（hybrid/dense-only/bm25-only）
- ✅ Candidate对象字段完整
- ✅ Evidence格式正确
- ✅ Speaker模态不受影响

### 性能验收
- ✅ P95延迟 < 50ms（1小时视频）
- ✅ 召回数量合理（5-20条）
- ✅ 错误率 < 0.1%

### 质量验收
- ✅ 中文查询正常（如"天气"）
- ✅ 英文查询正常（如"weather"）
- ✅ 中英混合查询正常（如"AI人工智能"）
- ✅ 长句子查询正常

### 兼容性验收
- ✅ Speaker-ASR引用有效性验证通过
- ✅ 时间戳对齐验证通过

### Legacy 清理验收
- ✅ `milvus_asr_candidates()`（全量 query 版本）已删除
- ✅ `_asr_for_video()` NPZ 路径及 `_candidates_for_video` 的 ASR 分支已删除
- ✅ `_reserve_asr_lexical_results` / `_asr_result_lexical_score` / `_ASR_LEXICAL_RESERVE_*` 已删除
- ✅ `BULK_QUERY_FIELDS` 移除 `"asr"`（现为空 dict）
- ✅ 未引入 `asr_use_hybrid` / `ASR_HYBRID_TRAFFIC_RATIO`（无灰度、无回滚开关）
- ✅ `create_asr_schema` 原地替换（无 `_v1`/`_v2` 后缀函数残留）
- ✅ 【v1.2 修正】`lexical_score()` / `_ocr_display_text()` / `_text_candidate_decision()` / `_text_candidate_evidence()` / `_semantic_chunk_scores()` / `_asr_chunks_from_npz()` / `_semantic_arrays()` 均已 `grep` 确认成为孤儿，一并删除（或明确标注"暂缓清理"）；**不得**以"OCR 依赖"为由保留 `lexical_score()`
- ✅ 【v1.2 新增】`_validate_existing_asr_collection` 已挂进 `_init_collections`（`milvus_client.py:281` 附近），legacy schema fail-fast 生效
- ✅ 【v1.2 新增】`_COLLECTION_CONFIGS["asr_embeddings"]` 改用 `indexes` 字典；`milvus_search.py` 的 `_STATIC_INDEX_TYPES["asr"]` 改为 `"DISKANN"` 且相关测试断言同步更新
- ✅ 【v1.2 新增】已显式声明并接受"删除 NPZ fallback 后 ASR 失去错误恢复通道"这一权衡（与 OCR 对齐）
- ✅ `above_threshold` 由 `search.py` 全局动态阈值统一设置，hybrid 函数内无硬编码阈值
- ✅ ASR-only 查询下 BM25 词面命中排序合理（已不依赖 `_reserve_asr_lexical_results`）

---

## 🎯 下一步行动

### 立即行动（第1天）
1. ✅ Review本优化计划，确认技术方案
2. ✅ 更新Schema和索引配置代码
3. ✅ 实现Schema校验函数
4. ✅ 编写单元测试

### 第2-3天
1. ✅ 实现混合检索函数
2. ✅ 更新调用侧代码
3. ✅ 移除bulk query逻辑
4. ✅ 编写集成测试

### 第4-5天
1. ✅ 删除旧ASR collection
2. ✅ 重启服务并验证新Schema
3. ✅ 重建ASR索引（批量或单个视频）
4. ✅ 验证索引构建成功

### 第6-7天
1. ✅ 运行完整测试套件
2. ✅ 性能基准测试
3. ✅ Speaker兼容性验证
4. ✅ 修复发现的问题

### 第8天（直接切换上线，无灰度）
1. ✅ 调用侧原地替换为 `milvus_asr_candidates_hybrid`，删除 legacy（无回滚开关）
2. ✅ 监控关键指标（错误率 / P95 / 召回数 / hybrid_score 分布）
3. ✅ 依据 hybrid_score 分布校准全局阈值系数
4. ✅ 参数调优（recall_size / semantic_weight / search_list）

### 后续优化（可选）
1. 🔄 根据实际数据调整权重和阈值
2. 🔄 优化chinese analyzer（如升级Milvus版本）
3. 🔄 实现动态权重调整（基于查询类型）
4. 🔄 添加更丰富的监控指标

---

## 📝 经验教训总结

### 从Visual优化中学到的
1. ✅ **DiskANN可靠**: Visual已成功使用DiskANN，ASR可以放心采用
2. ✅ **避免采样估算**: Visual尝试过采样估算分布，效果不佳，ASR直接使用服务端融合
3. ✅ **保留legacy字段**: Visual保留了NPZ fallback字段，ASR也保留`has_embedding`以应对未来需求

### 从OCR优化中学到的
1. ✅ **chinese analyzer有效**: OCR验证了`chinese` analyzer支持中英文双语
2. ✅ **BM25 Function无参数**: Milvus 2.6的BM25 Function不接受params，直接使用默认值
3. ✅ **词面优先策略**: OCR使用70%词面权重，效果很好；ASR因文本更长，使用65%语义权重
4. ✅ **完全移除NPZ fallback + 无灰度直接切换**: OCR 直接原地替换、无 flag、无回滚开关，ASR 照此执行
5. ✅ **绝不硬编码融合分阈值**: OCR 用 `above_threshold=True` 占位 + `search.py` 全局动态阈值；ASR 必须对称新增 ASR 段，切勿沿用 v1.0 的 0.8/0.6

### 特有的挑战
1. ⚠️ **Speaker依赖**: ASR与Speaker存在跨模态依赖，需特别注意数据一致性
2. ⚠️ **segment_idx稳定性**: 必须保持`segment_idx`的稳定性以保证Speaker引用有效
3. ⚠️ **权重策略不同**: ASR（语义优先）vs OCR（词面优先），需根据数据特性调整
4. ⚠️ **词面保底逻辑取舍**: ASR 独有 `_reserve_asr_lexical_results`（OCR 无）。hybrid 架构下它因 `lexical_score` 不再填充而失效，且 BM25 已在服务端保证词面召回，故**直接删除**而非修补
5. ⚠️ **写入路径易认错**: ASR 写入在 `AsrMilvusIndexer.upsert_from_memory()`（`milvus_indexer.py`），**不是** `asr.py` 里虚构的 `_save_asr_to_milvus()`——v1.0 在此处误导，实际写入侧几乎无需改动

---

**文档制定**: Claude Opus 4.8  
**制定日期**: 2026-08-03  
**修订日期**: 2026-08-04（v1.2，第二轮代码审核修订）  
**文档版本**: v1.2  
**状态**: 待实施
