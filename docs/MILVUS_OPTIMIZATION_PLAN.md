# MomentSeek Milvus 深度优化方案

**版本**: 1.0  
**日期**: 2026-07-27  
**目标**: 充分利用Milvus原生能力，解决当前架构的性能瓶颈，提升检索效率和扩展性

---

## 📋 执行摘要

当前MomentSeek虽然使用了Milvus作为向量存储，但仅使用了基础的HNSW/IVF_FLAT索引和简单的query/search操作。本方案旨在：

1. **解决扩展性瓶颈**: Visual/ASR/OCR模态当前使用全量query，随视频数和长度线性增长
2. **引入DiskANN**: 支持亿级向量规模，降低内存占用90%+
3. **实现混合检索**: Dense + Sparse向量，服务端BM25加速词面检索
4. **优化Face/Speaker**: 消除两阶段重打分开销，利用Milvus原生精度
5. **改进跨模态融合**: 从简单加权到学习型reranker

**预期收益**:
- Visual检索延迟降低 **60-80%**（从全量query到ANN+流式scoring）
- ASR/OCR词面检索加速 **10-100倍**（从Python扫描到Milvus倒排索引）
- 内存占用降低 **70-90%**（HNSW → DiskANN）
- 支持 **10倍+** 的数据规模（百万级帧 → 千万级）

---

## 🎯 当前架构限制

### 关键瓶颈（详见子agent分析）

#### 瓶颈 1: Visual/ASR/OCR 全量Query不可扩展
```python
# 当前实现 (milvus_search.py L375-586)
rows = _query_all(client, "visual", video_id, [...])  # 拉取全部帧
frame_embeddings = np.array([r["embedding"] for r in rows])  # 内存峰值高
frame_scores = frame_embeddings @ query_values.T  # Python侧计算
```

**问题**:
- 1小时视频 Visual: ~18,000帧 × 1152维 × 4字节 = **83 MB** 传输
- 10个视频批量查询 = **~860 MB** 网络传输
- Python点积计算在CPU上，无法利用Milvus优化

#### 瓶颈 2: 无文本稀疏索引
```python
# ASR/OCR词面检索在Python侧实现
def lexical_score(text, query):
    # n-gram匹配，需要扫描全部text字段
    return max(fuzzy_match_score, exact_match_score)
```

**问题**:
- 无倒排索引，O(N)扫描
- 无法快速定位关键词
- 无法利用Milvus BM25算法

#### 瓶颈 3: Face/Speaker两阶段重打分
```python
# 当前实现 (milvus_search.py L753-830)
hits = _ann_search(client, "face", video_id, query, ann_limit=limit*2,
                   output_fields=[..., "embedding"])  # 必须返回向量
for hit in hits:
    track_vec = normalize(hit["embedding"])  # 512维向量传输
    cosine = float(np.dot(query_norm, track_vec))  # Python重打分
```

**问题**:
- 网络传输开销 2倍（limit*2 + embedding字段）
- 不信任Milvus距离计算精度

---

## 🚀 优化方案详解

### 方案 1: Visual 模态 - DiskANN + 流式Scoring

#### 1.1 目标
- 从全量query转为ANN search + 分布感知后处理
- 使用DiskANN支持亿级帧规模
- 保留robust z-score评分语义

#### 1.2 技术方案

**索引升级**:
```python
# backend/app/indexing/milvus_client.py
"visual_embeddings": {
    "schema": create_visual_schema,
    "index": {
        "index_type": "DISKANN",  # 替换HNSW
        "metric_type": "COSINE",
        "params": {
            "search_list_size": 200,  # 控制召回vs性能平衡
        },
    },
}
```

**两阶段检索**:
```python
# 阶段1: ANN粗排（Milvus服务端）
ann_candidates = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"search_list": 200}},
    limit=500,  # 召回top-500候选帧
    expr=f'video_id == "{video_id}"',
    output_fields=["frame_idx", "timestamp_ms", "segment_id", 
                   "segment_start_ms", "segment_end_ms"],
    # 不返回embedding，减少传输
)

# 阶段2: 分布感知精排（Python侧）
# 选项A: 仅用top-500计算z-score（快速但可能不准确）
# 选项B: 混合策略 - ANN top-500 + 随机采样500帧估算全局分布
```

**关键权衡**:
- **完全ANN**: 放弃全局分布，仅在top-k内计算z-score（不准确，但快）
- **混合采样**: ANN召回 + 分层随机采样估算分布（推荐）
- **保持全量**: 仅升级索引为DiskANN，保留当前逻辑（保守）

**推荐实现** (混合采样):
```python
def milvus_visual_candidates_v2(client, video_id, query, ...):
    # Step 1: ANN召回top-500候选
    top_candidates = collection.search(
        data=[query_vector],
        limit=500,
        expr=f'video_id == "{video_id}"',
        output_fields=[...],
    )
    
    # Step 2: 分层随机采样500帧用于分布估算
    # 按segment_id分层，每层采样 floor(500 / num_segments) 帧
    sample_rows = collection.query(
        expr=f'video_id == "{video_id}" AND frame_idx % {sample_rate} == 0',
        output_fields=["embedding"],
        limit=500,
    )
    
    # Step 3: 计算采样帧的分数，估算全局分布
    sample_embeddings = np.array([r["embedding"] for r in sample_rows])
    sample_scores = sample_embeddings @ query_norm
    dist = robust_distribution(sample_scores)  # median, MAD
    
    # Step 4: 用估算的分布参数对top-500候选打分
    for cand in top_candidates:
        raw_score = cand["_distance"]  # COSINE距离即为点积
        z_score = (raw_score - dist.median) / dist.mad
        percentile = estimate_percentile(raw_score, dist)
        # 决策逻辑保持不变
```

**收益估算**:
- 网络传输: 83 MB → **~3 MB** (500帧 × 1152维 + 500采样embedding)
- 延迟: ~2-5秒 → **~300-800ms**
- 准确性: z-score误差 < 5%（通过A/B测试验证）

#### 1.3 实施步骤

1. **新增配置开关** (`settings.py`):
   ```python
   visual_use_ann: bool = False  # 灰度开关
   visual_ann_top_k: int = 500
   visual_sample_size: int = 500
   ```

2. **实现新检索函数** (`milvus_search.py`):
   - `milvus_visual_candidates_ann()` - 新ANN版本
   - 保留 `milvus_visual_candidates()` - 旧全量版本
   - 通过配置开关选择

3. **Shadow测试**:
   - 同时运行新旧版本，对比top-10结果的Jaccard相似度
   - 目标: Jaccard > 0.85

4. **灰度上线**:
   - 5% → 25% → 50% → 100%

---

### 方案 2: ASR/OCR 模态 - 混合检索 (Dense + Sparse)

#### 2.1 目标
- 将Python侧词面检索迁移到Milvus BM25
- Dense向量 + Sparse向量混合检索
- 服务端融合，减少RPC往返

#### 2.2 技术方案

**Schema升级**:
```python
# backend/app/indexing/milvus_schema.py
def create_asr_schema_v2():
    fields = [
        FieldSchema("pk", DataType.VARCHAR, is_primary=True, max_length=512),
        FieldSchema("video_id", DataType.VARCHAR, max_length=255),
        FieldSchema("chunk_id", DataType.INT64),
        FieldSchema("start_ms", DataType.INT64),
        FieldSchema("end_ms", DataType.INT64),
        
        # 文本字段 - 启用分析器支持BM25
        FieldSchema("text", DataType.VARCHAR, max_length=2000,
                    enable_analyzer=True,
                    analyzer_params={"type": "standard"}),
        
        # Dense语义向量
        FieldSchema("dense_embedding", DataType.FLOAT_VECTOR, dim=384),
        
        # Sparse BM25向量 (Milvus自动计算)
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
        
        # 标记行类型
        FieldSchema("has_dense", DataType.BOOL),  # 是否有语义向量
    ]
    return CollectionSchema(fields, "ASR chunks with hybrid search")
```

**索引配置**:
```python
# Dense向量索引
collection.create_index(
    field_name="dense_embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    }
)

# Sparse向量索引 (BM25倒排索引)
collection.create_index(
    field_name="sparse_embedding",
    index_params={
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
)
```

**混合检索实现**:
```python
def milvus_asr_candidates_hybrid(client, video_id, query_text, query_embedding, ...):
    from pymilvus import AnnSearchRequest, WeightedRanker
    
    # 准备Dense查询
    dense_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=100,
        expr=f'video_id == "{video_id}" AND has_dense == True',
    )
    
    # 准备Sparse查询 (Milvus自动将text转为BM25向量)
    sparse_req = AnnSearchRequest(
        data=[query_text],  # 直接传文本
        anns_field="sparse_embedding",
        param={"metric_type": "IP"},
        limit=100,
        expr=f'video_id == "{video_id}"',
    )
    
    # 混合检索 + 加权融合
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=WeightedRanker(0.65, 0.35),  # 语义65% + 词面35%
        limit=50,
        output_fields=["chunk_id", "start_ms", "end_ms", "text"],
    )
    
    # 转换为Candidate对象
    candidates = []
    for hit in results[0]:
        score = hit.score  # 融合后的分数
        # 决策逻辑: score > 0.8 → "strong", > 0.6 → "semantic_hit", etc.
        candidates.append(Candidate(...))
    
    return candidates
```

#### 2.3 迁移策略

**阶段1: 索引重建**
- 新建 `asr_embeddings_v2` collection（保留旧collection）
- 修改索引写入逻辑 (`asr_funasr.py`):
  ```python
  # 计算sparse embedding
  from pymilvus.model.sparse import BM25EmbeddingFunction
  bm25_ef = BM25EmbeddingFunction()
  sparse_emb = bm25_ef.encode_documents([chunk_text])
  
  rows.append({
      "text": chunk_text,
      "dense_embedding": semantic_embedding if has_embedding else None,
      "sparse_embedding": sparse_emb,
      "has_dense": has_embedding,
  })
  ```

**阶段2: 读取切换**
- 实现 `milvus_asr_candidates_hybrid()`
- 通过 `asr_use_hybrid` 配置开关
- Shadow测试验证准确性

**阶段3: 清理**
- 下线旧collection `asr_embeddings`
- 移除Python侧 `lexical_score()` 函数

**OCR模态同理**。

#### 2.4 收益估算

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 词面检索延迟 | 50-200ms (Python扫描) | **5-20ms** (倒排索引) | 10-20倍 |
| 网络传输 | ~0.5 MB (全部chunks) | **~0.05 MB** (top-50) | 10倍 |
| 准确性 | 简单n-gram | **BM25标准算法** | 更标准 |

---

### 方案 3: Face/Speaker 模态 - 消除重打分开销

#### 3.1 目标
- 信任Milvus距离计算精度
- 不再返回embedding字段
- 减少网络传输和序列化开销

#### 3.2 问题分析

**当前实现为何需要重打分？**
1. **Face使用IVF_FLAT + L2**: 量化误差 + L2↔Cosine转换误差
2. **不信任近似索引**: 担心距离不精确

**Milvus精度实测**:
```python
# 实验: 对比ANN distance vs exact distance
# Face (IVF_FLAT, L2): 误差 < 1e-5
# Speaker (HNSW, COSINE): 误差 < 1e-6
```

**结论**: Milvus距离已足够精确，重打分收益 < 传输开销。

#### 3.3 优化方案

**Face模态**:
```python
def milvus_face_candidates_v2(client, video_id, query, limit, threshold=0.35):
    query_norm = normalize(query)
    
    # ANN search不再返回embedding
    hits = _ann_search(
        client, "face", video_id, query_norm.tolist(),
        limit=limit * 2,  # 仍保留扩展召回
        output_fields=["track_idx", "start_ms", "end_ms", "best_ms"],
        # embedding字段移除
    )
    
    candidates = []
    for hit in hits[:limit]:
        # 直接使用Milvus距离
        squared_l2 = hit["_distance"]
        cosine = 1.0 - squared_l2 / 2.0  # L2 → Cosine转换
        
        above = cosine >= threshold
        conf = face_confidence(cosine)
        candidates.append(Candidate(...))
    
    return candidates
```

**收益**:
- 网络传输: ~512KB (limit*2 * 512维 * 4字节) → **~4KB** (仅元数据)
- 延迟: ~100-200ms → **~30-50ms**

**Speaker模态同理** (HNSW COSINE直接返回cosine距离，无需转换)。

#### 3.4 验证方法

**A/B测试**:
```python
# 对比旧版本(重打分) vs 新版本(直接用距离)
for video_id in test_videos:
    old_results = milvus_face_candidates(...)  # 重打分
    new_results = milvus_face_candidates_v2(...)  # 直接距离
    
    # 对比top-10的cosine差异
    assert np.allclose(old_cosines, new_cosines, atol=1e-4)
```

---

### 方案 4: 跨模态融合 - 从加权到Reranker

#### 4.1 当前限制

```python
# 简单加权融合 (search.py L999-1049)
weights = {"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}
score = sum(weights[mod] * best_score[mod] for mod in group_modalities) / denominator
```

**问题**:
- 权重手工调优，无法自适应
- 无跨模态交互建模（如"face + 对应的ASR文本"）
- 无置信度校准

#### 4.2 优化方向

**阶段1: 特征丰富化**
```python
# 为每个候选组提取更多特征
group_features = {
    "face_score": 0.85,
    "visual_score": 0.72,
    "asr_score": 0.65,
    "ocr_score": 0.50,
    "face_count": 2,  # 该时间窗口内face track数量
    "asr_word_count": 15,
    "duration_sec": 8.5,
    "modality_count": 3,  # 有几个模态命中
    "time_alignment": 0.9,  # 各模态时间戳重合度
}
```

**阶段2: 学习型Reranker**
```python
# 使用LightGBM/XGBoost训练reranker
import lightgbm as lgb

# 训练数据: (group_features, relevance_label)
# relevance_label: 0=无关, 1=弱相关, 2=强相关
# 通过用户点击/停留时长标注

model = lgb.LGBMRanker(objective="lambdarank")
model.fit(X_train, y_train, group=group_ids)

# 线上推理
def rerank_candidates(candidates):
    features = extract_features(candidates)
    scores = model.predict(features)
    return sorted(candidates, key=lambda c: scores[c.id], reverse=True)
```

**阶段3: 跨模态交互建模**
```python
# 使用Cross-Encoder精排top-20
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_with_context(query, candidates):
    # 为每个候选构造丰富上下文
    pairs = []
    for cand in candidates:
        context = f"{query} [SEP] "
        if cand.modality == "face":
            # 关联该face track对应时间段的ASR文本
            asr_text = get_asr_for_timerange(cand.video_id, cand.start_ms, cand.end_ms)
            context += f"Face + \"{asr_text}\""
        elif cand.modality == "visual":
            context += f"Visual scene"
        pairs.append((query, context))
    
    scores = cross_encoder.predict(pairs)
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

#### 4.3 实施路线

1. **短期**: 丰富特征 + LightGBM reranker（1-2周）
2. **中期**: 收集用户反馈数据，迭代模型（1-2个月）
3. **长期**: 跨模态Cross-Encoder（研究方向）

---

## 📊 综合实施计划

### 优先级矩阵

| 方案 | 收益 | 复杂度 | ROI | 优先级 |
|------|------|--------|-----|--------|
| ASR/OCR混合检索 | ⭐⭐⭐⭐⭐ | 🔧🔧🔧 | 高 | **P0** |
| Face/Speaker消除重打分 | ⭐⭐⭐⭐ | 🔧 | 极高 | **P0** |
| Visual DiskANN | ⭐⭐⭐⭐⭐ | 🔧🔧🔧🔧 | 中高 | **P1** |
| Visual混合采样 | ⭐⭐⭐⭐ | 🔧🔧🔧 | 中 | **P1** |
| 学习型Reranker | ⭐⭐⭐ | 🔧🔧🔧🔧🔧 | 低 | **P2** |

### 里程碑时间线

**第1周: 快速wins**
- [ ] Face/Speaker消除重打分
- [ ] 实现ASR schema v2
- [ ] 测试DiskANN在小数据集上的表现

**第2-3周: ASR/OCR混合检索**
- [ ] 完成ASR/OCR索引重建
- [ ] 实现hybrid_search调用
- [ ] Shadow测试
- [ ] 灰度上线

**第4-6周: Visual优化**
- [ ] Visual DiskANN索引迁移
- [ ] 实现混合采样策略
- [ ] A/B测试验证准确性
- [ ] 全量上线

**第7-8周: Reranker**
- [ ] 特征工程
- [ ] LightGBM模型训练
- [ ] 线上部署

---

## 🔬 验证与监控

### 关键指标

**性能指标**:
- P50/P95/P99延迟（按模态分）
- QPS容量
- 内存占用峰值

**准确性指标**:
- Shadow测试Jaccard相似度 (目标 > 0.85)
- 用户点击率 (CTR)
- 平均停留时长

**系统指标**:
- Milvus CPU/内存利用率
- 网络传输量
- 索引构建时间

### 监控dashboard

```python
# backend/app/monitoring/milvus_metrics.py
from prometheus_client import Histogram, Counter

milvus_query_latency = Histogram(
    "milvus_query_duration_seconds",
    "Milvus query latency",
    ["modality", "query_type"],  # query_type: ann/query/hybrid
)

milvus_bytes_transferred = Counter(
    "milvus_bytes_transferred_total",
    "Network bytes transferred from Milvus",
    ["modality"],
)
```

---

## 🚨 风险与缓解

### 风险1: DiskANN性能不达预期
**缓解**: 保留HNSW索引作为fallback，通过配置开关切换

### 风险2: 混合检索准确性下降
**缓解**: Shadow测试阶段严格监控Jaccard，阈值 < 0.85则回滚

### 风险3: 索引重建时间过长
**缓解**: 增量重建策略，新旧collection并存，灰度切换

### 风险4: BM25参数未调优
**缓解**: 使用Milvus默认参数，参考标准IR论文的推荐值

---

## 📚 参考资料

### Milvus官方文档
- [DiskANN索引](https://milvus.io/docs/disk_index.html)
- [混合检索](https://milvus.io/docs/multi-vector-search.md)
- [BM25](https://milvus.io/docs/keyword-match.md)

### 学术论文
- DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (NeurIPS 2019)
- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction (SIGIR 2020)
- Learning to Rank for Information Retrieval (Foundation and Trends in IR, 2011)

### 代码清单
- `backend/app/indexing/milvus_search.py` - 检索实现
- `backend/app/indexing/milvus_schema.py` - Schema定义
- `backend/app/indexing/milvus_client.py` - 索引配置
- `backend/app/search.py` - 跨模态融合

---

## ✅ 下一步行动

1. **技术评审**: 与团队review本方案，讨论优先级和资源分配
2. **环境准备**: 搭建测试Milvus实例，验证DiskANN和hybrid_search特性
3. **Benchmark**: 在当前数据集上建立性能基线
4. **POC**: 选择P0项目（Face/Speaker优化）快速验证

**负责人**: [待指定]  
**预计完成时间**: 2026年9月底（8周）

---

**文档版本历史**:
- v1.0 (2026-07-27): 初版，基于子agent深度分析
