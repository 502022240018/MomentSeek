# OCR模态DiskANN + BM25混合检索优化实施计划

## 目标
实现OCR模态的高性能混合检索，支持亿级数据规模，延迟降低90%+

## 核心技术方案
- **Dense向量**: DiskANN索引（语义检索）
- **Sparse向量**: Milvus BM25（词面检索，中文jieba分词）
- **混合检索**: Milvus服务端hybrid_search()融合
- **权重策略**: 词面优先（sparse 70% > dense 30%）

## 实施步骤

### 阶段1: Schema和配置（约1-2小时）

#### 1.1 添加环境变量到settings.py
```python
# OCR混合检索配置
ocr_hybrid_recall_size: int = 100    # Dense/Sparse各召回100帧
ocr_lexical_weight: float = 0.7      # 词面权重（语义权重自动为1-此值）
ocr_diskann_search_list: int = 100   # DiskANN搜索列表大小
```

**位置**: `backend/app/settings.py` 在visual配置后添加（L168之后）

**验证器**:
- ocr_hybrid_recall_size > 0
- ocr_lexical_weight ∈ [0.0, 1.0]
- ocr_diskann_search_list > 0

#### 1.2 创建OCR v2 Schema
```python
def create_ocr_schema_v2() -> CollectionSchema:
    """OCR v2: DiskANN + BM25混合检索，保留has_embedding。"""
    from pymilvus import Function, FunctionType
    
    fields = _common_fields("ocr") + [
        # 元数据字段（完全兼容v1）
        FieldSchema("frame_idx",     DataType.INT64),
        FieldSchema("region_idx",    DataType.INT64),
        FieldSchema("frame_ms",      DataType.INT64),
        FieldSchema("start_ms",      DataType.INT64),
        FieldSchema("end_ms",        DataType.INT64),
        FieldSchema("avg_box_score", DataType.FLOAT),
        
        # 文本字段（启用BM25分析器）
        FieldSchema("text", DataType.VARCHAR, max_length=2000,
                    enable_analyzer=True,
                    analyzer_params={"type": "chinese"}),
        
        # 保留has_embedding（处理embedding失败/禁用）
        FieldSchema("has_embedding", DataType.BOOL, default_value=True),
        
        # Dense向量（字段名保持不变，索引升级为DiskANN）
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
        
        # Sparse BM25向量（新增，Milvus Function自动生成）
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True),
    ]
    
    # BM25 Function定义
    bm25_function = Function(
        name="bm25_ocr",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
        params={"language": "zh"},
    )
    
    return CollectionSchema(
        fields,
        description="OCR v2: DiskANN + BM25混合检索",
        functions=[bm25_function]
    )
```

**位置**: `backend/app/indexing/milvus_schema.py` 在create_ocr_schema()之后添加

#### 1.3 添加OCR v2 Collection配置
```python
# 在_STATIC_INDEX_CONFIGS中添加
"ocr_embeddings_v2": {
    "index_type": "DISKANN",
    "metric_type": "IP",
    "params": {"search_list": 200},
},

# 在_COLLECTION_CONFIGS中添加
"ocr_embeddings_v2": {
    "schema": create_ocr_schema_v2,
    "indexes": {
        # Dense索引：DiskANN
        "embedding": {
            "index_type": "DISKANN",
            "metric_type": "IP",
            "params": {"search_list": 200},
        },
        # Sparse索引：BM25倒排索引
        "sparse_embedding": {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        },
    },
},
```

**位置**: `backend/app/indexing/milvus_client.py`

**注意**: 不修改_COLLECTION_FOR_MODALITY映射（开发阶段保持指向v1）

### 阶段2: 实现混合检索（约2-3小时）

#### 2.1 实现混合检索函数
创建新函数`milvus_ocr_candidates_hybrid()`，实现：
- Dense检索（DiskANN，仅has_embedding=True）
- Sparse检索（BM25，所有帧）
- 服务端融合（WeightedRanker）
- 空查询处理（fallback到dense-only）

**位置**: `backend/app/indexing/milvus_search.py` 在milvus_ocr_candidates()之后添加

**关键点**:
1. 从settings动态读取配置
2. 保留evidence兼容性：使用`[milvus_query_all]`前缀
3. 空查询fallback到dense-only检索
4. 返回Candidate对象，features包含hybrid_score

#### 2.2 修改现有milvus_ocr_candidates()
```python
def milvus_ocr_candidates(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
    profiler: RetrievalProfiler | None = None,
    rows: list[dict] | None = None,
    profile: str = "balanced",  # 新增参数
) -> list[Candidate]:
    """OCR召回入口（v2混合检索）。
    
    开发阶段：直接调用混合检索实现（当v2 collection存在时）
    """
    # 检测v2 collection是否存在
    try:
        col_v2 = client.collection_for_name("ocr_embeddings_v2")
        # v2存在，使用混合检索
        return milvus_ocr_candidates_hybrid(
            client, video_id, query_text, query_embedding,
            limit, profiler, rows, profile
        )
    except:
        # v2不存在，使用旧逻辑
        # 保持原有实现不变...
        if rows is None:
            rows = _query_all(...)
        # ... 原有逻辑 ...
```

**注意**: 开发阶段使用collection检测自动切换，生产环境通过修改_COLLECTION_FOR_MODALITY切换

### 阶段3: 创建和索引v2 Collection（约30分钟）

#### 3.1 创建测试脚本
创建`backend/scripts/create_ocr_v2_collection.py`:
```python
#!/usr/bin/env python3
"""创建OCR v2 collection（DiskANN + BM25）"""
from app.indexing.milvus_client import get_milvus_client
from pymilvus import utility

def main():
    client = get_milvus_client()
    
    # 检查v2是否已存在
    if utility.has_collection("ocr_embeddings_v2"):
        print("✓ ocr_embeddings_v2 already exists")
        return
    
    # 创建v2 collection（自动由_init_collections创建）
    # 触发创建的方式：重启应用或调用client._init_collections()
    client._init_collections()
    
    print("✓ ocr_embeddings_v2 created")
    
    # 验证schema
    col = client.collection_for_name("ocr_embeddings_v2")
    schema_fields = {f.name for f in col.schema.fields}
    assert "sparse_embedding" in schema_fields
    assert "has_embedding" in schema_fields
    print("✓ Schema verified")

if __name__ == "__main__":
    main()
```

#### 3.2 重建OCR索引到v2
开发阶段简化：手动触发少量视频重建测试
```bash
# 重建单个视频测试
python -c "
from app.indexing.milvus_client import get_milvus_client
from app.processors.build_job import build_ocr
# 触发重建...
"
```

**注意**: 开发阶段数据量少，无需批量重建脚本

### 阶段4: 测试验证（约1-2小时）

#### 4.1 单元测试
创建`backend/tests/test_ocr_hybrid_search.py`:
```python
def test_ocr_v2_schema():
    """验证v2 schema正确性"""
    from app.indexing.milvus_schema import create_ocr_schema_v2
    schema = create_ocr_schema_v2()
    field_names = {f.name for f in schema.fields}
    assert "sparse_embedding" in field_names
    assert "has_embedding" in field_names
    assert "embedding" in field_names

def test_ocr_hybrid_search():
    """测试混合检索功能"""
    # 需要实际Milvus连接和测试数据
    pass
```

#### 4.2 手动测试
```python
# 测试混合检索
from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_search import milvus_ocr_candidates
import numpy as np

client = get_milvus_client()
query_embedding = np.random.randn(384).astype(np.float32)
query_embedding = query_embedding / np.linalg.norm(query_embedding)

candidates = milvus_ocr_candidates(
    client=client,
    video_id="test_video",
    query_text="测试文字",
    query_embedding=query_embedding,
    limit=20,
)

for c in candidates[:5]:
    print(f"{c.start_time:.1f}s: {c.text[:50]} (score={c.score:.3f})")
```

### 阶段5: 文档更新（约30分钟）

#### 5.1 更新配置文档
在`.env.example`中添加：
```bash
# OCR混合检索配置（v2: DiskANN + BM25）
OCR_HYBRID_RECALL_SIZE=100
OCR_LEXICAL_WEIGHT=0.7
OCR_DISKANN_SEARCH_LIST=100
```

#### 5.2 创建OCR优化文档
创建`docs/OCR_HYBRID_SEARCH.md`，记录：
- 优化前后对比
- 配置说明
- 性能指标
- 故障排查

## 实施清单

### 必须完成
- [ ] 添加环境变量到settings.py（含validator）
- [ ] 创建create_ocr_schema_v2()函数
- [ ] 添加ocr_embeddings_v2 collection配置
- [ ] 实现milvus_ocr_candidates_hybrid()函数
- [ ] 修改milvus_ocr_candidates()支持v2切换
- [ ] 创建ocr_embeddings_v2 collection
- [ ] 测试混合检索功能
- [ ] 更新.env.example配置

### 可选完成
- [ ] 编写单元测试
- [ ] 性能基准测试
- [ ] 创建优化文档
- [ ] 权重A/B测试

## 风险与缓解

### 风险1: Milvus Function Field不支持
**概率**: 低（已确认Milvus 2.6）
**影响**: 阻塞实施
**缓解**: 已确认版本满足

### 风险2: 混合检索性能不达预期
**概率**: 中
**影响**: 延迟目标未达成
**缓解**: 调整recall_size和search_list参数

### 风险3: 中文分词效果不佳
**概率**: 中
**影响**: 词面检索召回率低
**缓解**: 后续通过A/B测试调整权重

## 预期收益

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 延迟（10小时视频） | 1-3秒 | **< 30ms** | **100倍** |
| 网络传输 | 50-150MB | **< 10KB** | **10,000倍** |
| 内存（亿级） | 500+GB | **< 80GB** | **90%降低** |

## 后续优化方向

1. **权重调优**: A/B测试确定最优lexical_weight
2. **性能监控**: 添加检索延迟和召回率监控
3. **批量重建**: 编写批量重建脚本（如需要）
4. **前端优化**: 展示混合分数详情（可选）

---

**创建时间**: 2026-07-30
**预计完成**: 4-6小时（核心功能）
**负责人**: Claude + 用户
