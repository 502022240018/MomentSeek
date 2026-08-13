# OCR_ASR_OPTIMIZATION_PLAN.md 严重错误清单（初步）

## 🔴 根本性架构错误

### 错误 1: Schema 字段名称完全错误

**方案中的设计**（错误）：
```python
FieldSchema("dense_embedding", DataType.FLOAT_VECTOR, dim=384),
FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR, is_function_output=True),
```

**实际代码**：
```python
# backend/app/indexing/milvus_schema.py L149
FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["asr"]),
FieldSchema("has_embedding", DataType.BOOL, default_value=True),
```

**影响**：
- ❌ 方案中所有使用 `dense_embedding` 的地方都是错误的（35处）
- ❌ 方案中所有使用 `sparse_embedding` 的地方都是错误的（35处）
- ❌ 检索代码 `anns_field="dense_embedding"` 会失败
- ❌ 写入代码需要完全重写

**错误出现位置**：
- L231: `FieldSchema("dense_embedding", ...)`
- L234: `FieldSchema("sparse_embedding", ...)`
- L261: `field_name="dense_embedding"` (索引)
- L283: `field_name="sparse_embedding"` (索引)
- L423: `anns_field="dense_embedding"` (检索)
- L455: `anns_field="sparse_embedding"` (检索)
- 以及其他30+处

---

### 错误 2: 方案要求删除 has_embedding 字段

**方案中的描述**（L204）：
```
✅ 删除 `has_embedding` 字段（所有行都有 dense embedding）
```

**实际情况**：
- ✅ 当前代码**已有** `has_embedding` 字段
- ✅ 写入逻辑依赖这个字段（L287-309 in milvus_indexer.py）
- ✅ 用于标记是否有真实 embedding（vs 零向量占位符）

**问题**：
- ❌ 方案假设"所有行都有 embedding"是错误的
- ❌ 实际上很多 ASR chunk 没有 embedding（lexical-only）
- ❌ 删除这个字段会导致无法区分真实 embedding 和占位符

---

### 错误 3: 文件路径完全错误

**方案中的文件路径**（L675, L716）：
```
backend/app/processors/asr_funasr.py
backend/app/processors/ocr_*.py
```

**实际情况**：
- ❌ `processors/` 目录**不存在**
- ✅ 写入逻辑在 `backend/app/indexing/milvus_indexer.py`
- ✅ 类名：`AsrMilvusIndexer` 和 `OcrMilvusIndexer`

**正确位置**：
```
backend/app/indexing/milvus_indexer.py
  - class AsrMilvusIndexer (L253-335)
  - class OcrMilvusIndexer (L338-458)
```

---

## 🟡 中等错误

### 错误 4: 写入逻辑假设错误

**方案描述**（L677-710）：
```python
# 方案假设需要这样写入
row = {
    "dense_embedding": emb.tolist(),
    "text": text,  # Function Field 会自动生成 sparse_embedding
}
```

**实际代码**（milvus_indexer.py L297-310）：
```python
row = {
    "embedding": emb.tolist() if has_emb else zero_vec,
    "has_embedding": has_emb,
    "text": texts[chunk_idx][:2000],
}
```

**差异**：
1. 字段名 `embedding` vs `dense_embedding`
2. 需要处理 `has_embedding` 标记
3. 需要处理零向量占位符

---

### 错误 5: 索引配置字段名错误

**方案中的索引配置**（L260-270, L282-289）：
```python
collection.create_index(
    field_name="dense_embedding",  # ❌ 错误
    index_params={"index_type": "DISKANN", ...}
)

collection.create_index(
    field_name="sparse_embedding",  # ❌ 错误
    index_params={"index_type": "SPARSE_INVERTED_INDEX", ...}
)
```

**应该是**：
```python
collection.create_index(
    field_name="embedding",  # ✅ 正确
    index_params={"index_type": "DISKANN", ...}
)

collection.create_index(
    field_name="sparse_embedding",  # 这个是新增的，OK
    index_params={"index_type": "SPARSE_INVERTED_INDEX", ...}
)
```

---

## 📊 错误统计

| 错误类型 | 出现次数 | 严重程度 |
|---------|---------|---------|
| `dense_embedding` 字段名 | ~20次 | 🔴 严重 |
| `sparse_embedding` 字段名 | ~15次 | 🟡 中等（新字段） |
| 删除 `has_embedding` | 多处 | 🔴 严重 |
| 错误文件路径 | 3次 | 🟡 中等 |
| 写入逻辑假设错误 | 多处 | 🔴 严重 |

---

## 🔧 修复策略

### 方案 A: 完全重写 schema 设计（推荐）

**核心思路**：保持现有字段名，仅新增 sparse 字段

```python
def create_asr_schema_v2() -> CollectionSchema:
    fields = _common_fields("asr") + [
        FieldSchema("segment_idx", DataType.INT64),
        FieldSchema("start_ms", DataType.INT64),
        FieldSchema("end_ms", DataType.INT64),
        FieldSchema("text", DataType.VARCHAR, max_length=2000,
                    enable_analyzer=True,
                    analyzer_params={"type": "chinese"}),
        
        # ✅ 保持原字段名 "embedding"
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
        
        # ✅ 保留 has_embedding 字段
        FieldSchema("has_embedding", DataType.BOOL, default_value=True),
        
        # ✅ 新增 sparse 字段（BM25）
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR,
                    is_function_output=True),
    ]
    
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

**优点**：
- ✅ 最小化变更，兼容现有写入逻辑
- ✅ 仅需修改写入代码：删除 sparse_embedding 赋值（Function 自动生成）
- ✅ 保留 has_embedding 语义

---

### 方案 B: 重命名字段（不推荐）

修改所有代码以适配 `dense_embedding` 命名。

**缺点**：
- ❌ 需要修改大量现有代码
- ❌ 破坏向后兼容性
- ❌ 风险高

---

## 🎯 下一步行动

1. **立即停止使用当前方案** - 架构错误太多
2. **全面重写 schema 部分** - 采用方案 A
3. **修正所有字段名引用** - `embedding` 代替 `dense_embedding`
4. **修正文件路径** - `milvus_indexer.py` 代替 `processors/`
5. **保留 has_embedding 逻辑** - 不删除此字段
6. **重新验证检索代码** - 使用正确的字段名

---

**创建时间**: 2026-07-30  
**发现者**: Claude Code  
**状态**: 🔴 阻塞实施
