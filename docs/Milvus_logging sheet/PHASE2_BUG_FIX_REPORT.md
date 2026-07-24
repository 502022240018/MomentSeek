# Phase 2 Bug 修复报告

**修复日期**: 2026-07-21  
**Bug 严重性**: 🔴 **Critical** - 导致 Speaker 功能文本错位  
**状态**: ✅ **已修复并验证**

---

## Bug 描述

### 问题代码

**文件**: `backend/app/indexing/milvus_indexer.py:165`

**错误**:
```python
"segment_idx": embed_idx,  # ❌ 使用了 embed_idx 而非 chunk_idx
```

### 根本原因

ASR semantic indexing 是**稀疏的**：只有有文本内容且 `semantic_eligible=True` 的 chunk 才生成 embedding。

**示例场景**:
```
原始 ASR chunks:  [chunk0, chunk1, chunk2, chunk3, chunk4]
                          ↓ chunk1, chunk3 为空或不 eligible
索引的 chunks:    [chunk0, chunk2, chunk4]
embedding_chunk_indices: [0, 2, 4]  # 保留原始索引
embeddings:       [emb0, emb2, emb4]
```

**错误映射（使用 embed_idx）**:
```python
embed_idx=0, chunk_idx=0 → segment_idx=0, text=texts[0] ✓
embed_idx=1, chunk_idx=2 → segment_idx=1, text=texts[2] ✗  # 应该 segment_idx=2
embed_idx=2, chunk_idx=4 → segment_idx=2, text=texts[4] ✗  # 应该 segment_idx=4
```

**导致结果（Milvus 数据错误）**:
```
segment_idx | text      | 实际对应 chunk
------------|-----------|---------------
0           | texts[0]  | chunk 0 ✓
1           | texts[2]  | chunk 2 ✗ (存储位置错误)
2           | texts[4]  | chunk 4 ✗ (存储位置错误)
```

**Phase 2 读取（基于错误数据）**:
```python
_texts_from_milvus() 按 segment_idx 排序构建稠密列表
result = ["texts[0]", "texts[2]", "texts[4]"]  # 长度=3

Speaker service 期望访问:
texts[0] = chunk 0 的文本 ✓
texts[1] = chunk 1 的文本 ✗ (实际得到 chunk 2 的文本)
texts[2] = chunk 2 的文本 ✗ (实际得到 chunk 4 的文本)
texts[3] = chunk 3 的文本 ✗ (IndexError: 列表长度不足)
texts[4] = chunk 4 的文本 ✗ (IndexError: 列表长度不足)
```

---

## 影响范围

### ❌ 受影响功能

1. **Speaker 功能** - 🔴 **严重受影响**
   - `video_speakers()` 返回的 utterance 文本会错位
   - `voice_search_vectors()` 返回的搜索结果文本会错位
   - 可能导致 IndexError（如果访问超出范围的索引）

### ✅ 不受影响功能

1. **Visual 检索** - Visual indexing 不是稀疏的
2. **ASR 检索** - `search.py` 中的 ASR 检索不依赖 `segment_idx`，直接使用 Milvus rows
3. **OCR/Face 检索** - 不使用 ASR texts

---

## 修复方案

### 代码修复

**文件**: `backend/app/indexing/milvus_indexer.py:165`

**修改**:
```python
# 修改前
"segment_idx": embed_idx,

# 修改后
"segment_idx": chunk_idx,  # Fixed: use chunk_idx to maintain original ASR chunk index
```

**完整上下文**:
```python
class AsrMilvusIndexer:
    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        # ... 省略前面代码 ...
        for embed_idx, chunk_idx in enumerate(indices):
            chunk_idx = int(chunk_idx)
            if chunk_idx < 0 or chunk_idx >= len(times):
                continue
            rows.append({
                "pk":            asr_pk(ctx.video_id, ctx.asset_version, embed_idx, model_ver),
                "video_id":      ctx.video_id,
                "asset_version": ctx.asset_version,
                "model_version": model_ver,
                "segment_idx":   chunk_idx,  # ✅ 修复：使用 chunk_idx
                "start_ms":      int(times[chunk_idx, 0]),
                "end_ms":        int(times[chunk_idx, 1]),
                "text":          texts[chunk_idx][:2000],
                "embedding":     embeddings[embed_idx].tolist(),
            })
        return _upsert_batched(col, rows)
```

---

## 修复后行为

**正确映射（使用 chunk_idx）**:
```python
embed_idx=0, chunk_idx=0 → segment_idx=0, text=texts[0] ✓
embed_idx=1, chunk_idx=2 → segment_idx=2, text=texts[2] ✓
embed_idx=2, chunk_idx=4 → segment_idx=4, text=texts[4] ✓
```

**Milvus 数据（正确）**:
```
segment_idx | text      | 对应 chunk
------------|-----------|----------
0           | texts[0]  | chunk 0 ✓
2           | texts[2]  | chunk 2 ✓
4           | texts[4]  | chunk 4 ✓
```

**Phase 2 读取（正确）**:
```python
_texts_from_milvus() 按 segment_idx 排序构建稠密列表
result = ["texts[0]", "", "texts[2]", "", "texts[4]"]  # 长度=5 ✓

Speaker service 访问:
texts[0] = chunk 0 的文本 ✓
texts[1] = ""            ✓ (chunk 1 为空，正确)
texts[2] = chunk 2 的文本 ✓
texts[3] = ""            ✓ (chunk 3 为空，正确)
texts[4] = chunk 4 的文本 ✓
```

---

## 验证测试

### 1. 代码层验证

**测试文件**: `backend/tests/test_phase2_fix.py`

**测试结果**: ✅ 全部通过

```bash
$ python tests/test_phase2_fix.py
======================================================================
Phase 2 Fix Verification
======================================================================

Testing segment_idx fix in AsrMilvusIndexer...
  [OK] segment_idx fix verified: 5 texts correctly aligned
Testing Phase 2 _texts_from_milvus() with fixed segment_idx...
  [OK] Phase 2 logic works correctly with fixed data

======================================================================
[SUCCESS] All tests passed!
======================================================================

Summary:
  - segment_idx fix verified (uses chunk_idx)
  - Phase 2 _texts_from_milvus() works correctly
  - Speaker service will get correct text alignment

Next steps:
  1. Rebuild ASR indexes to fix existing Milvus data
  2. Run integration tests with live Milvus
  3. Deploy to staging for validation
```

### 2. 逻辑验证

**简化验证脚本**:
```python
embedding_chunk_indices = [0, 2, 4]
texts_all = ['chunk0', '', 'chunk2', '', 'chunk4']

# 修复后：segment_idx = chunk_idx
rows_fixed = [(chunk_idx, texts_all[chunk_idx]) 
              for embed_idx, chunk_idx in enumerate(embedding_chunk_indices)]
max_idx = max(r[0] for r in rows_fixed)
result = [next((r[1] for r in rows_fixed if r[0] == i), '') 
          for i in range(max_idx + 1)]

assert result == texts_all  # ✅ 通过
```

---

## 数据迁移需求

⚠️ **已写入的 Milvus 数据需要重建**

### 影响

- 当前 Milvus ASR collection 中的所有数据的 `segment_idx` 字段都是**错误的**
- 必须重新索引才能修复

### 迁移方案

**选项 1: 删除并重建（推荐）**

```python
# 1. 清理旧数据
from app.indexing.milvus_client import get_milvus_client
client = get_milvus_client()
client.collection_for("asr").delete(expr='video_id != ""')  # 删除所有记录

# 2. 重建索引
# 设置 MILVUS_WRITE_ENABLED=true
# 重新运行所有视频的 ASR 索引构建
```

**选项 2: 选择性重建**

```python
# 只重建需要 Speaker 功能的视频
video_ids_with_speaker = [...]  # 有 Speaker 索引的视频列表

for video_id in video_ids_with_speaker:
    # 删除该视频的旧 ASR 数据
    client.collection_for("asr").delete(expr=f'video_id == "{video_id}"')
    
    # 重新索引该视频
    # rebuild_asr_index(video_id)
```

---

## 时间线

| 时间 | 事件 |
|------|------|
| 2026-07-21 早期 | Phase 2 初始实施（包含 bug） |
| 2026-07-21 下午 | 审核发现 bug，更新 PHASE2_VERIFICATION_RESULT.md |
| 2026-07-21 下午 | Bug 修复，代码验证通过 |

---

## 修改文件

```
M  backend/app/indexing/milvus_indexer.py      # 修复 segment_idx = chunk_idx
A  backend/tests/test_phase2_fix.py             # 新增修复验证测试
A  backend/tests/verify_segment_idx_bug.py      # Bug 演示脚本
A  docs/PHASE2_BUG_FIX_REPORT.md                # 本报告
```

---

## 部署建议

### 🚨 重要提示

**不要在修复前启用 `MILVUS_READ_ENABLED=true`！**

如果启用，Speaker 功能会显示错误的文本。

### 部署步骤

1. **应用代码修复**
   ```bash
   git pull  # 获取修复后的代码
   ```

2. **清理旧的 Milvus ASR 数据**
   ```bash
   # 删除所有 ASR collection 数据
   # 或者只删除有 Speaker 索引的视频
   ```

3. **重建 ASR 索引**
   ```bash
   export MILVUS_WRITE_ENABLED=true
   # 重新运行 ASR indexing
   ```

4. **验证修复**
   ```bash
   # 测试 Speaker 功能
   # 检查 utterance 文本是否正确
   ```

5. **启用 Milvus 读取**
   ```bash
   export MILVUS_READ_ENABLED=true
   # 逐步灰度
   ```

---

## 总结

### ✅ 修复完成

- ✅ Bug 已识别和确认
- ✅ 代码已修复（1 行修改）
- ✅ 修复已验证（测试通过）
- ✅ 文档已更新

### ⚠️ 待处理

- ⚠️ 需要重建 Milvus ASR 数据
- ⚠️ 需要集成测试验证
- ⚠️ 需要更新 PHASE2_COMPLETION_REPORT.md

### 教训

1. **稀疏索引需要特别注意** - ASR semantic indexing 的稀疏性是核心问题
2. **验证数据对齐** - 在实施阶段就应该验证 `segment_idx` 的映射正确性
3. **端到端测试** - 需要完整的集成测试覆盖 Speaker 功能

---

**修复者**: Claude Code  
**审核**: 待审核  
**部署**: 待部署
