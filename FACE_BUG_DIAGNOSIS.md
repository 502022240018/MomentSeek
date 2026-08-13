# Face 检索 Bug 诊断报告

## 🔴 问题描述

用户在前端测试 Face 检索时发现：

1. **聚合过于宽松**：召回的都是长片段（10-20s），片段内包含大量非目标人脸
2. **显示分数不一致**：
   - 前端显示"召回分数 99%, 92%, 91%"
   - 但展开详情看到：`[milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值`
   - 尽管显示"低于阈值"，片段仍在高分区，未被划分到"低于阈值"展示状态

## 🔍 根因分析

### 问题 1：聚合过于宽松 - 缺少分数兼容性检查

**定位**：`backend/app/retrieval/search.py` L485-513 `_should_merge` 函数

**现有逻辑**：
```python
def _should_merge(group: list[Candidate], candidate: Candidate, gap: float, max_duration: float) -> bool:
    # ...
    near = candidate.start_time <= group_end + gap  # gap=2 秒
    
    # OCR-only 有分数兼容性检查
    if candidate.modality == "ocr" and group_modalities == {"ocr"}:
        return _should_merge_ocr_only(group, candidate)
    
    # Visual 有特殊规则（只合并 overlap）
    if candidate.modality == "visual" and group_modalities == {"visual"}:
        return overlaps
    
    # ❌ Face 模态走到这里：只要在 2 秒内，无论分数多低都合并
    return near
```

**问题场景**：
```
时间轴：
10.0s - Face候选A: cosine=0.72, confidence=99%, above_threshold=True
11.5s - Face候选B: cosine=-0.01, confidence=0.4%, above_threshold=False

gap = 11.5 - 10.0 = 1.5 < 2 秒
→ 判定为 near=True
→ 无分数检查，直接合并到同一组
→ 结果片段：10.0-12.0s（包含目标人脸 + 非目标人脸）
```

**对比 OCR 逻辑**（L411-442）：
```python
def _ocr_scores_compatible(group: list[Candidate], candidate: Candidate) -> bool:
    group_scores = [float(item.score) for item in group if item.modality == "ocr"]
    best_score = max(group_scores)
    
    threshold = max(
        best_score * 0.90,  # 90% 比例门槛
        best_score - 0.10,  # 绝对差 0.10
    )
    return float(candidate.score) >= threshold  # ← 分数门槛
```

OCR 有分数兼容性检查，但 **Face 没有**。

---

### 问题 2：显示分数是组内最高分，evidence 包含所有候选

**定位**：`backend/app/retrieval/search.py` L658-696 `_fuse_candidate_groups` 函数

**现有逻辑**：
```python
for group in _groups(candidates, merge_gap, max_result_seconds):
    best_by_modality = {}
    for item in group:
        # ❌ 取组内每个模态的最高分
        best_by_modality[item.modality] = max(best_by_modality.get(item.modality, -1), item.score)
    
    # 用最高分做加权融合
    score = sum(weights.get(name, 1) * value for name, value in best_by_modality.items()) / denominator
    
    result = SearchResult(
        score=score,  # ← 显示的是组内最高分融合结果 = 99%
        above_threshold=any(item.above_threshold for item in group),  # ← 只要有一个超阈值就 True
        evidence=[_serialize_evidence(item) for item in group],  # ← evidence 包含所有候选（含 0.4% 的）
    )
```

**问题**：
- `score` = 组内最高分（99%）→ 前端显示 99%
- `evidence` = 组内**所有**候选（包括 cosine=-0.01 的低分候选）→ 前端展开看到 0.4% 低于阈值文字
- `above_threshold = any(...)` → 只要组内有一个候选超阈值，整个片段就标记为"超阈值"，不会进入"低于阈值"展示区

**这正是用户观察到的现象**：
- 99% 是组内最高分候选（目标人脸）的分数
- "cosine=-0.010 · 0.4% · 低于阈值" 是组内被错误合并进来的非目标人脸候选的 evidence
- 片段整体 `above_threshold=True`（因为目标人脸候选超阈值），所以不进"低于阈值"区

---

## 🎯 根因总结

两个问题是**同一个根因的两个表现**：

**根因：Face 模态的聚合缺少分数兼容性约束**，导致时间相邻但分数差距巨大的候选被合并到同一组。

1. 合并后片段变长（问题1的"长片段"）
2. 合并后组内混入非目标人脸（问题1的"人脸不都是目标"）
3. 显示分数取组内最高分，evidence 显示所有候选 → 分数与文字不符（问题2）
4. `above_threshold=any()` → 低分候选无法进入"低于阈值"展示区（问题2）

**为什么优化后才暴露**：
- 优化前 `ann_limit = limit * 2`（2倍扩召回）+ Python 重打分，召回集经过重排截断
- 优化后 `ann_limit = limit * 1`，且 DiskANN 是近似索引，召回的 track 分布可能更宽
- 但**核心问题（Face 聚合无分数门槛）在优化前就存在**，只是 IVF_FLAT 精确重排 + 2倍召回下不明显

---

## 🔧 修复方案

### 修复 1：Face 聚合增加分数兼容性检查（对齐 OCR 策略）

在 `_should_merge` 中为 face-only 场景增加分数门槛，避免低分候选拖长高分片段。

### 修复 2：显示分数与 evidence 一致性

融合后的 `above_threshold` 和 evidence 应保持一致性。修复 1 之后，被错误合并的低分候选不再进入同组，问题 2 自然大幅缓解。但仍需确保：
- Face track 本身是独立语义单元，不应盲目按时间链式合并

---

## ✅ 已实施修复

### 修复 1：新增 `_face_scores_compatible` 函数

**位置**：`backend/app/retrieval/search.py` L416-454

**实现**：
```python
_FACE_MERGE_MAX_COSINE_DROP = 0.15  # 阈值：组内最佳 cosine 与候选 cosine 差距上限

def _face_scores_compatible(group: list[Candidate], candidate: Candidate) -> bool:
    """Face-only 合并时，避免不同人脸（cosine 差距大）被拼进同一片段。

    规则：
    - candidate 必须是 face 模态；
    - group 里必须已有 face 命中；
    - candidate 的 cosine（raw_score）不能比 group 内最佳 face cosine 低太多，
      也不能高太多——对称带宽，保证同组 track 属于同一相似度层级。

    raw_score 缺失时（理论上 face 恒有）退化为仅按时间合并，返回 True。
    """
    if candidate.modality != "face":
        return False

    group_cosines = [
        float(item.raw_score)
        for item in group
        if item.modality == "face" and item.raw_score is not None
    ]
    if not group_cosines:
        return False
    if candidate.raw_score is None:
        return True

    cand_cosine = float(candidate.raw_score)
    best_cosine = max(group_cosines)
    worst_cosine = min(group_cosines)
    # 对称带宽：candidate 与组内最强/最弱 face 都不能相差超过阈值
    return (
        cand_cosine >= best_cosine - _FACE_MERGE_MAX_COSINE_DROP
        and cand_cosine <= worst_cosine + _FACE_MERGE_MAX_COSINE_DROP
    )
```

**设计要点**：
1. **对称带宽约束**：防止高分 track 被并入低分组，或反之
2. **阈值 0.15**：参考 ArcFace 语义层级（cosine 0.70 vs 0.55 是不同置信度层级）
3. **raw_score=None 兜底**：理论上 face 恒有 raw_score，但保留兜底逻辑防御性编程

### 修复 2：`_should_merge` 新增 face-only 分支

**位置**：`backend/app/retrieval/search.py` L547-553

**实现**：
```python
def _should_merge(group: list[Candidate], candidate: Candidate, gap: float, max_duration: float) -> bool:
    # ... 时间检查 ...
    near = candidate.start_time <= group_end + gap

    # ✅ 新增：Face-only 合并须额外满足 cosine 带宽约束
    if candidate.modality == "face" and group_modalities == {"face"}:
        return near and _face_scores_compatible(group, candidate)

    # Visual-only 分支
    if candidate.modality == "visual" and group_modalities == {"visual"}:
        return overlaps
    # ... 其他分支 ...
```

**逻辑变化**：
- **修复前**：face-only 走到 `return near`（L554），只要时间相邻就合并
- **修复后**：face-only 走到新分支（L551-552），须同时满足 `near=True` **且** `_face_scores_compatible=True`

**不影响混合模态**：
- 当组内已有 OCR/ASR/Visual 时，`group_modalities != {"face"}`，不走 face-only 分支
- 混合模态场景中，face 作为辅助证据，由其他模态锚定片段，不需要独立的分数门槛

### 验证结果

**容器内逻辑验证**（2026-08-11）：
```
✓ PASS  用户bug场景(0.72 vs -0.01)应拒绝合并
✓ PASS  同人相近(0.72 vs 0.68)应合并
✓ PASS  _face_scores_compatible大跌拒绝
✓ PASS  _face_scores_compatible小跌接受
✓ PASS  混合模态(face+ocr)不检查cosine仍合并
✓ PASS  raw_score=None兜底返回True
```

**服务状态**：
- 修改文件已复制到容器 `/app/backend/app/retrieval/search.py`
- 容器已重启（docker restart momentseek-0829-platform）
- Health 接口正常：http://127.0.0.1:8100/api/health → `{"status":"ok"}`

---

## 🧪 测试覆盖

新增测试文件：`backend/tests/test_face_merge_fix.py`

**测试用例**：
1. `test_face_scores_compatible_same_cosine` - 相同 cosine 应兼容
2. `test_face_scores_compatible_small_drop` - 小幅下降（0.07）应兼容
3. `test_face_scores_compatible_large_drop_rejected` - 大幅下降（0.73）应拒绝
4. `test_face_scores_compatible_symmetric_bandwidth` - 对称带宽约束
5. `test_face_scores_compatible_non_face_candidate` - 非 face 候选返回 False
6. `test_face_scores_compatible_no_raw_score` - 缺 raw_score 兜底
7. `test_should_merge_face_only_rejects_low_cosine` - Face-only 拒绝低分
8. `test_should_merge_face_only_accepts_compatible_cosine` - Face-only 接受兼容分
9. `test_should_merge_face_mixed_modality_no_score_check` - 混合模态不检查
10. `test_regression_user_bug_scenario` - 用户 bug 场景回归测试

---

## 📊 预期效果

### 修复前
```
召回片段 1: 10.0-12.5s (2.5秒)
  显示分数: 99%
  above_threshold: True (组内有目标人脸超阈值)
  evidence:
    - [milvus] face cosine=0.720 · confidence=99% · absolute_hit
    - [milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值  ← 混入非目标人脸
```

### 修复后
```
召回片段 1: 10.0-11.0s (1.0秒)
  显示分数: 99%
  above_threshold: True
  evidence:
    - [milvus] face cosine=0.720 · confidence=99% · absolute_hit

召回片段 2: 11.5-12.5s (1.0秒)
  显示分数: 0.4%
  above_threshold: False
  evidence:
    - [milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值
  → 进入"低于阈值"展示区
```

**关键改进**：
1. ✅ 片段变短（1.0s vs 2.5s），更精准定位目标人脸出现时刻
2. ✅ 显示分数与 evidence 一致（99% 对应 99% 的 evidence，0.4% 对应 0.4% 的 evidence）
3. ✅ 低分片段正确进入"低于阈值"区（`above_threshold=False` → 前端分类显示）
4. ✅ 不影响混合模态场景（face+ocr/asr 时仍按原逻辑合并）

---

## 🔍 对比 OCR 策略

| 维度 | OCR | Face（修复前） | Face（修复后） |
|------|-----|---------------|---------------|
| **语义单元** | 单帧文本片段 | 同人连续 track | 同人连续 track |
| **合并 gap** | 0.35s（帧级） | 2.0s（默认） | 2.0s（默认） |
| **分数门槛** | ✅ 90% 比例 / 0.10 绝对差 | ❌ 无 | ✅ 0.15 cosine 带宽 |
| **专用函数** | `_ocr_scores_compatible` | 无 | `_face_scores_compatible` |
| **专用分支** | `if ocr and {"ocr"}` | 无 | `if face and {"face"}` |

**差异点**：
- OCR 用 `confidence`（0-1）做门槛，Face 用 `cosine`（-1 到 1）做门槛
- OCR 只检查下界（candidate >= best*0.9），Face 检查上下界（对称带宽）
- Face 阈值 0.15 比 OCR 的 0.10 更宽松（因为 face cosine 分布更宽，且 track 本身已是聚合单元）

---

## ⚠️ 注意事项

### 1. 阈值调优
`_FACE_MERGE_MAX_COSINE_DROP = 0.15` 是初始值，可能需要根据真实数据分布调整：
- **过小**（如 0.05）：同一人不同角度/光照的 track 无法合并，片段过碎
- **过大**（如 0.30）：相似但非同人的 face 仍可能合并

**建议**：在真实检索场景中观察召回结果，统计"同人不同 track 的 cosine 分布"和"不同人的 cosine 分布"，找到最优分界点。

### 2. 混合模态不受影响
修复只针对 **face-only** 场景（`group_modalities == {"face"}`）。当组内已有 OCR/ASR/Visual 时：
- face 作为辅助证据，由其他模态锚定片段
- 不走 face-only 分支，不检查 cosine 兼容性
- 低分 face 仍可能被并入混合片段（这是符合预期的——OCR 命中锚定了时刻，face 提供补充证据）

### 3. DiskANN 近似性
修复后，低分 face track 会单独成为一个片段（`above_threshold=False`）。由于 DiskANN 是近似索引，召回集可能包含更多低分候选。这是正常现象：
- 修复前：低分候选被合并到高分片段，"隐藏"在高分片段的 evidence 里
- 修复后：低分候选独立成段，前端正确划分到"低于阈值"区

**不是召回变差，而是显示更诚实**。

### 4. Face track 本身的质量
本修复假设 **face track 本身是可信的语义单元**（同一 track 内的帧都是同一人）。如果上游 `faces.py` 的 track 构建逻辑有问题（把不同人的帧合并成一个 track），则本修复无法解决该问题。

**依赖**：`backend/app/execution/faces.py` L55/85/134 的 track 构建逻辑正确。

---

## 🚀 部署状态

- ✅ 代码已修改：`backend/app/retrieval/search.py`
- ✅ 测试已创建：`backend/tests/test_face_merge_fix.py`
- ✅ 容器已更新：文件已复制到 `momentseek-0829-platform:/app/backend/app/retrieval/search.py`
- ✅ 服务已重启：docker restart momentseek-0829-platform
- ✅ 健康检查通过：http://127.0.0.1:8100/api/health → `{"status":"ok"}`
- 🔄 待用户前端验证：检索目标人脸，观察召回片段长度、分数/文字一致性、低于阈值片段是否正确划分

---

## 🎯 验收标准

用户在前端测试时，应观察到以下改进：

1. **片段长度缩短**：
   - 修复前：10-20s 长片段（混入多个 track）
   - 修复后：1-5s 短片段（单个或相邻的同人 track）

2. **分数与文字一致**：
   - 修复前：显示 99%，但 evidence 有 "cosine=-0.01 · 0.4% · 低于阈值"
   - 修复后：显示 99% 的片段，evidence 全是高分项；显示 0.4% 的片段，evidence 全是低分项

3. **低于阈值片段正确划分**：
   - 修复前：低分 face 混入高分片段，整段停留在高分区
   - 修复后：低分 face 独立成段，进入"低于阈值"展示区

4. **召回的人脸都是目标人脸**（在高分区）：
   - 修复前：高分片段内混入非目标人脸
   - 修复后：高分片段内只有目标人脸（cosine > 0.35）
