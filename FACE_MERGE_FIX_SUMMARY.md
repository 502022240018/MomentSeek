# Face 检索聚合修复总结

**修复日期**：2026-08-11  
**问题来源**：用户前端测试发现 Face 检索结果异常  
**根因**：Face 模态的聚合逻辑缺少分数兼容性检查

---

## 🐛 问题现象

1. **召回片段过长**：10-20s 长片段，内含多个不同人脸
2. **分数与文字不符**：
   - 前端显示："召回分数 99%"
   - 展开详情看到：`[milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值`
3. **低分未正确划分**：尽管显示"低于阈值"，片段仍停留在高分区（未进入"低于阈值"展示区）

---

## 🔍 根因

**核心问题**：`_should_merge` 函数对 face-only 场景只检查时间相邻（gap≤2s），**完全不检查分数兼容性**。

**对比其他模态**：
- ✅ OCR：有 `_ocr_scores_compatible` 函数，检查分数门槛（90% 比例 / 0.10 绝对差）
- ✅ Visual：只合并 overlap，不链式合并邻近片段
- ❌ Face：走到 `return near`，只要时间相邻就合并

**导致**：
```
时间轴：
10.0s - Face候选A: cosine=0.72, confidence=99%, above_threshold=True  (目标人脸)
11.5s - Face候选B: cosine=-0.01, confidence=0.4%, above_threshold=False (非目标人脸)

gap = 11.5 - 10.0 = 1.5s < 2s → near=True → 合并到同一组

融合后：
  score = max(0.99, 0.004) = 0.99  ← 取组内最高分
  above_threshold = any([True, False]) = True  ← 只要有一个超阈值
  evidence = [候选A的文字, 候选B的文字]  ← 包含所有候选

结果：
  前端显示 99%，但展开看到 "cosine=-0.01 · 0.4% · 低于阈值"
  片段停留在高分区（above_threshold=True）
```

---

## ✅ 修复方案

### 1. 新增 `_face_scores_compatible` 函数

**位置**：`backend/app/retrieval/search.py` L416-454

**逻辑**：
- 检查候选 face 的 cosine（raw_score）与组内已有 face 的 cosine 是否在同一带宽内
- **对称带宽约束**：`|candidate_cosine - group_best_cosine| ≤ 0.15` 且 `|candidate_cosine - group_worst_cosine| ≤ 0.15`
- 防止高分 track 被并入低分组，或反之

**阈值选择**：
- `_FACE_MERGE_MAX_COSINE_DROP = 0.15`
- 参考 ArcFace 语义层级（cosine 0.70 vs 0.55 是不同置信度层级）
- 比 OCR 的 0.10 更宽松（face cosine 分布更宽，且 track 本身已是聚合单元）

### 2. `_should_merge` 新增 face-only 分支

**位置**：`backend/app/retrieval/search.py` L551-552

**修改前**：
```python
return near  # 只要时间相邻就合并
```

**修改后**：
```python
if candidate.modality == "face" and group_modalities == {"face"}:
    return near and _face_scores_compatible(group, candidate)
```

**关键点**：
- 只针对 **face-only** 场景（`group_modalities == {"face"}`）
- 须同时满足 `near=True`（时间相邻）**且** `_face_scores_compatible=True`（分数兼容）
- 混合模态（face+ocr/asr）不走此分支，不影响现有逻辑

---

## 🧪 验证

### 容器内逻辑验证（2026-08-11）

```bash
$ docker exec momentseek-0829-platform python3 /tmp/verify_face_fix.py
============================================================
  ✓ PASS  用户bug场景(0.72 vs -0.01)应拒绝合并
  ✓ PASS  同人相近(0.72 vs 0.68)应合并
  ✓ PASS  _face_scores_compatible大跌拒绝
  ✓ PASS  _face_scores_compatible小跌接受
  ✓ PASS  混合模态(face+ocr)不检查cosine仍合并
  ✓ PASS  raw_score=None兜底返回True
============================================================
ALL PASS
```

### 单元测试

新增测试文件：`backend/tests/test_face_merge_fix.py`（10 个测试用例）

**核心测试**：
- `test_regression_user_bug_scenario`：用户报告的真实场景（0.72 vs -0.01）应拒绝合并
- `test_should_merge_face_only_accepts_compatible_cosine`：同人相近 cosine 应合并
- `test_should_merge_face_mixed_modality_no_score_check`：混合模态不检查分数

---

## 📊 预期效果

### 修复前
```
召回片段 1: 10.0-12.5s (2.5秒)
  显示分数: 99%
  above_threshold: True
  evidence:
    - [milvus] face cosine=0.720 · confidence=99% · absolute_hit
    - [milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值  ← 问题：混入非目标人脸
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
  above_threshold: False  ← 正确进入"低于阈值"区
  evidence:
    - [milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值
```

**关键改进**：
1. ✅ 片段变短（1.0s vs 2.5s），精准定位目标人脸
2. ✅ 显示分数与 evidence 一致
3. ✅ 低分片段正确划分到"低于阈值"区
4. ✅ 高分片段内只有目标人脸

---

## 🚀 部署状态

| 项目 | 状态 | 说明 |
|------|------|------|
| 代码修改 | ✅ 完成 | `backend/app/retrieval/search.py` |
| 测试编写 | ✅ 完成 | `backend/tests/test_face_merge_fix.py` |
| 容器更新 | ✅ 完成 | 文件已复制到 `momentseek-0829-platform` |
| 服务重启 | ✅ 完成 | `docker restart momentseek-0829-platform` |
| 健康检查 | ✅ 通过 | `http://127.0.0.1:8100/api/health` → `{"status":"ok"}` |
| 前端验证 | 🔄 待用户 | 检索目标人脸，观察召回质量 |

---

## ⚠️ 注意事项

### 1. 阈值调优
`_FACE_MERGE_MAX_COSINE_DROP = 0.15` 是初始值，可能需要根据真实数据分布调整。

**建议**：观察真实检索结果，统计"同人不同 track 的 cosine 分布"和"不同人的 cosine 分布"，找到最优分界点。

### 2. 混合模态不受影响
修复只针对 **face-only** 场景。当组内已有 OCR/ASR/Visual 时：
- face 作为辅助证据，由其他模态锚定片段
- 不检查 cosine 兼容性
- 低分 face 仍可能被并入混合片段（符合预期）

### 3. 显示更诚实
修复后，低分 face track 会单独成为一个片段（`above_threshold=False`）。这不是召回变差，而是**显示更诚实**：
- 修复前：低分候选"隐藏"在高分片段的 evidence 里
- 修复后：低分候选独立成段，前端正确划分

### 4. 依赖上游 track 质量
本修复假设 **face track 本身是可信的语义单元**（同一 track 内的帧都是同一人）。如果上游 `faces.py` 的 track 构建有问题，本修复无法解决。

---

## 📝 相关文档

- 详细诊断报告：`FACE_BUG_DIAGNOSIS.md`
- 测试用例：`backend/tests/test_face_merge_fix.py`
- Face 优化计划：`FACE_OPTIMIZATION_PLAN.md`
- Face 实施记录：`Face_IMPLEMENTATION_RECORD.md`

---

## 🎯 验收标准

用户在前端测试时，应观察到：

1. **片段长度缩短**：1-5s 短片段（vs 修复前的 10-20s）
2. **分数与文字一致**：99% 对应 99% 的 evidence，0.4% 对应 0.4% 的 evidence
3. **低于阈值片段正确划分**：进入"低于阈值"展示区
4. **召回的人脸都是目标人脸**：高分区片段内只有目标人脸

---

**修复完成，等待用户前端验证。**
