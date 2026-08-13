# 混合模态聚合逻辑详解

**文档创建日期**: 2026-08-11  
**基于代码**: `backend/app/retrieval/search.py` (修复后版本)

---

## 📋 核心问题回答

### Q: 混合模态是先各自聚合再跨模态合并，还是一次性合并？

**A: 混合策略 — OCR 先聚合，然后非 OCR 候选逐个尝试合并到已有组**

具体流程:
1. **OCR 独立聚合** — 使用 `_groups_ocr_score_first()` 按分数优先算法形成纯 OCR 组
2. **非 OCR 逐个择优合并** — face/visual/asr 按时间排序后，遍历所有现存组（含 OCR 组），找到第一个满足合并条件的组加入，或独立成组
3. **最终按时间排序** — 保证返回的组列表按视频+时间递增

---

## 🔄 完整流程图

```
输入: candidates = [ocr1, ocr2, face1, asr1, visual1, ...]

┌─────────────────────────────────────────────────────────────┐
│ 1. _groups(candidates, gap=2.0, max_duration=15)            │
└─────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────┴───────────────┐
         │ 按 modality 分流               │
         └───────────────┬───────────────┘
                         ↓
         ┌───────────────┴───────────────┐
         │ ocr_candidates = [ocr1, ocr2] │
         │ non_ocr = [face1, asr1, ...]  │
         └───────────────┬───────────────┘
                         ↓
         ┌───────────────┴───────────────┐
         │ 场景判断                       │
         └───────────────┬───────────────┘
                         ↓
                ┌────────┴────────┐
                │                 │
         纯 OCR 场景          混合模态场景
         (non_ocr为空)         (non_ocr非空)
                │                 │
                ↓                 ↓
    直接返回                groups = []
    _groups_ocr_score_first      ↓
    的结果                  ┌─────────────────────┐
                            │ 2. OCR 先聚合        │
                            │ ocr_groups =        │
                            │  _groups_ocr_score_ │
                            │   first(ocr_cands)  │
                            │                     │
                            │ 返回:               │
                            │ [                   │
                            │   [ocr1, ocr3],    │ ← 组1: 种子ocr1(分0.95)
                            │   [ocr2],          │ ← 组2: 种子ocr2(分0.90)
                            │ ]                  │
                            └─────────┬───────────┘
                                      ↓
                            groups.extend(ocr_groups)
                            groups = [
                              [ocr1, ocr3],  ← 组1: 5-7s
                              [ocr2],        ← 组2: 10-12s
                            ]
                                      ↓
                            ┌─────────────────────┐
                            │ 3. 非OCR逐个合并     │
                            │ 按(video, time)排序  │
                            └─────────┬───────────┘
                                      ↓
                        for candidate in [face1(6-7s), asr1(11-12s), ...]:
                                      ↓
                        ┌─────────────┴──────────────┐
                        │ 遍历所有组找第一个可合并组  │
                        │ target = next(            │
                        │   g for g in groups       │
                        │   if _should_merge(g, c)  │
                        │ )                         │
                        └─────────────┬──────────────┘
                                      ↓
                              ┌───────┴───────┐
                              │ 找到?         │
                              └───────┬───────┘
                                      ↓
                        ┌─────────────┴─────────────┐
                        │ YES                  NO   │
                        ↓                           ↓
              target.append(candidate)      groups.append([candidate])
                        │                           │
              例: face1合并到组1                 例: 无组匹配，独立成组
              groups = [                       groups = [
                [ocr1, ocr3, face1],             [ocr1, ocr3, face1],
                [ocr2],              继续循环→    [ocr2, asr1],
              ]                                   [visual1],
                                                ]
                                      ↓
                            ┌─────────────────────┐
                            │ 4. 按时间排序        │
                            │ sorted(groups,      │
                            │   key=lambda g:     │
                            │     min(start_time))│
                            └─────────┬───────────┘
                                      ↓
                            返回最终组列表
```

---

## 🔍 关键函数详解

### 1. `_groups(candidates, gap, max_duration)` — 顶层聚合入口

**位置**: L705-745

**职责**: 协调整个聚合流程

**算法**:
```python
def _groups(candidates, gap, max_duration):
    # 步骤1: 分流
    ocr_candidates = [c for c in candidates if c.modality == "ocr"]
    non_ocr_candidates = [c for c in candidates if c.modality != "ocr"]
    
    groups = []
    
    # 步骤2: OCR 先聚合
    if ocr_candidates:
        if not non_ocr_candidates:
            # 纯OCR场景，直接返回
            return _groups_ocr_score_first(ocr_candidates)
        else:
            # 混合场景，OCR先聚合成组
            ocr_groups = _groups_ocr_score_first(ocr_candidates)
            groups.extend(ocr_groups)  # ← 注意：这些组按分数顺序排列，非时间
    
    # 步骤3: 非OCR逐个择优合并
    for candidate in sorted(non_ocr_candidates, key=lambda i: (i.video_id, i.start_time)):
        # 遍历所有现存组（包括OCR组）
        target_group = next(
            (g for g in groups if _should_merge(g, candidate, gap, max_duration)),
            None,
        )
        if target_group is not None:
            target_group.append(candidate)  # 合并到现存组
        else:
            groups.append([candidate])      # 独立成新组
    
    # 步骤4: 按时间排序（因为OCR组按分数插入，混合后可能乱序）
    return sorted(groups, key=lambda g: (g[0].video_id, min(item.start_time for item in g)))
```

**关键点**:
- **为何 OCR 先聚合?** OCR 使用特殊的"分数优先"算法（从高分种子向时间两边扩展），与其他模态的"时间优先"算法不兼容，必须独立处理
- **为何非 OCR 逐个合并而非批量?** 每个非 OCR 候选的合并规则不同（face 看 cosine 带宽、visual 看 overlap），需要动态判断
- **为何最终排序?** OCR 组按**分数**顺序返回（高分组在前），非 OCR 合并后组的相对位置可能乱序，必须重新按时间排序保证展示稳定

---

### 2. `_groups_ocr_score_first(ocr_candidates)` — OCR 独立聚合

**位置**: L605-702

**职责**: 将 OCR 候选按"分数优先、时间约束"聚合

**算法特点**:
- **种子选择**: 按分数降序，未聚合的最高分作为种子
- **双向扩展**: 从种子向时间两边扩展，加入分数兼容（≥ 种子分数*0.9 且 ≥ 种子分数-0.1）且时间相邻（gap ≤ 0.35s）的帧
- **分数锚定**: 阈值基于**种子分数**固定，不随扩展滑坡
- **返回顺序**: 按种子选择顺序返回，即**分数从高到低**

**示例**:
```python
输入 OCR 候选:
  ocr1: 5-7s, 分数 0.95
  ocr2: 10-12s, 分数 0.90
  ocr3: 6-7s, 分数 0.92

执行:
  1. 选种子 ocr1(0.95)，阈值 = max(0.95*0.9, 0.95-0.1) = 0.855
  2. 向右扩展，ocr3(6-7s, 0.92) 满足：分数 0.92 ≥ 0.855 ✓，gap = 6-7 = 0 ✓ → 加入
  3. ocr1组完成: [ocr1, ocr3]
  4. 选种子 ocr2(0.90)，无可扩展 → [ocr2]

返回: [[ocr1, ocr3], [ocr2]]  ← 按种子分数顺序，非时间顺序
```

---

### 3. `_should_merge(group, candidate, gap, max_duration)` — 合并判断

**位置**: L549-602

**职责**: 判断候选是否应合并到组内

**核心逻辑** (分支树):
```
_should_merge(group, candidate, gap, max_duration)
    ↓
1. 同一视频? → NO → return False
    ↓ YES
2. 合并后总时长 ≤ max_duration? → NO → return False
    ↓ YES
3. 计算邻接状态:
   - overlaps = 候选与组时间重叠
   - gap_between = max(候选在组前的间隙, 候选在组后的间隙, 0)
   - near = gap_between ≤ gap
    ↓
4. 模态分支判断:

   ├─ candidate=ocr AND group=纯OCR
   │  → return _should_merge_ocr_only(...)  ← 死代码分支，实际不会走到
   │
   ├─ candidate=face AND group=纯face
   │  → return near AND _face_scores_compatible(group, candidate)
   │     cosine 带宽检查: |candidate.cosine - group_best| ≤ 0.15
   │                      AND |candidate.cosine - group_worst| ≤ 0.15
   │
   ├─ candidate=visual AND group=纯visual
   │  → return overlaps  ← 只合并重叠，不链式合并邻近
   │
   ├─ candidate=visual OR group包含visual
   │  → return overlaps OR (near AND group有非visual模态)
   │     visual 与其他模态混合时，允许邻近合并
   │
   └─ 其他（face/asr 与 OCR 混合等）
      → return near  ← 纯时间邻接，无分数约束
```

**关键点**:
- **纯模态有专属规则**: 
  - 纯 face → cosine 带宽
  - 纯 visual → 只合并重叠
  - 纯 OCR → (已由 `_groups_ocr_score_first` 处理，此分支死代码)
- **混合模态宽松**: face+ocr、asr+ocr 等混合组只看时间邻接 `near`，无分数约束
- **双向间隙**: `gap_between = max(候选在组前, 候选在组后, 0)` 修复了旧版单向判断的 bug

---

## 🎯 混合模态的实际行为

### 场景1: OCR + Face 混合

```python
输入候选:
  ocr1: 5-7s, 分数 0.95
  ocr2: 10-12s, 分数 0.90
  face1: 6-7s, cosine 0.72

执行流程:
  1. OCR 先聚合 → groups = [[ocr1], [ocr2]]
  
  2. face1(6-7s) 遍历所有组:
     - 尝试 ocr1(5-7s):
       _should_merge([ocr1], face1, gap=2.0):
         同视频 ✓
         总时长 max(7,7)-min(5,6)=2 ≤ 15 ✓
         gap_between = max(6-7, 5-7, 0) = 0 ≤ 2 ✓ → near=True
         group={ocr}, candidate=face → 非face-only分支
         group包含visual? NO
         → return near = True ✓
       → 合并到 ocr1 组
  
  3. 最终 groups = [[ocr1, face1], [ocr2]]
  
  4. 按时间排序 → [[ocr1, face1], [ocr2]] (已按时间)
```

**关键**: face1 与 OCR 组合并时**不检查 cosine 带宽**（因为 `group_modalities != {"face"}`），只要时间邻接就合并。

---

### 场景2: 修复前的 Bug 场景

```python
输入候选:
  ocr1: 5-7s, 分数 0.95
  ocr2: 10-12s, 分数 0.90
  face1: 6-7s, cosine 0.72

旧算法（只看 groups[-1]）:
  1. OCR 先聚合 → groups = [[ocr1], [ocr2]]  ← 按分数顺序，非时间
  
  2. face1(6-7s) 只比对 groups[-1]=ocr2(10-12s):
     _should_merge([ocr2(10-12s)], face1(6-7s)):
       gap_between = max(6-12, 10-7, 0) = 3 ≤ 2? NO
       → return False
     → 独立成新组
  
  3. 最终 groups = [[ocr1], [ocr2], [face1]]
  
问题: face1(6-7s) 本应与 ocr1(5-7s) 合并，但因为只看了 groups[-1]=ocr2 而被错过

新算法（遍历所有组）:
  face1 会先尝试 ocr1，发现可合并 → 合并成功 ✓
```

---

### 场景3: 纯 Face 场景（无 OCR）

```python
输入候选:
  face1: 5-7s, cosine 0.72
  face2: 8-9s, cosine 0.70
  face3: 10-11s, cosine -0.01

执行流程:
  1. ocr_candidates = [] → 跳过 OCR 聚合
  
  2. 非OCR逐个合并:
     - face1(5-7s) → groups为空 → 独立成组: [[face1]]
     
     - face2(8-9s) 遍历:
       _should_merge([face1], face2):
         near: gap_between = 8-7 = 1 ≤ 2 ✓
         group={face}, candidate=face → face-only分支
         _face_scores_compatible([face1(0.72)], face2(0.70)):
           |0.70 - 0.72| = 0.02 ≤ 0.15 ✓
         → return True
       → 合并: [[face1, face2]]
     
     - face3(10-11s) 遍历:
       _should_merge([face1, face2], face3):
         near: gap_between = 10-9 = 1 ≤ 2 ✓
         _face_scores_compatible([0.72, 0.70], face3(-0.01)):
           |-0.01 - 0.72| = 0.73 > 0.15 ✗
         → return False
       → 独立成组: [[face1, face2], [face3]]
  
  3. 按时间排序 → [[face1, face2], [face3]]
```

**关键**: 纯 face 场景有 cosine 护栏，防止异人合并。

---

## 🔄 融合阶段 (_fuse_candidate_groups)

**位置**: L748-821

**职责**: 将组转换为 SearchResult，计算加权分数

**算法**:
```python
for group in _groups(candidates, merge_gap, max_result_seconds):
    # 1. 每个模态取组内最高分
    best_by_modality = 
    for item in group:
        best_by_modality[item.modality] = max(
            best_by_modality.get(item.modality, -1), 
            item.score
        )
    
    # 2. 加权融合（权重: face 0.55, visual 0.30, ocr 0.20, asr 0.15）
    weights = {"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}
    denominator = sum(weights.get(name, 1) for name in best_by_modality)
    score = sum(
        weights.get(name, 1) * value 
        for name, value in best_by_modality.items()
    ) / denominator
    
    # 3. 时间边界 = 组内所有候选的时间范围
    start_time = min(item.start_time for item in group)
    end_time = max(item.end_time for item in group)
    
    # 4. above_threshold = 组内任一候选超阈值
    above_threshold = any(item.above_threshold for item in group)
    
    # 5. evidence = 组内所有候选的证据列表
    evidence = [_serialize_evidence(item) for item in group]
```

**示例**:
```python
组 = [
  ocr1(5-7s, 分数 0.95, above_threshold=True, evidence="OCR: 目标文本"),
  face1(6-7s, 分数 0.99, above_threshold=True, evidence="face cosine=0.72")
]

融合:
  best_by_modality = {"ocr": 0.95, "face": 0.99}
  score = (0.20*0.95 + 0.55*0.99) / (0.20+0.55) = 0.976
  start_time = min(5, 6) = 5
  end_time = max(7, 7) = 7
  above_threshold = True OR True = True
  evidence = ["OCR: 目标文本", "face cosine=0.72"]

返回 SearchResult:
  video_id: ...
  start_time: 5.0
  end_time: 7.0
  score: 0.976
  modalities: ["face", "ocr"]
  above_threshold: True
  evidence: [...]
```

---

## 🎯 设计优缺点

### ✅ 优点

1. **OCR 特殊优化**: 分数优先算法防止低分拖长高分片段（帧级精细控制）
2. **模态隔离**: 各模态的合并规则独立（face 看 cosine、visual 看 overlap）
3. **灵活混合**: 混合模态放宽约束，允许不同模态互补

### ⚠️ 缺点与权衡

1. **混合时丢失分数护栏**: 
   - face+ocr 混合组不检查 face 的 cosine 带宽
   - 可能导致低分 face 被 OCR 锚定的时刻"拖入"
   - **设计意图**: OCR 锚定了文本命中时刻，face 作为辅助证据，不需要独立门槛
   
2. **OCR 组按分数排序引入复杂性**: 
   - 需要最后重新按时间排序
   - 非 OCR 合并时需要遍历所有组（O(n*m)复杂度）
   - **设计意图**: OCR 分数优先聚合的收益（高分片段不被低分拖长）大于复杂性代价

3. **非 OCR 合并顺序敏感**: 
   - 按时间排序后逐个处理，先到先得
   - face1(6s) 先合并到 ocr1(5-7s) 后，face2(6.5s) 再来可能因 face1 已在组内而被 cosine 检查拒绝
   - **设计意图**: 贪心算法简单高效，复杂场景下的最优解需要全局搜索（代价高）

---

## 📌 总结

**混合模态聚合 = 两阶段 + 择优合并**

1. **第一阶段**: OCR 独立聚合（分数优先，形成纯 OCR 组）
2. **第二阶段**: 非 OCR 候选按时间排序，逐个遍历所有组（含 OCR 组），找到第一个满足合并条件的组加入，或独立成组
3. **最终排序**: 按视频+时间重新排序保证展示稳定

**关键特性**:
- **模态专属规则**: 纯 face 看 cosine、纯 visual 看 overlap、纯 OCR 看分数+时间
- **混合模态宽松**: face+ocr / asr+ocr 等只看时间邻接，无分数约束
- **双向间隙判断**: 修复旧版单向判断的 bug（候选在组前也判为相邻）
- **遍历择优**: 修复旧版只看 groups[-1] 的 bug（混合模态下 OCR 组按分数排序，早期组被跳过）
