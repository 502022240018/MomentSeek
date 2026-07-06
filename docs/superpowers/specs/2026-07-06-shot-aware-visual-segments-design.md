# MomentSeek Shot-Aware Visual Segments Design

日期：2026-07-06

## 背景

MomentSeek 当前 visual 通道采用固定时间分段：

```text
5fps 抽帧 -> 5s bucket -> bucket 内 frame MaxSim -> 返回固定 5s 片段
```

这个方案简单稳定，也已经在 v3 索引中保存了帧级向量和 `segment_frame_offsets`。但影视后期粗剪/精剪场景更关心“镜头”而不是固定 5 秒窗口。固定窗口可能切断一个真实镜头，也可能把两个镜头混在一个结果里，影响剪辑师判断和后续自动粗剪。

本设计将 visual 分段扩展为可选的镜头级 segment，同时保持旧 v3 索引可搜索。

## 目标

1. 新建 visual 索引可以按镜头边界返回时间戳。
2. 旧 v3 visual 索引继续可用，不强制重跑。
3. visual 搜索继续使用现有 frame embedding、MaxSim、`visual_top1/top3/mean` evidence。
4. Face / ASR / OCR 索引结构第一版不改。
5. 文档、测试和代码注释明确说明改动内容、兼容规则和注意事项。

## 非目标

本阶段不做完整的智能粗剪 agent，不生成 EDL/XML 时间线，也不把 Face/ASR/OCR 改成镜头级索引。

本阶段也不做完整 shot card 数据库。后续可以新增结构化层，把 Face track、ASR chunk、OCR chunk 按时间投影到 shot 上；这不属于本次 visual 分段扩展。

## 当前结构

当前 `visual.npz` v3 字段：

```text
frame_embeddings       [num_frames, dim]
frame_times_ms         [num_frames]
segment_frame_offsets  [num_segments + 1]
```

当前 manifest 里保存：

```text
duration_ms
segment_ms
channels.visual.sample_fps
```

搜索时 `_visual_candidates()` 根据 `segment_frame_offsets` 找到每个 bucket 内的帧向量，再用：

```text
start_ms = segment_id * segment_ms
end_ms = min((segment_id + 1) * segment_ms, duration_ms)
```

推导返回时间。

## 目标结构

新 visual 索引继续保留原有字段，并新增可选字段：

```text
frame_embeddings        [num_frames, dim]
frame_times_ms          [num_frames]
segment_frame_offsets   [num_segments + 1]
segment_times_ms        [num_segments, 2]  可选，镜头或子段真实起止时间
```

字段语义：

| 字段 | 作用 |
|---|---|
| `segment_frame_offsets` | 检索加速结构，表示每个 segment 对应 `frame_embeddings` 的半开行区间 |
| `segment_times_ms` | 时间语义结构，表示每个 segment 在原视频里的真实起止时间 |

`segment_frame_offsets` 和 `segment_times_ms` 不重复。固定 5s 时 `segment_times_ms` 可以由 `segment_ms` 推导；镜头级分段时 segment 长度不固定，必须显式保存。

## 兼容规则

搜索层按字段存在与否决定行为：

```text
如果 visual.npz 包含 segment_times_ms:
  使用 segment_times_ms[segment_id] 作为返回 start/end
否则:
  使用旧 v3 行为，通过 segment_id * segment_ms 推导固定窗口
```

因此：

- 旧 v3 索引继续可搜。
- 新建或重建 visual 索引后，才返回镜头级时间。
- schema version 暂不强制升级到 v4。该能力记录为 v3 optional shot-aware extension。

manifest 的 visual channel 可增加可选字段：

```json
{
  "segment_strategy": "shot",
  "shot_detector": "content",
  "min_segment_ms": 800,
  "max_segment_ms": 8000,
  "segment_times": "explicit"
}
```

旧 manifest 没有这些字段时，默认视为：

```text
segment_strategy = fixed
segment_times = inferred_from_segment_ms
```

## 分段策略

第一版采用 hybrid 策略：

```text
shot detection -> raw shots -> normalize shots -> split long shots -> assign frames -> write visual.npz
```

规则建议：

1. 通过镜头切分算法得到 raw shot 边界。
2. 过短 shot 可并入相邻 shot 或保留但设置最小时长保护。
3. 过长 shot 按最大时长继续切成子段，避免返回几十秒长片段。
4. 每个 segment 至少应尽量包含一帧 embedding；没有帧的 segment 可跳过或保留空 offset，取决于实现复杂度。
5. 缩略图仍命名为 `visual_{segment_id:06d}.jpg`，第一版可取 segment 内第一帧或中点附近帧。后续如果要优化 shot card 或缩略图代表性，再单独增加 `segment_key_ms`。

第一版推荐默认参数：

```text
min_segment_ms = 800
max_segment_ms = 8000
fallback_segment_seconds = 5.0
```

如果镜头检测失败或返回结果不可用，回退到当前固定 `visual_segment_seconds` 分段，保证索引任务可完成。

## 代码改动范围

### 后端 visual indexing

主要文件：

```text
backend/app/indexing/visual.py
backend/app/media.py
backend/app/stage_runner.py
backend/app/indexing/pipeline_manifest.py
backend/app/schemas.py
backend/app/settings.py
```

改动：

- 新增或内聚一个 shot detection helper，输出 `[(start_ms, end_ms)]`。
- `build_visual_index()` 支持 `segment_strategy`。
- 固定分段继续走旧逻辑。
- shot 分段写出 `segment_times_ms`。第一版不实现 `segment_key_ms`。
- manifest 写入 visual 的 `segment_strategy`、`min_segment_ms`、`max_segment_ms` 等可选字段。
- API request 可以先只在后端支持，前端暂不暴露；后续再加 UI 控件。

### 后端 search

主要文件：

```text
backend/app/search.py
```

改动：

- `_visual_candidates()` 校验并读取可选 `segment_times_ms`。
- 有显式时间时使用 `segment_times_ms` 返回 start/end。
- 没有显式时间时保持旧 v3 固定窗口推导。
- evidence 增加可选 `segment_strategy` 或 `segment_time_source`，方便诊断结果来自 fixed 还是 shot。

### 文档

主要文件：

```text
docs/RETRIEVAL_CHANNELS.md
docs/CURRENT.md
docs/ISSUES_AND_ROADMAP.md
docs/VALIDATION.md
```

改动：

- `RETRIEVAL_CHANNELS.md` 说明 `segment_times_ms` 是 v3 optional extension。
- `CURRENT.md` 在实际默认启用后更新当前 visual 分段策略。
- `ISSUES_AND_ROADMAP.md` 更新镜头级 visual 的状态和后续 shot card 结构化事项。
- `VALIDATION.md` 增加搜索兼容性测试命令或 pytest 入口。

### 前端

第一版可不改前端。

如果要暴露配置，改：

```text
frontend/src/main.tsx
frontend/src/api.ts
```

新增：

```text
Visual 分段方式：固定时间 / 镜头切分
最大镜头子段秒数
```

为了降低风险，第一版建议后端先支持并通过配置或默认值启用，前端 UI 后续再补。

## 其他通道边界

Face / ASR / OCR 第一版不改索引结构：

| 通道 | 本阶段是否改索引 | 原因 |
|---|---|---|
| Face | 不改 | 已经返回 track 时间段，可在融合时自然和 visual 镜头段重叠 |
| ASR | 不改 | 仍使用 Whisper/FunASR chunk；后续可单独优化 chunk 合并/拆分 |
| OCR | 不改 | 仍使用 sampled-frame chunk；后续可单独优化 OCR 抽帧和文本连续性 |

搜索融合层已经按时间重叠或邻近关系合并 candidates，因此其他通道可以自然落到新的 visual 时间片段上。

如果后续要实现真正的“镜头级结构化素材库”，应新增 shot card 层，而不是把所有通道索引都改成 shot 索引。shot card 可以记录：

```text
shot_id
start_ms / end_ms
visual evidence
face track ids
asr chunk ids / transcript summary
ocr chunk ids / text summary
tags / captions / quality signals
```

这是下一阶段。

## 错误处理与回退

索引阶段：

- 镜头检测失败时回退固定分段。
- 镜头检测返回空结果时回退固定分段。
- 分段结果超出视频 duration 时裁剪到 `[0, duration_ms]`。
- 分段必须保持递增且不重叠；否则清洗或回退。
- 如果某个 segment 没有帧，可以跳过或保留空 offset；搜索层已能跳过空 segment。

搜索阶段：

- `segment_times_ms` shape 必须是 `[num_segments, 2]`。
- `len(segment_times_ms)` 必须等于 `len(segment_frame_offsets) - 1`。
- 如果新字段无效，抛出“请重跑 visual 索引”的清晰错误，不静默返回错误时间。
- 旧 v3 没有新字段时不报错。

## 测试计划

新增或扩展：

```text
backend/tests/test_index_schema_v3.py
backend/tests/test_search.py
```

重点测试：

1. 旧 v3 visual 索引没有 `segment_times_ms` 时，搜索结果仍按固定 `segment_ms` 返回。
2. 新 visual 索引有 `segment_times_ms` 时，搜索结果按显式时间返回。
3. `segment_times_ms` 长度和 offsets 不一致时报错。
4. shot 分段中存在空 segment 时搜索可跳过。
5. shot detection 失败时 build_visual_index 回退固定分段并写出可诊断的 manifest/status。
6. 相邻 visual-only segments 仍不会被无限合并成长视频。

验证命令：

```powershell
cd video_retrieval_mvp/backend
pytest tests/test_search.py tests/test_index_schema_v3.py -v
```

如改动 settings/schema/frontend，再跑：

```powershell
pytest
cd ../frontend
npm run build
```

## 注意事项

- 不要删除或重命名现有 v3 字段。
- 不要把 `segment_times_ms` 当成所有 v3 索引必有字段。
- 不要让旧索引因为缺少新字段而失效。
- 不要把 visual-only 相邻 segment 重新串成整段视频。
- shot-aware 只改善时间边界，不自动解决 visual 多视频误召；`RQ-001` 和 `RQ-002` 仍需要 ranking 校准。
- 长镜头必须做最大时长保护，否则“镜头级”可能返回过长片段。
- 镜头检测算法对闪光、运动模糊、快切、转场敏感，需要保留固定分段 fallback。
- 文档必须明确 Face/ASR/OCR 第一版不改索引结构。

## 验收标准

1. 旧 v3 visual 索引搜索结果不变。
2. 新 shot-aware visual 索引可返回非固定 5s 的 segment 时间段。
3. 搜索 evidence 能看出结果时间来自显式 segment 还是固定 fallback。
4. Face / ASR / OCR 搜索测试仍通过。
5. `docs/RETRIEVAL_CHANNELS.md` 清楚说明 v3 optional shot-aware extension。
6. 测试覆盖兼容路径、新字段路径和错误路径。
