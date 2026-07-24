# Speaker Diarization 与声纹检索设计

日期：2026-07-13

## 背景

MomentSeek 当前 ASR 通道可以把语音转写为带时间范围的 retrieval chunk，但不能回答“这一段是谁说的”，也不能使用参考语音跨视频查找同一说话人。

本设计在现有 ASR pipeline 上增加两层能力：

```text
Speaker Diarization：区分单个视频内谁在什么时候说话
Voiceprint Retrieval：为每个视频内 speaker 生成聚合声纹向量
```

第一阶段不建立跨视频永久 Speaker ID。参考语音检索时，对每个视频的本地 speaker embedding 分别比较，再对所有候选做全局排序。

## 目标

1. 给 ASR retrieval chunk 标记视频内 speaker。
2. 尽量按照可靠的 speaker 切换边界切分文本 chunk。
3. 在视频粒度汇总 speaker turns。
4. 每个视频内 speaker 只保存一个正式聚合声纹 embedding。
5. 支持参考语音在多个视频中搜索相似 speaker。
6. 正式索引保持精简，调试模式额外保存诊断产物。
7. Diarization 或声纹失败时不破坏已有 ASR 索引能力。

## 非目标

第一阶段暂不实现：

- 跨视频永久 `voice_cluster_id`。
- 自动把匿名 speaker 绑定为真实人物。
- 按人物姓名直接进行跨视频声纹检索。
- 一次绑定后自动合并该人物在所有视频中的 speaker。
- 把临时声纹样本 embedding 写入正式生产索引。

## 概念边界

```text
track_id
  单个视频、单次 speaker 索引内的局部编号。

speaker label
  UI 派生文案，例如 track_id=0 显示为 SPEAKER_00，不单独存储。

voice embedding
  当前视频内一个 speaker 的聚合声纹向量。

global speaker identity
  跨视频身份层，第一阶段不实现。
```

`SPEAKER_00` 只在当前视频和当前 speaker 索引版本内有效。重跑 diarization 后，局部编号可能变化。

## Pipeline

```text
原始视频
-> audio_extract
-> ASR model_transcribe + raw transcript parser
-> speaker diarization turns
-> speaker turn 平滑与短抖动处理
-> ASR timestamp 与 speaker turns 对齐
-> 按可靠 speaker 边界生成 retrieval chunks
-> 为每个 speaker 选择高质量单人语音窗口
-> 临时计算多个窗口 embedding
-> 过滤离群向量并稳健聚合
-> 每个 speaker 保存一个归一化 track embedding
-> speaker.npz
```

Diarization 和 voiceprint 应作为可独立重跑的 speaker 阶段。它可以读取已有 ASR 时间数据，但不应要求重新执行 ASR 模型转写。

## 文本 Chunk 与 Speaker 对齐

### 可靠 word timestamp

当 ASR word/character timestamp 和 speaker 边界都可靠时，在最接近 speaker 切换点的安全词边界切分文本。

### 只有 segment timestamp

没有可靠 word timestamp 时，不猜测文本内部字符边界。整段保留并归属给时间覆盖最多的主 speaker；如果明显跨 speaker，调试信息记录 `speaker_boundary_crossing`。

### 重叠语音

多人重叠或无法可靠归属时使用：

```text
chunk_track_indices = -1
```

第一阶段正式索引只保存单一主 track 或 `-1`。多 speaker overlap ratio、候选分布和切分决策写入可选 debug 产物。

### Speaker Turn 平滑

Diarization 的极短 turn 或几十毫秒抖动不能直接触发文本切分。正式对齐前需要：

- 合并同一 speaker 的短间隔 turn。
- 对极短、低可靠 turn 做吸附或标记。
- 不让噪声、音乐或瞬时误判产生大量碎片文本。
- 只在具有足够持续时间和边界证据时切换 speaker。

具体阈值必须通过真实视频和 debug 统计确定，不根据单个样例不断追加规则。

## 声纹生成

虽然正式索引中每个 speaker 只保存一个 embedding，生成该向量时仍应使用多个高质量语音窗口：

```text
speaker turns
-> 排除多人重叠、音乐、强噪声、非语音和过短片段
-> 从不同时间位置选择若干单人语音窗口
-> 临时生成并归一化 sample embeddings
-> 检查内部一致性并排除离群点
-> 稳健聚合
-> 再次归一化
-> track_embeddings[track_id]
```

候选窗口、最低有效语音时长、样本数量和离群阈值在评测后确定。有效材料不足或内部一致性过低的 track 不应参与声纹匹配。

## 正式索引

第一阶段把 diarization 与声纹合并为一个文件：

```text
runtime/indexes/{video_id}/speaker.npz
```

正式 schema：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `turn_times_ms` | `[num_turns, 2] int32` | 平滑后的 speaker turn 起止时间 |
| `turn_track_indices` | `[num_turns] int32` | 每个 turn 对应的局部 track ID |
| `chunk_track_indices` | `[num_asr_chunks] int32` | 每个 ASR retrieval chunk 的主 track，`-1` 表示未知或无法可靠归属 |
| `track_embeddings` | `[num_tracks, dim] float16` | 每个视频内 speaker 的聚合、归一化声纹向量 |
| `track_embedding_valid` | `[num_tracks] bool` | 该 track 是否有足够可靠的声纹材料 |

数组行号就是 `track_id`。不重复保存：

```text
speaker_label
track_durations_ms
speaker_name
sample_embeddings
sample_track_indices
sample_times_ms
```

`speaker_label` 由 UI 使用 `SPEAKER_{track_id:02d}` 派生。发言时长由 turns 计算。真实人物名称和未来身份绑定属于可变业务数据，不写入 NPZ。

## Manifest

`index_manifest.json` 的 speaker channel 保存不能从数组直接推断的少量信息：

```json
{
  "speaker": {
    "file": "speaker.npz",
    "diarization_model": "<model>",
    "voice_embedding_model": "<model>",
    "embedding_space": "<space>",
    "embedding_normalized": true,
    "tracks": 3,
    "turns": 47,
    "decode_status": "complete",
    "debug_artifacts": false
  }
}
```

不同 `embedding_space` 的声纹向量禁止直接比较。

## Debug 模式

配置：

```text
SPEAKER_DEBUG_ARTIFACTS=false
```

启用后额外写入：

```text
runtime/indexes/{video_id}/debug/speaker_debug.npz
runtime/indexes/{video_id}/debug/speaker_debug.json
```

可选 `speaker_debug.npz` 内容：

```text
raw_turn_times_ms
raw_turn_track_indices
sample_times_ms
sample_track_indices
sample_embeddings
sample_quality_scores
chunk_track_overlap_ratios
```

`speaker_debug.json` 记录模型配置、speaker 数、未归属 chunk、跨 speaker chunk、样本筛选原因、有效语音时长和 embedding 离散度等诊断信息。

Debug 开关只能控制是否额外导出中间数据，不能改变正式计算路径或 `speaker.npz` 的结果。删除 debug 目录不影响检索。

## 跨视频检索

第一阶段不复用局部 Speaker ID。查询时：

```text
参考语音
-> query voice embedding
-> 遍历每个视频的 speaker.npz
-> 只比较 track_embedding_valid=true 的 track_embeddings
-> 按 raw cosine 汇总所有 (video_id, track_id) 候选
-> 跨视频全局排序
-> 展开命中 track 的 turns 和对应 ASR chunks
```

候选唯一键为：

```text
(video_id, track_id, speaker index/model fingerprint)
```

跨视频排序使用同一声纹空间内可比较的 raw cosine。不能用每个视频内部 percentile 作为主排序分数。

在当前素材规模下逐视频比较足够简单。规模扩大后可以生成可重建的全局扁平缓存，但每个视频的 `speaker.npz` 仍是权威数据。

## 失败与降级

- Diarization 失败：ASR 文本仍可正常检索，不生成 speaker channel。
- Voice embedding 失败：保留 speaker turns 和文本归属，对应 `track_embedding_valid=false`。
- 部分 speaker 材料不足：只禁用这些 track 的声纹检索，不影响其他 track。
- Speaker 索引版本过旧或模型空间不一致：跳过声纹比较并返回明确诊断。
- 调试产物缺失或被删除：不得影响正式检索。

## 后续方向

确认视频内 diarization、文本对齐和声纹召回质量后，再评估：

- 跨视频匿名 `voice_cluster_id`。
- 人工确认的 speaker 合并。
- 声纹与人物 `entity_id` 绑定。
- Face 与 Voice 联合身份判断。
- 按人物姓名搜索发言内容。

## 2026-07-14 第一版调整：句级声纹为正式索引

真实综艺测试表明，自动 diarization 可能同时发生不同人物合并和同一人物拆分。因此第一版不再只保存每个自动 track 的单一声纹，而以可播放、可人工纠正的句级声纹作为检索基础。正式 schema 为：

```text
utterance_embeddings             [N, D] float16
utterance_times_ms               [N, 2] int32
utterance_refs                   [N, 2] int32  # asr_chunk_index, auto_track_index
track_embeddings                 [T, D] float16
track_representative_indices     [T] int32
```

前三个数组是权威索引；后两个数组是可重建的页面与粗筛缓存。Speaker 名称、句子移动、可检索状态、人物绑定和声音库样本元数据保存在 SQLite，不写回 NPZ。搜索以具体 utterance 的余弦相似度为主，自动 track 只用于展示、候选组织和人工管理。

后续即使增加全局身份层，`track_id` 仍保持视频内局部语义，不直接改成永久人物 ID。

## 验收重点

1. 文本不会因短 speaker 抖动被切成大量碎片。
2. 有可靠 word timestamp 时，跨 speaker 文本能在安全词边界切开。
3. 重叠或不确定文本不会被强行归属。
4. 每个有效 track 只在正式索引保存一个聚合 embedding。
5. 普通 ASR 搜索不加载或计算不需要的调试数组。
6. 参考语音可以跨多个视频召回同声纹 track，并按 raw cosine 全局排序。
7. 关闭 debug 后不产生额外 sample embedding 和诊断文件。
