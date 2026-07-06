# 检索通道协议

本文档是 MomentSeek visual / face / ASR / OCR 通道的权威协议说明。索引格式当前为 **schema v3**。

## 总览

每个视频的索引目录固定为：

```text
runtime/indexes/{video_id}/
  index_manifest.json
  visual.npz
  face.npz
  asr.npz
  ocr.npz
```

`index_manifest.json` 存小元信息，`.npz` 只存检索所需数组。v3 不兼容旧索引；如果只有旧的 `faces.npz`、`asr.json`、`ocr.json` 或 visual v2 文件，必须重跑对应通道索引。

| 通道 | 索引频率 | 模型 / 表征 | 存储文件 | 召回粒度 |
|---|---:|---|---|---|
| `visual` | 默认 5fps，5s bucket | SigLIP2/CLIP image-text embedding | `visual.npz` | 5s bucket，按 bucket 内最佳帧 MaxSim 排序 |
| `face` | 默认 1fps | InsightFace `buffalo_l`，ArcFace identity embedding | `face.npz` | 人脸 track |
| `asr` | ASR 模型 chunk | Whisper/FunASR 文本 + 可选 MiniLM semantic embedding | `asr.npz` | ASR chunk |
| `ocr` | 默认 0.05fps，约每 20s 一帧 | RapidOCR PP-OCRv4 文本框 + 可选 MiniLM semantic embedding | `ocr.npz` | OCR sampled-frame chunk |

不同向量空间不能混用：

- Visual SigLIP2/CLIP 是视觉-文本空间。
- Face ArcFace 是身份空间。
- ASR/OCR semantic 是 MiniLM 文本空间。

## Manifest

`index_manifest.json` 示例：

```json
{
  "schema_version": 3,
  "video_id": "video-1",
  "duration_ms": 120000,
  "segment_ms": 5000,
  "channels": {
    "visual": {
      "file": "visual.npz",
      "model_key": "siglip2-so400m-384",
      "embedding_space": "siglip2-image-text",
      "sample_fps": 5.0,
      "decode_status": "complete"
    }
  }
}
```

Manifest 只保存少量元信息，不保存每帧/每 chunk 重复字段。`embedding_dim` 和 `embedding_dtype` 从数组 shape/dtype 或模型注册表推断，不在 manifest 中重复存储。

## Visual

当前默认模型：

```text
siglip2-so400m-384
```

`visual.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `frame_embeddings` | `[num_frames, dim] float16` | 帧级视觉向量 |
| `frame_times_ms` | `[num_frames] int32` | 每个帧向量对应的视频时间戳 |
| `segment_frame_offsets` | `[segments_total + 1] int32` | 每个 5s bucket 在 `frame_embeddings` 中的半开区间 `[start, end)` |

`segment_frame_offsets` 覆盖完整视频 timeline。某个 bucket 没有成功解码帧时，offset start/end 相同；这允许部分解码失败或空 bucket 不破坏数组对齐。

召回逻辑：

```text
query text/image -> visual query embedding
-> 与 frame_embeddings 做 cosine
-> 每个 bucket 取 top1 frame similarity
-> raw_score = visual_top1
-> 返回 5s bucket
```

Visual evidence 主要字段：

```text
raw_score
visual_top1
visual_top3
visual_mean
best_time / best_ms
percentile
robust_z
unit_type = "segment"
unit_id = segment_id
features
```

已知质量风险：per-video percentile 能帮助找到每个视频内部最突出的片段，但在大量无关视频参与搜索时，可能把无关视频的“本视频内部最佳片段”排得过高。见 `docs/ISSUES_AND_ROADMAP.md` 的检索效果优化问题池。

## Face

当前模型：

```text
InsightFace buffalo_l
ArcFace identity embedding
```

`face.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `embeddings` | `[num_tracks, 512] float32` | 每条人脸 track 的平均身份向量 |
| `track_times_ms` | `[num_tracks, 3] int32` | `[start_ms, end_ms, best_shot_ms]` |

Track id 就是数组行号。缩略图固定为：

```text
runtime/thumbnails/{video_id}/face_{track_id:06d}.jpg
```

查询来源：

```text
参考图
查询文本命中的人物库 entity
```

当前阈值：

```text
cosine >= 0.35
```

Face evidence 主要字段：

```text
raw_score
best_time / best_ms
unit_type = "track"
unit_id = track_id
features.face_cosine
```

## ASR

当前服务器常用引擎：

```text
Whisper small
```

`asr.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `chunk_times_ms` | `[num_chunks, 2] int32` | `[start_ms, end_ms]` |
| `texts` | `[num_chunks] str` | ASR chunk 文本 |
| `embeddings` | `[num_semantic_chunks, 384] float16` | 可选 MiniLM semantic embedding；不可用时为空数组 |
| `embedding_chunk_indices` | `[num_semantic_chunks] int32` | 每条 semantic embedding 对应的 `texts` 行号 |

ASR semantic 是可选能力。语义模型不可用时仍保留 `chunk_times_ms/texts`，搜索自动退回 lexical。

召回逻辑：

```text
query text -> lexical score
可选 query text -> MiniLM embedding -> semantic cosine
combined_score = max(lexical_score, 0.65 * semantic_score + 0.35 * lexical_score)
```

ASR evidence 主要字段：

```text
lexical_score
semantic_score
semantic_cosine
text
unit_type = "chunk"
unit_id = chunk_id
features
```

## OCR

当前引擎：

```text
RapidOCR / PP-OCRv4
sample_fps = 0.05
decode_height = 720
min_confidence = 0.5
```

OCR 的 chunk 是一次 OCR sampled frame/time window；文本与 score 在 box 级保存，chunk 文本在搜索时由该 chunk 的 boxes 拼接得到。

`ocr.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `chunk_times_ms` | `[num_chunks, 3] int32` | `[start_ms, end_ms, frame_ms]` |
| `embeddings` | `[num_semantic_chunks, 384] float16` | 可选 MiniLM semantic embedding；不可用时为空数组 |
| `embedding_chunk_indices` | `[num_semantic_chunks] int32` | 每条 semantic embedding 对应的 chunk 行号 |
| `box_chunk_indices` | `[num_boxes] int32` | 每个 OCR box 属于哪个 chunk |
| `box_texts` | `[num_boxes] str` | OCR box 文本 |
| `box_scores` | `[num_boxes] float32` | OCR box 置信度 |
| `boxes` | `[num_boxes, 4, 2] float32` | 归一化到 `[0, 1]` 的四点框 |

缩略图固定为：

```text
runtime/thumbnails/{video_id}/ocr_{chunk_id:06d}.jpg
```

OCR 检索复用 ASR 的 lexical + semantic candidate 逻辑，返回 OCR chunk。

## 结果融合

每个通道先独立产生 candidate：

```text
visual -> 5s bucket
face   -> face track
asr    -> ASR chunk
ocr    -> OCR chunk
```

系统再按时间重叠或邻近关系合并成最终结果。Visual-only 相邻 bucket 不会无限串联，只在真实重叠或有其他模态锚点时合并。

最终结果字段包括：

```text
video_id
video_name
start_time
end_time
score
modalities
thumbnail_url
media_url
clip_url
decision
above_threshold
evidence[]
```

`evidence[]` 保留旧的展示字段，并新增统一定位字段：

```text
unit_type
unit_id
best_ms
text
features
```
