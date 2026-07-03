# 检索通道协议

本文档是 MomentSeek visual / face / ASR / OCR 通道的权威协议说明。

## 总览

| 通道 | 索引频率 | 模型 / 表征 | 存储文件 | 召回粒度 |
|---|---:|---|---|---|
| `visual` | 5fps，5s bucket | SigLIP2 So400m 384，1152-d float32 | `visual.npz` | 5s bucket，按最佳帧 MaxSim 排序 |
| `face` | 当前服务器 1fps | InsightFace `buffalo_l`，ArcFace 512-d float32 | `faces.npz` | 人脸 track |
| `asr` | ASR 模型 chunk | Whisper small + 可选 MiniLM 384-d semantic embedding | `asr.json`, `asr_semantic.npz` | ASR chunk |
| `ocr` | 0.05fps，约每 20s 一帧 | RapidOCR PP-OCRv4 + 可选 MiniLM 384-d semantic embedding | `ocr.json`, `ocr_semantic.npz` | OCR chunk |

不同向量空间不能混用：

- Visual SigLIP2/CLIP 是视觉-文本空间。
- Face ArcFace 是身份空间。
- ASR/OCR semantic 是 MiniLM 文本空间。

## Visual

当前模型：

```text
siglip2-so400m-384
```

索引流程：

```text
video -> 5fps 抽帧 -> decode height 256 -> SigLIP2 frame embeddings
-> 按 5s bucket 分组
-> 同时保存帧级 embedding 和 segment mean embedding
```

`visual.npz` schema v2：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `schema_version` | `[1] int16` | 当前为 2 |
| `segment_ids` | `[segments] int32` | 5s bucket id |
| `embeddings` | `[segments, 1152] float32` | segment mean embedding |
| `start_times` | `[segments] float32` | bucket 起始时间 |
| `end_times` | `[segments] float32` | bucket 结束时间 |
| `thumbnails` | `[segments] str` | 代表缩略图 |
| `frame_embeddings` | `[frames, 1152] float32` | 帧级 embedding |
| `frame_times` | `[frames] float32` | 帧时间戳 |
| `frame_segment_ids` | `[frames] int32` | 帧所属 bucket |
| `model` | `str` | 模型显示名 |
| `visual_model` | `str` | 模型 key |
| `model_backend` | `str` | 当前通常为 `hf` |
| `model_id` | `str` | Hugging Face 或本地模型 id |

当前召回逻辑：

```text
query text/image -> SigLIP2 query embedding
-> 与 frame_embeddings 做 cosine
-> 每个 5s bucket 取 top1 frame similarity
-> raw_score = visual_top1
-> 返回 5s bucket
```

返回 evidence 字段：

```text
raw_score
visual_top1
visual_top3
visual_mean
best_time
percentile
robust_z
distribution_median
distribution_mad
```

已知质量风险：per-video percentile 能帮助找到每个视频内部最突出的片段，但在大量无关视频参与搜索时，可能把无关视频的“本视频内部最佳片段”排得过高。见 `docs/ISSUES_AND_ROADMAP.md` 的 `RQ-001`。

## Face

当前模型：

```text
InsightFace buffalo_l
ArcFace 512-d float32
```

索引流程：

```text
video -> 抽帧 -> 人脸检测 -> ArcFace embeddings
-> embedding cosine + bbox IoU 做简单 track
-> 聚合 track embedding
-> 保存最佳人脸 crop
```

`faces.npz` 字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `embeddings` | `[tracks, 512] float32` | 每条 track 的平均身份向量 |
| `start_times` | `[tracks] float32` | track 起始时间 |
| `end_times` | `[tracks] float32` | track 结束时间 |
| `thumbnails` | `[tracks] str` | 最佳人脸 crop |
| `qualities` | `[tracks] float32` | 人脸质量 |
| `model` | `str` | 模型名 |

查询来源：

```text
参考图
查询文本命中的人物库 entity
```

当前阈值：

```text
cosine >= 0.35
```

face confidence 在 `search.py::face_confidence` 中用 logistic 函数校准。

## ASR

当前服务器引擎：

```text
Whisper small
```

`asr.json`：

```json
{
  "engine": "whisper",
  "model": "small",
  "language": "en",
  "chunks": [
    {
      "start_time": 12.34,
      "end_time": 18.9,
      "text": "..."
    }
  ]
}
```

可选 `asr_semantic.npz`：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `schema_version` | `[1] int16` | 当前为 1 |
| `embeddings` | `[chunks, 384] float32` | MiniLM 文本向量 |
| `chunk_indices` | `[chunks] int32` | 对应 `asr.json` 的 chunk index |
| `model` | `str` | semantic 模型名 |
| `device` | `str` | semantic 设备 |

召回逻辑：

```text
query text -> lexical score
可选 query text -> MiniLM embedding -> semantic cosine
combined_score = max(lexical_score, 0.65 * semantic_score + 0.35 * lexical_score)
```

返回 evidence 字段：

```text
lexical_score
semantic_score
semantic_cosine
```

## OCR

当前引擎：

```text
RapidOCR / PP-OCRv4 English mobile
sample_fps = 0.05
decode_height = 720
min_confidence = 0.5
```

`ocr.json`：

```json
{
  "schema_version": 1,
  "engine": "rapidocr",
  "sample_fps": 0.05,
  "chunks": [
    {
      "start_time": 40.0,
      "end_time": 60.0,
      "text": "...",
      "items": [
        {
          "text": "...",
          "score": 0.9231,
          "box": [[0, 0], [1, 0], [1, 1], [0, 1]]
        }
      ],
      "thumbnail": "ocr_000002.jpg",
      "score": 0.9231
    }
  ]
}
```

可选 `ocr_semantic.npz` 使用和 ASR semantic 类似的 schema。

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
