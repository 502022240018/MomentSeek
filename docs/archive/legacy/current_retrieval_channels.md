> Archived reference. Current documentation starts at `docs/README.md`.

# MomentSeek 当前四条检索通道索引与召回说明

更新时间：2026-07-03
适用版本：当前服务器与本地 `feat/asr-search-asset-improvements` 分支附近版本

本文档整理当前 MVP 系统中四条检索通道的索引频率、索引元数据格式、召回粒度、存储方式与当前服务器实际配置。

四条通道分别是：

- `visual_index`：视觉语义检索
- `face_index`：人脸 / 明星检索
- `asr_index`：语音转写文本检索
- `ocr_index`：画面文字检索

## 1. 总览表

| 通道 | 当前索引频率 | 主要模型 / 表征 | 元数据与向量格式 | 召回粒度 | 存储文件 |
|---|---:|---|---|---|---|
| `visual_index` | 5 fps，5 秒分桶 | SigLIP2 So400m 384 | `1152-d float32`，同时存帧级向量和 5s 桶均值向量 | 5s bucket，按 bucket 内最大相似帧召回 | `runtime/indexes/{video_id}/visual.npz` |
| `face_index` | 服务器当前 1 fps；代码默认 2 fps | InsightFace `buffalo_l` / ArcFace | `512-d float32`，按人脸 track 聚合 | 人脸 track，时长可变 | `runtime/indexes/{video_id}/faces.npz` |
| `asr_index` | 不按帧；由 ASR 模型输出 chunk | Whisper small；语义检索用 MiniLM | 文本 chunk + 可选 `384-d float32` 文本向量 | ASR chunk，时长可变 | `asr.json` + `asr_semantic.npz` |
| `ocr_index` | 0.05 fps，即每 20 秒抽 1 帧 | RapidOCR / PP-OCRv4 | OCR 文本框 JSON + 可选 `384-d float32` 文本向量 | OCR chunk，当前约 20 秒 | `ocr.json` + `ocr_semantic.npz` |

注意：这里的向量不是“128bit”。当前向量都是 `float32` 数组：

- Visual SigLIP2：`1152 * 4 bytes ≈ 4.5 KB / vector`
- Face ArcFace：`512 * 4 bytes ≈ 2 KB / vector`
- ASR/OCR semantic：`384 * 4 bytes ≈ 1.5 KB / vector`

## 2. 当前服务器实际配置

服务器当前读取到的配置如下：

```text
visual_sample_fps = 5.0
visual_segment_seconds = 5.0
visual_decode_height = 256
visual_model = siglip2-so400m-384

face_sample_fps = 1.0
face_decode_height = 720
face_model = buffalo_l
face_provider = cann

asr_engine = whisper
asr_model = small
asr_language = en
asr_semantic_enabled = true
asr_semantic_model = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
asr_semantic_device = cpu

ocr_sample_fps = 0.05
ocr_decode_height = 720
ocr_engine = rapidocr
ocr_device = auto
ocr_version = PP-OCRv4
ocr_det_lang = en
ocr_rec_lang = en
ocr_model_type = mobile
ocr_min_confidence = 0.5
ocr_semantic_enabled = true
```

其中有一个需要特别注意的点：

- 代码默认 `face_sample_fps = 2.0`
- 当前服务器环境实际覆盖为 `face_sample_fps = 1.0`

所以如果在别的服务器重新部署，除非 `.env` 或容器环境变量同步，否则 face 抽帧频率可能会回到代码默认值。

## 3. `visual_index`

### 3.1 索引流程

当前 visual 通道使用：

```text
siglip2-so400m-384
```

索引流程：

```text
输入视频
  → 以 5 fps 抽帧
  → 解码高度缩放到 256，避免直接处理原始 1080p / 4K 大图
  → 每帧送入 SigLIP2
  → 得到每帧 1152 维 float32 视觉向量
  → 按 timestamp // 5s 分桶
  → 每个 5s bucket 内：
      - 保存所有帧级向量
      - 计算并保存 bucket 平均向量
      - 保存 start_time / end_time
      - 保存代表缩略图
```

### 3.2 `visual.npz` 格式

文件位置：

```text
runtime/indexes/{video_id}/visual.npz
```

当前 schema v2 字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `schema_version` | `int16, [1]` | 当前为 2 |
| `segment_ids` | `int32, [num_segments]` | 5s bucket ID |
| `embeddings` | `float32, [num_segments, 1152]` | 每个 5s bucket 的平均视觉向量 |
| `start_times` | `float32, [num_segments]` | bucket 起始时间 |
| `end_times` | `float32, [num_segments]` | bucket 结束时间 |
| `thumbnails` | `str, [num_segments]` | bucket 缩略图文件名 |
| `frame_embeddings` | `float32, [num_frames, 1152]` | 帧级视觉向量 |
| `frame_times` | `float32, [num_frames]` | 每个帧向量对应的视频时间戳 |
| `frame_segment_ids` | `int32, [num_frames]` | 每帧所属 bucket |
| `model` | `str` | 模型显示名，例如 `SigLIP2 So400m patch14 384` |
| `visual_model` | `str` | 模型 key，例如 `siglip2-so400m-384` |
| `model_backend` | `str` | 当前为 `hf` |
| `model_id` | `str` | HuggingFace 模型 ID 或本地模型 ID |

### 3.3 召回逻辑

当前 visual 检索逻辑已经改成“最大相似帧召回”：

```text
query 文本 / query 图片
  → SigLIP2 query embedding
  → 与 frame_embeddings 逐帧做 cosine similarity
  → 每个 5s bucket 内取 top1，即最大帧相似度
  → raw_score = visual_top1
  → 以 bucket 为单位返回
```

也就是说，现在不是简单使用 5 秒平均向量做最终排序，而是：

```text
5 秒 bucket 内，只要有某一帧强相关，这个 bucket 就可以被召回。
```

返回 evidence 中会保留诊断信息：

```text
raw_score
visual_top1
visual_top3
visual_mean
best_frame
percentile
robust_z
```

其中：

- `visual_top1`：bucket 内最相似帧分数，也是当前 `raw_score`
- `visual_top3`：bucket 内前三相似帧平均分
- `visual_mean`：bucket 段均值向量分数
- `best_frame`：最相似帧对应的视频时间戳

### 3.4 当前服务器样例

以 31 分钟 1080p 视频为例，当前索引为：

```text
segments = 375
frame_embeddings = 9367
embedding dim = 1152
```

对应：

```text
visual.npz:
  embeddings:       (375, 1152), float32
  frame_embeddings: (9367, 1152), float32
  frame_times:      (9367,), float32
```

## 4. `face_index`

### 4.1 索引流程

当前 face 通道使用：

```text
InsightFace buffalo_l
ArcFace embedding
```

索引流程：

```text
输入视频
  → 当前服务器按 1 fps 抽帧
  → 解码高度 720
  → InsightFace 检测人脸
  → 每张脸提取 ArcFace embedding，512 维 float32
  → 使用 embedding cosine + bbox IoU 做简单人脸 track
  → 每条 track 聚合为一个人脸片段
  → 保存 track 平均 embedding、起止时间、最佳人脸 crop
```

### 4.2 `faces.npz` 格式

文件位置：

```text
runtime/indexes/{video_id}/faces.npz
```

字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `embeddings` | `float32, [num_tracks, 512]` | 每条人脸 track 的身份向量，来自 track 内多帧人脸 embedding 的平均 |
| `start_times` | `float32, [num_tracks]` | track 起始时间 |
| `end_times` | `float32, [num_tracks]` | track 结束时间 |
| `thumbnails` | `str, [num_tracks]` | 最佳人脸 crop 文件名 |
| `qualities` | `float32, [num_tracks]` | 人脸质量分 |
| `model` | `str` | 当前为 `buffalo_l` |

### 4.3 召回逻辑

face 查询来源有两类：

```text
1. 用户上传参考图
2. 查询文本命中人物库 entity 名称后，读取该 entity 的人脸 embedding
```

检索流程：

```text
参考图 / entity embedding
  → ArcFace 512-d embedding
  → 与 faces.npz 里的 track embeddings 做 cosine
  → 返回对应 track 的 start_time ~ end_time
```

当前 face 判定阈值：

```text
cosine >= 0.35
```

返回粒度是“人脸 track”，不是固定 5s 桶。track 的长度由人物连续出现时间和 tracking 结果决定。

### 4.4 当前服务器样例

31 分钟视频当前 face 索引：

```text
faces.npz:
  embeddings:  (1655, 512), float32
  start_times: (1655,), float32
  end_times:   (1655,), float32
  qualities:   (1655,), float32
```

## 5. `asr_index`

### 5.1 索引流程

当前服务器 ASR 使用：

```text
Whisper small
```

代码支持 FunASR / Paraformer，但当前服务器实际配置是 `whisper`。

索引流程：

```text
输入视频
  → 抽取音频
  → Whisper small 转写
  → 生成若干语音 chunk
  → 保存 asr.json
  → 如果语义模型可用，对每个 chunk 文本生成 384-d semantic embedding
  → 保存 asr_semantic.npz
```

ASR 不是按固定帧率或固定秒数抽样，而是由 ASR 模型根据语音内容输出自然 chunk。

### 5.2 `asr.json` 格式

文件位置：

```text
runtime/indexes/{video_id}/asr.json
```

格式：

```json
{
  "engine": "whisper",
  "model": "small",
  "language": "en",
  "chunks": [
    {
      "start_time": 12.34,
      "end_time": 18.90,
      "text": "..."
    }
  ]
}
```

已有视频的 `language` 不一定都一样，因为任务创建时可以覆盖语言参数。例如当前服务器已有视频里存在 `zh` 和 `en` 混合。

### 5.3 `asr_semantic.npz` 格式

文件位置：

```text
runtime/indexes/{video_id}/asr_semantic.npz
```

字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `schema_version` | `int16, [1]` | 当前为 1 |
| `embeddings` | `float32, [num_chunks, 384]` | 每个 ASR chunk 的文本语义向量 |
| `chunk_indices` | `int32, [num_chunks]` | 向量对应 `asr.json` 中第几个 chunk |
| `model` | `str` | 当前为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `device` | `str` | 当前为 `cpu` |

注意：`asr_semantic.npz` 是可选文件。如果语义模型缺失或加载失败，系统仍然保留 `asr.json`，搜索时自动退回 lexical 文本匹配。

### 5.4 召回逻辑

检索流程：

```text
query 文本
  → lexical 文本匹配
  → 如果存在 asr_semantic.npz：
      query 文本 → 384-d semantic embedding
      与 chunk embeddings 做 cosine
  → lexical + semantic 融合
  → 返回 ASR chunk 的 start_time ~ end_time
```

当前融合大致逻辑：

```text
combined_score = max(lexical_score, 0.65 * semantic_score + 0.35 * lexical_score)
```

返回 evidence 中会包含：

```text
lexical_score
semantic_score
semantic_cosine
```

### 5.5 当前服务器样例

世界杯广告视频当前 ASR 索引：

```text
asr.json:
  chunks = 90
  engine = whisper
  model = small
  language = en

asr_semantic.npz:
  embeddings:    (90, 384), float32
  chunk_indices: (90,), int32
```

## 6. `ocr_index`

### 6.1 索引流程

当前 OCR 通道使用：

```text
RapidOCR
PP-OCRv4 English mobile
```

当前抽帧频率：

```text
ocr_sample_fps = 0.05
```

也就是：

```text
每 20 秒抽 1 帧
```

索引流程：

```text
输入视频
  → 每 20 秒抽一帧
  → 解码高度 720
  → RapidOCR 检测文字框并识别文本
  → 只保留 confidence >= 0.5 的结果
  → 有文字的帧保存为 OCR chunk
  → 保存 ocr.json
  → 如果语义模型可用，对 OCR chunk 文本生成 384-d semantic embedding
  → 保存 ocr_semantic.npz
```

### 6.2 `ocr.json` 格式

文件位置：

```text
runtime/indexes/{video_id}/ocr.json
```

格式：

```json
{
  "schema_version": 1,
  "engine": "rapidocr",
  "device": "npu",
  "providers": {},
  "ocr_version": "PP-OCRv4",
  "det_lang": "en",
  "rec_lang": "en",
  "model_type": "mobile",
  "sample_fps": 0.05,
  "decode_height": 720,
  "min_confidence": 0.5,
  "chunks": [
    {
      "start_time": 40.0,
      "end_time": 60.0,
      "text": "...",
      "items": [
        {
          "text": "...",
          "score": 0.9231,
          "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        }
      ],
      "thumbnail": "ocr_000002.jpg",
      "score": 0.9231
    }
  ]
}
```

OCR chunk 的 `end_time` 当前用 `timestamp + 1 / sample_fps` 得到。所以在 `0.05 fps` 下，每个 OCR chunk 约 20 秒。

### 6.3 `ocr_semantic.npz` 格式

文件位置：

```text
runtime/indexes/{video_id}/ocr_semantic.npz
```

字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `schema_version` | `int16, [1]` | 当前为 1 |
| `embeddings` | `float32, [num_ocr_chunks, 384]` | 每个 OCR chunk 文本的语义向量 |
| `chunk_indices` | `int32, [num_ocr_chunks]` | 对应 `ocr.json` 的 chunk index |
| `model` | `str` | 当前为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `device` | `str` | 当前为 `cpu` |

### 6.4 召回逻辑

OCR 搜索复用了 ASR 文本候选逻辑：

```text
query 文本
  → OCR lexical 文本匹配
  → 如果存在 ocr_semantic.npz：
      query 文本 → 384-d semantic embedding
      与 OCR chunk embeddings 做 cosine
  → lexical + semantic 融合
  → 返回 OCR chunk
```

返回粒度是 OCR chunk。当前默认约 20 秒。

### 6.5 当前服务器样例

球星牛奶广告当前 OCR 索引：

```text
ocr.json:
  chunks = 3
  sample_fps = 0.05
  decode_height = 720
  ocr_version = PP-OCRv4
  det_lang = en
  rec_lang = en

ocr_semantic.npz:
  embeddings:    (3, 384), float32
  chunk_indices: (3,), int32
```

## 7. 全局存储结构

当前系统主要数据放在 `runtime/` 下：

```text
runtime/
  catalog.sqlite3
  uploads/
    {video_id}...
  indexes/
    {video_id}/
      visual.npz
      faces.npz
      asr.json
      asr_semantic.npz
      ocr.json
      ocr_semantic.npz
      work/
  thumbnails/
    {video_id}/
      visual_000000.jpg
      face_000000.jpg
      ocr_000000.jpg
  clips/
    {video_id}/
      {start_ms}_{end_ms}.mp4
```

其中：

- `catalog.sqlite3`：视频、任务、人物库等结构化元数据
- `uploads/`：原始视频
- `indexes/{video_id}/`：各通道索引文件
- `thumbnails/{video_id}/`：检索结果缩略图、人脸 crop、OCR 命中帧
- `clips/{video_id}/`：前端播放检索结果时生成的临时视频片段缓存

`.npz` 文件通过：

```python
np.savez_compressed(...)
```

压缩保存。

`.json` 文件是 UTF-8 JSON，通过原子写入方式保存。

## 8. 当前服务器已有视频索引状态

当前服务器可见 4 个视频。Visual 已全部重新索引为 SigLIP2。

| 视频 | visual | face | asr | ocr |
|---|---|---|---|---|
| 五哈团美食速度挑战纯享_31min_1080p.mp4 | 有，SigLIP2 | 有 | 有 | 无 |
| 世界杯广告.mp4 | 有，SigLIP2 | 有 | 有，含 semantic | 无 |
| 球星牛奶广告 | 有，SigLIP2 | 有 | 有，含 semantic | 有，含 semantic |
| 给阿嬷的情书预告片 | 有，SigLIP2 | 有 | 有，含 semantic | 无 |

注意：`asr_semantic.npz` 和 `ocr_semantic.npz` 是可选增强索引，不存在时不会影响基本文本检索，只会缺少语义检索能力。

## 9. 当前召回合并与返回形式

搜索时每条通道先独立产生 candidate：

- visual：5s bucket candidate
- face：face track candidate
- asr：ASR chunk candidate
- ocr：OCR chunk candidate

然后系统按时间把接近或重叠的 candidate 合并为最终结果。

合并规则要点：

- visual-only 结果不会把相邻 5s bucket 无限串起来，避免“整段视频都被召回”。
- visual-only 只在时间真的重叠时合并。
- 如果有 face / asr / ocr 等其他通道锚点，近邻片段可以与 visual 合并。
- 最终结果默认最大时长约 15 秒。

最终返回给前端的结果包含：

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

其中 `evidence[]` 会记录每条通道自己的原始分数和诊断信息，例如：

```text
visual:
  raw_score
  visual_top1
  visual_top3
  visual_mean
  best_time
  percentile
  robust_z

face:
  raw_score / cosine
  confidence

asr / ocr:
  lexical_score
  semantic_score
  semantic_cosine
```
