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
| `visual` | 默认 5fps，5s bucket；可选 shot-aware segment | SigLIP2/CLIP image-text embedding | `visual.npz` | 默认 5s bucket；可选镜头/子段，按 segment 内最佳帧 MaxSim 排序 |
| `face` | 默认 1fps | InsightFace `buffalo_l`，ArcFace identity embedding | `face.npz` | 人脸 track |
| `asr` | ASR 模型 chunk | Whisper/FunASR 文本 + 可选 MiniLM semantic embedding | `asr.npz` | ASR chunk |
| `ocr` | 后端回退 0.05fps；前端新任务默认 1fps | RapidOCR PP-OCRv6（CPU/CUDA/Ascend）文本框 + 可选 MiniLM semantic embedding | `ocr.npz` | OCR sampled-frame |

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
      "decode_status": "complete",
      "segment_strategy": "shot",
      "segment_times": "explicit",
      "min_segment_ms": 800,
      "max_segment_ms": 8000,
      "shot_detector": "pyscenedetect_content",
      "shot_threshold": 0.2
    }
  }
}
```

Manifest 只保存少量元信息，不保存每帧/每 chunk 重复字段。`embedding_dim` 和 `embedding_dtype` 从数组 shape/dtype 或模型注册表推断，不在 manifest 中重复存储。

`segment_strategy`、`segment_times`、`min_segment_ms`、`max_segment_ms`、`shot_detector` 和 `shot_threshold` 是 visual 的可选字段。旧 v3 visual 索引没有这些字段时，默认视为固定 `segment_ms` 分段。

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
| `segment_frame_offsets` | `[segments_total + 1] int32` | 每个 visual segment 在 `frame_embeddings` 中的半开区间 `[start, end)` |
| `segment_times_ms` | `[segments_total, 2] int32` | 可选，shot-aware segment 的真实 `[start_ms, end_ms]` |

`segment_frame_offsets` 覆盖完整视频 timeline。某个 segment 没有成功解码帧时，offset start/end 相同；这允许部分解码失败或空 segment 不破坏数组对齐。

`segment_times_ms` 是 v3 optional shot-aware extension，不是所有 v3 visual 索引的必有字段：

```text
有 segment_times_ms -> 搜索结果使用显式镜头/子段时间
无 segment_times_ms -> 搜索结果回退到 segment_id * segment_ms 的固定时间窗口
```

前端按视频类型提供参数预设，但写入索引任务的仍是这几个明确字段：

| 预设 | `visual_segment_strategy` | 推荐 `visual_shot_detector` | 典型用途 |
|---|---|---|---|
| 固定分段 | `fixed` | 不发送 | 旧 v3 兼容、快速基线、检测不稳定素材 |
| 通用镜头 | `shot` | `simple` | 常规素材、剧情/纪实类混合镜头 |
| 广告 / MV | `shot` | `pyscenedetect_adaptive` | 快切素材，阈值更敏感，最长段更短 |
| 访谈长镜头 | `shot` | `pyscenedetect_content` | 长镜头/机位稳定素材，最短段更长，阈值更保守 |
| 自定义 | `shot` | 用户选择 | 手动设置 detector/min/max/threshold |

`visual_shot_detector` 可选值为 `simple`、`pyscenedetect_content`、`pyscenedetect_adaptive`。后端默认仍是 `simple`；PySceneDetect 仅在 `visual_segment_strategy="shot"` 时参与镜头边界检测，后续索引格式和检索读取逻辑不变。

召回逻辑：

```text
query text/image -> visual query embedding
-> 与 frame_embeddings 做 cosine
-> 每个 visual segment 取 top1 frame similarity
-> raw_score = visual_top1
-> Candidate.score = visual_rank_score = clip((raw_score + 1) / 2, 0, 1)
-> percentile / robust_z 用于视频内阈值判定和诊断，不作为跨视频排序主分数
-> 返回显式 segment_times_ms，或旧 v3 固定 segment_ms bucket
```

Visual evidence 主要字段：

```text
raw_score
visual_top1
visual_top3
visual_mean
visual_rank_score
best_time / best_ms
percentile
robust_z
unit_type = "segment"
unit_id = segment_id
features
```

Visual 结果预览图使用 `best_ms` 对应的最高相似度采样帧；后端按时间戳首次请求时抽帧并缓存，不再固定展示 segment 的第一张代表图。

排序说明：跨视频排序使用 raw visual_top1 的校准值 `visual_rank_score`；per-video percentile 能帮助找到每个视频内部最突出的片段，但只作为阈值判定和诊断 evidence。单帧偶然相似仍可能造成误召，后续见 `docs/ISSUES_AND_ROADMAP.md` 的 RQ-002。

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

Track id 就是数组行号。缩略图不再预存，检索命中后用 `best_shot_ms` 实时抽帧：

```text
GET /api/videos/{video_id}/frame?ms={best_shot_ms}
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

Speaker Diarization 与声纹检索已作为 ASR 的可选子阶段实现，不是独立检索通道。模型的 1.5 秒重叠短窗只用于边界分析与聚类；正式 `speaker.npz` 保存连续同 speaker 窗口合并、再按 ASR 自然边界切分后的自适应 turn 声纹、时间/ASR 引用和 track 缓存，不保存短窗。人物库统一管理人脸身份、视频内 speaker 绑定及代表声音。视频 speaker 接口按总说话时长排序，每人最多返回 5 个接近该人声纹中心的代表片段，完整 turn 向量仍用于同声纹检索。跨视频永久 Speaker ID 暂不自动合并。

2026-07-09 起，新建 ASR v3 索引使用分层 pipeline：

```text
audio_extract
-> model_transcribe
-> raw_transcript parser
-> retrieval_chunk_builder
-> MiniLM semantic embedding
-> asr.npz
```

生产默认只保存最终检索需要的数据；raw transcript、retrieval chunks 和 repair report 仅在调试开关开启时写入 `runtime/indexes/{video_id}/debug/`。删除 debug 目录不影响搜索。

当前默认策略：

```text
asr_engine = auto
asr_zh_model = iic/SenseVoiceSmall
asr_model = turbo
asr_language = auto
asr_vad_strategy = silero_12s
```

默认 `asr_engine=auto` 先用 faster-whisper turbo 做轻量语言 probe。短视频使用一个窗口；长视频从开头、主体中段和后段取 3 个 20s 窗口，按语言票数优先、置信度和次之决定主语言，再路由：

```text
zh / yue / cmn 等中文或方言 -> SenseVoiceSmall + Silero external VAD 12s
en / es / pt 等非中文      -> faster-whisper turbo + 24s 连续窗口 + word timestamps + 局部 builtin-VAD fallback
显式 ASR_ENGINE=funasr      -> 强制 SenseVoice/FunASR 路径
显式 ASR_ENGINE=faster-whisper -> 强制 faster-whisper 路径
```

SenseVoiceSmall 路径使用外置 Silero VAD 先把音频切成最长约 12s 的语音 group，再逐段送入 SenseVoiceSmall。parser 以模型原始文本为权威内容，word/character timestamp 只用于选择文本切片边界，避免重新拼 token 时破坏 `don't`、`V1`、`2015年` 等原文；对齐失败时保留整段原文。当模型缺 timestamp 时，用外层 VAD group 起止时间兜底。该路径不再依赖 FunASR 内置 FSMN VAD 作为默认切分器；`funasr_fsmn` 仅作为显式 fallback。

Faster-Whisper 路径先用 VAD 找语音区间，但按最长约 24s 的连续**原音频时间轴**窗口解码，主解码使用 `vad_filter=False`，保留模型 `segment.text` 和正常句子边界。只有某个窗口出现连续超过 12s 的无句末文本或异常长低边界 segment 时，才对同一个原音频窗口执行一次 builtin-VAD 局部 fallback；fallback 只有在文本覆盖、边界和重复指标更好时才替换主结果。它不再把 word timestamp 重组文本，也不把整个视频先压成稀疏 VAD 音频。

`retrieval_chunk_builder` 负责面向搜索的文本边界。常规 gap 为 500ms，短文本和同 5s bucket gap 为 1000ms，合并后的 chunk 不超过 12s；带 `unit_id` 的模型输出不能跨 decode/VAD window 合并。单个模型原始完整句如果已经超过 12s，不做无证据硬切，只标记 `long_chunk`。策略保持保守：只做文本规范化、同一 decode unit 内短碎片/近邻有限合并和朴素文本拼接；不再做 CJK/Latin 断词猜测或 false timestamp gap 修复。manifest 中记录 `language_route`、`route_reason`、`vad_strategy`、`raw_items`、`retrieval_chunks` 和 `chunk_builder_stats`。

2026-07-13 起，正式 ASR 后处理只保留模型无关的必要规则，再生成 semantic embedding：

```text
raw ASR chunks
-> 文本归一化：NFKC、繁体转简体、重复语气词前缀清理
-> 模型 adapter：输出原始文本、时间和 decode unit_id
-> chunk 合并：只在同一 unit 内按时间间隔合并，5s bucket 只作为轻量正向信号
-> semantic eligibility：纯语气/连接词、过短文本和明显不可信文字/时长比不写入 embeddings
-> quality_flags：记录 non_terminal_boundary、long_chunk、embedding_ineligible:<reason>
-> MiniLM semantic embedding
```

通用 `retrieval_chunk_builder` 不再包含 CJK/Latin 断词猜测、false timestamp gap 修复或 word-boundary repair 计数；这些过拟合规则已从正式路径移除。Whisper/faster-whisper 调用固定使用 `task="transcribe"`，避免把原语音翻译成英文。`semantic_eligible/reason/quality_flags` 存在于流水线对象和可选 debug JSON；精简 NPZ 不重复保存这些诊断字段，而由 `embedding_chunk_indices` 表达哪些文本实际生成了向量。新 manifest 会记录：

```text
task
requested_language
detected_language
tag_source
chunk_builder_stats
text_profile
```

实验结论记录在：

```text
docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md
docs/experiments/asr/2026-07-13-asr-production-chunk-pipeline.md
docs/experiments/asr/2026-07-14-asr-dual-pool-retrieval.md
```

当前常用 ASR 引擎策略：

```text
auto route：中文/方言 -> SenseVoiceSmall；非中文 -> faster-whisper turbo
```

`asr.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `chunk_times_ms` | `[num_chunks, 2] int32` | `[start_ms, end_ms]` |
| `chunk_emotions` | `[num_chunks] utf8 bytes` | optional SenseVoice emotion tag, for example `neutral` or `happy`; empty string when unavailable |
| `chunk_audio_events` | `[num_chunks] utf8 bytes` | optional SenseVoice audio event tag, for example `speech`, `bgm`, or `speech|bgm`; empty string when unavailable |
| `texts` | `[num_chunks] str` | ASR chunk 文本 |
| `embeddings` | `[num_semantic_chunks, 384] float16` | 可选 MiniLM semantic embedding；不可用时为空数组 |
| `embedding_chunk_indices` | `[num_semantic_chunks] int32` | 每条 semantic embedding 对应的 `texts` 行号 |

ASR semantic 是可选能力。语义模型不可用时仍保留 `chunk_times_ms/texts`，搜索自动退回 lexical。

当前 `chunk_emotions` 和 `chunk_audio_events` 只作为可选元数据保存，尚不参与召回或排序。后续接入原则和人工听查结论统一维护在 `ISSUES_AND_ROADMAP.md` 的 RQ-003G；在验证前不得按标签字面做硬过滤，也不得把标签拼入 semantic embedding 文本。

召回逻辑：

```text
query text -> lexical score
可选 query text -> MiniLM embedding -> semantic cosine
combined_score = max(lexical_score, 0.65 * semantic_score + 0.35 * lexical_score)
```

当前候选判定使用 `semantic_score >= 0.55 or lexical_score >= 0.25`。2026-07-14 在 74 条有答案和 56 条无答案查询上的校准表明，现行无答案 FAR 为 100%；保持约 95% target recall 时，验证集 FAR 仍接近 90%。因此这两个阈值目前只能理解为宽松结果分区，不能对外宣称“有答案/无答案”，生产参数也没有因该实验调整。详见 `docs/experiments/asr/2026-07-14-asr-no-answer-threshold-calibration.md`。

Lexical 先做 NFKC、繁简和搜索标点归一化。完整 query 子串命中记为 `1.0`，否则计算 query bigram 覆盖率。Lexical 与 semantic 分别形成候选池；ASR-only 搜索保留 combined score 的主排序，同时对 `lexical_score >= 0.50` 的强字面候选做稀疏保底：先完整保留主排序前 3 条，此后每 8 条主结果最多插入 1 条尚未出现的 lexical 候选，候选池大小为 50。

保底只改变最终返回顺序，不改写 calibrated `score`。多通道搜索仍按原有跨通道融合执行，不应用 ASR-only lexical reserve。42 条当前素材查询的 A/B 表明，直接提高 CJK 局部覆盖分或使用 weighted RRF 都会降低 Top-1，因此未采用；详见 `docs/experiments/asr/2026-07-14-asr-dual-pool-retrieval.md`。

2026-07-14 的 25-source / 82-query 离线实验表明，`gte-multilingual-base` 在 ASR chunk 排序上明显优于当前 MiniLM；最佳离线融合为 90% GTE semantic + 10% legacy bigram，并把 `lexical_score >= 0.50` 的强字面命中放入独立优先区。GTE 为 768 维，主权重约 611 MB，本机缓存约 628 MB；当前已明确暂缓，**没有替换本节现行生产协议，也不重建索引**。若未来恢复该方向，仍需完成 GTE 专属 confidence/threshold、真实 API result-level A/B，以及 ASR/OCR semantic 模型配置解耦；详见 `docs/experiments/asr/2026-07-14-asr-retrieval-v2-six-stage.md`。

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
RapidOCR / PP-OCRv6（CPU/CUDA/Ascend 统一）
sample_fps = 1.0（前端新任务默认；后端未指定时回退 0.05）
decode_height = 720
min_confidence = 0.5
```

OCR v3 已重构为 frame-native 格式，不兼容旧 OCR v3；升级后必须重建 OCR 索引。文本与 score 在 box 级保存，帧文本在搜索时由该帧的 boxes 拼接得到。

`ocr.npz` v3：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `frame_times_ms` | `[num_frames] int32` | 每个 OCR 采样帧的时间戳 |
| `frame_windows_ms` | `[num_frames, 2] int32` | 每个采样帧对应的检索窗口 `[start_ms, end_ms]` |
| `embeddings` | `[num_semantic_chunks, 384] float16` | 可选 MiniLM semantic embedding；不可用时为空数组 |
| `embedding_frame_indices` | `[num_semantic_frames] int32` | 每条 semantic embedding 对应的 frame 行号 |
| `box_frame_indices` | `[num_boxes] int32` | 每个 OCR box 所属的 frame 行号 |
| `box_texts` | `[num_boxes] str` | OCR box 文本 |
| `box_scores` | `[num_boxes] float32` | OCR box 置信度 |
| `boxes` | `[num_boxes, 4, 2] float32` | 归一化到 `[0, 1]` 的四点框 |

缩略图不再预存，检索命中后用该 OCR 帧的 `frame_ms` 实时抽帧：

```text
GET /api/videos/{video_id}/frame?time={frame_ms / 1000}
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
