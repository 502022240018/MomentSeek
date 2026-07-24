# ASR Pipeline Refactor Design

日期：2026-07-09

## 背景

当前 ASR 通路已经经过多轮快速实验和修复，包含 Whisper 参数修正、SenseVoice/FunASR 接入、faster-whisper turbo 备选、本地模型优先、短 chunk 合并、幻觉/重复文本 guard、MiniLM 语义 embedding 等工作。这些改动让 ASR 整体可用性提升明显，但也把实验逻辑、生产逻辑和临时修复叠在了一条索引链上。

目前最明显的问题是：生产索引中出现大量不自然切分，甚至出现中文词内部断裂，例如 `孤 / 独敏感`、`永 / 远`、`很 / 难受`。用户听查确认这些断裂在原音频中不是实际停顿，说明当前索引不能继续把模型 timestamp gap 当成最终语义边界。

## 目标

1. 重构 ASR pipeline，让前处理、模型转写、原始转写解析、检索 chunk 构建、语义 embedding 各自职责清楚。
2. 默认生产索引只保存搜索必需字段，避免长期存储 raw 输出导致索引膨胀。
3. 调试期可通过开关保存 raw/debug artifact，用来定位模型、VAD、timestamp、parser、chunk builder 的具体问题。
4. 让实验路径和生产路径复用同一套核心 pipeline，避免同名方案在实验和生产中行为不一致。
5. 将最终检索文本从“raw segment 直接入库”改为“面向检索的 retrieval chunk”，优先保证语义完整和可搜索性。

## 非目标

1. 本次设计不重新选择最终 ASR 模型，只为 SenseVoiceSmall 和 faster-whisper turbo 提供清晰落地点。
2. 本次设计不要求长期保存完整 raw transcript。
3. 本次设计不改变 visual、face、ocr 通路。
4. 本次设计不把 LLM 引入运行时索引流程。

## 当前问题

### 前处理策略混杂

当前代码和实验中同时存在：

- 整段音频交给 FunASR 内置 FSMN-VAD。
- faster-whisper builtin VAD。
- 实验脚本中的 Silero VAD 外置切分。
- Whisper 60s/90s/120s window + overlap。
- sidecar 字幕路径。

这些策略没有统一抽象，导致实验结论和生产索引实际路径容易错位。

### parser 与 chunking 混在一起

FunASR/SenseVoice parser 目前既解析模型输出，又根据 timestamp、标点和最大时长主动切 chunk。parser 应该尽量忠实还原模型输出；最终检索 chunk 应由单独的 retrieval chunk builder 生成。

### raw transcript 与 retrieval chunk 没有分层

当前 `asr.npz` 主要保存最终 chunks，缺少调试时可追溯的 raw artifact。出问题时很难判断是模型听错、VAD 切错、timestamp 漂移、parser 切坏，还是后处理合并策略错误。

### 过度信任 timestamp gap

用户听查确认，索引中的部分大 gap 是假 gap，不是音频真实停顿。最终检索文本不能只依赖 timestamp gap 决定是否合并。

### 词边界保护不足

当前后处理主要看 gap、bucket 和时长，不理解 CJK 词边界，也不理解 latin 语言的单词边界。中文 `孤独`、`永远`，英文单词中间，都可能被错误切断。

## 目标架构

ASR pipeline 拆成以下层：

```text
audio_preprocess
  -> speech_units
  -> model_transcribe
  -> raw_transcript
  -> retrieval_chunk_builder
  -> semantic_embedding
  -> asr.npz
```

### audio_preprocess

职责：

- 使用 ffmpeg 抽取 16k mono wav。
- 记录音频抽取耗时和音频时长。
- 处理无音轨视频，返回空 ASR 索引。

输出：

- `audio.wav`
- `audio_seconds`
- metrics: `audio_extract_seconds`

### speech_units

职责：

- 决定模型输入音频单元。
- 支持不同策略：
  - `none`: 整段输入，主要用于 debug。
  - `funasr_fsmn`: FunASR 内置或配套 VAD。
  - `silero`: 外置 Silero VAD，作为重点产品化候选。
  - `faster_whisper_builtin`: faster-whisper 内置 VAD。
  - `fixed_window`: 固定窗口 + overlap，仅保留为实验和 fallback。

输出：

- 内存中的 `SpeechUnit[]`：
  - `unit_id`
  - `start_ms`
  - `end_ms`
  - `core_start_ms`
  - `core_end_ms`
  - `source`

生产默认不保存该数组。调试开启时保存 `debug/asr_speech_units.json`。

### model_transcribe

职责：

- 对每个 speech unit 调用指定模型。
- 支持：
  - `sensevoice`: 默认中文和中文主素材候选。
  - `faster_whisper_turbo`: 多语言和高效果备选。
  - `sidecar`: 字幕文件输入。
- 不在这一层做最终检索 chunk 合并。

输出：

- 内存中的 `RawTranscriptItem[]`。
- 每个 item 包含：
  - `unit_id`
  - `start_ms`
  - `end_ms`
  - `text`
  - `source`
  - 可选模型诊断字段，例如 `avg_logprob`、`no_speech_prob`、`compression_ratio`。

生产默认不保存完整 raw。调试开启时保存 `debug/asr_raw_transcript.json` 和可选模型原始输出摘要。

### raw_transcript parser

职责：

- 解析模型输出为 raw transcript items。
- 只做必要清洗，例如去 rich transcription 控制标签、空白归一化。
- 不在 parser 中因为 8s/12s 之类规则强制生成最终检索 chunk。

约束：

- parser 不负责“语义最佳切分”。
- parser 不因为 timestamp gap 自动判断语义断开。
- parser 需要记录异常：
  - timestamp 缺失。
  - timestamp 长度与文本长度不匹配。
  - timestamp 大跳变。
  - item 异常过长。

### retrieval_chunk_builder

职责：

- 从 raw transcript items 构建最终用于搜索的 retrieval chunks。
- 这是 ASR 检索质量的核心层。

输入：

- `RawTranscriptItem[]`
- language profile
- chunk builder config

输出：

- `RetrievalChunk[]`
- 每个 chunk 包含：
  - `start_ms`
  - `end_ms`
  - `text`
  - `source_item_ids`
  - `semantic_eligible`
  - `quality_flags`

核心规则：

1. 优先保证文本语义完整，不让 raw timestamp 直接决定最终文本边界。
2. CJK 文本需要词边界保护：
   - 避免 `孤 / 独`、`永 / 远`、`很 / 难受` 这类断裂。
   - 前一段以单个 CJK 字结尾、后一段以 CJK 开头、且前段无句末标点时，允许跨较大假 gap 修复。
   - 可使用轻量词典或 jieba 分词做辅助，但不要把重型 NLP 模型放入运行时。
3. Latin 文本需要单词边界保护：
   - 避免 `whatarew / edoing`、`Nobill / ion` 这类单词中断。
   - 按空格、标点、字母连续性判断。
4. 对明显连续但 timestamp 假 gap 的片段，允许 text-first 合并。
5. 对真实长停顿、话题切换、句末标点后的新句，仍保持分开。
6. 对过长 chunk 按标点、停顿和语义边界再切，而不是机械按固定秒数切。

生产建议参数初始值：

- 目标 chunk 时长：8s 到 18s。
- 软上限：25s。
- 硬上限：35s。
- 假 gap 修复候选上限：8s，只有文本边界强烈提示需要修复时使用。
- 单字断裂修复优先级高于普通 gap 限制。

### semantic_embedding

职责：

- 只对 retrieval chunks 生成 MiniLM embedding。
- 不对 raw transcript items 生成 embedding。
- 低信息、疑似幻觉、明显乱码的 chunk 可保留文本但不生成 semantic embedding，或在检索中降权。

## 存储策略

### 生产默认

`asr.npz` 只保存搜索必需字段：

```text
chunk_times_ms
texts
embeddings
embedding_chunk_indices
```

`index_manifest.json` 保存轻量元信息和统计：

```json
{
  "engine": "funasr",
  "model_key": "iic/SenseVoiceSmall",
  "language_route": "zh_sensevoice",
  "vad_strategy": "silero",
  "raw_items": 552,
  "retrieval_chunks": 410,
  "word_boundary_repairs": 37,
  "fake_gap_repairs": 12,
  "semantic_chunks": 398
}
```

生产默认不保存完整 raw transcript、speech units 或模型原始输出。

### 调试模式

通过配置开启：

```text
ASR_DEBUG_ARTIFACTS=true
ASR_SAVE_RAW_TRANSCRIPT=true
```

调试模式额外保存：

```text
runtime/indexes/{video_id}/debug/asr_speech_units.json
runtime/indexes/{video_id}/debug/asr_raw_transcript.json
runtime/indexes/{video_id}/debug/asr_retrieval_chunks.json
runtime/indexes/{video_id}/debug/asr_repair_report.json
```

调试 artifact 不作为搜索运行时依赖。删除 debug 目录不影响搜索。

## 语言路由

默认不再全局强制 `zh`。语言策略需要分层：

1. 用户明确指定语言时尊重用户选择。
2. 用户未指定时使用 `auto`。
3. 中文主素材优先 SenseVoiceSmall。
4. 明确英语、西语、多语言混杂素材优先 faster-whisper turbo 或实验确认后的多语言路由。
5. 方言素材单独作为评估样本，不假定普通话路径最佳。

manifest 中记录：

- `requested_language`
- `detected_language`
- `language_route`
- `route_reason`

## Metrics

ASR job metrics 需要拆阶段：

```text
audio_extract_seconds
speech_unit_seconds
model_load_seconds
decode_seconds
raw_parse_seconds
retrieval_chunk_seconds
semantic_embedding_seconds
save_seconds
total_elapsed_seconds
```

质量统计：

```text
raw_items
retrieval_chunks
merged_items
word_boundary_repairs
fake_gap_repairs
long_chunks
semantic_ineligible_chunks
timestamp_jump_warnings
language_route
vad_strategy
```

这些统计必须在 debug 关闭时仍写入 manifest/job metrics，方便长期观测。

## 实验与生产复用

后续 ASR 实验脚本不再复制 parser 和后处理逻辑。实验脚本应该调用生产 pipeline 的核心函数，只替换配置：

- 模型配置。
- VAD 策略。
- chunk builder 配置。
- debug artifact 输出目录。
- eval truth 路径。

这样实验结果才代表生产可落地行为。

## 迁移策略

1. 新代码只保证新 schema 和新 retrieval chunks 最干净。
2. 旧 ASR 索引不做自动兼容修复。
3. 切换后要求重跑 ASR 索引。
4. visual、face、ocr 现有索引不受影响。
5. 在全量重跑前，先用 3 到 5 条代表性视频验证：
   - 中文剧集：`电视剧昨夜降至04.mp4`
   - 长剧集：`天c游xi...mkv`
   - 综艺：五哈素材
   - 英语广告：世界杯广告
   - 方言预告：给阿嬷的情书预告片

## 测试计划

单元测试：

- parser 不在词内部强制切 retrieval chunk。
- CJK 单字断裂修复：`孤` + `独敏感`、`永` + `远`、`很` + `难受`。
- Latin 单词断裂修复。
- raw transcript debug 关闭时不写 debug 文件。
- debug 开启时写出 expected artifact。
- semantic embedding 只基于 retrieval chunks。

集成测试：

- 以小音频或 sidecar fixture 跑完整 ASR pipeline。
- 验证 `asr.npz` 字段保持搜索所需最小集合。
- 验证 manifest 写入阶段耗时和质量统计。

人工评估：

- 导出每个视频 retrieval chunks，一行一个 chunk。
- 对比旧索引和新索引中的断词数量。
- 抽查搜索 query 是否更容易命中自然短语。

## 成功标准

1. 当前发现的断词样例不再出现在 retrieval chunks 中：
   - `孤 / 独敏感`
   - `永 / 远`
   - `很 / 难受`
2. `asr.npz` 默认不长期保存 raw transcript。
3. debug 关闭时索引体积接近当前 ASR 索引。
4. debug 开启时可以定位 raw、parser、retrieval chunk 之间的差异。
5. 实验脚本和生产索引使用同一套核心 chunk builder。
6. ASR job metrics 能解释耗时差异，不再只给一个总耗时。

## 设计决策

第一版实施采用以下默认值，避免继续扩散范围：

1. VAD 路径先实现 Silero external 作为可选正式路径，保留 FunASR FSMN integrated 作为 fallback。
2. `retrieval_chunk_builder` 第一版使用规则为主，jieba 作为可选轻量 CJK 词边界辅助，不引入重型 NLP 依赖。
3. 默认语言恢复为 `auto`，并记录 route 诊断；对明确非中文素材允许手动指定 faster-whisper turbo。
4. 多语言自动模型路由第一版只做诊断和手动覆盖，不直接自动切换默认模型，避免在缺少足够评估前引入不可解释行为。
