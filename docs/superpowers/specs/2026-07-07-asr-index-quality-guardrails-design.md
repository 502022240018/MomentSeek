# MomentSeek ASR 索引质量与防再犯设计

日期：2026-07-07

## 背景

MomentSeek 当前 ASR 通道保存 Whisper/FunASR 或 sidecar transcript 的 chunk 文本，并为这些 chunk 生成 MiniLM 语义向量：

```text
audio / transcript
-> raw ASR chunks
-> asr.npz: chunk_times_ms, texts, embeddings, embedding_chunk_indices
-> search: lexical + semantic
```

现有索引暴露出两类质量问题：

1. chunk 粒度不稳定：存在大量 `<1s` 短 chunk、单字 chunk、异常长但文本很少的低信息 chunk。
2. ASR 输出语言可能不符合实际素材：例如中文台词视频的旧索引中出现英文翻译稿。当前仓库代码不显式使用 Whisper `translate`，但索引 manifest 也没有记录 `task=transcribe/translate`，因此旧索引无法追溯任务意图。

本设计先优化 ASR 索引质量底座，不更换 ASR 大模型，不引入额外语言检测模型，不增加二次 ASR 预检作为默认流程。

## 目标

1. ASR 主通道只保存原语言转写文本，不保存翻译稿。
2. 支持多语言素材，默认仍允许 `asr_language=auto`。
3. 在不明显增加索引耗时的前提下，记录 ASR 文本语言画像和质量 warning。
4. 在生成 semantic embedding 前，对 ASR chunk 做保守后处理：
   - 文本清洗。
   - 合并过短 chunk。
   - 处理异常长低信息 chunk。
   - 可选参考 visual segment / shot 信息，为同一局部上下文内的碎片合并提供正向加成。
5. semantic embedding 基于后处理后的 chunk 文本生成。
6. 在现有素材上跑多组 chunk 合并策略实验，输出对比报告，用人工/LLM 评审选择第一版默认参数。
7. 保持 v3 ASR 索引结构简洁，避免冗余数组和大字段。

## 非目标

本阶段不做：

- 自动翻译 transcript。
- 多语言字幕生成。
- 默认二次 ASR 预检。
- 更换 text embedding 模型。
- Whisper / FunASR 模型质量大评测。
- 前端复杂 ASR 诊断 UI。

翻译如果后续需要，应作为独立的 `transcript_translation` 或展示增强层，不污染 ASR 原文索引。

## 设计原则

ASR 通路按“原文转写优先”设计：

```text
task = transcribe
language = auto | zh | en | ...
```

`language` 是识别提示，不是输出语言转换。中文素材输出中文，英文素材输出英文，西语素材输出西语。系统不把所有素材强制转成中文，也不把所有素材强制转成英文。

默认策略：

```text
默认上传:
  asr_language = auto
  asr_task = transcribe
  做轻量文本画像，只 warning，不阻塞

用户知道素材语言:
  可指定 zh/en/auto
  指定语言时 mismatch warning 更严格

永不默认 translate:
  translate 只能作为未来独立能力
```

## ASR 任务元信息

Manifest 中 ASR channel 增加少量小元信息：

```json
{
  "file": "asr.npz",
  "engine": "whisper",
  "model_key": "small",
  "task": "transcribe",
  "requested_language": "auto",
  "detected_language": "zh",
  "semantic_model_key": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_space": "minilm-text-semantic",
  "decode_status": "complete",
  "semantic_status": "complete",
  "quality_warnings": ["language_profile_mismatch"]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `task` | 固定为 `transcribe`，明确 ASR 主索引不是翻译稿 |
| `requested_language` | 用户或任务请求语言，默认 `auto` |
| `detected_language` | ASR 引擎返回的语言；没有时为空或省略 |
| `quality_warnings` | 小列表，只记录需要人工注意的质量风险 |

`text_language_profile` 不写入 manifest 作为默认必备字段。画像统计可放在 result metrics 或诊断报告中；manifest 只保存 warning，保持小而稳定。

## 轻量质量画像

ASR 完成后，用已生成文本做字符级统计，不引入额外模型：

```text
cjk_ratio
latin_ratio
mixed_ratio
empty_ratio
short_chunk_ratio
low_info_ratio
```

画像只扫字符串，开销相对 ASR 转写可以忽略。

Warning 规则第一版保持保守：

```text
requested_language = zh 且 latin_ratio 很高 -> language_profile_mismatch
requested_language = en 且 cjk_ratio 很高 -> language_profile_mismatch
short_chunk_ratio 过高 -> many_short_chunks
low_info_ratio 过高 -> many_low_info_chunks
```

当 `requested_language=auto` 时，不做强判错，只记录 metrics。多语言视频可能自然出现 mixed 画像，不应误报。

## Chunk 后处理

后处理发生在 raw ASR chunks 和 semantic embedding 之间：

```text
raw ASR chunks
-> normalize text
-> merge short chunks
-> handle long low-info chunks
-> processed chunks
-> semantic embeddings
-> asr.npz
```

第一版规则保守，目标是减少明显坏 chunk，不重写 transcript 语义。

### 文本清洗

保留原语言文本，只做轻量清洗：

```text
去掉首尾空白
合并连续空白
移除空文本 chunk
保留中英文、数字、常见标点
中文 chunk 可统一简体
```

中文简体化优先使用 OpenCC；如果 OpenCC 不可用，再使用当前小型手写映射作为 fallback。OpenCC 只作为轻量字符串规范化，不引入额外推理开销。

存储与检索策略：

```text
中文 chunk:
  texts 存储简体化后的文本
  semantic embedding 使用简体化后的文本

非中文或混合语言 chunk:
  preserve 原文，仅做空白清洗

query:
  lexical 搜索前同样做 OpenCC / fallback 规范化
```

不做自动翻译，不做激进 stopword 删除，不改写语序。

### 短 chunk 合并

短 chunk 定义第一版：

```text
duration < 1.0s
或文本长度很短，例如 CJK 1-2 字、Latin 1-2 token
```

合并优先级：

1. 优先合并到时间最近的后一个 chunk。
2. 如果后一个 chunk 间隔过大，则尝试合并到前一个 chunk。
3. 合并需要满足：
   - 相邻 gap <= 1.2s。
   - 合并后总时长 <= 8s。
   - 合并后文本长度不超过合理上限。
   - 不对 group 内 chunk 数做硬限制；chunk 数只进入 metrics/debug。

合并后时间范围取合并组的最早 start 和最晚 end，文本按时间顺序拼接。

第一版合并应以语句间时间间隔为主信号：

```text
gap <= 700ms:
  正常合并候选

短碎片 gap <= 1500ms:
  可并入最近邻

gap > 2500ms:
  默认不合并
```

如果存在可用 visual segment 信息，它是合并的正向证据，不是阻断边界：

```text
same fixed 5s bucket:
  轻微放宽 gap

same shot-aware segment:
  更明显放宽 gap

different segment:
  不惩罚，只是不加成
```

建议第一版阈值：

```text
无 visual 加成:
  normal_gap <= 700ms
  short_gap <= 1500ms

同 fixed bucket:
  normal_gap <= 1200ms
  short_gap <= 2000ms

同 shot-aware segment:
  normal_gap <= 1500ms
  short_gap <= 2500ms

硬限制:
  merged_duration <= 8s
  merged_text 不超过长度上限
```

ASR 后处理函数应支持可选边界参数，但不能依赖 visual 索引存在：

```text
postprocess_asr_chunks(chunks, segment_times_ms=None, segment_kind="none|fixed|shot")
```

没有 visual 信息时只按 ASR 自身 gap 合并；有 visual 信息时，同 segment 内的碎片可以更积极地合并。

### 异常长低信息 chunk

异常长低信息 chunk 第一版定义：

```text
duration > 8s
且文本字符数很少
或 chars_per_second 极低
```

处理方式：

```text
保留 chunk 文本和时间，用于展示与 lexical fallback
不为该 chunk 生成 semantic embedding
记录 low_info 标记到内部处理结果或 metrics
```

这样避免 “26 秒只有 Sh” 这类文本污染 semantic 排名，同时不丢失原始可见信息。

### 长文本 chunk

第一版不主动拆分所有长文本。仅当 chunk 同时满足：

```text
duration 明显过长
文本较长
存在明显标点或句界
```

才按标点做保守拆分。没有可靠句界时不拆，避免制造错误时间戳。

## ASR v3 数组

保持 `asr.npz` 简洁：

| 字段 | shape | 说明 |
|---|---|---|
| `chunk_times_ms` | `[num_chunks, 2] int32` | 后处理后的 chunk 起止时间 |
| `texts` | `[num_chunks] str` | 后处理后的 ASR 原文文本 |
| `embeddings` | `[num_semantic_chunks, dim] float16` | 对可参与 semantic 的 chunk 生成 |
| `embedding_chunk_indices` | `[num_semantic_chunks] int32` | embedding 对应 `texts` 行号 |

不新增大数组。低信息 chunk 是否参与 semantic 由 `embedding_chunk_indices` 是否包含该行体现。

## 搜索影响

搜索层继续读取 v3 ASR：

```text
lexical: 对所有 texts 生效
semantic: 只对 embedding_chunk_indices 覆盖的 chunks 生效
combined: 沿用 lexical + semantic 融合
```

后处理后的 chunk 更适合 semantic retrieval：

- 短单字 chunk 会被合并到上下文里。
- 低信息异常长 chunk 不会靠 semantic 排前。
- 返回片段边界更接近一句完整表达。

## 验证方案

实现后至少验证：

1. 单元测试：
   - 短 chunk 合并。
   - 长低信息 chunk 不生成 semantic embedding。
   - `task=transcribe` 写入 manifest。
   - 多语言 `auto` 不触发强 mismatch。
2. 诊断导出：
   - 重导 ASR chunk 一行一条文本。
   - 对比短 chunk 数、低信息 chunk 数、semantic coverage。
   - 对比不同合并策略的样例输出，人工/LLM 辅助判断哪种更适合 semantic 聚合。
3. 真实视频 smoke：
   - 中文剧集重跑后 ASR 文本为中文。
   - 书籍纪录片重跑后按实际音频语言输出原语言文本。
   - 中文 query 与英文/多语言素材仍可通过 semantic 部分召回，但原文展示不被翻译污染。

## 风险

1. 过度合并可能让片段变长，降低定位精度。
2. 多语言混合视频的字符画像可能看起来异常，但实际正常。
3. 不做预检意味着长视频仍可能跑完后才发现 ASR 质量 warning。

第一版通过保守阈值和 warning 而非阻塞来降低风险。

## 调参方法

合并阈值不直接拍死。实现时必须包含一个离线参数实验工具，用现有素材跑多组策略：

```text
strategy_a: 只按 ASR gap 合并
strategy_b: ASR gap + fixed bucket 加成
strategy_c: ASR gap + shot segment 加成
strategy_d: 更保守 gap
strategy_e: 更积极 short-fragment merge
```

诊断报告输出每个策略的：

```text
chunk_count
short_chunk_count
semantic_coverage
mean / p50 / p95 duration
典型合并前后样例
被排除 semantic 的 low-info chunk 样例
同一原始片段在不同策略下的合并结果对照
```

参数选择流程：

```text
1. 对当前 runtime-server/indexes 中已有 ASR 索引运行策略实验。
2. 报告按视频列出指标和样例。
3. 抽取同一时间范围下不同策略的合并结果。
4. 用人工/LLM 作为离线评审，判断哪种结果更适合 semantic 聚合。
5. 把选中的参数写成默认配置，并把实验报告路径记录到 docs/experiments/asr/。
```

LLM 只作为离线评审辅助，用来判断“合并后的文本是否更像一个适合语义检索的表达单位”。这不进入运行时，不引入 API 依赖，也不影响索引速度。

## 后续可选增强

- 为特定长视频增加手动预检按钮。
- 增加 ASR 诊断页面展示 language profile 和 warning。
- 评估 BGE/E5/GTE 等文本 embedding 模型。
- 增加独立 transcript translation 通道，用于展示或跨语言辅助检索。
