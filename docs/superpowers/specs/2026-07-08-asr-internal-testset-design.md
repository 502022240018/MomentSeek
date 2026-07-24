# MomentSeek ASR 内部基础测试集设计

日期：2026-07-08

## 背景

MomentSeek 当前 ASR 通道支持两类输入：上传视频直接跑 ASR，或随上传视频提供 `json/srt/vtt` transcript sidecar。后者会在索引时作为 `engine="sidecar"` 进入 `asr.npz`，并参与后续 lexical 与 semantic 检索。

现有平台已经有一条来自真实上传视频的字幕 truth：

```text
eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.sidecar.json
eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.srt
eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.jsonl
```

该素材来自平台上传的 `电视剧昨夜降至04.mp4`，包含 653 条内嵌中文字幕片段，可作为真实长视频 ASR truth 样本。

本设计要建立一个面向平台检索场景的 ASR 内部基础测试集。测试集优先服务内部实验和回归，不作为公开可分发数据集。

## 目标

1. 构建一个中文优先、贴近平台真实上传内容的 ASR 基础测试集。
2. 总量以 12 小时量级为目标，硬上限 15 小时。
3. WenetSpeech 作为主干来源，覆盖中文视频、播客、多域真实语音，目标约 4 小时。
4. 包含一定比例长视频或长音频封装样本，长样本时长占比约 40%。
5. 纳入平台已有的 `昨夜降至04` 字幕素材，作为真实上传长视频样本。
6. 每个样本都能适配平台上传和 ASR sidecar 链路。
7. 保留可复现的抽样 manifest、来源信息、许可证或使用限制、hash 与 truth 文件。

## 非目标

本阶段不做：

- 训练或微调 ASR 模型。
- 自动评价所有 ASR 模型的 WER/CER 排行。
- 公开发布原始媒体或衍生视频包。
- 引入服务器 NPU 资源。
- 为每个开源数据集做完整下载镜像。
- 建立视觉、OCR、Face 的综合多模态评测集。

## 数据集定位

测试集是内部实验资产，由三层组成：

```text
source cache:
  本机数据目录中的原始音频/视频缓存，不进 git

generated media:
  静态画面 MP4 或平台已有真实视频，不进 git

repo assets:
  manifest、truth sidecar、srt/vtt、sources、生成说明，可进 git
```

对 WenetSpeech、GigaSpeech 等包含第三方原始媒体的来源，manifest 必须标注：

```text
internal_use_only = true
redistribute_original_media = false
notes = "仅内部实验，不外发原始媒体或生成后完整媒体包"
```

## 时长规划

第一版目标总量为 12 小时左右，最多不超过 15 小时。若抽样过程中某一来源下载或解析成本过高，可以先降级到 10 小时以上，但 manifest 中必须记录缺口。

| 维度 | 来源 | 目标时长 | 上限时长 | 形态 |
|---|---:|---:|---:|---|
| 中文视频/播客/多域真实语音 | WenetSpeech | 4.0h | 5.0h | 音频封装 MP4 |
| 中文真实长视频/字幕样本 | 昨夜降至04 | 1.0-1.5h | 1.5h | 原平台视频 + 已提取字幕 truth |
| 中文会议/多人/远场/重叠说话 | AliMeeting | 1.5h | 2.0h | 音频封装 MP4 |
| 中文干净朗读/普通话基线 | AISHELL-1 | 1.0h | 1.5h | 音频封装 MP4 |
| 中文口音/短句/众包鲁棒性 | Common Voice zh-CN/zh-HK/zh-TW | 1.0h | 1.5h | 音频封装 MP4 |
| 多语种 CJK/近邻语言 | FLEURS zh/ja/ko/yue | 1.0h | 1.5h | 音频封装 MP4 |
| 英文真实语音补充 | GigaSpeech 或 Earnings-22 | 0.5h | 1.0h | 音频封装 MP4 |
| 难例补位 | WenetSpeech 或平台已有字幕视频 | 1.0h | 2.0h | 原视频或封装 MP4 |

推荐首版落点：

```text
WenetSpeech: 4.0h
昨夜降至04: 1.2h
AliMeeting: 1.5h
AISHELL-1: 1.0h
Common Voice zh*: 1.0h
FLEURS CJK: 1.0h
GigaSpeech/Earnings-22: 0.8h
WenetSpeech/platform hard cases: 1.5h
total: about 12.0h
```

如需要逼近 15 小时，上调顺序为：

```text
1. WenetSpeech +1.0h
2. AliMeeting +0.5h
3. 平台已有字幕视频 +0.5h
4. Common Voice/FLEURS +0.5h
5. GigaSpeech/Earnings-22 +0.5h
```

## 长样本策略

长样本用于覆盖平台真实上传的长视频检索问题。第一版要求：

- 至少 1 条真实平台长视频：`昨夜降至04`。
- 至少 2 条 WenetSpeech 长封装样本，每条 20-40 分钟。
- 至少 1 条 AliMeeting 长会议样本，每条 15-30 分钟。
- 长样本累计时长约占总量 40%。
- 其余样本以 5-10 分钟中短封装视频为主，方便快速回归和定位问题。

音频封装 MP4 使用静态画面，不引入额外视觉语义。静态画面只显示数据源、语言、维度、样本编号和内部使用提示，便于人工识别。

## 候选来源

### WenetSpeech

官方页：https://wenet-e2e.github.io/WenetSpeech/

用途：中文视频、播客、多域真实语音主干。

官方说明其数据来自 YouTube 和 Podcast，包含高置信标注和弱标注，按说话风格和场景覆盖多个类别。该来源非常贴近平台上传内容，但官方也说明数据仅用于非商业目的，且音频版权仍归原始视频或音频所有者。

使用规则：

- 只用于内部实验。
- manifest 保留原始 URL、segment id、置信度、类别、抽样 seed。
- 不把原始音频或封装后完整视频提交到 git。
- 优先使用 high-label 或 dev/test 中高质量片段。

### AliMeeting

官方页：https://www.openslr.org/119/

用途：中文会议、多人、远场、近场、重叠说话。

OpenSLR 页面标注许可证为 CC BY-SA 4.0，并说明该数据来自真实会议，包含 2-4 人、15-30 分钟会议 session、远场麦克风阵列和近场头戴麦克风数据。

使用规则：

- 优先抽 eval/test 或小量 train session。
- 同一会议保留连续时段，避免只抽碎片导致场景失真。
- 标注远场/近场、说话人数、重叠说话比例。

### AISHELL-1

官方页：https://www.openslr.org/33/

用途：中文干净朗读、普通话基线、口音区域覆盖。

OpenSLR 页面标注许可证为 Apache License v2.0，包含安静室内环境下普通话录音和专业转写。

使用规则：

- 抽取多说话人、多口音区域。
- 拼接为 5-10 分钟短基线视频。
- 用于判断 ASR 基础能力和文本规范化是否退化。

### Common Voice Chinese

官方页：https://commonvoice.mozilla.org/en/datasets

用途：中文短句、众包口音、录音设备差异。

使用规则：

- 优先 zh-CN，少量 zh-HK、zh-TW。
- 仅使用 validated clips。
- 统一转为简体 truth，同时保留原始文本字段。
- 拼接时保留 clip 间短静音，避免句子粘连。

### FLEURS

官方页：https://huggingface.co/datasets/google/fleurs

用途：CJK 与近邻语言，多语种 ASR 和检索展示鲁棒性。

Hugging Face 数据卡标注许可证为 CC BY 4.0，覆盖 102 种语言，每个配置包含 `transcription`、`raw_transcription`、语言和性别等字段。

使用规则：

- 首版只抽 cmn_hans_cn、yue_hant_hk、ja_jp、ko_kr。
- 每种语言保持少量样本，不让多语种压过中文主干。
- truth 中保留原文，不自动翻译。

### GigaSpeech / Earnings-22

GigaSpeech 官方仓库：https://github.com/SpeechColab/GigaSpeech

Earnings-22 官方仓库：https://github.com/revdotcom/speech-datasets/tree/main/earnings22

用途：英文真实语音补充。GigaSpeech 覆盖 audiobook、podcast、YouTube；Earnings-22 覆盖财报电话会、实体、数字和多口音英文。

使用规则：

- 英文总量控制在 0.5-1.0 小时。
- 优先选择真实口语、实体密集、数字密集片段。
- 只用于补充跨语言和实体检索，不喧宾夺主。

### 平台已有字幕视频

用途：贴近真实上传、长视频、字幕 sidecar 验证。

第一版必须包含：

```text
eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.sidecar.json
```

使用规则：

- 标注为平台内部已有上传样本。
- 不外发原视频。
- 可直接用现有 sidecar 走索引链路。
- 如果后续发现其他带内嵌软字幕视频，可以加入补位池。

## 产物布局

推荐布局：

```text
eval/asr/internal_testset/
  README.md
  sources.md
  manifest.jsonl
  splits/
    v1_manifest.jsonl
  truth/
    <sample_id>.sidecar.json
    <sample_id>.srt
    <sample_id>.vtt
  reports/
    build_summary.json
    validation_summary.json
```

本机缓存和生成媒体默认放在仓库外或 gitignored runtime/data 目录：

```text
data/asr_internal_testset/cache/
data/asr_internal_testset/generated_media/
```

若这些目录位于仓库内，必须确保原始媒体和生成后 MP4 不进入 git。

## Manifest Schema

每条封装后视频或真实平台视频对应一条 manifest 记录：

```json
{
  "sample_id": "asr_v1_wenet_long_001",
  "version": "v1",
  "source_dataset": "WenetSpeech",
  "source_url": "https://wenet-e2e.github.io/WenetSpeech/",
  "source_item_id": "PODCAST_OR_YOUTUBE_ID",
  "language": "zh",
  "text_script": "Hans",
  "scenario_tags": ["zh", "podcast", "long", "real_audio"],
  "duration_seconds": 1800.0,
  "media_kind": "generated_static_mp4",
  "generated_media_path": "data/asr_internal_testset/generated_media/asr_v1_wenet_long_001.mp4",
  "truth_sidecar_path": "eval/asr/internal_testset/truth/asr_v1_wenet_long_001.sidecar.json",
  "truth_srt_path": "eval/asr/internal_testset/truth/asr_v1_wenet_long_001.srt",
  "license": "CC BY 4.0 / internal restrictions as noted",
  "internal_use_only": true,
  "redistribute_original_media": false,
  "source_hash": "sha256:...",
  "generated_hash": "sha256:...",
  "sampling_seed": 20260708,
  "notes": "internal ASR retrieval test sample"
}
```

Truth sidecar 使用平台现有 `load_sidecar()` 可读结构：

```json
[
  {
    "start_time": 0.0,
    "end_time": 2.4,
    "text": "这是一条转写文本"
  }
]
```

如果原始标注是词级时间戳，构建时按语义和时长聚合为句级或短段级 sidecar。默认 chunk 目标：

```text
单条 1.5-8s
中文 8-40 字左右
英文 3-25 tokens 左右
长静音不生成空文本 chunk
```

## 构建流程

第一版构建工具应拆成三个小阶段，方便失败后重跑：

```text
1. source prepare
   下载或定位来源数据，生成 source inventory

2. sample select
   按维度、时长、语言、长短样本要求抽样，生成 manifest draft

3. media build
   拼接音频，生成静态画面 MP4、sidecar、srt/vtt、hash 和 summary
```

抽样必须可复现：

- 固定 seed。
- 每个来源记录原始 id。
- 每个生成样本记录由哪些原始片段组成。
- 对长样本保留连续上下文，短样本允许多片段拼接。

## 平台接入

测试集接入平台时使用两种路径：

```text
原平台视频:
  上传或复用已有 runtime-server/uploads 视频
  提供已生成 sidecar truth

开源音频封装视频:
  上传 generated static MP4
  同时上传 .sidecar.json 或 .srt/.vtt transcript
```

这样可以验证：

- 上传 transcript sidecar 能否被正确保存。
- 索引 job 能否发现并使用 sidecar。
- `asr.npz` 中的 chunk 时间和文本是否与 truth 对齐。
- ASR/OCR/visual 融合检索时，ASR evidence 是否稳定返回。

## 评测维度

第一版只定义数据集和基础校验，不强行定义完整 WER 排行。每条样本至少覆盖以下标签之一：

```text
zh_clean_read
zh_real_video_podcast
zh_long_video
zh_meeting_farfield
zh_meeting_overlap
zh_accent_short_clip
cjk_multilingual
en_real_speech
entity_number_heavy
platform_subtitle_truth
```

后续 ASR 回归可以按这些维度分别统计：

- sidecar chunk 加载成功率。
- chunk 数、短 chunk 比例、长低信息 chunk 比例。
- query 命中片段是否覆盖 truth 时间窗。
- ASR lexical 与 semantic evidence 的排名变化。
- 长视频中跨视频排序是否被少数高分片段误导。

## 错误处理

构建工具遇到以下情况应跳过样本并写入 report：

- 原始音频缺失或 hash 不匹配。
- 标注为空或时间戳不可解析。
- 音频时长和标注时长偏差过大。
- 拼接后 sidecar 时间戳非单调。
- 生成 MP4 失败。
- 来源许可证或内部限制缺失。

对 `昨夜降至04`，如果原始平台视频不存在但 truth 文件存在，测试集仍可保留 truth 记录，并在 manifest 中标注 `media_available=false`，避免误以为完整样本可上传。

## 验证计划

构建完成后至少验证：

1. `manifest.jsonl` 可逐行解析，sample_id 唯一。
2. 总时长在 10-15 小时之间，推荐 12 小时左右。
3. WenetSpeech 时长不少于 4 小时，除非 report 明确说明缺口。
4. 中文样本时长占比不少于 75%。
5. 长样本时长占比在 35%-45% 之间。
6. 每条 sample 的 sidecar 可被 `backend/app/indexing/asr.py::load_sidecar()` 读取。
7. 每个 sidecar 时间戳单调，文本非空。
8. 生成媒体文件 hash 与 manifest 一致。
9. 随机抽查 5 条样本上传并索引成功。
10. `昨夜降至04` 的 653 条字幕 truth 仍能完整加载。

## 风险

1. WenetSpeech 原始媒体版权不归数据集维护者所有，因此只适合内部实验，不适合外发。
2. 长样本会提高索引和上传验证成本，需要和短样本搭配。
3. 音频封装成静态画面 MP4 不能覆盖真实视觉变化，但本测试集目标是 ASR，不评估视觉通道。
4. 不同来源文本规范化差异较大，首版应同时保留 `raw_text` 和 `normalized_text`。
5. Common Voice 和 FLEURS 多语种样本可能和平台主场景不完全一致，比例必须受控。

## 后续扩展

- 增加更多平台已有带字幕长视频。
- 针对 ASR 错词、实体、数字、专名建立 query set。
- 增加基于 truth 时间窗的检索命中评估脚本。
- 增加 ASR 模型 A/B 对比报告。
- 从 WenetSpeech 类别中进一步分出新闻、访谈、知识讲解、娱乐内容等子维度。
