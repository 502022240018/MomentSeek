# ASR 短窗口 60/90/120 秒测试集评估

日期：2026-07-08

## 目的

在 `电视剧昨夜降至04.mp4` 的内嵌中文字幕真值集上，对比 Whisper 短窗口解码的窗口长度：

- `60s window + 5s overlap`
- `90s window + 5s overlap`
- `120s window + 5s overlap`

本实验只评估转写质量和速度，不覆盖正式 ASR 索引。

## 测试集

- 视频：`电视剧昨夜降至04.mp4`
- video_id：`a293b5981126444182208da7ba6274f5`
- 真值来源：视频内嵌中文字幕提取后转简体
- 真值文件：`eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.jsonl`
- 真值条数：653
- 真值时间范围：`00:00:24.160` 到 `00:30:18.880`

## 实验参数

- 模型：Whisper `small`
- 环境：本地 Docker CUDA 后端 `momentseek-mvp-app`
- language：`auto`
- overlap：`5s`
- 后处理：当前 ASR chunk 清洗/合并策略
- 对比基线：当前完整 Whisper ASR 索引 `runtime-server/indexes/a293b5981126444182208da7ba6274f5/asr.npz`

## 指标说明

- `elapsed_s`：各窗口 Whisper 解码耗时求和。短窗口实验不包含完整索引写入和 embedding 生成。
- `max_window_s`：单个窗口最慢耗时，用于估计任务卡顿风险。
- `raw_high`：raw segment 中被重复/幻觉 guard 标为 high risk 的数量。
- `bad_ngram`：后处理结果里仍出现明显重复 n-gram 的 chunk 数。
- `duplicate_runs`：连续相同文本 chunk 的 run 数。
- `recall@2s`：每条字幕真值在 `+-2s` 时间范围内 ASR 文本的平均字符召回。
- `hitR60@2s`：`recall@2s >= 0.60` 的字幕比例。
- `f1@2s`：字符 F1。由于 ASR chunk 通常比单条字幕更长，该值主要用于横向比较，不宜单独解释为绝对准确率。

## 结果

| source | windows | elapsed_s | max_window_s | chunks | raw_high | bad_ngram | duplicate_runs | known_phrases | recall@2s | f1@2s | hitR60@2s | recall@10s | f1@10s | hitR60@10s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `current_full_whisper` | 0 | 494.894 | 0.000 | 270 | - | 3 | 1 | `{"你跟她说":17,"你还不说":1}` | 0.771 | 0.265 | 0.767 | 0.823 | 0.130 | 0.836 |
| `short_60s` | 37 | 368.970 | 31.637 | 246 | 39 | 1 | 1 | `{}` | 0.726 | 0.238 | 0.721 | 0.829 | 0.137 | 0.836 |
| `short_90s` | 24 | 348.813 | 31.587 | 241 | 8 | 0 | 0 | `{}` | 0.807 | 0.277 | 0.809 | 0.854 | 0.136 | 0.859 |
| `short_120s` | 18 | 496.156 | 65.860 | 241 | 50 | 1 | 0 | `{"你还不说":1}` | 0.721 | 0.248 | 0.709 | 0.818 | 0.133 | 0.821 |

速度相对基线：

| source | elapsed_s | compared with current_full_whisper |
|---|---:|---:|
| `short_60s` | 368.970 | 快约 25.4% |
| `short_90s` | 348.813 | 快约 29.5% |
| `short_120s` | 496.156 | 慢约 0.3% |

## 观察

`short_90s` 在这组测试里表现最好：

- 总耗时最低：`348.813s`
- `recall@2s` 最高：`0.807`
- `hitR60@2s` 最高：`0.809`
- 后处理后没有明显 `bad_ngram` 和连续重复 run
- raw high-risk segment 最少：`8`

`short_60s` 能解决目标片段中严重的 `你跟她说/你还说` 循环，但整体召回低于 90s。推测原因是 60s 上下文偏短，部分句子边界、语气词和上下文承接更容易丢失。

`short_120s` 不推荐作为当前默认值：总耗时接近完整转写，单窗口最慢达到 `65.860s`，并且 raw high-risk segment 增多，测试集召回也低于 90s。

## 风险与例外

90s 虽然指标最好，但在目标窗口 `03:50-04:40` 出现了 `language=auto` 的英文泄漏：

```text
You...
Your girlfriend is awake. 他说你们是停车之后吵架 然后才喝的酒
```

这类问题不会被“重复幻觉”指标直接捕获。因此不能只凭本次自动指标就把 `90s + language=auto` 定为最终生产配置。

## 诊断字段有效性检查

本次额外检查了 raw ASR diagnostics 是否能反映实际文本问题。这里把“实际问题”先按轻量规则定义为：

- raw segment 出现连续重复 run；
- raw segment 内部出现明显重复 n-gram；
- 已知中文视频里出现 3 个以上连续 ASCII 字母，即疑似英文/外语泄漏。

诊断命中情况：

| source | raw segments | text problems | `hallucination_risk=high` 命中 | high 漏报 | high 误报 |
|---|---:|---:|---:|---:|---:|
| `short_60s` | 684 | 51 | 39 | 12 | 0 |
| `short_90s` | 643 | 25 | 8 | 17 | 0 |
| `short_120s` | 676 | 79 | 50 | 29 | 0 |

结论：

- `hallucination_risk=high` 很准，当前样本里没有误报；它抓到的基本都是真重复。
- 但它更像“连续重复 detector”，不是完整质量评分器。
- 英文泄漏、外语乱码、低置信错词、短重复 run 仍可能漏掉。

几个漏报样例：

```text
90s raw [0035] risk=none avg_logprob=-4.025 compression=0.784 | Double check the phone.
90s raw [0037] risk=none avg_logprob=-0.983 compression=1.027 | Your girlfriend is awake.
90s raw [0307-0309] risk=none avg_logprob=-0.856 compression=1.429 | 我穿的 / 我穿的 / 我穿的
120s raw [0029] risk=none avg_logprob=-3.632 compression=0.833 | Ben burada konuşuyorum.
```

Whisper 自带字段的单独效果也有限：

| rule | 60s recall | 90s recall | 120s recall | 观察 |
|---|---:|---:|---:|---|
| `avg_logprob < -1.0` | 0.06 | 0.16 | 0.29 | 能抓部分外语/乱码，但漏掉很多重复 |
| `avg_logprob < -0.8` | 0.49 | 0.60 | 0.52 | 召回提高，但误报明显增多 |
| `compression_ratio > 2.4` | 0.00 | 0.00 | 0.00 | 对这些短窗口样本几乎无效 |
| `no_speech_prob > 0.6` | 0.00 | 0.00 | 0.00 | 不适合检测这类幻觉 |
| `high OR avg_logprob<-1.0 OR 英文泄漏` | 0.94 | 0.56 | 0.92 | 组合规则更有用，但 90s 仍会漏短重复 run |

processed chunk 里仍有问题文本被标为可生成 embedding：

```text
90s processed [0017] semantic_eligible=True | Your girlfriend is awake. 他说你们是停车之后吵架 然后才喝的酒
90s processed [0015] semantic_eligible=True | Double check the phone.
60s processed [0159] semantic_eligible=True | 一表人才啊 怎么来 ... 来 来 来 来 来 ...
120s processed [0162] semantic_eligible=True | 现在成大老板了 ... 来 来 来 来 来 ...
```

因此后续正式索引不应只保存 raw diagnostics，还要把诊断结果反馈到最终 chunk：

1. raw segment 保留 `avg_logprob/compression_ratio/no_speech_prob/temperature/hallucination_reasons`。
2. processed chunk 聚合 source raw diagnostics，生成 `quality_flags`。
3. 对 `repetition_loop`、`language_leak`、`very_low_logprob`、`repeated_ngram` 等 chunk 设置 `semantic_eligible=false` 或检索降权。
4. 原文仍保留用于人工查看，不要直接删除。

下一步建议补做：

1. 在同一真值集上跑 `90s + language=zh`。
2. 比较 `90s auto` 与 `90s zh` 的英文泄漏、字幕召回、速度。
3. 如果 `zh` 明显更稳，则对已知中文视频使用显式 `language=zh`；多语言/未知语言视频仍走语言检测或 `auto`。

## 当前结论

候选默认窗口长度优先级：

1. `90s + 5s overlap`：当前最佳候选，但需要补测语言策略。
2. `60s + 5s overlap`：保守备选，抗长上下文循环有效，但召回偏低。
3. `120s + 5s overlap`：暂不推荐。

正式改索引流程前，推荐先补齐 `90s zh` A/B，并把“英文泄漏率/非目标语言比例”加入 ASR 评估指标。

## 产物路径

```text
runtime-server/analysis/asr_short_window_eval_20260708/
runtime-server/analysis/asr_short_window_eval_20260708_yesterday_90s/
runtime-server/analysis/asr_short_window_eval_20260708_yesterday_120s/
runtime-server/analysis/asr_window_ab_truth_eval_20260708/
runtime-server/analysis/asr_window_ab_truth_eval_20260708/asr_window_ab_truth_eval.md
runtime-server/analysis/asr_window_ab_truth_eval_20260708/asr_window_ab_truth_eval.json
```
