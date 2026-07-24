# ASR 短窗口方案追加实验：天c游xi 与书籍纪录片

日期：2026-07-08

## 目的

在 `电视剧昨夜降至04.mp4` 之外，继续验证短窗口 Whisper 解码对不同素材的效果：

- `天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv`
- `书籍纪录片.mp4`

本次仍是实验验证，不覆盖正式索引，不删除旧 ASR 文本。

## 参数

```text
window_seconds=60
overlap_seconds=5
language=zh
model=whisper small
condition_on_previous_text=True within each window
no cross-window prompt carry
```

输出目录：

```text
runtime-server/analysis/asr_short_window_eval_20260708_tian/
runtime-server/analysis/asr_short_window_eval_20260708_book/
```

每个目录包含：

```text
short_window_raw.json
short_window_processed.json
short_window_text.txt
short_window_comparison.md
window_reports.json
```

## 速度对比

注意：旧 `job_elapsed_seconds` 来自正式 ASR job，包含 Whisper、后处理、semantic embedding 和写索引；短窗口实验的 `elapsed_seconds` 主要统计窗口转写时间，不包含正式 semantic embedding。因此短窗口正式接入后的总耗时会略高于本表中的短窗口耗时。

| 视频 | 旧正式 ASR job | 短窗口实验 | 变化 |
|---|---:|---:|---:|
| `天c游xi...mkv` | 777.435s | 917.058s | +139.623s / +18.0% |
| `书籍纪录片.mp4` | 461.797s | 481.879s | +20.082s / +4.3% |

结论：短窗口方案在 `电视剧昨夜降至04.mp4` 上更快，但在这两条素材上变慢。尤其 `天c游xi` 属于更长、对话密集、部分窗口低信息/异常的素材，60 秒窗口会带来明显重复解码成本。

## 幻觉与重复指标

### 天c游xi

旧当前索引：

```text
chunks=709
bad_ngram_chunks=9
exact_duplicate_runs=6
典型问题：
- `在这儿` 连续重复
- `你还不赶紧` 连续重复
- `有时最美晚` 连续重复
- 单字 `你` 连续 run
```

短窗口 processed：

```text
raw_chunks=1772
processed_chunks=640
high_risk_raw_segments=145
bad_ngram_chunks=1
exact_duplicate_runs=0
```

目标窗口 `00:04:00-00:05:30` 中，当前完整 Whisper 索引存在：

```text
你这些人 你这些人 你这些人 你这些人
```

短窗口结果中该重复消失，文本回到更接近逐句识别的状态。

仍有一个残留可疑 chunk：

```text
[0013] 00:01:39.160 --> 00:01:41.400  也许有许有许有许有许有许
```

另有一次 `你还不说`，但不是连续 loop：

```text
[0242] 00:34:07.800 --> 00:34:15.640  除了肢体而来不太介绍 跳进舞来像个妖怪啊 那也比你强啊 你还不说
```

判断：短窗口显著降低了重复幻觉，但 guard 还需要处理“单个 chunk 内局部重复”的情况。正式接入时应把这种 chunk 标记为 `semantic_eligible=false` 或降权，而不是直接删除原文。

### 书籍纪录片

重建前备份索引开头存在 prompt 泄漏：

```text
这次是普通话简体中文转写。
普通话简体中文转写。
```

当前完整 Whisper 索引已无 prompt 泄漏，短窗口结果也未出现 prompt 泄漏。

短窗口 processed：

```text
raw_chunks=824
processed_chunks=295
high_risk_raw_segments=9
bad_ngram_chunks=0
exact_duplicate_runs=0
```

判断：书籍纪录片在当前索引中本来已经比较干净；短窗口方案没有引入新的重复幻觉，但文本仍有繁简混杂和一些听错词。这类问题属于后续文本规范化/纠错层，不是本次 repetition loop 的主要问题。

## 综合结论

短窗口方案对 ASR 重复幻觉有效，尤其能消除长上下文连续解码导致的 loop：

- `电视剧昨夜降至04`：严重 `你跟她说` loop 消失。
- `天c游xi`：多个连续重复 run 大幅减少，`你这些人` loop 消失。
- `书籍纪录片`：无 prompt 泄漏复发，整体保持干净。

代价是速度不稳定：

- 对容易进入长上下文 failure loop 的视频，短窗口可能更快。
- 对长剧集和纪录片，60 秒窗口会带来 4%-18% 以上的额外转写成本；正式接入后还要加上 semantic embedding。

## 下一步建议

正式实现时不要只做固定 60 秒窗口一种模式。建议做成可配置策略：

1. 默认短窗口参数先用 `60s window + 5s overlap`，保证质量优先。
2. 保留 `asr_window_seconds`、`asr_window_overlap_seconds` 配置，方便 A/B。
3. 引入更强的 hallucination guard：
   - 连续相同短句；
   - 单 chunk 内重复 n-gram；
   - 高 compression ratio + 重复；
   - 低 avg_logprob + 重复；
   - 长时间跨度低信息文本。
4. 对高风险 chunk 保留原文展示，但默认不参与 semantic embedding。
5. 后续做速度优化实验：
   - `90s/120s window + 5s overlap`
   - 只对高风险长上下文失败片段局部重跑
   - VAD 预切分
   - 批量/并行窗口调度
