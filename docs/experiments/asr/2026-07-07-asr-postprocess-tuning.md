# ASR 后处理参数调优实验，2026-07-07

## 目的

在现有素材上比较 ASR chunk 后处理策略，选择一个适合默认索引的合并参数。目标是减少过短碎片，让 semantic embedding 的文本单元更完整，同时避免把间隔较大的两句话合成一个过宽片段。

## 输入

- 本地 runtime：`runtime-server`
- ASR 索引数量：8 个 `runtime-server/indexes/*/asr.npz`
- 原始 ASR chunk 总数：6304
- 报告脚本：`scripts/asr_postprocess_report.py`
- HTML 报告：`runtime/analysis/asr_postprocess_report_20260707-162855.html`

## 策略

| 策略 | 说明 |
|---|---|
| `gap_only` | 只按 ASR chunk 间隔合并 |
| `bucket_bonus` | 以 5s bucket 作为轻量正向信号，短碎片在同 bucket 内可稍宽松合并 |
| `shot_bonus` | 用更宽松的镜头/段落信号，当前素材没有真实 shot hint 时只作为实验上限 |
| `conservative` | 更保守，保留更多原始短 chunk |
| `aggressive_short` | 更激进地合并短碎片，用于观察过度合并风险 |

## 汇总指标

| 策略 | processed chunks | merged chunks | short chunks | semantic chunks | semantic skipped |
|---|---:|---:|---:|---:|---:|
| `gap_only` | 2303 | 4001 | 808 | 1992 | 311 |
| `bucket_bonus` | 2282 | 4022 | 798 | 1971 | 311 |
| `shot_bonus` | 2238 | 4066 | 747 | 1927 | 311 |
| `conservative` | 2500 | 3804 | 975 | 2189 | 311 |
| `aggressive_short` | 2129 | 4175 | 640 | 1818 | 311 |

说明：第一版报告里 `gap_only` 与 `bucket_bonus` 完全一致。复查发现 `gap_only` 的配置仍然使用了 same-segment 放宽阈值，导致实验对照不纯。已新增回归测试 `test_gap_only_strategy_does_not_use_same_segment_bonus` 并修正配置，上表是修正后的结果。

## LLM 裁判抽样

评分维度：

```text
语义连贯：合并后是否像同一段话
跨话题风险：是否把不同语境拼在一起
检索适配：作为搜索命中片段是否清楚
```

| 策略 | 抽样判断 | 代表样例 | 结论 |
|---|---|---|---|
| `gap_only` | 最保守，边界较安全，但比 `bucket_bonus` 多留下 21 个 chunk，短碎片略多 | 无明显过合并风险 | 可作为安全 fallback |
| `bucket_bonus` | 多合并出的样例多为同一语境补全，例如 “Guys, what are you doing? / Where are you going?”、“我昨天晚上 / 让人威胁了 / 我怀疑是” | 少量综艺口语片段会拉到 7-8s，但仍在同一话题 | 推荐默认 |
| `shot_bonus` | 多合并样例开始出现边界风险，例如 “好不容易 / 没牌不可怕 / 我第一次看到你”、“可以了 / 音乐老师 / 给你谈一句轻无乱舞” | 对没有真实 shot boundary 的固定 bucket 不可靠 | 暂不默认 |
| `conservative` | 过短 chunk 保留过多，semantic 单元仍碎 | short chunks = 975，最高 | 不推荐默认 |
| `aggressive_short` | 明显过合并，常把重复口水词、广告串词、多人抢话压成一个 chunk | “我说过”重复 7 次、“赵正宵”重复 9 次、综艺口语连续 8-10 个碎片 | 不推荐默认 |

## 结论

默认策略选择 `bucket_bonus`，但参数已收紧：同一个 5s bucket 内不会因为 2 秒级停顿就继续合并。修正后的对照显示它比纯 `gap_only` 多合并 21 个 chunk，属于轻量放宽；LLM 抽样判断这些额外合并大多是同一语境的补全句，适合提升 semantic embedding 聚合质量。

不选择 `conservative`，因为它留下的短 chunk 明显更多，semantic 单元仍偏碎。

不选择 `aggressive_short`，因为它虽然减少短 chunk 最明显，但更容易把语义边界不同的短句合并。

不选择 `shot_bonus` 作为当前默认，因为现阶段 ASR 合并只使用固定 5s bucket hint，还没有接入真实 shot boundary。后续如果 visual shot-aware 索引稳定，可以重新跑本报告评估。

## 注意

- 这次调参只改变新建 ASR 索引；旧索引需要 ASR-only 重跑才会应用。
- LLM/人工裁判只用于离线查看 HTML 样例，不进入运行时检索链路。
- 部分现有原始 ASR chunk 本身超过 8s，最长样例约 22.8s。当前规则只把低信息长 chunk 排除出 semantic embedding；有信息的长 chunk 暂不硬拆，后续可单独做长句切分实验。

## 后续

- 在更多中英混合、多语种视频上复跑该报告。
- 如果接入真实 shot boundary，再比较 `bucket_bonus` 和 `shot_bonus`。
- ASR-only 重跑后，用中文 query 和英文 query 各做一组搜索 smoke test，确认翻译型输出不再出现。
