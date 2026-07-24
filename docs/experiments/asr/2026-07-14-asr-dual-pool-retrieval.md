# ASR lexical / semantic 双候选池实验

日期：2026-07-14

## 目标

解决短实体扩展 query 被 semantic 噪声淹没的问题，同时避免为单个中文例子持续增加 lexical 启发式。核心回归是：

```text
query: 昆仑山
target: 说实话,我们天山不好进的,一般都去昆仑。
```

## 数据与方法

- 语料：当前本地 9 个完整视频的 ASR v3 索引。
- chunk：2080 条，其中 2011 条有 MiniLM semantic embedding。
- 查询：42 条，`tune=21`、`holdout=21`，覆盖原句、语义改写、跨语言、短实体扩展和强字面命中。
- 固定查询集：`eval/asr/hybrid_retrieval_queries_v1.jsonl`。
- 可复现脚本：`scripts/asr_hybrid_retrieval_eval.py`。
- 完整逐查询结果：`runtime-server/analysis/asr_hybrid_retrieval_eval_20260714/`。

`holdout` 用于参数选择后的回归观察，但数据规模仍小，且包含已知的“昆仑山”失败案例，不应当作无偏线上指标。

## 对比方案

1. `combined_legacy`：原 bigram lexical 与 semantic calibrated score 直接混合。
2. `combined_cjk`：额外提高纯 CJK query 的最长连续片段覆盖率。
3. `dual_rrf`：lexical / semantic 各取候选后做 weighted RRF。
4. `semantic reserve`：semantic 作主序，按固定槽位插入 lexical 候选。
5. `rescored reserve`：两池独立产候选，现有 combined score 保持主序，只为强 lexical 候选保留稀疏槽位。

## 结果

全量 42 条：

| 方案 | MRR | Hit@1 | Hit@5 | Hit@50 |
|---|---:|---:|---:|---:|
| combined legacy | 0.864 | 0.833 | 0.905 | 0.929 |
| CJK 连续片段补偿 | 0.814 | 0.738 | 0.905 | 0.952 |
| weighted RRF | 0.803 | 0.714 | 0.905 | 0.952 |
| semantic-primary reserve | 0.832 | 0.762 | 0.929 | 0.952 |
| **combined-primary lexical reserve** | **0.869** | **0.833** | **0.929** | **0.952** |

最终方案在不降低 Hit@1 的前提下提高了 MRR、Hit@5 和 Hit@50。`昆仑山` 的指定目标由第 72 提升到第 4；本地真实 `/api/search` 同样返回第 4。

## 最终参数

```text
primary pool: 原 combined score 排序
lexical pool: lexical_score >= 0.50
pool size: 50
preserve first primary results: 3
then: 每 8 个 primary 结果最多插入 1 个尚未出现的 lexical 结果
scope: ASR-only 搜索最终排序
```

保底排序不改写 `score`，因此 API 中的 score 继续表达原始 calibrated confidence，而不是人为抬高后的排名分。

## 未采用方案

- 不保留 CJK 连续片段补偿：它修复单例，但把弱局部重合普遍抬高，Hit@1 明显下降。
- 不采用 weighted RRF：当前 MiniLM 的跨语言和短低信息 semantic 噪声较大，RRF 把弱 lexical 也变成有效投票，整体 Top-1 下降。
- 不采用 semantic-only 主序：它能提高召回，但丢失现有 combined 排序已经利用好的精确词面信号。

## 剩余失败

以下两条在当前 MiniLM 下仍排在 Top-50 外，属于 embedding / 跨语言能力问题，不应继续用排序启发式修补：

```text
姆巴佩射门，赫尔南德斯跟进
很久没有见到你，我特别想你
```

后续应在同一查询集上对比 `multilingual-e5-small`、`gte-multilingual-base`，并补充更多 hard negatives 与真实用户 query。
