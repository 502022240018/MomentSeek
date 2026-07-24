# ASR 无答案阈值校准

## 目标

校准当前 MiniLM ASR 检索的 `above_threshold`，判断能否在尽量保留正确答案的前提下，把无答案查询稳定放到“低于阈值”区域。本实验只调整离线判定，不修改正式索引、模型或线上阈值。

## 数据

| 项目 | 数量 |
|---|---:|
| source | 25 |
| 实际 ASR chunk | 6854 |
| 有答案查询 | 74 |
| 无答案查询 | 56 |
| 总查询 | 130 |

原评测集只有 8 条无答案查询。本次新增 48 条，按 tune/dev/holdout 各 16 条，覆盖易负例、同领域困难负例和跨语言负例。

人工抽查了困难负例的最高匹配。例如“图书馆把八千本书全部卖掉”匹配到“图书馆收入了八千多本书”；二者语义相近但事实相反，负例标签成立。

校准前还修正了 dev 查询 `o028` 的 qrel：原 truth 时间落在相邻的错误 ASR chunk，改为用实际目标文本“每个人一个格子间”定位。该修正只用于纠正标注，不参与阈值选参。

## 现行逻辑

```text
semantic_score = 0.7 * sigmoid(cosine) + 0.3 * per-video percentile
above_threshold = semantic_score >= 0.55 or legacy_bigram >= 0.25
```

在 6854 个 chunk 中采用“任一候选超过阈值”会产生极值效应；per-video percentile 又保证每个视频总有相对高分候选。现行逻辑在 56 条无答案查询上的误接收率为 100%。

## A/B 范围

语义门槛比较了现行 calibrated score、不含 percentile 的 absolute confidence、raw cosine 及 calibrated score + raw cosine 双门槛。Lexical 门槛比较了正式代码的去空格字符 bigram coverage 与 IDF 加权的英文 token / 中文 bigram coverage。

参数只在 tune 选择，dev 和 holdout 只用于验证。`target gate recall` 表示相关 chunk 未被门槛挡掉，FAR 表示无答案查询仍有候选被判为有效。

## 结果

| operating point | tune recall/FAR | dev recall/FAR | holdout recall/FAR | 全量 recall/FAR |
|---|---:|---:|---:|---:|
| 现行门槛 | 1.000/1.000 | 1.000/1.000 | 0.968/1.000 | 0.986/1.000 |
| tune recall >= 0.95：raw cosine 0.43 + IDF lexical 0.50 | 0.970/0.737 | 1.000/0.889 | 0.935/0.895 | 0.959/0.839 |
| 严格观察点：raw cosine 0.60 + legacy lexical 0.35 | 0.879/0.421 | 0.900/0.333 | 0.968/0.421 | 0.919/0.393 |

分数分布有明显重叠：有答案 target raw cosine 的 p10/median/p90 为 0.495/0.729/0.895；无答案查询全库最高 raw cosine 为 0.413/0.538/0.661。

现行英文 lexical 还有结构性问题：去空格字符 bigram 会把常见字母组合当成匹配。例如查询 `Mbappe announced his retirement from football` 对普通足球解说得到约 0.649 lexical，即使 semantic cosine 约 0.028 仍会被接收。IDF lexical 能缓解跨语言误接收，但不能识别“买书/卖书”一类反事实关系。

## 结论

**本轮不修改生产阈值。** 单一 lexical/semantic 阈值无法同时满足高召回和可靠拒答：保持约 95% 正确答案通过率时，holdout 无答案 FAR 仍约 89.5%；将 FAR 压到约 40% 时，全量正确答案通过率降到约 91.9%。因此 `0.43` 和 `0.60` 都不是可直接上线的“无答案阈值”。

后续把两个问题分开处理：

1. `above_threshold` 只表达“高置信候选”，不对用户宣称语料中一定存在答案。
2. 收集真实用户零结果 query，再按语言、query 类型和 corpus 规模校准。
3. 用 result-level reranker / entailment 判断实体、数值、否定和事实冲突；困难反事实不能靠 embedding cosine 解决。
4. IDF lexical gate 先做真实 API shadow A/B；排序协议暂不随本实验改变。

## 复现

```text
eval/asr/hybrid_retrieval_queries_v1.jsonl
eval/asr/retrieval_v2/open_queries.jsonl
eval/asr/retrieval_v2/no_answer_queries_v1.jsonl
scripts/asr_no_answer_threshold_eval.py
runtime-server/analysis/asr_no_answer_threshold_20260714/
```
