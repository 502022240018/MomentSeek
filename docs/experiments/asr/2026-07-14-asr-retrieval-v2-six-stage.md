# ASR 检索评测集 v2 与六阶段实验

日期：2026-07-14

## 目标

在不重新转写、不修改 retrieval chunk 的前提下，系统比较：

1. 低信息 semantic eligibility。
2. semantic score calibration。
3. MiniLM、multilingual-e5-small、gte-multilingual-base。
4. character bigram、IDF weighted coverage、BM25。
5. semantic / lexical 候选融合。

## Stage 1：联合评测集

Corpus 使用真实 ASR 输出，不使用 truth 文本冒充索引文本：

- 当前平台 9 条视频的正式 ASR v3 索引。
- `asr_internal_eval_20260709` 的 16 条开放样本 SenseVoice 识别结果。
- 开放样本音频 10.214 小时；排除与平台视频重复的 `asr_v1_platform_yesterday_ep04`。
- 合计 25 个 source、6854 个 ASR chunk。

查询共 82 条：

| 类型 | 数量 |
|---|---:|
| 有答案 | 74 |
| 无答案 | 8 |
| 平台人工查询 | 42 |
| 开放集查询及无答案查询 | 40 |
| tune / dev / holdout | 36 / 12 / 34 |

开放查询用 truth 时间段建立 qrel，再对齐到实际 SenseVoice chunk；truth 不参与检索。全部 qrel 已解析，开放 source 按素材隔离 split。人工审阅文件：

```text
runtime-server/analysis/asr_retrieval_benchmark_20260714/stage1_dataset/query_target_review.jsonl
```

## Stage 2：低信息过滤

固定 MiniLM 与当前 70/30 校准，只改 eligibility：

| 方案 | 全量 MRR | H@1 | H@5 | H@50 |
|---|---:|---:|---:|---:|
| current | 0.760 | 0.703 | 0.851 | 0.919 |
| obvious hard filter | 0.760 | 0.703 | 0.851 | 0.919 |
| hard filter + soft penalty | 0.764 | 0.703 | 0.851 | 0.919 |

新增 hard filter 只比当前多拒绝 6/6641 个 semantic chunk；soft penalty 降权 218 个 chunk，但所有 Hit 指标不变。tune 加权目标仅提高 0.462 个百分点，低于预设的 0.5 个百分点最小实质收益门槛。

结论：**保留 current eligibility，不增加新启发式。**

## Stage 3：semantic calibration

| 方案 | 全量 MRR | H@1 | H@5 | H@50 |
|---|---:|---:|---:|---:|
| absolute cosine confidence | 0.761 | 0.703 | 0.851 | 0.919 |
| current absolute 70% + per-source percentile 30% | 0.760 | 0.703 | 0.851 | 0.919 |
| percentile only | 0.150 | 0.027 | 0.257 | 0.811 |

Gated percentile 的各组参数没有超过 absolute。纯 percentile 会让每个 source 都产生一个相对高分，跨 source 排序明显失真。

结论：**实验路径选择 absolute，取消 per-source percentile 对主排序分的贡献。** percentile 仍可保留为诊断字段。

## Stage 4：embedding 模型

E5 按模型要求使用 `query:` / `passage:` 前缀。三模型均重新编码相同 6854 个 chunk；排序使用 Stage 2/3 选出的 current eligibility + absolute calibration。

| 模型 | dim | 全量 MRR | H@1 | H@5 | H@50 | 6854 chunks GPU 编码 | float16 索引 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | 384 | 0.761 | 0.703 | 0.851 | 0.919 | 4.51s | 5.02 MB |
| multilingual-e5-small | 384 | 0.749 | 0.716 | 0.784 | 0.824 | 5.67s | 5.02 MB |
| **gte-multilingual-base** | 768 | **0.869** | **0.824** | **0.932** | **0.959** | 12.48s | 10.04 MB |

GTE 的 tune/dev/holdout MRR 分别为 0.849/0.725/0.937，三个 split 均高于 MiniLM。E5-small 提高少量 H@1，但 H@5/H@50 和泛化稳定性不足，不作为替换候选。

## Stage 5：lexical

| 方案 | tune MRR | 全量 MRR | H@1 | H@5 | holdout MRR |
|---|---:|---:|---:|---:|---:|
| **legacy character bigram** | **0.714** | 0.730 | 0.689 | 0.770 | 0.710 |
| IDF weighted coverage | 0.694 | 0.729 | 0.689 | 0.784 | 0.726 |
| BM25 | 0.695 | 0.708 | 0.649 | 0.784 | 0.675 |

IDF 在 holdout 略好但 tune 下降，BM25 也没有稳定收益。

结论：**继续使用现有 character bigram coverage。**

## Stage 6：融合

最终候选使用 GTE + absolute semantic 与 legacy bigram。对照参考使用当前 MiniLM + 70/30 calibration + max-blend / lexical reserve 公式，并在同一 82-query corpus 上重新计算。

| 方案 | tune MRR | dev MRR | holdout MRR | 全量 MRR | H@1 | H@5 | H@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 MiniLM reserve 参考 | 0.821 | 0.850 | 0.833 | 0.831 | 0.784 | 0.919 | 0.946 |
| MiniLM absolute + priority 90/10 | 0.781 | 0.760 | 0.876 | 0.818 | 0.757 | 0.892 | 0.946 |
| GTE semantic only | 0.849 | 0.725 | 0.937 | 0.869 | 0.824 | 0.932 | 0.959 |
| GTE + naive 90/10 linear | 0.901 | 0.750 | 0.913 | 0.886 | 0.851 | 0.932 | 0.959 |
| **GTE + priority 90/10 linear** | **0.894** | **0.850** | **0.933** | **0.899** | **0.865** | **0.946** | **0.973** |
| weighted RRF | 0.830 | 0.804 | 0.866 | 0.842 | 0.757 | 0.959 | 0.973 |

Naive linear 曾让一个 semantic-ineligible、lexical 全库第 1 的目标从第 1 掉到第 6642。最终方案增加模型无关的安全契约：

```text
base_score = 0.90 * semantic + 0.10 * lexical
if lexical >= 0.50:
    进入 strong lexical priority band，band 内按 lexical 排序
else:
    按 base_score 排序
```

参数先在 tune 选择，再用 tune+dev 检查强 lexical 安全约束：若目标本身是 lexical 全局第 1 且 coverage >= 0.50，融合结果必须保留在 Top-5。最终方案没有违反项，未使用 holdout 调参。

补充的尺寸受限对照表明，GTE 上选出的 90/10 参数不能直接移植给 MiniLM；MiniLM absolute + priority 低于当前 MiniLM reserve。因此若暂不更换 embedding 模型，正式排序应继续保留当前 combined + lexical reserve。

分类收益：

| category | 当前参考 MRR | 最终 MRR | 当前 H@1 | 最终 H@1 |
|---|---:|---:|---:|---:|
| cross_lingual | 0.521 | 0.833 | 0.375 | 0.750 |
| semantic_paraphrase | 0.830 | 0.850 | 0.794 | 0.824 |
| truth_query | 0.867 | 0.969 | 0.812 | 0.938 |
| lexical_exact | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical_rewrite | 1.000 | 1.000 | 1.000 | 1.000 |

## 不更换高维 embedding 的阶段汇总

结合 retrieval v2、双候选池、拼音 fallback 和无答案校准，目前可以把 ASR 检索问题拆成“候选召回”和“候选精排”两层：

| 实验 | 关键结果 | 当前决定 |
|---|---|---|
| 低信息过滤 | MRR 0.760 -> 0.764，但 Hit@1/5/50 不变 | 不增加新 eligibility 启发式 |
| semantic calibration | absolute MRR 0.761；percentile-only H@1 仅 0.027 | percentile 只保留诊断价值，不参与跨视频主排序 |
| 384 维模型对比 | MiniLM MRR/H@1/H@5/H@50 = 0.761/0.703/0.851/0.919；E5-small = 0.749/0.716/0.784/0.824 | 不切换 E5-small，继续使用 MiniLM |
| MiniLM 最终参考排序 | MRR/H@1/H@5/H@50 = 0.831/0.784/0.919/0.946 | 保留 combined 主序 + sparse lexical reserve |
| 双候选池 | 42 条平台查询 H@1 保持 0.833，H@5 0.905 -> 0.929，H@50 0.929 -> 0.952 | 保留 lexical reserve；否决全量 RRF 和 CJK 局部覆盖加分 |
| 拼音 fallback | 41 条人工样例中，Top-20 lexical/pinyin 命中 25/29，额外救回 5 条；Top-50 pinyin 命中 32，额外救回 7 条 | 只作为受保护的实体/近音候选池，不进入无条件主排序 |
| 无答案阈值 | 现行 FAR 100%；保持约 95% recall 时 holdout FAR 仍约 89.5% | 不用单一 cosine/lexical 阈值宣称“有答案/无答案” |

74 条有答案查询中，当前参考方案约有 70 条进入 Top-50、68 条进入 Top-5、58 条排到 Top-1。由此可见：

1. 主要剩余空间是把已经进入候选池的正确 chunk 从 Top-50/Top-5 提升到 Top-1，而不是继续反复调整单一向量分数。
2. 小型 multilingual cross-encoder 精排 Top-30/50 不改变 MiniLM 384 维索引，最适合作为下一阶段主实验。
3. 精排不能救回 Top-50 外的约 4 条查询；这部分继续评估语言感知 lexical、受保护 pinyin、原 chunk + 邻接上下文多视图和 query expansion。
4. BM25/IDF 单独排序没有稳定获胜，不等于不能提供候选；后续只把它们作为候选来源，不再让弱分支通过全量 RRF 获得等价投票权。
5. 若精排和候选扩展仍不足，再用当前数据挖掘 hard negatives，对同一 384 维 MiniLM 做领域微调或教师蒸馏；不先扩大 embedding 维度。

精排待办与验收标准统一维护在 `docs/ISSUES_AND_ROADMAP.md` 的 `RQ-003H`。

## 结论与上线边界

六阶段实验支持以下方向：

1. 不增加新的低信息过滤启发式。
2. 主排序移除 per-source percentile，改为模型专属 absolute calibration。
3. ASR semantic 候选模型选择 GTE multilingual base，不选择 E5-small。
4. lexical 继续用现有 character bigram。
5. 融合使用 semantic-primary 90/10，并为 coverage >= 0.50 的 strong lexical 建独立优先区。

**本实验未直接修改生产配置或重建正式索引。** 当前 9 条视频仍为 MiniLM 384 维。

资源补充记录：GTE 输出 768 维；本地主权重文件为 610,753,338 bytes，Hugging Face 缓存合计约 627,964,743 bytes，约为当前 384 维 float16 索引向量空间的两倍。2026-07-14 决策为暂缓该替换，只保留候选与实验结论。

上线前还需要：

- 为 GTE 单独校准 confidence / above-threshold；不能沿用 MiniLM 的 sigmoid 和阈值。8 条无答案查询上，GTE semantic-only 平均最高分约 0.926，说明拒答阈值尚不可靠。
- 在真实 API 的 per-video candidate、clip merge 和 final result 层复跑 A/B；本实验主体是 chunk-level 排序。
- 将 ASR 和 OCR semantic model 配置解耦，或先补 OCR 评测；不同模型空间绝不能混用。
- 确认 CPU 建库、冷启动、常驻内存和批量重建成本，再决定默认部署设备。

## 复现

固定资产：

```text
eval/asr/hybrid_retrieval_queries_v1.jsonl
eval/asr/retrieval_v2/open_queries.jsonl
scripts/asr_retrieval_benchmark.py
```

完整产物：

```text
runtime-server/analysis/asr_retrieval_benchmark_20260714/
```

复现说明：本报告完成后，`open_queries.jsonl` 的 `o028` qrel 因 truth/ASR 时间错位改为实际文本定位；本页聚合数值保留原始六阶段运行结果，重新运行时该单条 query 的名次可能改善。

脚本按 `--stage 1` 到 `--stage 6` 顺序运行。Stage 4 首次运行需要允许下载 E5/GTE；之后复用本地模型与 corpus embedding cache。
