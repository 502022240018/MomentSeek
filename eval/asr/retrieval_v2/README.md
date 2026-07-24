# ASR 检索评测集 v2

这套评测集用于比较 ASR 文本的 semantic、lexical 与融合召回策略，不评价语音识别 CER 本身。

## 组成

- `../hybrid_retrieval_queries_v1.jsonl`：当前 9 条平台视频上的 42 条人工业务查询。
- `open_queries.jsonl`：10.769 小时开放 ASR 测试集上的人工查询，以及最初的 8 条无答案查询。
- `no_answer_queries_v1.jsonl`：48 条分层无答案查询，覆盖易负例、同领域困难负例和跨语言负例。
- 开放语料使用 `asr_internal_eval_20260709` 中 SenseVoice 的真实识别结果；truth 只用于确定相关时间段，不作为被检索文本。
- `asr_v1_platform_yesterday_ep04` 与当前平台视频重复，因此开放语料侧排除该条。

最终 corpus 为 9 条当前平台视频加 16 条开放样本。查询按素材级分配到 `tune/dev/holdout`；同一开放素材不会跨 split。平台 v1 查询保留原 split，作为既有回归集。

## 评价原则

- 主要指标：MRR、Hit@1/5/10/50，并分别报告平台与开放语料结果。
- 参数只在 `tune` 选择，`dev` 用于比较方向，`holdout` 只报告最终泛化结果。
- 开放查询通过 truth 时间区间和实际 ASR chunk 的时间重叠建立 qrel，允许同一 truth 片段对应多个识别 chunk。
- 无答案查询用于观察 semantic 分数校准和误接收，不混入 MRR 分母。
- 无答案阈值实验共使用 74 条有答案查询和 56 条无答案查询；结果不能反向用于修改 holdout 标签或选参。
- 运行产物写入 `runtime-server/analysis/`，不提交模型 embedding 和逐查询大结果。

## 运行

统一脚本：`scripts/asr_retrieval_benchmark.py`。按 `--stage 1` 到 `--stage 6` 顺序执行；各阶段只在 `tune` 选参数，`dev` 用于安全约束和方向验证，`holdout` 只报告最终结果。完整命令与结果见：

```text
docs/experiments/asr/2026-07-14-asr-retrieval-v2-six-stage.md
runtime-server/analysis/asr_retrieval_benchmark_20260714/
```

无答案阈值校准使用 `scripts/asr_no_answer_threshold_eval.py`，结论见：

```text
docs/experiments/asr/2026-07-14-asr-no-answer-threshold-calibration.md
runtime-server/analysis/asr_no_answer_threshold_20260714/
```
