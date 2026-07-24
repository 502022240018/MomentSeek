# 问题池与后续路线

这是 MomentSeek 唯一的活跃问题池和后续优化列表。

状态：

```text
open / investigating / planned / in_progress / done / deferred
```

优先级：

```text
P0 = 阻塞安全使用或演示
P1 = 重要质量/稳定性问题
P2 = 有价值的改进
P3 = 后续打磨
```

每条记录建议格式：

```text
ID:
优先级:
状态:
范围:
问题或目标:
影响:
证据或上下文:
下一步:
相关文件或实验:
```

## 1. 检索质量与用户体验

### RQ-001 Visual 多视频搜索误召靠前

```text
优先级：P1
状态：done
范围：visual search ranking
问题或目标：
  SigLIP2 + 5s bucket MaxSim 提高了局部目标召回，但当搜索范围包含大量无关视频时，无关视频的本视频内部最佳 bucket 可能被排得过高。
影响：
  例如搜索“绿茵足球场有人在踢球”时，综艺视频片段可能排在真正足球场片段前面。
证据或上下文：
  已于 2026-07-07 修复：visual candidate 的 raw_score = visual_top1，Candidate.score 改为 raw cosine 的跨视频校准值 visual_rank_score = clip((raw_score + 1) / 2, 0, 1)。per-video percentile / robust_z 继续用于视频内 strong/fuzzy/weak 判定和诊断 evidence，不再作为跨视频排序主分数。
下一步：
  继续用真实素材观察“烤包子”和“绿茵足球场有人在踢球”等 query 的排序；如果仍出现单帧偶然相似误召，进入 RQ-002 的 top3/mean 一致性抑制。
相关文件或实验：
  backend/app/search.py
  backend/tests/test_search.py::test_visual_ranking_score_prefers_cross_video_raw_similarity_over_local_percentile
  docs/RETRIEVAL_CHANNELS.md
```

### RQ-002 Visual 单帧尖峰抑制

```text
优先级：P1
状态：open
范围：visual search ranking
问题或目标：
  MaxSim 可能因为某一帧偶然相似而抬高整个 5s bucket。
影响：
  周围 5s 内容并不相关时，误召仍可能看起来很强。
证据或上下文：
  当前 evidence 已包含 visual_top1、visual_top3、visual_mean。真实命中通常应该比偶然尖峰有更好的 top3/mean 一致性。
下一步：
  评估最终跨视频排序是否应结合 raw visual_top1 和 top3/mean consistency。
相关文件或实验：
  backend/app/search.py
  docs/experiments/visual/
```

### RQ-003 ASR chunk 后处理

2026-07-07 更新：

```text
状态：done for first pass / keep monitoring
已完成：
  - Whisper 强制 task="transcribe"，manifest 记录 requested/detected language。
  - ASR raw chunks 写入索引前做文本归一化、短 chunk 合并、低信息 chunk semantic 跳过。
  - retrieval_chunk_builder 只在同一 decode unit 内合并，合并结果最长 12s；模型原始完整长句不硬切，只标记 long。
  - SenseVoiceSmall 默认使用 `silero_12s` 外置 VAD，parser 保留原始文本，timestamp 只负责选择边界。
  - 2026-07-13：默认 `asr_engine=auto` 对长视频做开头/中段/后段 3 窗口语言投票；中文/方言路由到 SenseVoiceSmall，非中文路由到 faster-whisper turbo。
  - 2026-07-13：faster-whisper 使用 24s 连续原音频窗口；只对异常无句末窗口执行局部 builtin-VAD fallback。
  - 2026-07-10：移除正式 retrieval_chunk_builder 中的 CJK/Latin 断词猜测、false gap 修复和 word-boundary repair 计数。
  - 2026-07-13：semantic eligibility 拒绝纯语气/连接词、过短项和极端不可信文字/时长比，原因写入已有 quality_flags。
  - 2026-07-13：删除已退出正式路径的 asr_postprocess.py、旧策略报告及其测试；删除未读取的 hard_max_duration_ms，以及与 chunk_builder_stats 重复的 postprocess_strategy/postprocess_stats 元数据。
后续：
  - 启发式规则消融和新素材验证统一跟踪于 RQ-003F，不再恢复旧多策略后处理路径。
  - 本地 9 个视频已全部 ASR-only 重跑；其他环境的旧 ASR 索引仍需重跑。
相关实验：
  docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md
  runtime-server/analysis/asr_dual_path_final_scheme_20260709/
```

```text
优先级：P2
状态：open
范围：asr search quality
问题或目标：
  ASR chunk 可能过短、过长或边界不适合 semantic search。
影响：
  语义检索质量和播放片段边界都会受影响。
证据或上下文：
  当前 chunks 主要来自 Whisper 或 sidecar transcript。
下一步：
  设计短 chunk 合并、长 chunk 拆分和可选滑窗 semantic chunk。
相关文件或实验：
  backend/app/indexing/asr.py
  backend/app/search.py
```

### RQ-003C ASR pipeline 分层与 false timestamp gap
```text
优先级：P1
状态：done for first pass / keep monitoring
范围：asr search quality / transcript reliability
问题或目标：
  生产 ASR 索引曾出现中文词内断裂和不自然切分，例如“孤/独敏感”“很/难受”“永/远”。
  用户听查确认部分 timestamp gap 并不是音频真实停顿，因此不能把模型 timestamp gap 直接作为最终检索文本边界。
影响：
  检索文本不自然会降低 semantic embedding 质量，也会让用户按自然短语查询时漏召回或误召回。
已完成：
  1. ASR pipeline 拆为 raw transcript parser 和 retrieval_chunk_builder。
  2. parser 不再按固定 8s/12s 规则生成最终检索 chunk。
  3. retrieval_chunk_builder 负责短碎片合并、近邻合并、同 5s bucket 内有限合并和低信息 chunk 标记。
  4. asr.npz 默认仍只保存 chunk_times_ms、texts、embeddings、embedding_chunk_indices。
  5. debug 开启时才保存 raw transcript、retrieval chunks 和 repair report。
  6. 落地双路径方案：SenseVoiceSmall + Silero external VAD 12s；faster-whisper turbo + 24s 连续原音频窗口与异常窗口局部 fallback。
  7. 2026-07-10 默认自动语言路由：中文/方言走 SenseVoiceSmall，英文/西语/葡语等走 faster-whisper turbo。
  8. 2026-07-10 SenseVoice parser 支持 word-level timestamp，timestamp 缺失时使用 VAD group bounds 兜底，避免 0 秒 chunk。
  9. 2026-07-13 SenseVoice word timestamp 不再重拼文本；faster-whisper 直接保留 segment.text。
  10. 2026-07-13 retrieval chunk 不跨 decode unit 合并，合并上限 12s；完整 raw 长句不硬切。
  11. 2026-07-13 长视频语言 probe 改为 3 个位置投票，书籍纪录片由错误 `en` 路由恢复为 `zh`。
仍需观察：
  - SenseVoiceSmall 与 faster-whisper turbo 在同一 chunk builder 下的真实检索召回差异。
  - 混合语言视频目前仍按整片主语言路由；未来只有在真实查询评估证明必要时再考虑分段路由。
  - 完整索引仍保留 4 个 `>12s` 模型原始完整句（最长 19.56s）；它们未找到更可靠边界，因此按“不硬切”原则保留并标记 long。
  - 暂不做 CJK/Latin 断词猜测和 false gap 修复，优先从模型、VAD、timestamp 与切分策略上减少原始断裂。
相关文件或实验：
  backend/app/indexing/asr.py
  backend/app/indexing/asr_transcript_parser.py
  backend/app/indexing/asr_retrieval_chunks.py
  backend/app/indexing/asr_debug.py
  docs/superpowers/specs/2026-07-09-asr-pipeline-refactor-design.md
  docs/superpowers/plans/2026-07-09-asr-pipeline-refactor.md
  docs/experiments/asr/2026-07-13-asr-production-chunk-pipeline.md
```

### RQ-003E ASR semantic 跨视频近分误排

```text
优先级：P2
状态：in_progress / offline experiment complete
范围：asr/ocr semantic ranking / embedding model evaluation / threshold calibration
问题或目标：
  MiniLM 在纯 semantic、跨语言或低分区间会出现弱相关结果与真实结果近分并列。
  已完成 MiniLM / multilingual-e5-small / gte-multilingual-base 的 ASR chunk-level 对比；GTE 明显更好，但模型专属阈值和生产 API A/B 尚未完成，当前默认仍是 MiniLM。
影响：
  中文查询“姆巴佩应该多传球”时，世界杯广告中的真实英文台词在全库 API 排第 2；Top-1 是无明显语义关系的中文综艺短句，两者校准分均约 0.62。
证据或上下文：
  2026-07-13 正式 embedding 评估 18 条 query：目标 chunk Top-1/3/5 = 16/18/18；两条 Top-1 失败都在 Top-2。
  2026-07-14 retrieval v2：25 个 source、6854 个真实 ASR chunk、74 条有答案查询和 8 条无答案查询。GTE semantic-only 全量 MRR/H@1/H@5 = 0.869/0.824/0.932，MiniLM 为 0.761/0.703/0.851；E5-small 没有形成稳定收益。
  最终离线融合候选为 GTE absolute + legacy bigram，90/10 semantic-primary，并为 lexical coverage >= 0.50 建独立优先区；全量 MRR/H@1/H@5/H@50 = 0.899/0.865/0.946/0.973。
  GTE-base 为 768 维，主权重文件 610,753,338 bytes，本机 Hugging Face 缓存合计约 627,964,743 bytes；相比当前 384 维 MiniLM，float16 索引向量空间翻倍。当前决定是只保留实验记录，暂不改默认模型、依赖或正式索引。
  2026-07-14 无答案校准扩充到 56 条负例。现行 MiniLM threshold 的 FAR 为 100%；保持 tune target recall >= 95% 时，holdout recall/FAR 为 0.935/0.895。raw cosine 0.60 的严格观察点全量 recall/FAR 为 0.919/0.393，仍不适合直接作为“无答案”判定，因此生产阈值未修改。
下一步：
  1. 用真实用户零结果 query 扩充无答案、困难负例和多语言 query；如果未来恢复 GTE 评估，必须单独拟合 confidence / above-threshold，不得沿用 MiniLM sigmoid 和阈值。
  2. 在真实 API 的 per-video candidate、clip merge、final result 层复跑 shadow A/B，并记录视频级 MRR、冷热查询延迟、模型常驻内存与 CPU/GPU 建库吞吐。
  3. 将 ASR 与 OCR semantic model 配置解耦，或先完成 OCR 固定评测；禁止混用 MiniLM 与 GTE 向量空间。
  4. 上述验证通过后，才修改默认模型并重建全部 ASR semantic embedding；当前正式 9 条视频仍保留 MiniLM 384-d。
  5. 不要通过继续修改 chunk 切分来掩盖向量排序问题。
相关文件或实验：
  backend/app/search.py
  backend/app/indexing/text_semantic.py
  runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/REPORT.md
  docs/experiments/asr/2026-07-13-asr-production-chunk-pipeline.md
  docs/experiments/asr/2026-07-14-asr-retrieval-v2-six-stage.md
  docs/experiments/asr/2026-07-14-asr-no-answer-threshold-calibration.md
  eval/asr/retrieval_v2/
  scripts/asr_retrieval_benchmark.py
  scripts/asr_no_answer_threshold_eval.py
  runtime-server/analysis/asr_retrieval_benchmark_20260714/
```

### RQ-003H ASR 候选精排

```text
优先级：P1
状态：planned / evidence ready
范围：ASR-only candidate reranking / multilingual relevance / result-level evaluation
问题或目标：
  当前 MiniLM + combined 主序 + lexical reserve 的候选召回已经较高，但 Top-1 仍明显低于 Top-50。
  下一阶段保持 MiniLM 384 维索引不变，对现有 Top-30/50 候选增加小型 multilingual cross-encoder 精排，优先改善语义改写、跨语言和同主题近分误排。
实验依据：
  retrieval v2 的 74 条有答案查询中，当前参考方案 MRR/H@1/H@5/H@50 = 0.831/0.784/0.919/0.946，约 70 条已进入 Top-50、68 条进入 Top-5、58 条位于 Top-1。
  这说明多数失败不是“完全没有召回”，而是正确 chunk 已在候选池中但排名不够靠前；精排存在明确的可提升上限。
  42 条平台查询的 combined-primary lexical reserve 保持 H@1 0.833，并将 H@5/H@50 提升到 0.929/0.952；全量 RRF 和通用 CJK 连续片段加分会降低 Top-1，不能作为精排替代品。
  无答案实验中，单阈值保持约 95% recall 时 holdout FAR 仍约 89.5%；“收入八千本书/卖掉八千本书”等困难反事实需要 query-chunk 联合判断，不能继续只调 cosine。
候选输入基线：
  1. 先固定现有 combined 主序和 lexical reserve，取去重后的 Top-30/50，避免同时改召回与精排导致归因不清。
  2. 第二阶段再加入语言感知 lexical、受保护 pinyin 和多视图候选，比较 candidate recall 上限变化。
  3. 每个候选保留原始 score、lexical/semantic/pinyin 来源和 chunk_id；reranker 只新增独立 relevance score，不覆盖原始 evidence。
实验步骤：
  1. 冻结 25-source corpus、74 条正例、56 条无答案及 tune/dev/holdout；先输出每条 query 的固定 Top-50 候选快照。
  2. 选择 2-3 个体量可控的 multilingual cross-encoder，只做离线 Top-30/50 rerank；记录模型大小、CPU/GPU 常驻内存、批量吞吐和单 query p50/p95 延迟。
  3. 输入先使用 query + 当前 chunk；再单变量 A/B query + 当前 chunk + 邻接上下文，禁止同时加入新的 chunk 合并规则。
  4. 比较无精排、纯 reranker、reranker + strong lexical safety 三组；strong exact/lexical 命中不得被明显不相关候选挤出 Top-5。
  5. 按 semantic_paraphrase、cross_lingual、lexical、短实体和语言分别报告 MRR、H@1/5/50，不能只看全量平均。
  6. 在 56 条无答案中单独报告 FAR，并人工检查实体、数字、否定和反事实 hard negatives；精排分数不能未经校准直接改成“有答案”判断。
验收原则：
  1. tune 只用于选择模型和参数，dev 用于方向与安全约束，holdout 只做最终报告。
  2. 相比当前参考，H@1 至少提高 3 个百分点，H@5/H@50 不下降，且 lexical exact/rewrite 不出现明确回归。
  3. holdout 与跨语言分类不能出现方向相反的明显退化；收益不能只来自当前平台的少量已知 query。
  4. 延迟和常驻内存必须单列；若精排成本不适合交互搜索，则缩小 Top-K、量化或蒸馏，而不是静默接受高延迟。
边界：
  精排只能重排已召回候选，不能救回 Top-50 外的约 4 条正例。漏召继续由 RQ-003A 的 pinyin/实体容错、语言感知 lexical、多视图或 query expansion 处理。
  本事项不更换高维 embedding、不重建 ASR 文本，也不把 ColBERT/SPLADE 等新索引体系混入第一轮实验。
相关文件或实验：
  backend/app/search.py
  eval/asr/hybrid_retrieval_queries_v1.jsonl
  eval/asr/retrieval_v2/
  docs/experiments/asr/2026-07-14-asr-dual-pool-retrieval.md
  docs/experiments/asr/2026-07-14-asr-retrieval-v2-six-stage.md
  docs/experiments/asr/2026-07-14-asr-no-answer-threshold-calibration.md
  runtime-server/analysis/asr_retrieval_benchmark_20260714/
```

### RQ-003F ASR 启发式规则消融与准入

```text
优先级：P2
状态：planned
范围：asr chunk builder / semantic eligibility / faster-whisper fallback
问题或目标：
  当前正式文本已可作为稳定基线，但仍包含 same_bucket、gap、连接词过滤和局部 fallback 等经验规则。
  后续不再根据单个错误样例追加规则，而是量化每条规则的收益、触发范围和误伤风险。
执行步骤：
  1. 为每条正式启发式规则统计触发次数、影响的 chunk 数和对应视频/语言分布。
  2. 分别关闭 same_bucket 合并、连接词过滤及其他可疑规则，做单变量消融，不同时修改多个条件。
  3. 使用现有人工听查结果和固定 18 条 semantic query 检查边界质量与检索指标是否退化。
  4. 增加未参与现有参数选择的新语言、新节目类型视频作为 holdout，避免只适配当前 9 条素材。
  5. 规则只有在当前评估集与 holdout 上都改善，且没有明确误伤时才进入或继续留在正式路径。
验收原则：
  最终通用后处理只保留空白/字符规范化、可靠边界合并和极端异常过滤；模型或 adapter 负责其特有输出格式。
已完成的消融：
  2026-07-14 在 25-source / 82-query retrieval v2 上比较 current eligibility、obvious hard filter、hard filter + soft penalty。新增规则只多拒绝 6/6641 个 semantic chunk，soft penalty 降权 218 个 chunk；所有 Hit 指标不变，tune 加权目标增益仅 0.462 个百分点，未达到 0.5 个百分点最小实质收益门槛，因此保留 current eligibility，不新增规则。
相关文件或实验：
  backend/app/indexing/asr_retrieval_chunks.py
  backend/app/indexing/asr_text.py
  backend/app/indexing/asr.py
  runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/
  docs/experiments/asr/2026-07-13-asr-production-chunk-pipeline.md
  docs/experiments/asr/2026-07-14-asr-retrieval-v2-six-stage.md
```

### RQ-003A ASR 错词容错与专有名词召回

```text
优先级：P2
状态：investigating
范围：asr search quality / semantic retrieval / lexical fallback
问题或目标：
  当前 ASR 原文里存在较多听错字、同音近音错词、专有名词误识别。
  MiniLM semantic embedding 可以缓解主题型 query 的召回，但不能可靠恢复被 ASR 听错的人名、地名、片名、书名、品牌名等关键信息。
影响：
  用户搜索具体实体词时，semantic embedding 可能不稳定；lexical 搜索也会因为原文错词而漏召回。
证据或上下文：
  现有素材里出现过类似“黄拔”“赵正宵”“冰气”等疑似 ASR 错词。
  embedding 对完整上下文的主题相似度有效，但对短 chunk 或关键名词错误无能为力。
已完成：
  2026-07-14：基于当前 9 个完整 ASR 索引建立 42 条 lexical / semantic 查询集。否决通用 CJK 连续片段加分和直接 weighted RRF，改为 combined 主序 + 强 lexical 独立候选池保底；全量 Hit@1 保持 0.833，Hit@5 从 0.905 提升到 0.929，Hit@50 从 0.929 提升到 0.952。“昆仑山”指定原句由离线第 72、真实 API Top-50 外提升到真实 API 第 4，无需重建索引。
未来优化方向：
  1. ASR 模型质量对比：按中文/英文/多语种素材比较 FunASR/Paraformer、Whisper small、Whisper medium 或其他更强转写模型。
  2. 发音容错索引：为中文 ASR 文本增加 pinyin/近音检索 fallback，补人名、地名、片名、专有名词听错导致的漏召回。
  3. 实体词保护：对字幕、OCR、文件名、用户标注里出现的实体词建立词表，用于 ASR 后处理提示、纠错候选或搜索扩展。
  4. 多路融合：ASR semantic 负责主题召回，ASR lexical 负责精确词，pinyin/近音负责错词容错，最终在 evidence 中标明命中来源。
  5. 评估集：构造一组“正确 query -> 错误 ASR 文本”的样例，单独评估 semantic、lexical、pinyin fallback 的召回贡献。
下一步：
  已完成第一版只读候选导出和 pinyin fallback seed eval。下一步从候选 HTML 中人工听查 30-50 条，补 correct_text/manual_label/真实 query，并增加 negative controls 评估误召风险。
相关文件或实验：
  backend/app/indexing/asr.py
  backend/app/indexing/asr_text.py
  backend/app/search.py
  docs/experiments/asr/
  docs/experiments/asr/2026-07-07-asr-pinyin-fallback-seed.md
  docs/experiments/asr/2026-07-14-asr-dual-pool-retrieval.md
  eval/asr/asr_pinyin_seed_eval_20260707.jsonl
  eval/asr/hybrid_retrieval_queries_v1.jsonl
  scripts/asr_error_candidates.py
  scripts/asr_hybrid_retrieval_eval.py
  scripts/asr_pinyin_fallback_eval.py
```

### RQ-003B ASR 重复幻觉过滤与风险标注
```text
优先级：P1
状态：open
范围：asr search quality / transcript reliability
问题或目标：
  删除中文 initial_prompt 并全量重建 ASR 后，prompt 泄漏已消失，但 Whisper 局部重复幻觉仍存在。
  当前 retrieval_chunk_builder 会有限合并短碎片，但不会主动删除或降权重复幻觉文本。
影响：
  用户按 ASR 文本检索时，重复幻觉片段可能被错误召回；播放时也会看到不存在或错位的台词。
证据或上下文：
  2026-07-07 本地全量 ASR 重建后：
  - 书籍纪录片 prompt 泄漏短语计数为 0。
  - 天c游xi 08:40 左右不再出现旧索引中的连续“我去找他”。
  - 电视剧昨夜降至04 04:03-04:27 仍出现连续“你跟她说”。
  - 天c游xi 04:22 左右仍出现“你这些人”短语级重复，并存在连续单字“你” run。
下一步：
  1. 保存 Whisper segment 诊断字段：avg_logprob、compression_ratio、no_speech_prob。
  2. 增加只读报告脚本，统计 chunk 内重复 n-gram、连续单字循环和低置信片段。
  3. 先对明显低信息重复 chunk 标记 semantic_eligible=False 或检索降权，避免直接误删真实重复台词。
  4. 用已导出的全文和人工听查样例评估误杀风险，再决定是否在索引阶段过滤。
相关文件或实验：
  docs/experiments/asr/2026-07-07-asr-full-rebuild-no-prompt.md
  runtime-server/analysis/asr_rebuild_20260707_all_summary.json
  runtime-server/analysis/asr_full_texts_after_rebuild_20260707/
  backend/app/indexing/asr.py
  backend/app/indexing/asr_retrieval_chunks.py
```

### RQ-003D ASR 模型 adapter 边界

```text
优先级：P2
状态：open
范围：asr architecture / model extensibility
问题或目标：
  当前 ASR 可用 SenseVoiceSmall/FunASR 和 faster-whisper turbo，未来可能继续更换或增加模型。
  需要保持平台通用 pipeline 简洁，避免把某个模型的输出格式、tag、timestamp 怪癖或参数补丁扩散到通用检索层。
影响：
  如果模型分支继续堆在 backend/app/indexing/asr.py，后续新增模型会让加载、VAD、timestamp、language metadata、tag parser 和 retrieval chunk 逻辑互相干扰。
  如果过早抽象，也可能在模型实验还没稳定时增加不必要复杂度。
设计原则：
  1. 模型特异逻辑只放在 model adapter / parser 层，例如模型加载、VAD 参数、raw output 解析、tag 剥离和必要 metadata 映射。
  2. 通用层保持模型无关：RawTranscriptItem、retrieval_chunk_builder、semantic embedding、asr.npz schema 和 search 不写某个模型专属分支。
  3. 当前 schema 的 chunk_emotions/chunk_audio_events 是可选通用增强字段；有能力的模型填值，没有能力的模型写空字符串。
触发条件：
  1. asr.py 中新增第三个以上模型分支，或单个模型需要大量专属参数。
  2. timestamp/tag/language metadata 的解析逻辑开始明显膨胀。
  3. 测试很难单独覆盖某个模型输出到 RawTranscriptItem 的转换。
  4. 通用 retrieval chunk 合并逻辑被迫判断具体模型名。
下一步：
  触发后把模型调用拆为 backend/app/indexing/asr_adapters/ 下的 sensevoice.py、faster_whisper.py、whisper.py、sidecar.py 等小模块。
  每个 adapter 输出统一的 raw_items 和少量 metadata，主流程继续只负责 extract audio -> adapter.transcribe -> build_retrieval_chunks -> semantic embedding -> save asr.npz。
相关文件或实验：
  backend/app/indexing/asr.py
  backend/app/indexing/asr_transcript_parser.py
  backend/app/indexing/asr_retrieval_chunks.py
  docs/RETRIEVAL_CHANNELS.md
```

### RQ-003G SenseVoice 情绪/音效标签检索利用

```text
优先级：P2
状态：planned
范围：asr emotion/audio-event metadata / query intent / ranking
问题或目标：
  SenseVoice 已为部分 ASR chunk 生成 emotion 和 audio_event 标签，但当前搜索尚未使用。需要先确认各标签在真实视频中的含义和可靠性，再决定如何参与召回、重排和结果解释。
当前状态：
  1. asr.npz 已保存 chunk_emotions 和 chunk_audio_events；faster-whisper 路径写空字符串。
  2. 当前 SenseVoice 使用 Silero 人声 VAD，因此标签主要描述“台词及其伴随声音”，不能覆盖所有纯音乐、纯掌声等无语音片段。
  3. 标签保持独立结构化 metadata，不拼入 ASR semantic embedding 文本。
2026-07-13 初轮人工听查：
  1. 每个实际出现标签抽样 5 条，共 40 条，音频包含 chunk 前后各 2 秒。
  2. happy 的清晰样本几乎都伴随笑声，更像正向/高唤醒或笑意线索，与 laughter 高度相关。
  3. angry 约一半不符合字面“愤怒”，可能混入激动、着急、大声或综艺式夸张表达，暂不能作为愤怒硬条件。
  4. sad 和 surprised 相对准确，可作为中等强度的正向证据。
  5. 标为 bgm 的样本均能听到 BGM；但 speech 样本也出现实际带 BGM，说明 event 更接近主导标签，speech 不能解释为“没有 BGM”。
  6. laughter 基本准确，是当前最可靠的音效标签之一。
暂定使用原则：
  1. 只奖励明确标签命中，不因标签缺失或其他标签而扣分；缺失表示 unknown，不表示不匹配。
  2. laughter、bgm 可作为较强正向证据；sad、surprised 作为中等证据；happy 只做弱证据；angry 暂停按字面接入；speech 仅展示。
  3. happy 与 laughter 等相关线索取最大值或设置统一加分上限，禁止重复累加同一声学现象。
  4. 查询没有明确情绪/音效意图时，标签不得改变普通文本检索排序。
  5. 初期只做候选重排和 evidence 展示，不做硬过滤；纯音频事件检索若成为明确需求，再评估独立且不依赖人声 VAD 的 audio-event 通道。
下一步：
  1. 再抽 20-30 条 angry，人工区分真愤怒、激动、着急、大声和其他，确认是否应改解释为 high-arousal 候选。
  2. 从 speech 中抽样至少 20 条检查实际 BGM，估计 bgm 标签漏检率；不能只评估 bgm 阳性精度。
  3. 扩大 happy 样本，区分真实开心、笑声驱动、兴奋和误判。
  4. 固定标签 query 与负例，对“不使用标签 / 正向弱加分 / 较强加分”做排序 A/B，再决定正式权重。
相关文件或实验：
  backend/app/indexing/asr_transcript_parser.py
  backend/app/indexing/asr_retrieval_chunks.py
  backend/app/indexing/asr.py
  backend/app/search.py
  runtime-server/analysis/sensevoice_tag_review_20260713/review.html
  runtime-server/analysis/sensevoice_tag_review_20260713/samples.jsonl
```

### RQ-004 OCR chunk 质量

```text
优先级：P2
状态：open
范围：ocr search quality
问题或目标：
  OCR 当前 0.05fps 抽帧，chunk 约 20s，粒度偏粗。
影响：
  OCR 命中片段可能过长，也可能错过短暂文字。
证据或上下文：
  当前 OCR chunk end_time = timestamp + 1 / sample_fps。
下一步：
  评估自适应 OCR 抽帧或基于文本连续性的 chunk 合并。
相关文件或实验：
  backend/app/indexing/ocr.py
  docs/RETRIEVAL_CHANNELS.md
```

### RQ-005 统一搜索质量回归基线

```text
优先级：P1
状态：in_progress / unified evaluator implemented
范围：retrieval evaluation / quality regression
问题或目标：
  Visual、ASR 和 sequence retrieval 已有多份数据集、脚本与实验记录，但尚未形成覆盖核心用户查询的统一固定回归门槛。
影响：
  排序、阈值、chunk 或融合策略调整后，容易只改善少量手工样例，同时让其他查询或其他通道退化。
证据或上下文：
  当前 eval/visual、eval/asr 和 docs/experiments 已积累可复用资产；ASR 已有固定 semantic query 评估，Visual 仍需要把典型误召查询纳入稳定基线。
下一步：
  1. 已新增 `scripts/retrieval_quality_eval.py` 和 `eval/RETRIEVAL_QUALITY.md`，统一输出 Recall@K、MRR、nDCG@K、Top-K 误召率、无答案错误接受率和时间段 tIoU。
  2. 固定真实剪辑 query、目标视频/时间段、负样本和 tune/dev/holdout 划分；当前 Visual seed 中未人工确认的 positives 不得充当正式门槛。
  3. 先把 RQ-002 的 top1/top3/mean 候选策略做单变量离线对比。
  4. 为关键指标定义允许波动和回归门槛，并在搜索策略变更时运行。
相关文件或实验：
  eval/visual/
  eval/asr/
  scripts/visual_clip_eval.py
  scripts/visual_clip_eval_report.py
  scripts/asr_eval_report.py
  docs/experiments/
```

### RQ-006 Speaker Diarization 与声纹检索

```text
优先级：P2
状态：deferred（当前基线已实现，后续优化暂缓）
范围：asr speaker attribution / voiceprint retrieval
问题或目标：
  当前基线已支持视频内说话人区分、自适应 speaker turn、逐 turn 声纹索引、
  跨视频同声纹搜索，以及在人物库中查看、修正和绑定 speaker。
已知不足：
  1. 谱聚类仍可能欠分，UMAP + HDBSCAN fallback 仍可能过分。
  2. 尚未可靠处理重叠说话、音乐和低信噪比片段。
  3. track 中心为简单均值，缺少质量加权、离群点过滤和多原型表达。
  4. 缺少固定人工 truth，尚未系统测量 DER、B-cubed F1、EER、Recall@K 和误报率。
  5. 第一阶段不建立跨视频永久 Speaker ID。
下一步：
  当前暂不继续开发。恢复时先建立多人综艺、电视剧、广告、纪录片及重叠语音人工标注集，
  再比较谱聚类、HDBSCAN、AHC/VBx、重叠语音过滤、turn 质量评分和稳健 track 聚合；
  不针对当前局部视频继续追加补丁式阈值。
相关文件或实验：
  docs/superpowers/specs/2026-07-13-speaker-diarization-voiceprint-design.md
  backend/app/indexing/speaker.py
  backend/app/speaker_service.py
  eval/speaker/
```

## 2. 性能、资源与推理效率

### PERF-001 模型加载和释放开销

```text
优先级：P1
状态：open
范围：indexing performance
问题或目标：
  Ascend 产品模式已改为单个串行 indexer daemon；Visual 和 Face 已接入 ModelPool 并可常驻。
  SenseVoice/FunASR 的 AutoModel、Silero VAD，以及 ASR/OCR 文本 embedding encoder 尚未接入池，
  因此每次进入 ASR 通道仍会重新构造模型，尚未实现完整的索引模型常驻复用。
影响：
  连续构建多个视频的 ASR 索引仍会重复承担模型加载和 NPU 预热成本。
证据或上下文：
  2026-07-20 已在 Ascend 部署中启用 resident daemon 和串行通道调度；当前 `_funasr()`
  仍在每次调用时执行 `AutoModel(...)`，与已经支持 encoder 注入的 Visual/Face 不同。
下一步：
  抽象 SenseVoiceEngine，将 AutoModel、Silero VAD session 和文本 embedding encoder 以明确生命周期
  注入索引流程；每个任务使用独立推理 cache，任务结束清理状态但保留权重。补充连续多视频回归，
  测量冷启动、热启动、常驻 HBM、异常恢复和容器退出释放，再决定 faster-whisper probe 是否一并常驻。
相关文件或实验：
  backend/app/model_pool.py
  backend/app/indexer_daemon.py
  docs/experiments/visual/
```

### PERF-002 Visual 预处理瓶颈

```text
优先级：P2
状态：open
范围：visual indexing speed
问题或目标：
  visual 索引速度很多时候卡在 CPU 解码/resize/预处理，而不是 NPU encoder。
影响：
  720p 和长视频索引耗时高于预期。
证据或上下文：
  历史 benchmark 显示 cv2 resize 有收益；合并 visual+face 解码曾导致同卡运行时互抢。
下一步：
  继续测量预处理，避免重新引入 visual+face 同进程 NPU 互抢。
相关文件或实验：
  backend/app/indexing/visual.py
  docs/experiments/visual/2026-07-01-clip-910b.md
```

### PERF-003 ASR 速度和模型策略

```text
优先级：P2
状态：open
范围：asr indexing performance
问题或目标：
  ASR 默认切到 auto 路由；ASR 耗时仍受语言 probe、语音密度、VAD、timestamp 对齐和 chunk 切分影响很大。
影响：
  长视频或语音密集视频可能由 ASR 主导总耗时。
证据或上下文：
  Whisper medium 在共享 NPU 2 上曾 OOM；2026-07-08 实验显示 SenseVoiceSmall 速度快、中文/方言文本可读性好；2026-07-10 起非中文默认路由到 faster-whisper turbo，并开启 word timestamp 拆长段。
下一步：
  继续测量 auto probe、SenseVoice 和 faster-whisper 在真实索引任务里的分阶段耗时；评估是否需要缓存 probe 模型或分段并行。
相关文件或实验：
  backend/app/indexing/asr.py
  docs/OPERATIONS.md
```

### PERF-004 NPU 内存管理和共享资源安全

```text
优先级：P0
状态：open
范围：server resources
问题或目标：
  MomentSeek 必须在共享 NPU 资源上运行，不能影响 ComfyUI、VLLM 或其他用户进程。
影响：
  不安全的清理或 broad kill 会中断他人任务。
证据或上下文：
  共享服务器存在无关 VLLM 和 python 进程，只能操作明确归属 MomentSeek 的目标。
下一步：
  严格执行只读检查，并把事故经验写入 `docs/LESSONS_LEARNED.md`。
相关文件或实验：
  docs/OPERATIONS.md
  docs/LESSONS_LEARNED.md
```

## 3. 工程稳定性与运维

### ENG-001 文档体系整理

```text
优先级：P1
状态：done
范围：docs
问题或目标：
  项目知识曾分散在旧 handoff、当前状态、报告和实验笔记中。
影响：
  新会话需要读很多文件，问题列表也重复出现。
证据或上下文：
  文档体系已收敛到 `docs/README.md` 下的固定文件，旧文档已归档到 `docs/archive/`。
下一步：
  后续按 `docs/README.md` 的更新规则维护，避免新增平行的问题池或重复 handoff。
相关文件或实验：
  docs/README.md
  docs/superpowers/specs/2026-07-03-docs-experiments-consolidation-design.md
```

### ENG-002 公网入口稳定性

```text
优先级：P2
状态：open
范围：public access
问题或目标：
  当前 Cloudflare quick tunnel 是临时入口，且可能依赖 PC 转发。
影响：
  后端健康时，前端也可能因为 tunnel/SSH 断开而 `failed to fetch`。
证据或上下文：
  Quick tunnel 域名会变；当前项目只给自己和少数同学测试。
下一步：
  短期保留当前方案；后续可评估服务器侧 quick tunnel、ngrok dev domain 或有域名后的 Cloudflare named tunnel。
相关文件或实验：
  docs/OPERATIONS.md
```

### ENG-003 公网演示鉴权

```text
优先级：P1
状态：open
范围：public access security
问题或目标：
  当前公网 demo 可能暴露上传、搜索、删除能力，没有鉴权。
影响：
  不适合敏感视频或扩大分享。
证据或上下文：
  当前只面向自己和少数同学测试。
下一步：
  在更广泛分享前增加 Basic Auth、简单访问密码或 Cloudflare Access；同时限制上传类型、文件大小、请求频率和高成本操作权限，并记录关键写操作审计信息。
相关文件或实验：
  backend/app/main.py
  frontend/src/api.ts
```

### ENG-004 前端组件拆分

```text
优先级：P2
状态：open
范围：frontend maintainability
问题或目标：
  `frontend/src/main.tsx` 承担了大部分 UI 行为。
影响：
  搜索、素材、索引和播放器逻辑继续增长后会更难协作。
证据或上下文：
  当前前端可用，但集中在单个大文件中。
下一步：
  按 workflow 拆分：upload/indexing、search、assets、player、shared controls。
相关文件或实验：
  frontend/src/main.tsx
  frontend/src/api.ts
```

### ENG-005 索引状态和完整性工具

```text
优先级：P2
状态：open
范围：tooling
问题或目标：
  需要快速查看每个视频有哪些通道索引、哪些 semantic 文件，以及 schema、模型版本、向量数量和文件可读性。
影响：
  排查 ASR/OCR/semantic 缺失时需要手动看文件。
证据或上下文：
  当前服务器 OCR 覆盖不完整，ASR/OCR semantic 也是可选文件。
下一步：
  增加索引完整性检查和状态导出脚本；在素材页展示每个通道的 available/missing/stale/corrupt 状态及重建建议。
相关文件或实验：
  runtime/indexes/
  backend/app/db.py
```

### ENG-006 Job cancel 和 stale job 清理

```text
优先级：P2
状态：open
范围：job lifecycle
问题或目标：
  中断后可能出现 stale running job，UI 也没有 cancel 功能。
影响：
  用户和运维人员会被错误 job 状态误导。
证据或上下文：
  历史上 face job 出现过 running 但 worker 已不存在。
下一步：
  增加安全 cancel、worker 进程归属、API 启动恢复和脚本化 stale-job cleanup；取消时只终止明确属于该 job 的进程。
相关文件或实验：
  backend/app/worker.py
  backend/app/db.py
  frontend/src/main.tsx
```

### ENG-007 多人开发与可复制部署第一阶段

```text
优先级：P1
状态：in_progress
范围：development workflow / deployment
问题或目标：
  GitHub clone 后可以按 dev.cpu/dev.cuda profile 开发和验证；staging/prod/new-server 可以通过 release manifest、model manifest、models lock 和 env profile 可复制部署。
影响：
  降低多人协作接手成本，减少服务器手工步骤和模型缓存漂移。
证据或上下文：
  第一阶段已新增 dev.cpu/dev.cuda/staging.ascend/prod.ascend profile 和 manifest，并补充 development、deployment、models 文档。
下一步：
  completing docs/env profile/model manifest/bootstrap/smoke/health metadata，并在实际 staging/prod 发布中记录 deployment record。
相关文件或实验：
  docs/DEVELOPMENT.md
  docs/DEPLOYMENT.md
  docs/MODELS.md
  deploy/env/
  deploy/models/
  scripts/bootstrap_dev.ps1
  scripts/bootstrap_dev.sh
  scripts/smoke_check.py
  backend/app/deployment.py
```

### ENG-008 CI/CD 与镜像化部署

```text
优先级：P2
状态：open
范围：deployment automation
问题或目标：
  Phase 1 保持 manual manifest/scripts；Phase 2 标准化 Dockerfile、compose、GitHub Actions、自动 publish 和 rollback。
影响：
  减少人工部署差异，提高 staging/prod/new-server 的一致性和回滚速度。
证据或上下文：
  当前可复制部署依赖 release manifest、env profile、model manifest、models lock 和手动脚本。
下一步：
  在第一阶段稳定后设计镜像构建、制品发布、部署记录写入和回滚自动化。
相关文件或实验：
  docs/DEPLOYMENT.md
  deploy/releases/
  scripts/write_release_manifest.py
```

### ENG-009 Runtime 容量与缓存治理

```text
优先级：P1
状态：open
范围：storage lifecycle / public access safety
问题或目标：
  上传视频、查询图片、缩略图和 preview clip 会持续写入本地 runtime；当前缺少统一的容量配额、缓存淘汰和磁盘水位保护。
影响：
  长时间使用或公网误用可能耗尽磁盘；无界上传也会放大解析耗时、模型任务和拒绝服务风险。
下一步：
  1. 为视频、字幕和查询图片增加类型、单文件大小和请求体限制。
  2. 为 clip/query 临时文件定义 TTL/LRU 清理策略和最大容量。
  3. 创建索引任务和生成 clip 前检查磁盘水位，低空间时拒绝高成本写操作并返回明确错误。
  4. 在 health 或独立诊断接口暴露 runtime 使用量、缓存量和剩余空间。
相关文件或实验：
  backend/app/main.py
  backend/app/media.py
  backend/app/settings.py
  runtime/uploads/
  runtime/clips/
```

### ENG-010 Catalog 与 API 可扩展性

```text
优先级：P2
状态：open
范围：catalog / API completeness / scale readiness
问题或目标：
  当前视频、任务和人物列表按全量返回，人物库只有创建和查看；SQLite 同步访问适合当前 MVP，但需要明确扩容边界。
影响：
  素材和任务数量增长后，列表、轮询和前端渲染成本会上升；人物参考信息错误时缺少修改、删除和重建入口。
下一步：
  1. 为 videos/jobs/entities 增加分页、筛选和稳定排序协议。
  2. 增加人物重命名、删除、替换参考图及 embedding 重建能力。
  3. 测量 SQLite/NPZ 在目标素材规模下的查询延迟和并发写入，再决定是否迁移到数据库向量存储。
  4. 保持当前 Search/API 概念稳定，为未来 pgvector、Milvus 或 Qdrant 适配预留存储接口边界。
相关文件或实验：
  backend/app/main.py
  backend/app/db.py
  backend/app/search.py
  frontend/src/api.ts
  frontend/src/main.tsx
  docs/ARCHITECTURE.md
```
