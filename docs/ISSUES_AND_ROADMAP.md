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
  - 新增 scripts/asr_postprocess_report.py，可在现有素材上比较 gap_only/bucket_bonus/shot_bonus/conservative/aggressive_short。
  - 当前默认策略为收紧后的 bucket_bonus。
后续：
  - 更多多语种素材上复跑调参报告。
  - 真实 shot-aware visual segment 稳定后，重新评估 shot_bonus。
  - 旧 ASR 索引需要 ASR-only 重跑才会应用新后处理。
相关实验：
  docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md
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
  eval/asr/asr_pinyin_seed_eval_20260707.jsonl
  scripts/asr_error_candidates.py
  scripts/asr_pinyin_fallback_eval.py
```

### RQ-003B ASR 重复幻觉过滤与风险标注
```text
优先级：P1
状态：open
范围：asr search quality / transcript reliability
问题或目标：
  删除中文 initial_prompt 并全量重建 ASR 后，prompt 泄漏已消失，但 Whisper 局部重复幻觉仍存在。
  当前 bucket_bonus 后处理会合并短碎片，但不会主动删除或降权重复幻觉文本。
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
  backend/app/indexing/asr_postprocess.py
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

## 2. 性能、资源与推理效率

### PERF-001 模型加载和释放开销

```text
优先级：P1
状态：open
范围：indexing performance
问题或目标：
  当前子进程模型生命周期更安全，但每个阶段/任务都要重新加载模型。
影响：
  长视频和重复建索引会付出明显 cold-start 成本。
证据或上下文：
  历史 910B benchmark 观察到固定模型加载成本，并实现过 warm pool / indexer daemon。
下一步：
  在测量常驻资源和共享服务器安全性的前提下，决定是否部署 `indexer_daemon.py` 和 model pool。
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
  ASR 默认切到 SenseVoiceSmall/FunASR；ASR 耗时仍受语音密度、VAD、timestamp 对齐和 chunk 切分影响很大。
影响：
  长视频或语音密集视频可能由 ASR 主导总耗时。
证据或上下文：
  Whisper medium 在共享 NPU 2 上曾 OOM；2026-07-08 实验显示 SenseVoiceSmall 速度快、文本可读性好，但 timestamp/chunk 切分仍需打磨；faster-whisper turbo 作为多语言/高效果备选。
下一步：
  优化 SenseVoice timestamp 对齐、VAD 和 chunk 合并策略；保留 faster-whisper turbo 对特殊语言或高效果场景的可切换路径。
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
  在更广泛分享前增加 Basic Auth、简单访问密码或 Cloudflare Access。
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
  需要快速查看每个视频有哪些通道索引、哪些 semantic 文件。
影响：
  排查 ASR/OCR/semantic 缺失时需要手动看文件。
证据或上下文：
  当前服务器 OCR 覆盖不完整，ASR/OCR semantic 也是可选文件。
下一步：
  增加索引完整性检查和状态导出脚本。
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
  增加安全 cancel 和脚本化 stale-job cleanup。
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
