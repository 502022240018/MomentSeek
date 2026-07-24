# ASR 全量重建：删除中文 initial_prompt + bucket_bonus 后处理

日期：2026-07-07

## 目的

将本地 `runtime-server` 中全部视频的 ASR 索引统一重建到当前 ASR 方案：

- Whisper 强制 `task="transcribe"`。
- `language="zh"` 时不再传中文 `initial_prompt`。
- 保留 Whisper 默认 `condition_on_previous_text=True`。
- raw chunks 入库前执行 `bucket_bonus` 后处理：文本归一化、短 chunk 合并、低信息 chunk semantic 跳过。
- ASR semantic embedding 仍使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。

## 环境

- 本地 Docker CUDA 后端：`momentseek-mvp-app`
- 端口：`127.0.0.1:18301 -> container:8000`
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- ASR 模型：Whisper `small`
- ASR engine：`whisper`
- ASR device：`auto`，容器内解析为 CUDA
- ASR semantic device：CPU

说明：本次没有完整重建 Docker image。因为 `docker compose build` 开始重新下载 Torch CUDA 大包，耗时过高，因此中断该 build，并用 `docker cp backend/app/. momentseek-mvp-app:/app/backend/app/` 将已测试的新后端代码同步到当前容器，再通过 API 的 ASR-only job 串行重建索引。仓库代码已经包含同样修改。

## 备份与输出

备份目录：

```text
runtime-server/analysis/asr_rebuild_backup_20260707_all/
```

包含：

- `catalog.sqlite3`
- 每个视频旧的 `asr.npz`
- 每个视频旧的 `index_manifest.json`

重建脚本与验证脚本：

```text
runtime-server/analysis/asr_rebuild_20260707_all_runner.py
runtime-server/analysis/asr_rebuild_20260707_verify.py
```

验证摘要：

```text
runtime-server/analysis/asr_rebuild_20260707_all_summary.json
```

重建后全文导出，一个 chunk 一行：

```text
runtime-server/analysis/asr_full_texts_after_rebuild_20260707/
```

## 重建参数

语言策略：沿用每个视频最近一次 ASR 任务的语言意图。

- `书籍纪录片.mp4`：`zh`
- `天c游xi.2026...mkv`：`zh`
- 其他视频：`auto`

理由：保留多语言素材的自动识别能力，同时让此前手动补跑中文的两条视频继续明确走中文转写；区别是本次不再传中文 prompt。

## 结果

| video_id | 视频 | requested_language | detected_language | raw_chunks | processed_chunks | semantic_chunks | elapsed_seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| `8e43cd0b84b74077b7f652b09374da9e` | 球星牛奶广告 | auto | es | 32 | 12 | 9 | 43.278 |
| `b5d17b3d0c904626893bd9043008959c` | 给阿嬷的情书预告片 | auto | zh | 44 | 15 | 15 | 46.231 |
| `8c6d9d47f1a74e4497a4654162c1ce2d` | 世界杯广告.mp4 | auto | en | 84 | 39 | 33 | 176.121 |
| `196c6422395241e8bb508ecf84b99289` | 五哈团美食速度挑战纯享_31min_1080p.mp4 | auto | zh | 940 | 325 | 270 | 521.637 |
| `a293b5981126444182208da7ba6274f5` | 电视剧昨夜降至04.mp4 | auto | zh | 693 | 270 | 240 | 494.894 |
| `b09c33148400467e856802ef59e8e479` | 书籍纪录片.mp4 | zh | zh | 1003 | 302 | 293 | 461.797 |
| `e8f92255e704482883da3beaea986000` | 2025-04-20 第2期下：五哈版决战天山之巅 够癫！.mkv | auto | zh | 1906 | 751 | 529 | 647.727 |
| `e96d218007fe43d4a1d39973ba55de93` | 天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv | zh | zh | 1841 | 709 | 610 | 777.435 |

## 验证

代码层测试：

```text
PYTHONPATH=backend python -m pytest backend/tests/test_transcript.py::test_whisper_zh_language_does_not_pass_initial_prompt -q
1 passed

PYTHONPATH=backend python -m pytest backend/tests/test_transcript.py backend/tests/test_asr_postprocess.py backend/tests/test_asr_semantic_filtering.py backend/tests/test_asr_text.py backend/tests/test_index_schema_v3.py -q
28 passed
```

容器内代码检查：

```text
app.indexing.asr._whisper()
```

已确认 `options` 只包含 `fp16`、`task` 和可选 `language`，不再包含 `initial_prompt`。

磁盘验证：

- 8 个视频均存在 `asr.npz` 和 `index_manifest.json`。
- `asr.npz` 字段均为 `chunk_times_ms`、`texts`、`embeddings`、`embedding_chunk_indices`。
- `chunk_times_ms.shape == (len(texts), 2)`。
- `embeddings.shape[0] == len(embedding_chunk_indices)`。
- `embedding_chunk_indices` 未越界。
- manifest 中 ASR 均有 `postprocess_strategy="bucket_bonus"`。
- manifest 中 ASR `semantic_status="complete"`。
- prompt 泄漏短语计数均为 0：
  - `这次是普通话简体中文转写`
  - `普通话简体中文转写`

## 抽查结论

正向结果：

- `书籍纪录片.mp4` 不再出现 prompt 泄漏。
- `天c游xi...mkv` 08:40 左右不再出现旧索引中的连续 `我去找他` 幻觉。
- 全量 ASR chunk 数明显减少，短碎片合并生效。

残留问题：

- Whisper 局部重复幻觉仍存在，删除 prompt 和 chunk 合并不能完全解决。
- `电视剧昨夜降至04.mp4` 04:03-04:27 仍出现连续 `你跟她说`：

```text
[0021] 00:04:03.040 --> 00:04:10.320  你刚刚一直在昏迷 我同事已经联系了 你跟她说 你跟她说 你跟她说 你跟她说
[0022] 00:04:10.320 --> 00:04:17.480  你跟她说 你跟她说 你跟她说 你跟她说 你跟她说 你跟她说
[0023] 00:04:17.480 --> 00:04:24.240  你跟她说 你跟她说 你跟她说 你跟她说 你跟她说
[0024] 00:04:24.240 --> 00:04:27.320  你跟她说 你跟她说
```

- `天c游xi...mkv` 04:22 左右仍有短语级重复：

```text
你这些人 你这些人 你这些人 你这些人
```

- `天c游xi...mkv` 仍有单字重复 run，例如连续 `你`。这类文本被后处理合并后更容易被看到，但目前还没有被主动删除。

## 结论

本次全量重建完成了 ASR 索引格式和后处理策略统一，并验证了中文 prompt 泄漏已经消除。

但 ASR 的局部重复幻觉仍需要单独处理。下一步不应继续靠简单重跑，而应增加轻量的重复/幻觉检测层，例如：

- 保存 Whisper segment 诊断字段：`avg_logprob`、`compression_ratio`、`no_speech_prob`。
- 检测 chunk 内重复 n-gram 或连续单字循环。
- 对明显低信息重复 chunk 标记 `semantic_eligible=False`，必要时从检索候选中降权或过滤。
- 对疑似幻觉片段保留原文但在 evidence 中标注风险，避免误删真实重复台词、口号、笑声或语气词。

