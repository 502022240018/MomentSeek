# ASR Dual Path Final Scheme, 2026-07-09

目的：把已讨论的两条 ASR 路线落成可维护默认方案，并验证 chunk 时长是否适合语义检索。

实验产物：

```text
runtime-server/analysis/asr_dual_path_final_scheme_20260709/
```

最终策略：

```text
SenseVoiceSmall:
  VAD = Silero external VAD
  max_group_seconds = 12
  FunASR merge_vad = false
  parser = timestamp + safe punctuation split

faster-whisper:
  VAD = builtin VAD
  min_silence_duration_ms = 500
  condition_on_previous_text = true
  parser = raw segments + safe punctuation split for long raw items

retrieval_chunk_builder:
  normal_gap_ms = 500
  short_gap_ms = 1000
  same_bucket_gap_ms = 1000
  target_max_duration_ms = 8000
  soft_max_duration_ms = 12000
  hard_max_duration_ms = 15000
```

关键结果：

| Variant | Layer | Count | p90 duration | Max duration | >30s |
|---|---:|---:|---:|---:|---:|
| `sensevoice_current_fsmn` | retrieval | 5 | 68.55s | 68.55s | 5 |
| `sensevoice_final_silero12` | retrieval | 43 | 7.80s | 14.40s | 0 |
| `faster_whisper_current` | retrieval | 19 | 32.88s | 33.98s | 4 |
| `faster_whisper_final_8_12_15` | retrieval | 35 | 14.34s | 14.92s | 0 |

结论：

- 默认 ASR 采用 `SenseVoiceSmall + Silero external VAD 12s`。
- 多语言或更强效果候选采用 `faster-whisper turbo + builtin VAD`。
- 两条路径共用同一个 retrieval chunk builder，默认窗口为 `8/12/15`。
- 旧 ASR 索引需要 ASR-only 重跑后才会应用该策略。

验证：

```text
python -m py_compile /app/runtime/analysis/asr_dual_path_final_scheme_20260709_run.py
python /app/runtime/analysis/asr_dual_path_final_scheme_20260709_run.py
```

实验脚本完成输出：`done total_s=86.60`。
