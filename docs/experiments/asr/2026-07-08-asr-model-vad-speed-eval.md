# ASR 模型 / VAD / 速度分阶段实验

日期：2026-07-08

运行位置：本地 Docker `momentseek-mvp-app`，RTX 3060 Laptop GPU。未使用服务器/NPU。

评估素材：`电视剧昨夜降至04.mp4`

真值：`eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.*`，653 条字幕真值。

完整输出目录：

- `runtime-server/analysis/asr_model_vad_ab_20260708_full/summary.md`
- `runtime-server/analysis/asr_model_vad_ab_20260708_full/summary.json`
- `runtime-server/analysis/asr_model_vad_ab_20260708_full/truth_alignment.jsonl`
- 每个 run 的 `processed.txt`、`summary.json`、`unit_reports.json` 在 `runtime-server/analysis/asr_model_vad_ab_20260708_full/runs/<run>/`

## 实验命令

```powershell
docker exec momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_model_vad_ab_20260708_full --runs openai_whisper_small_90s_auto,openai_whisper_small_90s_zh,openai_whisper_small_90s_zh_silero_vad,faster_whisper_small_90s_zh,faster_whisper_small_builtin_vad_zh,funasr_sensevoice_small_silero_vad_zh,funasr_paraformer_zh_vad_punc
```

Paraformer 第一次解析有问题：FunASR 返回 `text + timestamp`，旧实验脚本把 timestamp 文本折成了整片一个超长 chunk，导致 `processed=0`。已修正实验脚本的 timestamp 切分逻辑，并重跑 Paraformer：

```powershell
docker exec momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_model_vad_ab_20260708_paraformer_fixed --runs funasr_paraformer_zh_vad_punc
```

修正版结果已合并回 full 目录；旧错误解析结果备份在：

`runtime-server/analysis/asr_model_vad_ab_20260708_full/runs/funasr_paraformer_zh_vad_punc_broken_parse/`

## 分阶段结果

| run | total_s | model_load_s | vad_detect_s | decode_s | encoder_s | decoder_s | chunks | raw_high | recall@2s | f1@2s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI Whisper small auto 90s | 482.501 | 15.284 | 0.000 | 466.949 | 11.179 | 407.447 | 236 | 20 | 0.774 | 0.260 |
| OpenAI Whisper small zh 90s | 393.112 | 13.434 | 0.000 | 379.409 | 8.298 | 332.595 | 237 | 42 | 0.780 | 0.265 |
| OpenAI Whisper small zh + Silero VAD | 241.209 | 17.067 | 25.684 | 197.778 | 6.405 | 172.467 | 226 | 17 | 0.774 | 0.270 |
| faster-whisper small zh 90s | 120.462 | 14.607 | 0.000 | 102.981 | 0.000 | 0.000 | 240 | 20 | 0.786 | 0.268 |
| faster-whisper small zh builtin VAD | 65.465 | 3.308 | 0.000 | 61.953 | 0.000 | 0.000 | 202 | 5 | 0.779 | 0.285 |
| SenseVoiceSmall zh + Silero VAD | 76.886 | 8.631 | 33.619 | 22.284 | 0.000 | 0.000 | 94 | 2 | 0.780 | 0.177 |
| Paraformer zh + VAD + punc | 95.746 | 36.403 | 0.000 | 49.688 | 0.000 | 0.000 | 187 | 9 | 0.838 | 0.236 |

说明：

- OpenAI Whisper 的 `encoder_s` / `decoder_s` 是通过包装 `model.encoder.forward` 和 `model.decoder.forward` 得到的近似 GPU forward 时间。
- faster-whisper / FunASR 当前只记录总 decode 时间；底层没有在脚本里拆 encoder/decoder。
- `model_load_s` 是热缓存后的结果。冷启动下载成本见下节。

## 冷启动成本

首次拉模型会显著影响总耗时：

- faster-whisper small 首次 60 秒 smoke：`model_load_seconds=250.485`，主要是 HuggingFace 模型下载；设置 `HF_ENDPOINT=https://hf-mirror.com` 后可下载成功。
- SenseVoiceSmall 首次 60 秒 smoke：`model_load_seconds=314.523`，主要是 ModelScope 模型下载。
- Paraformer + punc 首次 full run：旧错误解析 run 中 `model_load_seconds=643.729`，包含 Paraformer、FSMN-VAD、CT-punc 大模型下载；热缓存后修正版 `model_load_seconds=36.403`。

部署说明里应区分 cold cache 和 warm cache；真实索引吞吐更接近 warm-cache 的 `decode_s + postprocess_s`。

## 结论

当前最适合继续验证的默认候选：

- 速度/质量综合：`faster_whisper_small_builtin_vad_zh`。65.5s 完成约 33.3 分钟音频，raw_high 最低，F1 最高。
- 召回最高：`funasr_paraformer_zh_vad_punc`。recall@2s 最高，但模型体积大，时间戳切分仍需进一步校准。
- 如果继续用 OpenAI Whisper：`openai_whisper_small_90s_zh_silero_vad` 比固定 90s zh 快约 39%，recall 基本持平，raw_high 更低。

不建议作为当前默认：

- `asr_language=auto` 用于中文剧集会有窗口误判 English 的风险，且总耗时最高。
- `SenseVoiceSmall + Silero VAD` 很快，但缺少细粒度时间戳；当前只能按外部 VAD group 给近似时间，chunk 数少、precision/F1 偏低。

## 后续事项

- 把 faster-whisper small builtin VAD 和 Paraformer + punc 放进小规模人工听查，对比错词、漏句、时间戳偏移。
- Paraformer timestamp 文本切分需要继续优化，当前存在个别 chunk 时间跨度偏长、开头残留 `e/the` 之类噪声的问题。
- 如果生产索引引入 faster-whisper 或 FunASR，需要把冷启动模型下载、模型缓存目录、GPU/CPU fallback 写进部署说明。
