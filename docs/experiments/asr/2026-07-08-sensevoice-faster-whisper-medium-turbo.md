# ASR 新版 SenseVoice 与 faster-whisper medium/turbo 对比

日期：2026-07-08

环境：本地 Docker `momentseek-mvp-app`，NVIDIA RTX 3060 Laptop GPU 6GB。未使用服务器 NPU。

评估素材：`电视剧昨夜降至04.mp4`，`video_id=a293b5981126444182208da7ba6274f5`，音频 1998.368 秒。

真值：`eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.*`，653 条字幕。

## 目的

1. 验证 `funasr 1.3.14` 下 SenseVoiceSmall 是否能返回 timestamp、音效/情绪标签、说话人/音纹信息。
2. 对比 `faster-whisper small / medium / turbo` 的速度、检索评估指标和幻觉风险。
3. 判断下一阶段 ASR 主索引模型和可选辅助通道怎么选。

## 版本与模型缓存

容器内版本：

```text
funasr 1.3.14
faster-whisper 1.2.1
torch 2.6.0
torchaudio 2.6.0+cu124
modelscope 1.38.1
onnxruntime 1.27.0
```

`faster-whisper medium` 和 `turbo` 首次下载很慢，不能把 cold download 算进日常索引速度：

| 模型 | 首次下载+加载 smoke | 180s 解码 | 主权重大小 |
|---|---:|---:|---:|
| `medium` | 946.647s | 4.968s | 1,527,906,378 bytes |
| `turbo` | 702.544s | 2.601s | 1,617,884,929 bytes |

下载后 `/app/models/faster-whisper` 总缓存约 3.4G。

## 命令

faster-whisper 与 SenseVoice 基础对比：

```powershell
docker exec -e HF_ENDPOINT=https://hf-mirror.com momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_model_vad_extension_20260708_full --runs faster_whisper_small_builtin_vad_zh,faster_whisper_medium_builtin_vad_zh,faster_whisper_turbo_builtin_vad_zh,funasr_sensevoice_small_silero_vad_zh,funasr_sensevoice_small_zh_vad
```

SenseVoice timestamp 版：

```powershell
docker exec momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_sensevoice_timestamp_20260708_full --runs funasr_sensevoice_small_ts_zh_vad,funasr_sensevoice_small_ts_silero_vad_zh
```

## 结果

| run | total_s | model_load_s | vad_detect_s | decode_s | chunks | raw_high | repeated | recall@2s | f1@2s | hitR60@2s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `faster_whisper_small_builtin_vad_zh` | 94.201 | 7.368 | 0.000 | 84.226 | 202 | 5 | 0 | 0.779 | 0.285 | 0.784 |
| `faster_whisper_medium_builtin_vad_zh` | 153.032 | 15.713 | 0.000 | 137.004 | 234 | 11 | 0 | 0.634 | 0.208 | 0.669 |
| `faster_whisper_turbo_builtin_vad_zh` | 105.957 | 16.378 | 0.000 | 89.307 | 225 | 22 | 0 | 0.806 | 0.284 | 0.806 |
| `funasr_sensevoice_small_zh_vad` | 55.292 | 8.534 | 0.000 | 38.021 | 0 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| `funasr_sensevoice_small_silero_vad_zh` | 51.493 | 5.863 | 20.252 | 22.349 | 94 | 2 | 2 | 0.780 | 0.177 | 0.772 |
| `funasr_sensevoice_small_ts_zh_vad` | 68.034 | 6.628 | 0.000 | 54.473 | 174 | 4 | 0 | 0.196 | 0.049 | 0.078 |
| `funasr_sensevoice_small_ts_silero_vad_zh` | 64.903 | 6.717 | 19.969 | 34.823 | 160 | 3 | 1 | 0.760 | 0.247 | 0.755 |

原始输出：

```text
runtime-server/analysis/asr_model_vad_extension_20260708_full/
runtime-server/analysis/asr_sensevoice_timestamp_20260708_full/
```

每个 run 目录下有 `summary.json`、`processed.txt`、`raw.json`、`unit_reports.json`。

## SenseVoice 关键发现

- `SenseVoiceSmall` 默认不一定返回 timestamp。必须显式传入 `output_timestamp=True` 和 `return_time_stamps=True`。
- timestamp 版输出包含 `text`、`timestamp`、`words`。解析时应优先使用 `words + timestamp`，不要用带 `<|SAD|>`、`<|BGM|>`、`<|Speech|>` 标签的 `text` 做字符对齐。
- `punc_model=ct-punc` 不适合直接接在 SenseVoice rich transcription 上：smoke 中出现了标签被拆成 `< | zh | >` 的现象，并提示 punc 与 timestamp 长度不匹配。
- `SenseVoiceSmall + cam++ + spk_mode=vad_segment + return_spk_center=True` 的 180s smoke 返回了 `timestamp`、`words`、`sentence_info`、`spk_embedding_center`。这说明 FunASR 工具链可以提供说话人分段和音纹中心，但它更适合作为辅助元数据通道，而不是直接替代主 ASR 文本检索。

180s `cam++` smoke：

```text
load_s 15.579
decode_s 7.583
keys: key, text, timestamp, words, spk_embedding_center, sentence_info
timestamp_len: 251
spk_embedding_center len: 5
sentence_info_len: 9
```

## 结论

当前默认 ASR 检索主线仍建议使用 `faster-whisper small + builtin VAD`：幻觉风险低，`f1@2s` 最高，速度可接受。

`faster-whisper turbo` 是最值得继续测试的高召回候选：`recall@2s=0.806` 最高，速度接近 small，但 `raw_high=22` 高于 small，需要在更多视频上确认是否有额外幻觉风险。

`faster-whisper medium` 在这条中文剧集上不推荐：更慢，召回和 F1 都明显低于 small/turbo。

SenseVoice 不建议现在作为主 ASR 检索模型，但建议保留为辅助通道候选：

- 用 `output_timestamp=True` 获取 `words/timestamp`。
- 用 rich tags 做情绪/音效元数据，例如 BGM、Speech、SAD。
- 用 `cam++` 做说话人分段和音纹中心。
- 先不要接 `ct-punc` 直接处理 SenseVoice rich text。

## 后续建议

1. 在更多视频上追加 `faster-whisper small` vs `turbo` 对比，重点人工检查 `raw_high` 对应片段是否真实幻觉。
2. 将 SenseVoice 作为辅助实验通道设计，不要和主 ASR 语义检索混在同一份索引字段里。
3. 如果要引入音效/情绪/音纹，先定义独立 metadata schema：`audio_events`、`emotion_tags`、`speaker_segments`、`speaker_embedding_centers`。
4. 后续可测试 `Fun-ASR-Nano + cam++`，但它依赖更重，应该单独评估，不和 SenseVoiceSmall 混为一谈。
