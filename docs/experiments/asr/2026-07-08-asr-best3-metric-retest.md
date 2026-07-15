# ASR 三模型复测与指标校准

日期：2026-07-08

## 实验目的

复测以下三条 ASR 方案在 `电视剧昨夜降至04.mp4` 上的速度和文本质量：

- `faster-whisper small + builtin VAD`
- `faster-whisper turbo + builtin VAD`
- `SenseVoiceSmall timestamp + Silero VAD`

同时核查一个重要问题：现有 `recall@2s / f1@2s` 是否能代表人工肉眼看到的 ASR 文本准确率。

## 环境

- 本地 Docker：`momentseek-mvp-app`
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- `faster-whisper 1.2.1`
- `ctranslate2 4.8.1`
- `funasr 1.3.14`
- `torch 2.6.0+cu124`

未使用服务器/NPU。

## 输入与真值

- 视频：`电视剧昨夜降至04.mp4`
- `video_id`: `a293b5981126444182208da7ba6274f5`
- 音频长度：约 1998s
- 真值：内嵌中文字幕导出的 `653` 条 subtitle truth

真值路径：

```text
eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.jsonl
runtime-server/analysis/asr_truth_alignment_20260708/yesterday_ep04_truth_vs_60s_90s_120s.jsonl
```

## 执行记录

第一次三模型同跑时，`faster-whisper` 两组失败，原因是虽然模型已经有本地缓存，但 `WhisperModel("small")` / `WhisperModel("turbo")` 仍尝试访问 Hugging Face 查询 revision，当前网络出现 SSL EOF。

后续使用离线缓存重跑 faster-whisper：

```bash
docker exec -e HF_HUB_OFFLINE=1 momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_best3_retest_fw_offline_20260708 --runs faster_whisper_small_builtin_vad_zh,faster_whisper_turbo_builtin_vad_zh
```

SenseVoice 本次结果来自：

```bash
docker exec momentseek-mvp-app python /app/runtime/analysis/asr_model_vad_ab_20260708_run.py --output-dir /app/runtime/analysis/asr_best3_retest_20260708 --runs faster_whisper_small_builtin_vad_zh,faster_whisper_turbo_builtin_vad_zh,funasr_sensevoice_small_ts_silero_vad_zh
```

其中 faster-whisper 两组在该命令中失败，SenseVoice 成功。

## 原始指标

现有 `recall@2s / f1@2s` 的计算方式：

- 对每一条很短的 subtitle truth；
- 找到与该字幕时间范围 `+/-2s` 相交的 ASR chunk；
- 拼接候选文本；
- 用 compact 后的字符 LCS 计算 recall、precision、F1。

这不是纯文字准确率指标。它会同时惩罚：

- 时间戳漂移；
- chunk 过长导致候选文本混入额外句子；
- ASR 切分方式与 subtitle 切分方式不同；
- 同义但不同字的识别结果。

| model | total_s | model_load_s | vad_detect_s | decode_s | chunks | raw_high | recall@2s | precision@2s | f1@2s | hitR60@2s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper small + builtin VAD | 75.646 | 3.435 | 0.000 | 71.149 | 202 | 5 | 0.779 | 0.197 | 0.285 | 0.784 |
| faster-whisper turbo + builtin VAD | 89.187 | 15.747 | 0.000 | 73.079 | 229 | 9 | 0.801 | 0.189 | 0.284 | 0.802 |
| SenseVoiceSmall timestamp + Silero VAD | 41.558 | 4.975 | 11.540 | 17.929 | 160 | 3 | 0.760 | 0.163 | 0.247 | 0.755 |

## 补充文本指标

为避免只看逐字幕 `+/-2s` 指标造成误判，补充两类复算：

- `global`：忽略时间戳，把全片字幕文本和全片 ASR 文本拼起来，计算 CER / LCS recall / precision / F1。
- `window30`：按 30 秒窗口聚合 truth 和 ASR 文本，衡量文本是否出现在大致正确的时间区域。

| model | global CER lower better | global recall | global precision | global F1 | window30 midpoint CER lower better | window30 midpoint F1 | window30 overlap CER lower better | window30 overlap F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper small + builtin VAD | 0.218 | 0.794 | 0.871 | 0.831 | 0.273 | 0.769 | 0.352 | 0.750 |
| faster-whisper turbo + builtin VAD | 0.253 | 0.823 | 0.844 | 0.833 | 0.285 | 0.758 | 0.346 | 0.755 |
| SenseVoiceSmall timestamp + Silero VAD | 0.237 | 0.803 | 0.879 | 0.839 | 0.341 | 0.737 | 0.416 | 0.728 |

## 解读

用户肉眼认为 SenseVoice 文本比 faster-whisper small 更自然，这个观察是合理的。补充指标显示：

- SenseVoice 的 `global F1=0.839` 最高，说明全片文本覆盖和精度综合并不差；
- SenseVoice 的 `global precision=0.879` 最高，文本冗余相对少；
- faster-whisper small 的 `global CER=0.218` 最低，逐字编辑距离更好；
- SenseVoice 的 `window30` 指标偏低，主要问题不是纯文本识别，而是 timestamp / chunk 切分和长 chunk 对齐；
- 原来的 `recall@2s/f1@2s` 不能单独作为 ASR 模型优劣判断依据。

因此，这不是“召回指标完全错了”，而是指标目标不同：它更接近“检索时能否在精确时间附近命中短字幕”，不等同于“全文 ASR 是否读起来准确”。

## 结论

当前更合理的判断方式应当是多指标组合：

- 文字底稿质量：`global CER`、`global F1`、人工抽查；
- 检索时间定位：`window30`、subtitle-local `recall@2s/f1@2s`；
- 稳定性：幻觉/重复/英文泄漏；
- 性能：`total_s`、`decode_s`、VAD 时间；
- 实际搜索：人工 query 检索命中率。

在这条视频上：

- `SenseVoiceSmall timestamp + Silero VAD` 速度最快，文本可读性很有竞争力，但时间戳和切分需要继续打磨。
- `faster-whisper turbo + builtin VAD` 检索召回略高，但速度慢于 SenseVoice，且部分指标不稳定。
- `faster-whisper small + builtin VAD` 仍然是较稳的基线，CER 最低，但肉眼可读性未必最好。

## 后续建议

1. 不再只用 `recall@2s/f1@2s` 判断 ASR 方案。
2. ASR eval 固定输出四类指标：全文文本质量、窗口级文本质量、字幕局部命中、速度/稳定性。
3. SenseVoice 下一步优先优化 timestamp 对齐和 chunk 切分，而不是先否定模型。
4. 产品默认 ASR 方案采用 `SenseVoiceSmall + FunASR`，`faster-whisper turbo` 作为多语言/高效果备选；两条路径都必须优先使用本地模型目录或本地 snapshot，避免已缓存模型仍因网络 revision 查询失败。

## 原始输出

```text
runtime-server/analysis/asr_best3_retest_fw_offline_20260708/
runtime-server/analysis/asr_best3_retest_20260708/
runtime-server/analysis/asr_best3_metric_review_20260708/
```
