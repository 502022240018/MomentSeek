# 实验总结

本目录保存人类可读的实验结论、指标摘要和建议。

目录分工：

```text
docs/experiments/ -> 实验结论、推荐方案、摘要指标
eval/             -> 可复现实验资产、manifest、schema、query 文件和运行说明
```

## 命名规则

```text
docs/experiments/<area>/<YYYY-MM-DD>-<topic>.md
```

示例：

- `docs/experiments/visual/2026-07-01-clip-910b.md`
- `docs/experiments/visual/2026-07-03-siglip2-31min-index.md`

## 当前实验总结

| 日期 | 方向 | 文档 |
|---|---|---|
| 2026-07-01 | visual | `visual/2026-07-01-clip-910b.md` |
| 2026-07-03 | visual | `visual/2026-07-03-siglip2-31min-index.md` |
| 2026-07-07 | asr | `asr/2026-07-07-asr-postprocess-tuning.md` |
| 2026-07-07 | asr | `asr/2026-07-07-asr-pinyin-fallback-seed.md` |
| 2026-07-07 | asr | `asr/2026-07-07-whisper-params-abtest.md` |
| 2026-07-07 | asr | `asr/2026-07-07-asr-full-rebuild-no-prompt.md` |
| 2026-07-08 | asr | `asr/2026-07-08-whisper-context-loop-diagnostic.md` |
| 2026-07-08 | asr | `asr/2026-07-08-asr-short-window-eval.md` |
| 2026-07-08 | asr | `asr/2026-07-08-asr-short-window-extra-videos.md` |
| 2026-07-08 | asr | `asr/2026-07-08-asr-window-ab-truth-eval.md` |
| 2026-07-08 | asr | `asr/2026-07-08-asr-model-vad-speed-eval.md` |
| 2026-07-08 | asr | `asr/2026-07-08-sensevoice-faster-whisper-medium-turbo.md` |
| 2026-07-08 | asr | `asr/2026-07-08-asr-best3-metric-retest.md` |

## 每份实验总结应包含

- 实验目的。
- 实验环境。
- 数据集或输入。
- 使用的命令或脚本。
- 关键指标。
- 结论。
- 建议。
- 原始输出位置或链接。
