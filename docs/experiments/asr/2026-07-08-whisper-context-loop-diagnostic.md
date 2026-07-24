# Whisper 长上下文重复幻觉诊断

日期：2026-07-08

## 目的

定位 `电视剧昨夜降至04.mp4` 在 04:03-04:27 左右出现连续重复文本的问题，判断它是否由 `task="transcribe"`、`language`、`condition_on_previous_text` 或长音频上下文触发。

本次只做诊断实验，不修改正式 ASR 索引。

## 输入

- 视频：`电视剧昨夜降至04.mp4`
- video_id：`a293b5981126444182208da7ba6274f5`
- 原始视频：`runtime-server/uploads/a293b5981126444182208da7ba6274f5.mp4`
- 模型：OpenAI Whisper `small`
- 环境：本地 Docker CUDA 后端 `momentseek-mvp-app`
- Whisper 版本：`20250625`
- 目标坏窗口：`230s-280s`，即约 `03:50-04:40`

输出目录：

```text
runtime-server/analysis/asr_hallucination_diag_20260708/
```

关键输出：

```text
summary.md
all_results.json
clip_timestamp_probe.md
*.json
*.silencedetect.txt
```

## 实验设计

裁剪两种输入：

| window | 范围 | 目的 |
|---|---:|---|
| `local_0350_0440` | `230s-280s` | 只看坏片段本身是否会触发重复 |
| `context_0000_0500` | `0s-300s` | 保留前文上下文，观察连续解码是否触发重复 |

参数组：

| config | 说明 |
|---|---|
| `auto_prev_default` | 当前默认方向：auto language + previous text |
| `auto_no_prev` | auto language + 关闭 previous text |
| `zh_prev_default` | 显式中文 + previous text |
| `zh_no_prev` | 显式中文 + 关闭 previous text |
| `auto_prev_word_hal2` | auto + previous text + `word_timestamps=True` + `hallucination_silence_threshold=2` |
| `auto_prev_temp0` | auto + previous text + 固定 `temperature=0` |

补充实验：

在同一个 `context_0000_0500.wav` 上使用 Whisper `clip_timestamps` 只解局部区间：

- `clip_timestamps="230,280"`
- `clip_timestamps="200,280"`

## 结果摘要

### 1. 单独 50 秒局部窗口没有复现重复循环

`local_0350_0440` 的 6 组参数均未出现连续 `你跟她说`、`你还说` 或其他明显 loop。

典型输出接近：

```text
你女朋友已经醒了
她说你们是停车之后吵架
然后才喝的酒
你刚刚一直在昏迷
我同事已经联系下属了
下次开车优着点
这大晚上这多危险啊
出了什么事怎么办啊
下次注意啊
在这顶上签个字
走吧
```

这说明坏片段本身不是完全不可识别；单独短窗口转写可以得到相对正常的文本。

### 2. 0-5 分钟连续解码会复现不同形式的重复幻觉

`context_0000_0500` 在多个参数组中复现重复，但重复词并不固定：

| config | 现象 |
|---|---|
| `auto_prev_default` | `你还不说` 连续重复约 18 次，`compression_ratio=2.378`，接近默认失败阈值 `2.4` |
| `auto_no_prev` | `你女朋友已经醒了` 被重复，说明全局关闭 previous text 不能根治 |
| `zh_prev_default` | 没有明显 loop，但出现 `237.52-263.70` 的长段 `她说`，内容丢失严重 |
| `zh_no_prev` | `跟着我/跟着你` 循环重复，`compression_ratio=2.26` |
| `auto_prev_word_hal2` | 文本 loop 基本消失，但出现多个空文本、零时长 segment，需要清洗 |
| `auto_prev_temp0` | 固定温度仍出现 `我同意你` 连续重复，不能作为单独修复 |

结论：重复幻觉不是某个中文短语的固定问题，而是 Whisper 连续解码在该窗口附近进入了不同形式的 failure loop。

### 3. `clip_timestamps` 局部解码可以避免这次 loop

在同一个 `context_0000_0500.wav` 上只解 `230s-280s` 或 `200s-280s`，输出没有重复循环，结果接近单独 50 秒裁剪。

这进一步说明问题主要来自连续长上下文解码状态，而不是音频文件、`task="transcribe"` 或某个固定 `language` 参数本身。

### 4. 静音/低信息间隔是诱因之一，但不是唯一判据

`silencedetect` 显示目标区域附近存在大量静音或低能量间隔：

```text
230.56-231.63
231.81-233.81
233.98-235.99
236.18-238.17
244.81-245.61
249.66-250.17
...
```

但 Whisper 输出中的 `no_speech_prob` 往往很低，因此不能只靠 `no_speech_prob` 过滤。`compression_ratio`、重复 n-gram、时间戳异常、空文本/零时长段需要一起判断。

## 判断

`task="transcribe"` 不是本问题主因。Whisper 默认任务本来就是 `transcribe`；本次实验中，同样使用 `task="transcribe"`，局部窗口和 `clip_timestamps` 均可得到正常输出。

`condition_on_previous_text=False` 也不是银弹。它能改变重复形态，但在 `context_0000_0500` 中仍会产生新的重复循环。

当前更可信的根因是：

1. Whisper 对整段较长音频连续解码时，在静音/低信息/时间戳边界附近进入 failure loop。
2. previous text、temperature fallback 和时间戳预测会改变 loop 的文本形态。
3. 当前索引流程没有保存 Whisper 原始诊断字段，也没有在 semantic embedding 前过滤明显 loop，因此问题会进入检索索引。

## 下一步建议

优先做 ASR 解码层重构：

1. 将整段音频一次性 Whisper 转写改为短窗口解码，例如 `30-60s` 窗口，带 `2-5s` overlap。
2. 使用独立短窗口或 `clip_timestamps`，避免前文状态无限传递。
3. 合并 overlap 时做去重和时间戳归一化。
4. 保存 raw segment 诊断字段：`avg_logprob`、`compression_ratio`、`no_speech_prob`、`temperature`。
5. 在 ASR 后处理前增加轻量 hallucination guard：
   - 重复 n-gram 或短语循环；
   - 空文本/零时长 segment；
   - 高 `compression_ratio` 或低 `avg_logprob`；
   - 异常长时间戳跨度但文本极短。
6. 对高风险 chunk 保留可查看文本，但默认不生成 semantic embedding 或在检索中降权。

备选增强：

- `word_timestamps=True + hallucination_silence_threshold` 对本例有效，但会产生空文本/零时长段，并可能增加耗时；适合作为后续 A/B，而不是第一优先级。
- 固定 `temperature=0` 不能解决本例，不建议作为单独修复。
