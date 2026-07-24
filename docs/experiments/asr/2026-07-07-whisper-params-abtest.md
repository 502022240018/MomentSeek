# Whisper 参数 A/B 实验，2026-07-07

## 目的

验证后来 ASR-only 补跑的两条中文视频中出现的 prompt 泄漏、重复幻觉和局部时间漂移，是否主要由 `initial_prompt` 或 `condition_on_previous_text` 引起。

本实验只跑 10 分钟窗口，不覆盖现有索引。

## 输入

从本地 Docker CUDA 后端 `momentseek-mvp-app` 里切出 10 分钟 WAV：

- `runtime-server/analysis/asr_abtest_20260707/book_0000_1000.wav`
- `runtime-server/analysis/asr_abtest_20260707/drama_0000_1000.wav`

视频：

- `书籍纪录片.mp4`
- `天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv`

模型：

- Whisper `small`
- device: CUDA, `NVIDIA GeForce RTX 3060 Laptop GPU`

## 参数组

| 组 | initial_prompt | condition_on_previous_text | 说明 |
|---|---|---|---|
| A `old_prompt_prev` | `以下是普通话简体中文转写，请输出简体中文。` | 默认 `True` | 复现后补 ASR 的可疑路径 |
| B `new_no_prompt_no_prev` | 无 | `False` | 去 prompt，并禁止前文续写 |
| C `no_prompt_prev` | 无 | `True` | 只去 prompt，保留 Whisper 默认前文上下文 |

## 输出文件

目录：

- `runtime-server/analysis/asr_abtest_20260707/`

结果 JSON：

- `book_0000_1000_old_prompt_prev.json`
- `book_0000_1000_new_no_prompt_no_prev.json`
- `book_0000_1000_no_prompt_prev.json`
- `drama_0000_1000_old_prompt_prev.json`
- `drama_0000_1000_new_no_prompt_no_prev.json`
- `drama_0000_1000_no_prompt_prev.json`

每个 JSON 保存：

- `segments[].text`
- `segments[].start/end`
- `segments[].avg_logprob`
- `segments[].compression_ratio`
- `segments[].no_speech_prob`
- `elapsed_seconds`

## 汇总

### 书籍纪录片，00:00-10:00

| 组 | segments | elapsed_seconds | prompt leak | repeat runs >= 3 |
|---|---:|---:|---:|---:|
| A `old_prompt_prev` | 194 | 68.57 | 2 | 0 |
| B `new_no_prompt_no_prev` | 183 | 104.02 | 0 | 0 |
| C `no_prompt_prev` | 207 | 87.75 | 0 | 0 |

A 组开头出现 prompt 泄漏：

```text
[0.00-2.00] 请输出简体中文转写，
[30.00-32.00] 请输出简体中文转写，
```

B/C 组均未出现 `普通话简体中文转写` 或 `请输出简体中文转写`。

### 天c游xi，00:00-10:00

| 组 | segments | elapsed_seconds | prompt leak | repeat runs >= 3 |
|---|---:|---:|---:|---:|
| A `old_prompt_prev` | 124 | 83.21 | 3 | 1 |
| B `new_no_prompt_no_prev` | 172 | 94.93 | 0 | 2 |
| C `no_prompt_prev` | 169 | 99.15 | 0 | 2 |

A 组 prompt 泄漏形成连续重复：

```text
run count=3, text=请输出简体中文转写，, 30.0-100.0s
```

08:10-09:05 关键窗口：

- A 组没有复现现有索引里的 `我去找他` 连续重复，但存在 prompt 泄漏。
- B 组去 prompt 且关闭 previous text 后，在 490-520s 出现明显低置信异常片段，例如 `我从小想憋到 user`、`我明年一大 länger`，对应 `avg_logprob=-4.086`。
- C 组去 prompt 但保留 previous text，关键窗口最稳定：

```text
[518.86-519.86] 你好啊
[520.86-521.86] 刘全伦
[523.86-524.86] 你是陈伦
[527.86-528.86] 我知道你想要他
[528.86-530.86] 就帮你捡回来了
[531.86-533.86] 是你自己卡掉了
[533.86-534.86] 我不是故意帮我冲你
```

## 结论

1. `initial_prompt` 是 prompt 泄漏的直接原因。A 组在两条视频里都出现了 `请输出简体中文转写` 类文本；B/C 去掉 prompt 后消失。
2. `condition_on_previous_text=False` 不一定更好。在剧集关键窗口，B 组产生低置信中英混杂乱码，说明直接全局关闭 previous context 可能损害片段稳定性。
3. 当前小窗口实验里，推荐优先改为 C 组策略：删除 `initial_prompt`，暂时保留 `condition_on_previous_text=True`。
4. 现有完整索引里的 `我去找他` 连续重复没有在本次 10 分钟 A 组稳定复现，可能与完整长音频上下文、历史运行环境、随机性或当时具体代码路径有关。因此不能把它完全归因于 `condition_on_previous_text=True`。
5. 后续仍需要保存 Whisper 诊断字段，并基于 `avg_logprob/no_speech_prob/compression_ratio` 与连续重复规则做轻量过滤。

## 建议

近期代码改动：

```text
- 删除中文 initial_prompt。
- 保留 language="zh"。
- 暂不全局设置 condition_on_previous_text=False。
- 保存 Whisper segment 诊断字段，为后处理过滤做准备。
```

验证顺序：

1. 修改代码后，只重跑 `书籍纪录片.mp4` ASR，确认 prompt 泄漏消失。
2. 重跑 `天c游xi...mkv` ASR，检查 `我去找他`、`赵正宵` 等连续重复是否还存在。
3. 如仍存在重复幻觉，再加连续重复过滤和低置信过滤。
