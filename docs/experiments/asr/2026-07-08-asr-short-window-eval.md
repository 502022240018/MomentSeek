# ASR 短窗口转写实验

日期：2026-07-08

## 目的

验证“短窗口 Whisper 解码 + overlap 合并 + 连续重复 guard”是否能缓解 `电视剧昨夜降至04.mp4` 的 ASR 重复幻觉问题。

本次是实验验证，不覆盖正式索引，不删除旧 ASR 文本。

## 输入与输出

- 视频：`电视剧昨夜降至04.mp4`
- video_id：`a293b5981126444182208da7ba6274f5`
- 模型：Whisper `small`
- 环境：本地 Docker CUDA 后端 `momentseek-mvp-app`
- 参数：`window_seconds=60`，`overlap_seconds=5`，`language=auto`

旧结果保留位置：

```text
runtime-server/indexes/a293b5981126444182208da7ba6274f5/asr.npz
runtime-server/analysis/asr_rebuild_backup_20260707_all/a293b5981126444182208da7ba6274f5/asr.npz
runtime-server/analysis/asr_full_texts_after_rebuild_20260707/电视剧昨夜降至04.mp4__a293b5981126444182208da7ba6274f5.txt
```

新实验输出位置：

```text
runtime-server/analysis/asr_short_window_eval_20260708/
```

关键文件：

```text
short_window_raw.json
short_window_processed.json
short_window_text.txt
short_window_comparison.md
window_reports.json
```

## 实验流程

1. 从原视频抽取 16k mono WAV。
2. 按 60 秒窗口、5 秒 overlap 切分。
3. 每个窗口独立调用 Whisper，窗口内保留 `condition_on_previous_text=True`，窗口之间不传递 previous text。
4. 只保留每个窗口 core 区间的 segment，避免 overlap 重复。
5. 记录 raw segment 的 `avg_logprob`、`compression_ratio`、`no_speech_prob`、`temperature`。
6. 在 postprocess 前标记连续相同短句重复，并将高风险 raw segment 排除出 processed chunks。
7. 使用现有 `bucket_bonus` ASR 后处理策略合并短 chunk。
8. 生成旧索引、当前索引、新短窗口结果的对比报告。

## 结果摘要

完整视频共 37 个短窗口：

```text
short_window_raw=684
short_window_processed=246
elapsed_seconds=368.999
high_risk_raw_segments=39
```

对比指标：

| source | chunks | chars | max_3gram | max_4gram | known phrases |
|---|---:|---:|---|---|---|
| `backup_before_rebuild` | 681 | 4007 | `你你你 x20` | `你你你你 x18` | `{"你还说": 9}` |
| `current_full_whisper` | 270 | 4029 | `她说你 x17` | `你跟她说 x17` | `{"你跟她说": 17, "你还不说": 1}` |
| `short_window_raw` | 684 | 4194 | `我老婆 x11` | `我老婆林 x11` | `{}` |
| `short_window_processed` | 246 | 3992 | `来来来 x6` | `怎么回事 x5` | `{}` |

目标窗口 `03:50-04:40`：

- 重建前备份索引存在连续 `你还说/你还说我还说`。
- 当前完整 Whisper 索引存在连续 `你跟她说`。
- 短窗口结果不再出现上述重复幻觉。
- 短窗口结果时间戳整体落在 `04:10` 左右，更接近此前人工听查发现的“旧时间戳偏早十几秒”的现象。

短窗口后处理结果示例：

```text
[0017] 00:04:10.400 --> 00:04:16.680  你女朋友已经醒了 她说 你们是停车之后吵架 然后才喝的酒
[0018] 00:04:18.560 --> 00:04:25.760  你刚刚一直在昏迷 我同事已经联系下属了 下次开车优势点
[0019] 00:04:25.920 --> 00:04:33.000  这大晚上这多危险啊 出了什么事怎么办啊 下次注意啊 在这顶上签个字
[0020] 00:04:33.000 --> 00:04:38.560  走吧 你别这么赶啊 在这儿顶上签个字 走吧 你干吗呢
[0021] 00:04:38.560 --> 00:04:45.680  我就给我几个情意的喝酒呢 对不对 走 没喝多少酒呢 就来 一小口
```

## 额外发现

短窗口本身能显著减少长上下文 failure loop，但仍可能在个别窗口中出现短句连续重复。增强 guard 后，以下类型被识别并从 processed chunks 中排除：

```text
《一切不成人》连续重复
我老婆林彪连续重复
我还不认定连续重复
磕了一个连续重复
```

仍有少量类似 `来 来 来` 的口语重复保留在文本中。正式接入时不建议直接删除这类文本；更稳妥的做法是保留原文展示，但对明显低信息/重复 chunk 设置 `semantic_eligible=false` 或检索降权。

## 结论

短窗口方案有效缓解了本例中最严重的 ASR 重复幻觉，且不会依赖 `task`、全局关闭 `condition_on_previous_text` 或固定 `temperature=0` 这类不稳定参数。

推荐作为下一步正式 ASR 索引改造方向：

1. 将 Whisper 整段转写改为短窗口解码。
2. 保留 overlap core 合并逻辑。
3. 保存 Whisper raw diagnostics。
4. 在 ASR 后处理前增加连续重复 guard。
5. 对高风险文本保留可查看原文，但默认不生成 semantic embedding。

正式化前还需要在 `天c游xi...mkv` 和 `书籍纪录片.mp4` 上重复同样实验，确认中文剧集和纪录片都能受益。
