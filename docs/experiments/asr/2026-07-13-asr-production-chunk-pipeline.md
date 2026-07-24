# ASR 正式 chunk pipeline 与全量重建

日期：2026-07-13

## 目标

把前期 SenseVoice / faster-whisper 边界实验收敛到正式 `build_asr_index`，重点验证：句子完整性、时间轴可靠性、embedding 适配性、多语言路由和全量运行稳定性。

## 最终方案

```text
audio_extract
-> auto language probe
   short video: one window
   long video: 3 x 20s position voting
-> zh/yue/cmn: SenseVoiceSmall + Silero 12s groups
   source text is authoritative; timestamps select boundaries only
-> other languages: faster-whisper turbo
   24s contiguous original-audio windows
   suspicious window only -> local builtin-VAD fallback
-> unit-aware retrieval chunk builder
   no cross-unit merge; merged duration <= 12s; no blind hard split
-> semantic eligibility + quality_flags
-> MiniLM 384-d float16 embedding
-> schema v3 asr.npz
```

Semantic eligibility 只拒绝整段纯语气/连接词、少于 2 个有效字符的片段，以及 `<500ms` 内出现明显不可能语音单位数的片段。英文按单词而不是字母计数；正式回归发现 `Your head!` 只有 460ms 但完全合理，因此修正了第一版字符计数误伤。

## 正式路径回归

9 份片段覆盖中文综艺、中文剧集、纪录片、方言、英语广告/剧集和西语广告。结果目录：

```text
runtime-server/analysis/asr_formal_regression_20260713_rerun1/
runtime-server/analysis/asr_formal_regression_20260713_worldcup_eligibility_fix/
runtime-server/analysis/asr_formal_regression_20260713_final/
```

结果：

- 9/9 正式入口成功生成 NPZ 和 384-d semantic embedding。
- 抽样最长 retrieval chunk 为 11.76s，没有 `>12s` 项。
- 修正 eligibility 后，世界杯片段 `Your head!` 正常生成 embedding；`I mean...`、`No?` 仍按低信息拒绝。
- 生产 embedding 检索 18 条 query：目标 chunk Top-1/3/5 为 `16/18/18`，目标视频 Top-1/5 为 `17/18`。
- 两条 Top-1 失败均在 Top-2，后续归入 ASR semantic 排序/阈值问题，不继续修改 chunk 边界。

人工听查包：

```text
runtime-server/analysis/asr_formal_review_eval_20260713/review.html
runtime-server/analysis/asr_formal_review_eval_20260713/review.jsonl
runtime-server/analysis/asr_formal_review_eval_20260713/REPORT.md
runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/review.html
runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/review.jsonl
runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/REPORT.md
```

## 全量重建

本地 catalog 共 9 个视频，全部使用 `asr_engine=auto / asr_model=turbo / asr_language=auto` 串行重建。重建过程中发现书籍纪录片只探测片头会误判为英文；停止自动后续提交，改为长视频 3 位置投票后，真实 probe 得到 `zh / 0.877`，重新索引为 SenseVoice。

最终路由：

- `funasr / SenseVoiceSmall / zh`：五哈美食、昨夜降至、书籍纪录片、五哈长综艺、天才游戏剧集。
- `funasr / SenseVoiceSmall / zh`：给阿嬷的情书方言预告片。
- `faster-whisper / turbo / en`：世界杯广告、世界杯比赛集锦。
- `faster-whisper / turbo / es`：球星牛奶广告。

最终验证：

- `/api/health = ok`
- 9/9 视频 `ready`
- 9/9 包含 ASR v3
- active jobs = 0
- 9/9 NPZ 的 `chunk_times_ms/texts/embeddings/embedding_chunk_indices` 可读
- 6 条平台 ASR API 中/英/西语 smoke query 的目标视频均为 Top-1
- 后端测试 `146 passed`；另有 6 个只覆盖已退出正式路径的 `asr_postprocess`/旧报告测试随死代码一起删除。

最终全量重建记录：

```text
runtime-server/analysis/asr_reindex_final_20260713.json
```

最终 9 段回归为 9/9 semantic complete，共 291 个 embedding chunk，最长 11.76s，没有 `>12s` 项。按最终 chunk 编号同步人工 query 标签后，18 条 query 的目标 chunk Top-1/3/5 仍为 `16/18/18`，目标视频 Top-1/5 为 `17/18`。

完整视频里保留 4 个 `>12s` 模型原始完整句，最长 19.56s。局部 fallback 没有找到更可靠边界；按照“不为满足长度而切断完整句”的结论，保留并标记 long。

## 后续

1. 从 `review.html` 完成人工听查，特别关注 non-terminal boundary、长原始句和方言错词。
2. 在固定 query 集上调 ASR semantic 阈值与 lexical 融合，解决低分区跨视频近分误排。
3. 混合语言视频先继续按主语言路由；只有评估证明必要时再引入分段语言路由。
