# ASR 拼音容错 seed 实验，2026-07-07

## 目的

验证中文 ASR 错词、同音字和近音字场景下，拼音 fallback 是否能补足 lexical search 的漏召回。

本实验只读运行，不改变生产检索排序。

## 输入

- 本地 runtime：`runtime-server`
- ASR 索引数量：8 个 `runtime-server/indexes/*/asr.npz`
- 错词候选导出脚本：`scripts/asr_error_candidates.py`
- 拼音 fallback 评估脚本：`scripts/asr_pinyin_fallback_eval.py`
- seed eval：`eval/asr/asr_pinyin_seed_eval_20260707.jsonl`
- 依赖：`pypinyin==0.54.0`

本轮 seed eval 已使用两个新重跑中文 ASR 素材：

- `书籍纪录片.mp4`
- `天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv`

## 候选导出

命令：

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
python scripts/asr_error_candidates.py --runtime runtime-server --out runtime/analysis --limit-per-video 250
```

输出：

- `runtime/analysis/asr_error_candidates_20260707-165005.jsonl`
- `runtime/analysis/asr_error_candidates_20260707-165005.html`

候选数量：

| 视频 | 候选数 |
|---|---:|
| `五哈团美食速度挑战纯享_31min_1080p.mp4` | 250 |
| `2025-04-20 第2期下：五哈版决战天山之巅 够癫！.mkv` | 250 |
| `天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv` | 250 |
| `电视剧昨夜降至04.mp4` | 245 |
| `书籍纪录片.mp4` | 153 |
| `给阿嬷的情书预告片` | 7 |
| `世界杯广告.mp4` | 5 |
| `球星牛奶广告` | 4 |

候选原因分布：

| 原因 | 数量 |
|---|---:|
| `short_cjk_token` | 1140 |
| `very_short_normalized` | 450 |
| `mixed_script` | 14 |
| `repeated_phrase` | 5 |
| `long_sparse_text` | 1 |

说明：候选导出是人工复查入口，不代表这些 chunk 一定是错词。

## 人工听查包

命令：

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
python scripts/asr_manual_review_pack.py `
  --candidates runtime/analysis/asr_error_candidates_20260707-170123.jsonl `
  --out runtime/analysis/asr_error_review_20260707.html `
  --upload-dir runtime-server/uploads `
  --sample-size 40 `
  --focus-video-id b09c33148400467e856802ef59e8e479 `
  --focus-video-id e96d218007fe43d4a1d39973ba55de93
```

输出：

- `runtime/analysis/asr_error_review_20260707.html`

这个 HTML 是单文件人工标注工作台，内嵌 JSON 数据，表格字段可编辑，并提供“导出 JSONL”和“复制 JSONL”按钮。标注完成后导出的 JSONL 可作为后续 pinyin fallback 正式评估输入。

抽样分布：

| 视频 | 条数 |
|---|---:|
| `天c游xi.2026.1080p[云视网yuntv.net]国语中字.mkv` | 12 |
| `书籍纪录片.mp4` | 8 |
| `五哈团美食速度挑战纯享_31min_1080p.mp4` | 6 |
| `电视剧昨夜降至04.mp4` | 4 |
| `2025-04-20 第2期下：五哈版决战天山之巅 够癫！.mkv` | 4 |
| `世界杯广告.mp4` | 3 |
| `球星牛奶广告` | 2 |
| `给阿嬷的情书预告片` | 1 |

| 原因 | 条数 |
|---|---:|
| `short_cjk_token` | 18 |
| `mixed_script` | 12 |
| `very_short_normalized` | 11 |
| `repeated_phrase` | 6 |
| `long_sparse_text` | 4 |

## Seed Eval

命令：

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
python scripts/asr_pinyin_fallback_eval.py --runtime runtime-server --eval eval/asr/asr_pinyin_seed_eval_20260707.jsonl --out runtime/analysis --top-k 20
```

输出：

- `runtime/analysis/asr_pinyin_fallback_eval.json`
- `runtime/analysis/asr_pinyin_fallback_eval.html`

结果：

| top-k | cases | lexical target hits | pinyin target hits | rescued |
|---:|---:|---:|---:|---:|
| 20 | 7 | 0 | 6 | 6 |
| 50 | 7 | 0 | 7 | 7 |

补充 top-k 50 命令：

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
python scripts/asr_pinyin_fallback_eval.py --runtime runtime-server --eval eval/asr/asr_pinyin_seed_eval_20260707.jsonl --out runtime/analysis/asr_pinyin_top50 --top-k 50
```

输出：

- `runtime/analysis/asr_pinyin_top50/asr_pinyin_fallback_eval.json`
- `runtime/analysis/asr_pinyin_top50/asr_pinyin_fallback_eval.html`

## 观察

1. 拼音 fallback 对人名、书名、短实体词有明显补召回价值。比如 `谷晓君 -> 顧小君`、`巴厘书院 -> 《八里書園》`、`留成隆 -> 刘承龙` 都能在 lexical target miss 时被找回。
2. 两字短 query 风险较高。比如 `孟天` 的拼音近似会把 `慢点`、`很香`、`每天去` 这类片段排到前面，说明不能把拼音分支作为无条件强排序信号。
3. 对“已有大量正确 lexical 命中”的 query，拼音只能补充候选，不一定能把疑似错词片段排进很前面。`陈伦 -> 你是成轮` 在 top-k 20 未命中、top-k 50 命中，就是这种情况。
4. 当前 seed eval 包含真实 ASR chunk 和同音扰动 probe，但还不是完整人工听查评估集。后续需要把 `correct_text` 和 `manual_label` 补齐后再作为正式指标。

## 人工标注结果

标注文件：

- `eval/asr/manual_review/asr_error_review_20260707.filled.jsonl`

40 条人工听查结果：

| manual_label | 条数 |
|---|---:|
| `correct` | 16 |
| `hallucination` | 11 |
| `unclear` | 8 |
| `minor_error` | 3 |
| `wrong_entity` | 1 |
| empty | 1 |

补充字段情况：

| 字段 | 非空数量 |
|---|---:|
| `correct_text` | 26 |
| `query_should_hit` | 0 |

说明：本轮人工标注补了大量 `correct_text` 和 notes，但没有填写 `query_should_hit`。因此还不能直接作为“真实用户 query -> 目标 chunk”的正式检索评估集。

## 自动代理评估

为了先观察趋势，临时生成了一个自动代理评估集：

- `eval/asr/manual_review/asr_error_review_20260707.auto_eval.jsonl`

生成规则：

- 只取 `manual_label` 属于 `minor_error`、`wrong_word`、`wrong_entity`、`hallucination`、`language_issue` 且 `correct_text` 非空的行。
- 因为 `query_should_hit` 为空，临时使用 `correct_text` 作为 query。
- 这不是最终正式评估，只用于判断 pinyin fallback 是否有继续价值。

结果：

| top-k | cases | lexical target hits | pinyin target hits | rescued |
|---:|---:|---:|---:|---:|
| 20 | 10 | 4 | 5 | 3 |
| 50 | 10 | 5 | 6 | 2 |

输出：

- `runtime/analysis/asr_manual_auto_top20/asr_pinyin_fallback_eval.json`
- `runtime/analysis/asr_manual_auto_top20/asr_pinyin_fallback_eval.html`
- `runtime/analysis/asr_manual_auto_top50/asr_pinyin_fallback_eval.json`
- `runtime/analysis/asr_manual_auto_top50/asr_pinyin_fallback_eval.html`

逐条目标排名观察：

| id | query | ASR text | lexical target rank | pinyin target rank | 观察 |
|---|---|---|---:|---:|---|
| `manual-05` | `《西汉会要》` | `一看話要` | >500 | 1 | 拼音对书名/实体错词非常有效 |
| `manual-29` | `一个写经` | `一個血精` | 36 | 1 | 拼音能把近音错字提前 |
| `manual-03` | `不要 不要 不要` | `不 不 不 不` | >500 | 14 | 重复短语能补召，但容易和其他“要不要”片段竞争 |
| `manual-10` | `天，却怎么都不亮` | `对 却怎么都不掉` | 1 | 2 | lexical 已能命中，pinyin 不一定更靠前 |
| `manual-15` | `看到啦` | `看到了吗` | 1 | 1 | 两者都能命中，pinyin 提高分数 |
| `manual-20` | `（我就和我几个）兄弟喝酒呢` | `wo乾酒呢` | 13 | >500 | 中英混杂/严重错听不是拼音能稳定解决的 |
| `manual-27` | `（爱他美冠军卓奥）爱他卓越可见` | `童谣传选` | >500 | >500 | 品牌广告类错听需要更强 ASR 或实体词保护 |
| `manual-32` | `我去兄弟` | `出 very good` | >500 | >500 | 严重跨语言错听无法靠拼音恢复 |
| `manual-40` | `你妈适配的骨髓找到了` | `严 Simpson` | >500 | >500 | 严重 hallucination/错听需要 ASR 质量或置信度过滤 |

关键结论：

- pinyin fallback 对“中文实体词/近音错字”有明确收益，最典型是 `《西汉会要》 -> 一看話要`。
- 对无声/音效 hallucination、中英混杂错听、长句严重错听，pinyin 帮助有限。
- 当前标注样本显示 ASR 质量问题不只是一类“同音错字”，还包括无台词误生成、音效误识别、外语/方言听不懂、广告品牌词错听。
- 下一版人工评估需要补 `query_should_hit`，并增加 negative controls，才能判断生产排序是否会误召。

## 五哈 5 分钟人工标注

标注输入：

- `eval/asr/manual_review/wuha_5min_annotated_20260707.txt`

结构化输出：

- `eval/asr/manual_review/wuha_5min_annotated_20260707.parsed.jsonl`
- `eval/asr/manual_review/wuha_5min_eval_20260707.jsonl`

统计：

| 项 | 数量 |
|---|---:|
| 标注 chunk | 166 |
| 可评估纠错样例 | 41 |
| `wrong_word` | 26 |
| `minor_error` | 15 |
| `hallucination` | 1 |

评估结果：

| top-k | cases | lexical target hits | pinyin target hits | rescued |
|---:|---:|---:|---:|---:|
| 20 | 41 | 25 | 29 | 5 |
| 50 | 41 | 25 | 32 | 7 |

输出：

- `runtime/analysis/wuha_5min_top20/asr_pinyin_fallback_eval.json`
- `runtime/analysis/wuha_5min_top20/asr_pinyin_fallback_eval.html`
- `runtime/analysis/wuha_5min_top50/asr_pinyin_fallback_eval.json`
- `runtime/analysis/wuha_5min_top50/asr_pinyin_fallback_eval.html`

top-20 被 pinyin 救回的样例：

| query | ASR text |
|---|---|
| `迟到了` | `知道了` |
| `勉勉` | `免面` |
| `勉勉快` | `免面快` |
| `勉勉` | `免面` |
| `勉勉` | `免面` |

top-50 额外救回：

| query | ASR text |
|---|---|
| `超哥` | `张哥` |
| `阿勒泰的` | `阿达的` |

仍未命中的样例多是语义差距较大或极短词，例如：

- `志胜 -> 这事`
- `往起起啊 -> 王勉起`
- `你倒是起啊 -> 你干什么`
- `有点害怕 -> 太怕了`
- `漏啦 -> 弄完`

这组真实标注支持当前判断：pinyin fallback 对近音错词有稳定增益，但不能解决所有 ASR 错词，仍需要 ASR 参数修复、重复/幻觉过滤和更强模型对比。

## 结论

拼音 fallback 值得继续，但不应直接全量接入生产排序。

推荐后续方向：

- 先继续作为离线实验工具，扩充人工确认样例。
- 生产设计上只作为 ASR lexical 的弱补充证据，分数上限低于精确 lexical。
- 对短 query 加保护：结合实体词迹象、文件名/OCR/字幕词表、semantic 支持或更严格的拼音连续匹配。
- evidence 中明确标记 `asr_pinyin` 命中来源，便于 UI 和调试识别。

## 后续

- 从 `asr_error_candidates_20260707-165005.html` 人工听查 30-50 条，补 `correct_text`、`manual_label` 和真实 query。
- 增加 negative controls，专门评估拼音分支带来的误召。
- 设计生产级候选融合策略：`semantic` 负责主题，`lexical` 负责精确词，`pinyin` 只补同音/近音实体漏召。
