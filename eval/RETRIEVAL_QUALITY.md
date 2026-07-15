# 统一检索质量回归协议

这个协议为 Visual、ASR、OCR、Face、未来的 Shot Card 和多模态融合提供同一个离线裁判。它不负责运行模型，只评估搜索接口或实验策略已经产生的排序结果。

## Query/Qrel 格式

UTF-8 JSONL，每行一条查询。`query_id` 也兼容旧 ASR 数据的 `id`；`positives` 也兼容 `targets`。

```json
{"query_id":"shot_001","query":"人物望向远方的特写","split":"holdout","query_type":"shot_language","modalities":["visual"],"positives":[{"video_id":"video-1","start":12.4,"end":18.2}]}
{"query_id":"none_001","query":"雪地里的红色直升机","split":"holdout","query_type":"no_answer","positives":[]}
```

时间可使用秒字段 `start/end`、API 风格 `start_time/end_time`，或毫秒字段 `start_ms/end_ms`。只有 `video_id` 而没有时间的 target 用于素材级命中评估。

必须保留无答案查询。它们的 `positives/targets` 是空数组，用于计算错误接受率，不能混入 MRR 分母。

## Result 格式

UTF-8 JSONL，每行对应一条查询：

```json
{"query_id":"shot_001","results":[{"video_id":"video-1","start_time":13.0,"end_time":18.0,"score":0.83,"above_threshold":true}]}
```

`results` 按实际展示顺序排列。`above_threshold=false` 的诊断结果不计入正式排序，也不会导致无答案错误接受。

## 命中与指标

默认要求同一视频且时间重叠至少 1 秒。可同时设置最低 time IoU。

- Recall@1/5/10：第一个相关结果是否在前 K。
- MRR：第一个相关结果的倒数排名。
- nDCG@K：多个相关时间段在 Top-K 中的排序质量。
- False-positive rate@K：有答案查询已返回结果中的非相关比例。
- Mean first-hit tIoU：第一个相关结果的时间边界质量。
- No-answer false-accept rate：无答案查询中仍返回阈值以上结果的比例。

运行：

```powershell
python scripts/retrieval_quality_eval.py `
  --queries eval/retrieval/queries.local.jsonl `
  --results runtime/analysis/retrieval/results.jsonl `
  --output runtime/analysis/retrieval/quality.json `
  --min-overlap-seconds 1 `
  --min-tiou 0
```

第一阶段不要用同一批查询同时调参和验收。至少分为 `tune/dev/holdout`；策略只在 tune 选择，dev 用于发现退化，holdout 只用于最终报告。

## 建议首批查询覆盖

- 全局场景、人物主体、小物体、动作、镜头景别、运镜、风格/氛围；
- 精确对白、语义改写、跨语言、专有名词；
- OCR 字幕、招牌和边缘文字；
- 多条件组合、否定条件、时长条件；
- 同领域困难负例和跨领域无答案查询。

脚本只定义统一度量，不替代人工标注。Visual seed 中尚未填写的 `positives` 必须经人工确认后才能作为正式回归门槛。
