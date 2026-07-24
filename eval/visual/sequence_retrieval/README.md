# Sequence-level visual retrieval evaluation

这个子集专门回答：

> 一段图片序列/视频片段应该怎样表达，才能被文本准确检索出来？

这里才评估抽帧、segment、shot、聚合策略和时间片段返回。

## 适合比较什么

- 抽帧 FPS：1fps / 2fps / 5fps；
- segment 长度：5s / 10s / 15s；
- 聚合方式：mean、top1 MaxSim、top3 MaxSim、mean + topK；
- 多视图融合：center crop、padding、tile；
- shot-aware keyframe；
- 检索后片段合并策略。

## Segment manifest 格式

`segments.local.json` 示例：

```json
{
  "schema_version": 1,
  "segments": [
    {
      "segment_id": "worldcup_ad_720p_s000010_000015",
      "group_id": "worldcup_ad",
      "variant_id": "worldcup_ad_720p",
      "start": 10.0,
      "end": 15.0,
      "duration": 5.0,
      "resolution_label": "720p",
      "source_video_path": "runtime/uploads/example.mp4"
    }
  ]
}
```

## Query 格式

`queries.local.jsonl` 每行一条：

```json
{
  "query_id": "seq_goal_001",
  "group_id": "worldcup_ad",
  "query": "a player scores a goal",
  "language": "en",
  "query_type": "action",
  "positives": [
    {"start": 120.0, "end": 135.0, "confidence": "confirmed"}
  ],
  "notes": "标完整动作时间段，不只标一帧。"
}
```

注意：sequence query 绑定 `group_id`，不是绑定某个具体分辨率。这样同一个 query 和 positive 时间段可以复用于 360p/720p/1080p/4K 变体。

## 命中规则

第一版建议简单一点：

```text
返回片段和 positive 时间段重叠 >= 1 秒，就算 hit。
```

后续再升级：

```text
time IoU >= 0.3
```

## 主要指标

- Recall@1；
- Recall@5；
- Recall@10；
- MRR；
- overlap hit rate；
- mean time IoU；
- 平均返回片段长度；
- 索引耗时；
- 搜索耗时；
- embedding 数量；
- 索引大小。

