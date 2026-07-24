# Image-level visual retrieval evaluation

这个子集专门回答：

> 如果正确画面已经被抽成图片，CLIP 能不能用文本把它找出来？

它不评估视频切片、不评估 segment 合并，也不评估多帧聚合。它只测单张图片/单帧的图文匹配能力。

## 适合比较什么

- CLIP / OpenCLIP / SigLIP / Chinese-CLIP 等模型差异；
- center crop / padded full / horizontal crop / tile 的差异；
- 360p / 720p / 1080p / 4K 分辨率差异；
- 小物体、边缘目标、文字画面是否被裁掉或压没；
- 中英文 query 对同一画面的匹配能力。

## Manifest 格式

`frames.local.json` 示例：

```json
{
  "schema_version": 1,
  "frames": [
    {
      "image_id": "worldcup_ad_360p_t000010_000",
      "group_id": "worldcup_ad",
      "variant_id": "worldcup_ad_360p",
      "path": "runtime/eval/visual/frames/worldcup_ad/worldcup_ad_360p/t000010_000.jpg",
      "time": 10.0,
      "width": 636,
      "height": 360,
      "resolution_label": "360p",
      "view_type": "original_frame"
    }
  ]
}
```

## Query 格式

`queries.local.jsonl` 每行一条：

```json
{
  "query_id": "worldcup_small_001",
  "group_id": "worldcup_ad",
  "query": "a football on the field",
  "language": "en",
  "query_type": "small_object",
  "positive_image_ids": ["worldcup_ad_720p_t000012_000"],
  "hard_negative_image_ids": [],
  "notes": "足球很小，适合测分辨率和 tile。"
}
```

## 初期标注方法

第一版可以不用给每张图精细打标签。推荐流程：

1. 从视频每 2-5 秒抽一张 frame；
2. 人工打开 frame gallery；
3. 给每条 query 标 3-20 个 positive image；
4. 对容易混淆的图标 hard negative；
5. 按 query_type 分组看 Recall@K。

## 主要指标

- Recall@1；
- Recall@5；
- Recall@10；
- MRR；
- 按 query_type / resolution_label / view_type 分组的 Recall。

