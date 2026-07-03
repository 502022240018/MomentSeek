# MomentSeek visual evaluation set

本目录保存可复现的 visual 评测资产：数据集计划、manifest、query 文件、schema 和运行说明。

人类可读的实验结论和建议放在：

```text
docs/experiments/visual/
```

这个目录用于沉淀 visual 检索评估集。它先服务于我们当前讨论的核心问题：

- 不同 CLIP 预处理方案：center crop、padding/full view、horizontal crops、tiles；
- 不同抽帧与解码高度：例如 1fps/5fps、decode height 256/512/720；
- 不同视频分辨率：360p、720p、1080p、4K；
- 不同查询类型：全局场景、小物体、边缘目标、动作事件、文字/字幕画面。

## 设计原则

评估集分成两个层级：

```text
content group：同一个原始内容，例如“世界杯广告”
  ├── source/original variant：原始视频
  ├── 360p variant：同内容低分辨率版本
  ├── 720p variant：同内容 720p 版本
  ├── 1080p variant：同内容 1080p 版本
  └── 2160p variant：同内容 4K 版本
```

query 和人工标注绑定到 `group_id`，而不是某个具体视频文件。这样同一个查询、同一组正确时间段，可以横向比较不同分辨率变体的召回效果。

举个例子：

```json
{
  "query_id": "worldcup_edge_001",
  "group_id": "worldcup_ad",
  "query": "logo on the top right of the screen",
  "query_type": "edge_object",
  "positives": [
    {"start": 12.0, "end": 18.0}
  ]
}
```

这条标注会同时用于：

```text
worldcup_ad_360p
worldcup_ad_720p
worldcup_ad_1080p
worldcup_ad_2160p
```

如果 4K/1080p 是从低清源上采样出来的，要在 manifest 里标记 `is_upscaled=true`。这类结果只能用来验证工程链路，不适合当作真实“高分辨率收益”结论。

## 文件说明

```text
eval/visual/
├── README.md
├── DATASET_SOURCING.md
├── CUSTOM_VIDEO_CHECKLIST.md
├── AUTO_ANNOTATION.md
├── auto_annotation.schema.example.json
├── videos.example.json
├── queries.seed.jsonl
├── resolution_eval_config.example.json
├── image_retrieval/
│   ├── README.md
│   ├── frames.example.json
│   └── queries.seed.jsonl
└── sequence_retrieval/
    ├── README.md
    ├── segments.example.json
    └── queries.seed.jsonl
```

- `videos.example.json`：视频 manifest 示例。
- `queries.seed.jsonl`：初始 query 种子，包含建议查询类型，但时间段还需要人工标注。
- `resolution_eval_config.example.json`：后续跑不同 CLIP/预处理方案的评测配置示例。
- `DATASET_SOURCING.md`：开源数据集复用策略。
- `CUSTOM_VIDEO_CHECKLIST.md`：如果需要自建素材，应上传什么视频。
- `image_retrieval/`：单帧/图片级图文检索评估。
- `sequence_retrieval/`：视频片段/图片序列检索评估。

本地生成文件建议使用：

```text
eval/visual/videos.local.json
eval/visual/videos.variants.local.json
eval/visual/queries.local.jsonl
```

这些文件已加入 `.gitignore`，因为它们通常包含本机路径和本地素材信息。

## 两层评估集

我们把 visual 评估拆成两层：

```text
image_retrieval
  测 CLIP 对单张图片/单帧的检索能力。

sequence_retrieval
  测一段图片序列/视频片段的表达能力。
```

如果 image-level 很好但 sequence-level 很差，说明主要问题在抽帧、聚合、segment 合并；如果 image-level 本身就差，说明主要问题在模型、预处理、分辨率、小物体/边缘目标。

## 推荐评估指标

第一阶段先看这些：

- `Recall@1`：第 1 个结果是否命中正确时间段；
- `Recall@5`：前 5 个结果是否至少有一个命中；
- `Recall@10`：前 10 个结果是否至少有一个命中；
- `MRR`：正确结果排得靠不靠前；
- `time IoU / overlap`：返回片段和标注片段是否有足够时间重叠；
- 索引耗时、搜索耗时、embedding 数量、索引大小。

## Query 类型建议

评估时不要只写“容易被 CLIP 命中”的全局场景查询，要刻意覆盖会暴露问题的类型。

| query_type | 目的 |
|---|---|
| `global_scene` | 测全局场景语义，例如球场、厨房、舞台 |
| `main_subject` | 测主体人物/主体物体 |
| `small_object` | 测小物体，例如杯子、球、手机、鞋 |
| `edge_object` | 测边缘目标，例如右上角 logo、左侧广告牌 |
| `action` | 测动作/事件，例如进球、举手、开门 |
| `text_overlay` | 测画面中文字/字幕相关检索 |
| `style_scene` | 测画面风格/氛围，例如夜景、动画、黑白画面 |

## 人工标注规则

每条 query 至少标 1 个 positive 时间段：

```json
{"start": 10.0, "end": 15.0, "confidence": "confirmed"}
```

建议：

- 时间段宁可稍宽一点，不要只标一帧；
- 若目标只出现瞬间，可以给 1~3 秒窗口；
- 对动作事件，标完整动作发生区间；
- 对边缘/小物体，记录 notes，说明目标在哪个位置；
- 如果 query 是为了测试分辨率敏感性，设置 `resolution_sensitivity` 为 `high`。

## 本地生成当前视频清单

从当前 `runtime/catalog.sqlite3` 生成本地 manifest：

```powershell
python scripts/visual_eval_dataset.py scan-catalog `
  --db runtime/catalog.sqlite3 `
  --out eval/visual/videos.local.json
```

如果当前 shell 里的 Python 没有 OpenCV，可使用项目外层虚拟环境：

```powershell
..\.venv\Scripts\python.exe scripts/visual_eval_dataset.py scan-catalog `
  --db runtime/catalog.sqlite3 `
  --out eval/visual/videos.local.json
```

也可以从任意视频目录生成 manifest，例如：

```powershell
$env:VIDEO_TEST_DIR="C:\Users\29154\Videos\视频检索测试"
..\.venv\Scripts\python.exe scripts/visual_eval_dataset.py scan-directory `
  --root "$env:VIDEO_TEST_DIR" `
  --out eval/visual/videos.video_search_test.local.json
```

## 生成不同分辨率变体

从 manifest 生成 360p/720p/1080p/4K 变体：

```powershell
python scripts/visual_eval_dataset.py make-variants `
  --manifest eval/visual/videos.local.json `
  --out-root runtime/eval/visual/resolution_variants `
  --out-manifest eval/visual/videos.variants.local.json `
  --heights 360 720 1080 2160
```

默认不会把低清源强行上采样成 1080p/4K。若只是为了测试工程链路，可以加：

```powershell
  --allow-upscale
```

但真正比较“高分辨率是否更好”时，应优先使用真实 1080p/4K 源视频，再向下生成 720p/360p。

## 抽帧生成 image-level manifest

```powershell
python scripts/visual_eval_dataset.py extract-frames `
  --manifest eval/visual/videos.variants.local.json `
  --out-root runtime/eval/visual/frames `
  --out eval/visual/image_retrieval/frames.local.json `
  --interval-seconds 5
```

## 生成 sequence-level segment manifest

```powershell
python scripts/visual_eval_dataset.py make-segments `
  --manifest eval/visual/videos.variants.local.json `
  --out eval/visual/sequence_retrieval/segments.local.json `
  --segment-seconds 5 `
  --stride-seconds 5
```
