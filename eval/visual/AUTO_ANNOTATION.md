# Visual auto-annotation plan

我们可以用 ChatGPT / 视觉大模型给抽出的图片和 5 秒短片做自动预标注，但评估集里要区分：

```text
auto label：模型自动生成的候选标注，用于加速人工审核
gold label：人工确认后的标注，用于正式计算 Recall/MRR/IoU
```

不要直接把 auto label 当 ground truth，否则后续评估会变成“用模型标的数据评估另一个模型”，容易污染结论。

## 总体流程

```text
视频
  ↓
生成分辨率变体：source / 720p / 360p ...
  ↓
image_retrieval：抽单帧图片
  ↓
sequence_retrieval：切 5 秒片段，并为每段抽 5-6 张代表帧
  ↓
视觉大模型自动预标注
  ↓
生成 auto labels / candidate queries
  ↓
人工快速审核
  ↓
写入 gold labels
  ↓
跑不同 CLIP 方案评估
```

## 为什么 5 秒短片也建议转成图片序列

直接把视频喂给大模型依赖具体 API/模型能力，成本和稳定性都不如图片序列可控。

对 5 秒片段，推荐抽：

```text
0.0s / 1.0s / 2.0s / 3.0s / 4.0s / 5.0s
```

或者抽 6 张图做成一张 contact sheet：

```text
┌──────┬──────┬──────┐
│ 0.0s │ 1.0s │ 2.0s │
├──────┼──────┼──────┤
│ 3.0s │ 4.0s │ 5.0s │
└──────┴──────┴──────┘
```

这样可以让模型看出一点时间变化，同时工程上仍然是 image input。

## Image-level 自动标注

目标：给单张 frame 生成结构化描述。

输出字段建议：

```json
{
  "item_id": "worldcup_ad_720p_t000012_000",
  "item_type": "image",
  "caption_en": "A football match in a stadium with players on the field.",
  "caption_zh": "体育场中的足球比赛，球员在场上奔跑。",
  "query_types": ["global_scene", "main_subject"],
  "objects": [
    {"name": "football player", "location": "center", "visibility": "clear"},
    {"name": "football", "location": "lower left", "visibility": "small"}
  ],
  "actions": ["running"],
  "scene": "football stadium",
  "text_overlay": [],
  "edge_objects": [
    {"name": "logo", "location": "top right", "visibility": "partial"}
  ],
  "small_objects": ["football"],
  "suggested_queries": [
    {"query": "a football match in a stadium", "query_type": "global_scene"},
    {"query": "a football on the field", "query_type": "small_object"},
    {"query": "logo on the top right", "query_type": "edge_object"}
  ],
  "confidence": 0.82,
  "needs_review": true
}
```

## Sequence-level 自动标注

目标：给 5 秒片段生成“片段级描述”和候选 query。

输出字段建议：

```json
{
  "item_id": "worldcup_ad_720p_s000010_000015",
  "item_type": "segment",
  "start": 10.0,
  "end": 15.0,
  "caption_en": "A football player runs with the ball while other players move across the field.",
  "caption_zh": "一名足球运动员带球奔跑，其他球员在场上移动。",
  "temporal_events": [
    {"time": "10.0-12.0", "event": "players running on the field"},
    {"time": "12.0-15.0", "event": "the ball moves toward the side"}
  ],
  "query_types": ["action", "main_subject", "small_object"],
  "objects": ["football player", "football", "stadium"],
  "actions": ["running", "kicking"],
  "text_overlay": [],
  "edge_objects": [],
  "small_objects": ["football"],
  "suggested_queries": [
    {"query": "a football player runs with the ball", "query_type": "action"},
    {"query": "a football on the field", "query_type": "small_object"}
  ],
  "confidence": 0.78,
  "needs_review": true
}
```

## 人工审核方式

建议做一个简单 review queue：

```text
左侧：图片或 5 秒片段 contact sheet
右侧：模型生成 caption / objects / suggested_queries
按钮：accept / edit / reject
```

第一版也可以不用 UI，直接用 JSONL/CSV：

```text
auto_annotations.local.jsonl
reviewed_annotations.local.jsonl
```

审核规则：

- 明显正确：accept；
- 时间范围不准：edit start/end；
- 模型看错：reject；
- 模型描述太泛：保留 caption，但不要生成 gold query；
- 小物体/边缘目标：人工必须确认，因为这正是 CLIP 容易失败的区域。

## Auto label 与正式评估的关系

正式评估只使用：

```text
source = human_verified
```

或者：

```text
review_status = accepted / edited
```

不使用：

```text
review_status = pending
```

这样可以保证评估结果可信。

## 推荐 Prompt：图片

```text
You are annotating frames for a video retrieval benchmark.

Return only valid JSON. Describe visible content, not guesses.
Focus on:
1. main scene
2. people and actions
3. small objects
4. objects near edges/corners
5. visible text/OCR
6. useful search queries

Use concise English captions and optional Chinese captions.
Mark uncertainty explicitly.
```

## 推荐 Prompt：5 秒片段 contact sheet

```text
You are annotating a 5-second video segment represented as a contact sheet.
Each cell is labeled with a timestamp.

Return only valid JSON. Describe the whole segment and temporal changes.
Do not invent events that are not visible.
Pay special attention to small objects, edge/corner objects, text on screen, and actions.
Generate search queries that a user might use to find this segment.
```

## 成本控制建议

第一轮不要全量标所有帧/片段。建议：

```text
image-level：
  每个视频每 5-10 秒抽 1 帧

sequence-level：
  每个视频每 5 秒一个 segment
  但只抽部分 segment 做自动标注，例如每 2-3 个 segment 取 1 个

高价值视频：
  真实 4K / 1080p 素材优先
```

对长视频，先标：

- 前 3-5 分钟；
- 或用户指定的关键时间段；
- 或通过 shot detection/场景变化挑代表片段。

## ChatGPT/API 与本地 VLM

可以有两条路线：

```text
OpenAI/ChatGPT API：
  效果通常更稳，适合高质量预标注和 query 生成；
  需要 API key，涉及上传图片/帧，注意隐私和成本。

本地 VLM：
  例如 Qwen2.5-VL / InternVL / LLaVA 类模型；
  不上传数据，适合隐私场景；
  但部署、显存、速度和标注质量要单独评估。
```

建议先用少量样本 A/B：

```text
30 张 image frame
30 个 5 秒 segment contact sheet
```

看模型输出质量，再决定是否全量跑。

## 用服务器 Qwen2.5-VL 预标注

如果服务器上的 Qwen2.5-VL 以 OpenAI-compatible API 暴露，例如 vLLM/LMDeploy/Ollama 网关一类，可以直接使用：

```powershell
$env:VLM_BASE_URL="http://SERVER_HOST:PORT/v1"
$env:VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
$env:VLM_API_KEY="EMPTY"

python scripts/visual_auto_annotate.py `
  --manifest eval/visual/image_retrieval/frames.video_search_test.local.json `
  --kind image `
  --out eval/visual/image_retrieval/auto_annotations.qwen25vl.local.jsonl `
  --sample 30 `
  --resume
```

先 dry-run 检查请求格式：

```powershell
python scripts/visual_auto_annotate.py `
  --manifest eval/visual/sequence_retrieval/contact_sheets.video_search_test.local.json `
  --kind segment `
  --out eval/visual/sequence_retrieval/qwen25vl_requests.local.jsonl `
  --limit 3 `
  --dry-run
```

如果模型服务不需要 API key，可以设置：

```powershell
$env:VLM_API_KEY=""
```

正式输出仍然只是 `auto_annotations.*.local.jsonl`，默认 `review_status=pending`。人工确认后才能进入正式 gold labels。

## 当前本地评测集配置

当前使用的视频目录：

```text
C:\Users\29154\Videos\视频检索测试
```

已生成：

```text
eval/visual/videos.video_search_test.local.json
eval/visual/image_retrieval/frames.video_search_test.local.json
eval/visual/sequence_retrieval/segments.video_search_test.local.json
```

sequence contact sheet 当前采用：

```text
5 秒 segment
每秒 2 帧
每张 contact sheet 共 10 帧
```

生成命令：

```powershell
..\.venv\Scripts\python.exe scripts\visual_eval_dataset.py make-contact-sheets `
  --segments eval\visual\sequence_retrieval\segments.video_search_test.local.json `
  --out-root runtime\eval\visual\contact_sheets_video_search_test_2fps `
  --out eval\visual\sequence_retrieval\contact_sheets.video_search_test.2fps.local.json `
  --sheet-sample-fps 2 `
  --cell-width 240 `
  --cell-height 135
```

当前 Qwen fallback 队列：

```text
qwen3.6-plus
qwen3.7-plus
qwen3-vl-235b-a22b-thinking
qwen3-vl-32b-thinking
```

设置方式：

```powershell
$env:VLM_MODELS="qwen3.6-plus,qwen3.7-plus,qwen3-vl-235b-a22b-thinking,qwen3-vl-32b-thinking"
```

如果 `qwen3.6-plus` 额度耗尽，脚本会在请求错误时尝试后续模型。已经成功解析的 item 会被 `--resume` 跳过。
