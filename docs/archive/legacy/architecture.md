> Archived reference. Current documentation starts at `docs/README.md`.

# MVP 架构

## 数据流

```text
视频上传
  ├─ Visual：1 fps 抽帧 → 5 秒窗口 → OpenCLIP 向量
  ├─ Face：2 fps 检测 → 人脸关联 → ArcFace tracklet 向量
  └─ ASR：16 kHz 音频 → Whisper 分句与时间戳

查询
  ├─ 文本/参考图 → visual_index
  ├─ 参考脸或已登记姓名 → face_index
  └─ 文字 → asr_index
       ↓
  时间相邻候选合并 + 模态权重融合
       ↓
  start/end、分数、缩略图、证据、播放器链接
```

## 存储

- `catalog.sqlite3`：视频、任务和人物实体元数据。
- `indexes/<video_id>/visual.npz`：视觉片段向量与时间段。
- `indexes/<video_id>/faces.npz`：人脸 tracklet 向量与时间段。
- `indexes/<video_id>/asr.json`：转写文本与句级时间戳。
- `thumbnails/<video_id>/`：结果预览图。

MVP 采用单机文件索引，便于快速验证。达到多机或百万级片段后，可将相同接口替换为 pgvector、Milvus 或 Qdrant。

## 模型生命周期

API 不导入 NPU 模型。每个索引阶段由单独 Python 子进程执行；子进程结束后，驱动上下文与 HBM 一并释放。为防止多个上传同时抢卡，编排层使用独占 worker lock 串行执行索引任务。

在线检索仅将轻量查询 encoder 缓存在 CPU 内存，不占 NPU。

## 已知 baseline 限制

- OpenAI CLIP 的中文文本能力有限，中文场景查询需评估 Chinese-CLIP/SigLIP。
- Face CPU 建索引较慢，应降低抽帧率、先跟踪后识别，或迁移到 CANN provider。
- ASR 文本当前使用精确/字符近似检索，后续增加 BM25 与多语言文本 embedding。
- 动作和事件依赖时序，后续在 CLIP 候选上加入 InternVideo2 精排。
