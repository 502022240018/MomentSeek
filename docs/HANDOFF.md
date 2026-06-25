# MomentSeek MVP 开发交接文档

> 面向下一位共同开发同学：这份文档记录从项目启动到当前版本的设计、实现、运行方式、数据组织、索引/检索 pipeline、已知问题和后续建议。  
> 当前项目仓库目录：`video_retrieval_mvp/`。GitHub 仓库：`https://github.com/502022240018/MomentSeek`。

## 1. 项目目标

MomentSeek 是一个私有视频片段检索 MVP baseline。目标不是先做复杂产品，而是先把“上传视频 → 建三路索引 → 输入查询 → 返回可播放时间片段”这条闭环跑通，并且能方便后续迁移到服务器/GitHub/其他机器。

当前 MVP 聚焦三个核心索引：

| 索引 | 模型/算法 | 解决的问题 | 当前状态 |
| --- | --- | --- | --- |
| `face_index` | InsightFace / ArcFace | 参考图找人、明星/人物出现片段 | 已实现；默认 CPU ONNXRuntime，支持 CANN provider 入口但未充分验证 |
| `visual_index` | OpenCLIP ViT-B/32 | 文本/参考图找场景、物体、视觉语义 | 已实现；默认 5fps 抽帧、5 秒逻辑分段 |
| `asr_index` | Whisper baseline，保留 FunASR 适配器 | 文本找语音内容 | 已实现；前端可选 ASR 模型和语言 |

当前支持三类核心查询：

1. 参考图 + 文本找人 / 明星出现。
2. 文本或文本 + 参考图找场景/物体。
3. 文本找语音内容。

前端风格参考 TwelveLabs Playground，当前是 React + Vite 单页应用；后端是 FastAPI。

## 2. 当前版本状态

本地项目已经具备：

- 上传视频和可选字幕文件。
- 对视频建立 Visual / Face / ASR 三路索引。
- 索引任务排队、运行、失败、完成状态。
- 索引阶段耗时展示：Visual / Face / ASR 每阶段耗时、总耗时。
- 搜索耗时展示。
- Visual 分段召回，不再把连续视觉桶一路合并成整段视频。
- Visual 检索使用分布式判定：raw cosine + percentile + robust z，而不是固定 `cos >= 0.12`。
- ASR 可手动选择语言：`zh / en / auto`。
- ASR 可手动选择模型：`base / small / medium / large-v3`。前端当前默认 `medium`，后端无前端覆盖时默认 `small`。
- 中文 ASR 搜索做了部分简繁折叠：例如搜“白痴”可以命中旧索引里的“白癡”。
- 本地 CUDA 可用时，CLIP visual 和 Whisper ASR 可跑在本机 3060 上。
- 模型索引阶段采用子进程，阶段结束后释放显存/内存。
- 支持 Cloudflare Tunnel 暂时公网展示前端。

需要注意：

- GitHub 上可能只包含较早 baseline。当前本地存在较多未提交修改，交接前建议整理 commit 并 push。
- `runtime/` 和 `models/` 不进 Git，迁移机器时要单独拷贝或重新生成。
- 服务器 `110.126.0.52` 是共享环境，部署前必须检查资源，避免占用别人 GPU/NPU。

## 3. 代码结构

```text
video_retrieval_mvp/
├─ backend/
│  ├─ app/
│  │  ├─ main.py             # FastAPI 入口、API 路由、静态前端挂载
│  │  ├─ settings.py         # 环境变量与路径配置
│  │  ├─ schemas.py          # 请求/响应校验模型
│  │  ├─ db.py               # SQLite catalog 数据库
│  │  ├─ worker.py           # 索引任务编排；每阶段启动独立子进程
│  │  ├─ stage_runner.py     # 子进程阶段入口：visual / face / asr
│  │  ├─ search.py           # 三路召回、融合、分段合并
│  │  ├─ media.py            # 视频探测、抽帧、抽音频、缩略图
│  │  └─ indexing/
│  │     ├─ visual.py        # OpenCLIP 视觉索引
│  │     ├─ faces.py         # InsightFace 人脸索引
│  │     ├─ asr.py           # Whisper/FunASR/字幕 ASR 索引
│  │     └─ common.py        # 原子保存、向量 normalize 等
│  ├─ tests/                 # pytest 单测
│  └─ requirements*.txt
├─ frontend/
│  ├─ src/
│  │  ├─ main.tsx            # React UI，所有页面当前集中在这里
│  │  ├─ api.ts              # 前端 API client 和类型定义
│  │  └─ styles.css          # 页面样式
│  ├─ package.json
│  └─ vite.config.ts
├─ docs/
│  ├─ HANDOFF.md             # 本交接文档
│  ├─ architecture.md
│  ├─ server-operations.md
│  └─ validation.md
├─ runtime/                  # 运行时数据，不进 Git
├─ models/                   # 模型权重，不进 Git
├─ DEPLOY.md                 # 部署与迁移说明
├─ README.md                 # 项目概览
├─ Dockerfile.cpu
├─ Dockerfile.ascend
├─ compose.yml
├─ compose.server.yml
└─ compose.ascend.yml
```

## 4. 运行环境与启动方式

### 4.1 Windows 本地开发

当前本地虚拟环境在 `prototype/.venv`，项目代码在 `prototype/video_retrieval_mvp`。

后端启动示例：

```powershell
cd C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp

$env:APP_DATA_DIR=(Resolve-Path ".\runtime").Path
$env:APP_MODEL_DIR=(Resolve-Path ".\models").Path
$env:CUDA_ENABLED="true"
$env:ASR_ENGINE="whisper"
$env:ASR_MODEL="small"
$env:ASR_LANGUAGE="zh"
$env:ASR_DEVICE="cuda"

C:\Users\29154\Projects\video-removal-system\prototype\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

前端开发服务：

```powershell
cd C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp\frontend
npm install
npm run dev -- --host 0.0.0.0 --host 127.0.0.1 --port 5173
```

访问：

- 前端 dev：`http://127.0.0.1:5173`
- 后端 API：`http://127.0.0.1:8000`
- API docs：`http://127.0.0.1:8000/docs`

### 4.2 前端公网临时展示

当前试过 Cloudflare Tunnel：

```powershell
runtime\tools\cloudflared.exe tunnel --url http://127.0.0.1:5173 --no-autoupdate --protocol http2
```

注意：

- quick tunnel 地址会变，适合临时演示。
- 默认没有登录鉴权，任何拿到 URL 的人都可以访问、上传、检索。
- Vite 已设置 `allowedHosts: true`，否则 Cloudflare 域名会被 Vite 拦截。

### 4.3 Docker / 服务器部署

CPU 模式：

```bash
cp .env.example .env
docker compose -f compose.yml up -d --build
```

共享服务器上先不映射 NPU：

```bash
docker compose -f compose.yml -f compose.server.yml up -d --build
```

昇腾/NPU 模式必须先确认空闲卡：

```bash
./scripts/check_resource.sh
NPU_DEVICE_ID=7 docker compose -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d --build
```

详细迁移见 `DEPLOY.md`。

## 5. 配置项

主要配置在 `.env.example` 和 `backend/app/settings.py`。

| 配置 | 说明 | 当前建议 |
| --- | --- | --- |
| `APP_DATA_DIR` | 运行时数据目录 | 本地用 `runtime/`，容器用 `/app/runtime` |
| `APP_MODEL_DIR` | 模型目录 | 本地用 `models/`，容器用 `/app/models` |
| `CUDA_ENABLED` | 是否优先用 CUDA | 本地 3060 可设 `true` |
| `NPU_ENABLED` | 是否启用 NPU | 共享服务器默认 `false`，确认卡后再开 |
| `NPU_DEVICE_ID` | NPU 卡号 | 服务器上按实际空闲卡选择 |
| `CLIP_MODEL` | OpenCLIP 模型 | `ViT-B-32` |
| `CLIP_PRETRAINED` | CLIP 权重来源/路径 | 本地可 `openai`，服务器可指定权重文件 |
| `VISUAL_SAMPLE_FPS` | Visual 抽帧 fps | 默认 `5.0` |
| `VISUAL_SEGMENT_SECONDS` | Visual 逻辑分段长度 | 默认 `5.0` 秒 |
| `FACE_SAMPLE_FPS` | Face 抽帧 fps | 默认 `2.0` |
| `FACE_PROVIDER` | Face 推理 provider | 默认 `cpu` |
| `ASR_ENGINE` | ASR 引擎 | 当前默认 `whisper` |
| `ASR_MODEL` | Whisper 模型 | 后端默认 `small`；前端创建任务时当前默认覆盖为 `medium` |
| `ASR_LANGUAGE` | ASR 语言 | 中文视频默认 `zh` |
| `ASR_DEVICE` | ASR 推理设备 | 本地可 `cuda`，服务器按实际配置 |

前端建索引时可以覆盖：

- `visual_sample_fps`
- `visual_segment_seconds`
- `face_sample_fps`
- `asr_model`
- `asr_language`

这些参数会写入 `jobs.options`，并由 worker 子进程读取。

## 6. 数据管理与数据库

运行时数据全部在 `runtime/`，不提交到 Git。

```text
runtime/
├─ catalog.sqlite3              # SQLite 元数据数据库
├─ uploads/                     # 上传视频和可选字幕
├─ indexes/{video_id}/
│  ├─ visual.npz                # Visual 分段向量
│  ├─ faces.npz                 # Face track 向量
│  ├─ asr.json                  # ASR 带时间戳文本
│  └─ work/                     # 阶段临时文件，如 audio.wav
├─ thumbnails/{video_id}/       # visual/face 缩略图
├─ entities/                    # 人物库参考图和 embedding
├─ queries/                     # 搜索时上传的临时参考图
└─ job-{job_id}.log             # 索引任务日志
```

SQLite 数据库：`runtime/catalog.sqlite3`。

表结构：

### `videos`

记录上传视频元数据。

| 字段 | 说明 |
| --- | --- |
| `id` | 视频 UUID |
| `name` | 原始文件名 |
| `file_path` | 上传视频路径 |
| `duration` | 时长秒 |
| `fps` | 视频帧率 |
| `width` / `height` | 分辨率 |
| `status` | `uploaded / indexing / ready / failed` |
| `indexed_modalities` | JSON 数组，如 `["visual","face","asr"]` |
| `created_at` / `updated_at` | 时间 |

### `jobs`

记录索引任务。

| 字段 | 说明 |
| --- | --- |
| `id` | job UUID |
| `video_id` | 关联视频 |
| `status` | `queued / running / completed / failed` |
| `stage` | 当前阶段：`starting / visual / face / asr / completed / failed` |
| `progress` | 0 到 1 |
| `modalities` | 本任务需要跑的阶段 |
| `options` | 前端传入的索引参数 |
| `metrics` | 阶段耗时和模型信息 |
| `error` | 失败错误文本 |
| `worker_pid` | worker 进程号 |

`metrics` 示例：

```json
{
  "stages": {
    "visual": {
      "elapsed_seconds": 22.896,
      "status": "completed",
      "segments": 30,
      "frames": 750,
      "device": "cuda"
    },
    "face": {
      "elapsed_seconds": 369.712,
      "status": "completed",
      "tracks": 7,
      "detections": 96,
      "provider": "cpu"
    },
    "asr": {
      "elapsed_seconds": 133.327,
      "status": "completed",
      "chunks": 23,
      "engine": "whisper",
      "model": "small",
      "language": "zh"
    }
  },
  "total_elapsed_seconds": 525.984
}
```

### `entities`

人物库。

| 字段 | 说明 |
| --- | --- |
| `id` | entity UUID |
| `name` | 人物名，大小写不敏感唯一 |
| `reference_path` | 参考图 |
| `embedding_path` | 参考脸向量 `.npz` |
| `created_at` | 创建时间 |

## 7. API 概览

后端入口：`backend/app/main.py`。

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查，返回 CUDA/NPU 状态 |
| `POST` | `/api/videos` | 上传视频，可选字幕 JSON/SRT/VTT |
| `GET` | `/api/videos` | 视频列表 |
| `GET` | `/api/videos/{video_id}` | 视频详情和任务 |
| `GET` | `/api/videos/{video_id}/media` | 原视频流式播放 |
| `POST` | `/api/videos/{video_id}/index` | 创建索引任务 |
| `GET` | `/api/jobs` | 索引任务列表 |
| `GET` | `/api/jobs/{job_id}` | 任务详情，失败时带 log tail |
| `POST` | `/api/entities` | 登记人物参考图 |
| `GET` | `/api/entities` | 人物库列表 |
| `GET` | `/api/entities/{entity_id}/reference` | 人物参考图 |
| `POST` | `/api/search` | 三路检索 |
| `GET` | `/api/thumbnails/{video_id}/{filename}` | 缩略图 |

### 创建索引任务请求

```json
{
  "modalities": ["visual", "face", "asr"],
  "visual_sample_fps": 5.0,
  "visual_segment_seconds": 5.0,
  "face_sample_fps": 2.0,
  "asr_model": "small",
  "asr_language": "zh"
}
```

### 搜索请求

`POST /api/search` 使用 `multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `query_text` | 查询文本，可空但必须有文本或图片之一 |
| `query_image` | 参考图，可空 |
| `modalities` | 逗号分隔，如 `visual,face,asr` |
| `video_ids` | JSON 数组，限定检索视频 |
| `alpha` | 文本/图片组合的文本权重，0 到 1 |
| `limit` | 返回上限 |

返回结果包含：

- `video_id`
- `video_name`
- `start_time`
- `end_time`
- `score`
- `modalities`
- `thumbnail_url`
- `media_url`
- `decision`
- `evidence`
- `elapsed_seconds`

检索结果返回的是原视频的时间片段，不是物理切出来的小视频文件。前端播放时通过原视频 URL 加 `currentTime` 跳转。

## 8. 索引 pipeline 详解

整体索引流程：

```text
上传视频
  ↓
写入 runtime/uploads/{video_id}.mp4
  ↓
probe_video 读取 duration/fps/width/height
  ↓
写入 SQLite videos
  ↓
用户点击建立索引
  ↓
写入 SQLite jobs，状态 queued
  ↓
launch_job 启动 app.worker 子进程
  ↓
worker 获取文件锁，串行执行各阶段
  ↓
每个阶段再启动 app.stage_runner 子进程
  ↓
阶段进程结束，模型释放
  ↓
写入 indexes / thumbnails / metrics
  ↓
video.status = ready
```

为什么这么设计：

- API 进程常驻但不应该长期占用显存。
- 每个模型阶段在独立子进程中加载模型，阶段结束进程退出，显存释放。
- 共享服务器上可以减少“模型常驻占卡”的风险。
- `index-worker.lock` 保证当前实例一次只跑一个索引任务，避免本地/共享机器同时拉满资源。

### 8.1 Visual index

代码：`backend/app/indexing/visual.py`。

默认参数：

- `sample_fps = 5.0`
- `segment_seconds = 5.0`
- `batch_size = 32`
- 模型：OpenCLIP `ViT-B-32`

流程：

```text
iter_sampled_frames(video, sample_fps)
  ↓
按 timestamp // segment_seconds 分桶
  ↓
每个 bucket 保存一张缩略图
  ↓
CLIP encode_image 批量编码抽样帧
  ↓
每个 bucket 内帧向量求平均
  ↓
normalize
  ↓
保存 visual.npz
```

`visual.npz` 内容：

| key | 说明 |
| --- | --- |
| `embeddings` | 每个逻辑片段的 CLIP 向量 |
| `start_times` | 每段开始时间 |
| `end_times` | 每段结束时间 |
| `thumbnails` | 缩略图文件名 |
| `model` | 模型名 |

当前 Visual 抽帧不是物理切片；它只是按时间桶组织向量。召回结果仍指向原视频时间段。

### 8.2 Face index

代码：`backend/app/indexing/faces.py`。

默认参数：

- `face_sample_fps = 2.0`
- 模型：InsightFace `buffalo_l`
- provider：默认 CPU；如果 CANNExecutionProvider 可用可切 CANN

流程：

```text
iter_sampled_frames(video, face_sample_fps)
  ↓
InsightFace detect + embedding
  ↓
基于 embedding cosine + bbox IoU 做简易 track
  ↓
为每条 track 维护 start/end/best_crop/embedding 列表
  ↓
track 结束后平均 embedding
  ↓
保存 faces.npz 和 face 缩略图
```

`faces.npz` 内容：

| key | 说明 |
| --- | --- |
| `embeddings` | 每条人脸 track 的身份向量 |
| `start_times` | track 开始时间 |
| `end_times` | track 结束时间 |
| `thumbnails` | 最佳人脸 crop 缩略图 |
| `qualities` | 人脸质量分 |
| `model` | 模型名 |

当前瓶颈：Face 阶段默认 CPU，很慢。一次 150 秒视频，Face 可能数分钟。后续可优化：

- 降低 `face_sample_fps`。
- 先只跑 Visual/ASR，Face 按需跑。
- 在服务器上验证 InsightFace CANN provider。
- 改成 RetinaFace/ArcFace 更轻量组合或批处理。

### 8.3 ASR index

代码：`backend/app/indexing/asr.py`。

支持三种输入：

1. 上传时带 JSON/SRT/VTT 字幕：直接解析，不跑模型。
2. Whisper：当前主要 baseline。
3. FunASR：保留适配器，但本地环境尚未安装/验证完整依赖。

Whisper 当前前端可选：

- 模型：`base / small / medium / large-v3`
- 语言：`zh / en / auto`

建议：

- 快速冒烟：`base + zh`
- 正常测试：`small + zh` 或 `medium + zh`
- 英文视频：`small + en` 或 `base + en`
- 混杂语种：`auto`

流程：

```text
如果有 sidecar 字幕
  → load_sidecar
否则
  → ffmpeg/imageio-ffmpeg 抽取 16k mono wav
  → Whisper/FunASR 转写
  → 保存 chunks
```

`asr.json` 内容：

```json
{
  "engine": "whisper",
  "model": "small",
  "language": "zh",
  "chunks": [
    {
      "start_time": 0.92,
      "end_time": 2.24,
      "text": "示例字幕"
    }
  ]
}
```

已处理的问题：

- 本地未装系统 ffmpeg 时，使用 `imageio-ffmpeg` fallback。
- 视频无音轨时，不再把 ASR 任务判失败，而是写入空 `chunks`，engine 为 `no_audio`。
- Whisper 原本内部调用 ffmpeg；当前改成先抽 wav，再用 Python `wave + numpy` 加载音频数组给 Whisper，规避 PATH 问题。
- 中文搜索加入部分简繁折叠，旧 ASR 结果中的繁体字也能被简体查询命中。

ASR 当前不足：

- Whisper 小模型中文质量可能不稳，容易乱码/串语言。
- `small/medium` 明显更稳，但更慢，首次运行会下载模型。
- 真正中文长视频建议评估 FunASR/Paraformer，并加入语义检索或拼音/错别字容错。

## 9. 检索 pipeline 详解

代码：`backend/app/search.py`。

整体流程：

```text
前端提交 query_text / query_image / modalities / video_ids
  ↓
后端按 modality 构建查询向量或查询文本
  ↓
遍历候选视频 indexes/{video_id}
  ↓
分别召回 visual / face / asr candidates
  ↓
按时间合并 candidates 为 SearchResult
  ↓
返回片段、分数、证据、缩略图、播放 URL
```

### 9.1 Visual 检索

输入：

- 纯文本：`query_text`
- 纯图片：`query_image`
- 文本 + 图片：按 `alpha` 加权组合

流程：

```text
CLIP encode_text / encode_image
  ↓
query vector normalize
  ↓
与 visual.npz embeddings 做 cosine
  ↓
对每个视频内部的 scores 做 robust distribution
  ↓
按 percentile / robust_z 判定 strong/fuzzy/fallback
  ↓
返回 visual candidates
```

为什么不用固定 cosine 阈值：

- CLIP raw cosine 在不同视频/查询之间分布差异很大。
- 固定 `0.12` 区分度不足，容易“所有段都像”或“全都不够”。
- 当前改成 per-query/per-video 分布判断：
  - `median`
  - `MAD`
  - `robust_z`
  - `empirical percentile`

当前 profile：

- `recall`
- `balanced`
- `precision`

API 暂未把 profile 暴露到前端，后续可以加。

### 9.2 Face 检索

输入方式：

1. 上传参考图：直接用 InsightFace 提取参考脸向量。
2. 文本包含人物库名称：例如人物库有“张三”，查询文本出现“张三”，就读取该 entity embedding。

流程：

```text
reference image / entity embedding
  ↓
与 faces.npz embeddings 做 cosine
  ↓
阈值当前约 0.35
  ↓
返回 face candidates
```

注意：

- 当前 Face 搜索编码仍默认 CPU。
- 参考图必须有清晰人脸；否则会返回“未检测到人脸”。

### 9.3 ASR 检索

输入：`query_text`。

流程：

```text
读取 asr.json chunks
  ↓
normalize_text(query) 和 normalize_text(chunk.text)
  ↓
子串命中得 1
  ↓
否则用 char bigram coverage 做近似分
  ↓
score >= 0.25 返回
```

当前 `normalize_text` 支持：

- 大小写折叠。
- NFKC 标准化。
- 去掉标点空白，只保留 `isalnum`。
- 部分繁简折叠，如 `白癡 → 白痴`、`來這邊 → 来这边`。

ASR 检索是 lexical，不是语义检索。也就是说：

- 搜“电影投资”必须 ASR 文本里真的有这些字或近似字。
- 如果 ASR 把台词识别错了，搜索不会神奇命中。
- 后续可加文本 embedding / BM25 / 拼音 / edit distance。

### 9.4 片段合并逻辑

候选结果类型：`Candidate`。

候选合并为 `SearchResult` 时会按时间聚合，关键规则：

- 结果返回的是连续时间片段，包含 `start_time/end_time`。
- Face/ASR 相邻候选可按 gap 合并。
- Visual bucket 已经是展示粒度，visual-only 结果不再因为相邻就无限合并。
- Visual 与 Face/ASR 如果时间重叠或被其他 modality 锚定，可以合并。
- 默认最大结果时长约 15 秒，避免召回整个视频。

这是之前修过的一个重要点：早期 Visual 相邻 bucket 会被链式合并，导致“整个视频都被召回”。现在 visual-only 只在重叠时合并。

## 10. 前端结构与页面

代码集中在 `frontend/src/main.tsx`，API 类型在 `frontend/src/api.ts`，样式在 `frontend/src/styles.css`。

页面：

| 页面 | 功能 |
| --- | --- |
| 概览 | 展示资产、人物、任务数量 |
| 视频资产 | 上传视频/字幕、设置索引参数、创建索引任务 |
| 索引任务 | 查看任务状态、阶段、进度、耗时、错误 |
| 人物库 | 上传参考脸，登记人物 |
| 检索 | 选择视频范围、输入查询、上传参考图、选择 modality、展示结果 |

当前前端已增加：

- Visual fps 输入框。
- Visual 分段秒数输入框。
- Face fps 输入框。
- ASR 模型下拉框：`base / small / medium / large-v3`，当前默认 `medium`。
- ASR 语言下拉框：`中文 / English / Auto`。
- 索引阶段耗时展示。
- 搜索耗时展示。
- Cloudflare Tunnel 兼容。

前端结果卡片展示：

- 缩略图。
- 视频名。
- 命中时间段。
- modality chips。
- evidence detail。
- 点击后打开 modal 播放原视频，并跳到 `start_time`。

## 11. 当前性能观察

本地机器：NVIDIA GeForce RTX 3060 Laptop GPU，6GB；torch CUDA 可用。

示例完整任务曾观察到：

| 阶段 | 示例耗时 | 说明 |
| --- | --- | --- |
| Visual | 约 22.9s / 150s 视频 / 750 frames | CUDA，5fps |
| Face | 约 369.7s | CPU，当前最大瓶颈 |
| ASR | 约 133.3s | Whisper baseline |
| 总计 | 约 526s | Face 占主要耗时 |

另一个 ASR-only 历史任务使用 Whisper tiny 跑 117s 视频约 21.8s，但质量较差；当前前端已不再提供 tiny 作为常规选项。

结论：

- 搜索阶段通常很快，瓶颈在索引。
- 首次搜索/首次索引会加载模型，首个请求会慢。
- Face CPU 最慢，后续优化优先级高。
- ASR 需要做质量/耗时曲线：`base/small/medium/large-v3`。

## 12. GPU / NPU 策略

### 本地 3060

当前可用：

- CLIP visual：`CUDA_ENABLED=true` 时优先 CUDA。
- Whisper ASR：`ASR_DEVICE=cuda` 时可用 CUDA。
- Face：默认 CPU ONNXRuntime。

6GB 显存建议：

- CLIP ViT-B/32：通常可跑。
- Whisper base/small/medium：可试；large-v3 质量高但对 6GB 3060 压力大。
- Whisper large-v3：很可能吃紧，需观察或放到服务器验证。
- 不建议同时跑多个大模型进程。

### 服务器 NPU

项目设计原则：

- 默认 `NPU_ENABLED=false`。
- 只有确认无人使用的卡后，才开启指定 `NPU_DEVICE_ID`。
- 使用独立容器隔离。
- 索引阶段进程退出后释放显存。
- 不要让模型常驻共享 NPU。

粗略 NPU 显存估计：

- CLIP ViT-B/32：约 2–4GB HBM。
- Whisper base/small/medium/large-v3：显存差异较大，取决于模型大小和实现；large-v3 不建议在 6GB 卡上作为默认。
- Face 当前默认 CPU，不占 NPU；若启用 CANN provider 需重新测。

部署前务必：

```bash
npu-smi info
ps -ef | grep -E "python|uvicorn|torch|npu|ascend"
docker ps
```

并与同事确认卡号。

## 13. Git 与迁移

仓库：`https://github.com/502022240018/MomentSeek`。

推荐 Git 策略：

- Git 只放代码、配置模板、文档、测试。
- 不放：
  - `runtime/`
  - `models/`
  - 上传视频
  - 生成索引
  - 大模型权重
  - Cloudflare 临时工具

迁移到另一台机器：

```bash
git clone https://github.com/502022240018/MomentSeek.git
cd MomentSeek
cp .env.example .env
# 按机器修改 .env
docker compose -f compose.yml up -d --build
```

如果要迁移已有数据：

```text
拷贝 runtime/
拷贝 models/
保持 APP_DATA_DIR / APP_MODEL_DIR 指向正确
```

如果只迁移代码，不迁移数据：

- 视频需要重新上传。
- 索引需要重新建立。
- 人物库需要重新登记。

当前本地有未提交改动，建议交接前：

```bash
git status
git add .
git commit -m "Update MomentSeek ASR options and handoff docs"
git push
```

如果 GitHub 连接失败，本机曾使用过代理：

```bash
git config http.proxy http://127.0.0.1:7890
git config https.proxy http://127.0.0.1:7890
```

## 14. 测试与验证

后端测试：

```powershell
cd C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp\backend
C:\Users\29154\Projects\video-removal-system\prototype\.venv\Scripts\python.exe -m pytest tests
```

当前结果：`8 passed`。

前端构建：

```powershell
cd C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp\frontend
npm run build
```

当前结果：通过。

建议每次改动后至少跑：

- 后端 pytest。
- 前端 `npm run build`。
- 手工上传一个短视频，跑 Visual/ASR。
- 搜索一个确实出现在 ASR 文本中的词。
- 检查 `runtime/indexes/{video_id}` 里三个索引文件是否生成。

## 15. 已知问题与坑

### 15.1 ASR 质量

Whisper 小模型中文质量可能不稳，曾出现乱码、繁体、混合语言。现在前端已支持手动选 `small/medium + zh`，但旧索引需要重跑 ASR 才能改善转写内容。

### 15.2 ASR 搜不到不一定是搜索坏

如果 `asr.json` 里没有对应文字，lexical 搜索就不会返回。排查方式：

```powershell
Get-Content runtime\indexes\{video_id}\asr.json -Raw -Encoding UTF8
```

先搜 `asr.json` 里真实存在的词，确认链路是否正常。

### 15.3 PowerShell 中文编码

通过 PowerShell 管道测中文 API 时，可能把中文变成 `??`。建议测试脚本中用 unicode escape 或文件方式，避免误判。

### 15.4 Face 阶段慢

目前 Face 默认 CPU，长视频会很慢。MVP 阶段建议：

- 演示时可先只跑 Visual + ASR。
- Face fps 调低到 0.5–1。
- 后续优先验证 CANN/CUDA provider 或替换更轻量 pipeline。

### 15.5 前端没有鉴权

Cloudflare Tunnel 公网展示时，任何人拿到链接都可以操作系统。不适合放敏感视频。

### 15.6 运行时路径

之前出现过 API 和 worker 工作目录不同，导致相对路径 `runtime/...` 找不到。现在通过 `APP_DATA_DIR` / `APP_MODEL_DIR` 传绝对路径给 worker。后续新增子进程时也要继承这两个环境变量。

### 15.7 搜索返回的是时间片段，不是切片文件

当前没有实际切视频。返回结果是：

```text
media_url + start_time + end_time
```

如果后续需要导出片段，需要新增 ffmpeg clip export 功能。

## 16. 推荐下一步开发路线

优先级从高到低：

1. 整理当前本地改动，commit + push 到 GitHub。
2. 在前端加入“只重建 ASR / 只重建 Visual / 只重建 Face”的显式选项，而不是默认全跑。
3. 加一个 ASR 文本预览页：显示每个视频的 `asr.json` chunks，便于判断搜不到是 ASR 问题还是搜索问题。
4. 加搜索 profile 控件：`recall / balanced / precision`。
5. 给 ASR 加语义检索：
   - 文本 embedding；
   - 或 BM25；
   - 或拼音/编辑距离容错。
6. 对 Whisper `base/small/medium/large-v3` 做质量-耗时表。
7. 验证 FunASR/Paraformer 中文 ASR，尤其服务器/NPU部署可行性。
8. 优化 Face：
   - 降低 fps；
   - 批处理；
   - CANN/CUDA provider；
   - 或改用更轻量检测 + ArcFace。
9. 加 job cancel / cleanup 功能。
10. 加用户鉴权或最小访问密码，避免公网 tunnel 被误用。
11. 加导出片段功能：按 `start_time/end_time` 物理切出 mp4。
12. 把前端 `main.tsx` 拆成组件，降低继续开发成本。

## 17. 快速排查手册

### 后端是否活着

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### 看当前任务

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/jobs
```

### 看 ASR 索引内容

```powershell
Get-ChildItem runtime\indexes -Recurse -Filter asr.json
Get-Content runtime\indexes\{video_id}\asr.json -Raw -Encoding UTF8
```

### 看数据库数量

```powershell
python - <<'PY'
import sqlite3
conn = sqlite3.connect("runtime/catalog.sqlite3")
for table in ["videos", "jobs", "entities"]:
    print(table, conn.execute(f"select count(*) from {table}").fetchone()[0])
PY
```

### 搜 ASR API

```python
import urllib.request, urllib.parse, json
q = "白痴"
data = urllib.parse.urlencode({"query_text": q, "modalities": "asr", "limit": "5"}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/search",
    data=data,
    method="POST",
    headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
)
payload = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
print(payload["count"], payload["results"][:2])
```

### 看项目相关进程

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'video_retrieval_mvp|uvicorn|vite|cloudflared' } |
  Select-Object ProcessId,Name,CommandLine
```

## 18. 交接时一句话总结

这套系统当前是“可跑通的多模态视频片段检索 baseline”：  
Visual 用 CLIP 按时间桶建向量，Face 用 InsightFace 建人脸 track，ASR 用 Whisper/字幕建带时间戳文本；搜索时分别召回 candidates，再按时间合并成可播放片段。当前主要工程风险不是搜索慢，而是索引耗时和 ASR/Face 质量，需要下一阶段围绕模型选型、耗时记录、可解释调试和服务器隔离部署继续迭代。
