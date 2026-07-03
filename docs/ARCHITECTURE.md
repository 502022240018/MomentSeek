# MomentSeek 架构

## 总览

MomentSeek 是一个单机文件索引的视频检索 MVP：

```text
React/Vite 前端
  -> FastAPI 后端
  -> SQLite catalog
  -> runtime 文件资产和索引
  -> 模型索引/检索 pipeline
```

主要 runtime 结构：

```text
runtime/
  catalog.sqlite3
  uploads/
  indexes/{video_id}/
  thumbnails/{video_id}/
  clips/{video_id}/
```

当前 MVP 把索引保存在本地 `.npz` 和 `.json` 文件中。后续如果扩展到多机或百万级片段，可以在保持 API 概念不变的前提下，把存储层替换为 pgvector、Milvus、Qdrant 等系统。

## 数据流

```text
上传视频
  -> 写入 SQLite videos
  -> 保存源视频到 runtime/uploads
  -> 创建索引任务
  -> worker / stage_runner 构建所选通道索引
  -> search 加载索引并返回时间段、证据、缩略图、媒体和 clip URL
```

索引通道：

```text
visual -> 抽帧 -> SigLIP2/CLIP embeddings -> visual.npz
face   -> 抽帧 -> InsightFace/ArcFace tracks -> faces.npz
asr    -> 音频 -> Whisper chunks -> asr.json + 可选 asr_semantic.npz
ocr    -> 抽帧 -> RapidOCR chunks -> ocr.json + 可选 ocr_semantic.npz
```

搜索时，各通道先独立产生 candidates，然后按时间重叠或邻近关系合并为最终可播放片段。

## 模型生命周期

默认设计是 API 进程不长期加载重型索引模型。索引阶段在子进程中运行，阶段结束后释放 NPU 上下文和显存。

当前服务器 health 显示：

```text
model_idle_policy = process_exit
```

仓库里已有 warm model pool 和 indexer daemon 相关代码：

```text
backend/app/model_pool.py
backend/app/indexer_daemon.py
```

是否启用它们是部署决策，因为它会改变显存常驻和队列行为。相关性能/资源问题记录在 `docs/ISSUES_AND_ROADMAP.md`。

## 后端模块

| 路径 | 职责 |
|---|---|
| `backend/app/main.py` | FastAPI app、API 路由、静态前端挂载 |
| `backend/app/schemas.py` | 请求/响应校验模型 |
| `backend/app/db.py` | SQLite catalog，管理 videos / jobs / entities |
| `backend/app/worker.py` | 索引任务编排和 worker lock |
| `backend/app/stage_runner.py` | 各通道索引子进程入口 |
| `backend/app/search.py` | visual / face / ASR / OCR 召回和结果融合 |
| `backend/app/media.py` | 视频探测、抽帧、抽音频、缩略图、clip |
| `backend/app/indexing/visual.py` | visual encoder 和 visual 索引构建 |
| `backend/app/indexing/faces.py` | face encoder 和人脸 track 索引构建 |
| `backend/app/indexing/asr.py` | Whisper / FunASR / 字幕 ASR 索引构建 |
| `backend/app/indexing/ocr.py` | RapidOCR 索引构建 |
| `backend/app/indexing/text_semantic.py` | ASR/OCR semantic text embedding |

## 前端模块

| 路径 | 职责 |
|---|---|
| `frontend/src/main.tsx` | 上传、建索引、搜索、播放、素材管理等主 UI |
| `frontend/src/api.ts` | 前端 API client 和 TypeScript 类型 |
| `frontend/src/styles.css` | 样式 |

`main.tsx` 当前较大，后续应按 upload/indexing、search、assets、player、shared controls 等职责拆分。该事项记录在 `docs/ISSUES_AND_ROADMAP.md`。

## 部署元信息

`/api/health` 除健康和设备状态外，也返回部署元信息，便于确认当前服务是否与 release manifest 一致。关键字段包括：

```text
env_profile
release_id
git_commit
image_tag
model_manifest
```

staging/prod 验证时应把这些字段与 `docs/DEPLOYMENT.md` 中的 release manifest 和 deployment record 对齐。

## API Surface

运行中的 FastAPI 后端可通过 `/docs` 和 `/openapi.json` 查看自动文档。本表维护“接口到代码/前端调用”的人工索引。

| Method | Path | 后端函数 | 功能 | 前端调用 | 相关模块 |
|---|---|---|---|---|---|
| `GET` | `/api/health` | `main.py::health` | 健康检查、设备状态和部署元信息 | smoke check | `settings.py`, `schemas.py`, `deployment.py` |
| `POST` | `/api/videos` | `main.py::upload_video` | 上传视频和可选字幕 | `api.ts::uploadVideo` | `media.py`, `db.py`, `runtime/uploads` |
| `GET` | `/api/videos` | `main.py::list_videos` | 视频列表 | `api.ts::videos` | `db.py` |
| `GET` | `/api/videos/{video_id}` | `main.py::get_video` | 视频详情和任务 | 详情/轮询流程 | `db.py` |
| `PATCH` | `/api/videos/{video_id}` | `main.py::rename_video` | 重命名视频 | `api.ts::renameVideo` | `schemas.py`, `db.py` |
| `DELETE` | `/api/videos/{video_id}` | `main.py::delete_video` | 删除视频、索引、缩略图、任务 | `api.ts::deleteVideo` | `db.py`, runtime 清理 |
| `GET` | `/api/videos/{video_id}/media` | `main.py::video_media` | 原视频流式播放 | 播放器/结果卡 | `runtime/uploads` |
| `GET` | `/api/videos/{video_id}/clip` | `main.py::video_clip` | 生成/返回命中时间段 clip | 播放器/结果卡 | `media.py`, `runtime/clips` |
| `POST` | `/api/videos/{video_id}/index` | `main.py::create_index_job` | 创建索引任务 | `api.ts::indexVideo` | `schemas.py`, `db.py`, `worker.py` |
| `GET` | `/api/jobs` | `main.py::list_jobs` | 任务列表，可按 video 过滤 | `api.ts::jobs` | `db.py` |
| `GET` | `/api/jobs/{job_id}` | `main.py::get_job` | 任务详情、失败信息、日志上下文 | UI 轮询/详情 | `db.py` |
| `POST` | `/api/entities` | `main.py::create_entity` | 登记人物参考图 | `api.ts::createEntity` | `faces.py`, `db.py`, `runtime/entities` |
| `GET` | `/api/entities` | `main.py::list_entities` | 人物库列表 | `api.ts::entities` | `db.py` |
| `GET` | `/api/entities/{entity_id}/reference` | `main.py::entity_reference` | 返回人物参考图 | entity UI | `runtime/entities` |
| `POST` | `/api/search` | `main.py::search` | 多模态搜索 | `api.ts::search` | `search.py`, indexes, thumbnails/clips |
| `GET` | `/api/thumbnails/{video_id}/{filename}` | `main.py::thumbnail` | 返回缩略图 | 结果卡 | `runtime/thumbnails` |
| `GET` | `/` | `main.py::root` | 返回前端入口 | 浏览器 | frontend build |
| `GET` | `/{path:path}` | `main.py::frontend` | 返回前端路由/静态资源 | 浏览器 | frontend build |

## 请求模型

重要请求模型在 `backend/app/schemas.py`：

- `IndexRequest`：`modalities`、visual 参数、face fps、OCR fps、ASR 模型和语言。
- `VideoRenameRequest`：校验视频名称。
- `HealthResponse`：health 响应结构。

搜索接口 `main.py::search` 使用 `multipart/form-data`，字段包括：

```text
query_text
query_image
modalities
video_ids
alpha
limit
```
