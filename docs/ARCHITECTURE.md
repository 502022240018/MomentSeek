# MomentSeek 平台架构

> 本文是平台交付仓的权威架构文档，对应 `backend/app` 重构后的新目录结构。
> 接口明细以运行实例的 `/docs`（OpenAPI）为准；部署流程见
> [DEPLOYMENT_ASCEND.md](DEPLOYMENT_ASCEND.md)。

## 1. 一句话定位

MomentSeek 是面向私有视频素材的多通道视频片段检索平台：视频上传后按
**visual / face / asr / speaker / ocr** 五条通道建立索引，用户用文字、图片或
声音查询，平台返回可播放的连续时间片段及证据。

## 2. 系统拓扑

```text
┌──────────────┐   HTTP    ┌─────────────────────────────┐
│ React 前端    │ ────────► │ FastAPI 后端 (app.main)      │
│ frontend/     │           │  api/ 路由层                 │
└──────────────┘           └──────┬──────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
┌───────────────┐        ┌────────────────┐         ┌────────────────┐
│ SQLite catalog │        │ runtime/ 文件区 │         │ Milvus 向量库   │
│ (视频/任务/    │        │ uploads 索引NPZ │         │ (etcd + MinIO)  │
│  人物/声纹)    │        │ 帧缓存 clip缓存 │         │ 五通道 collection│
└───────────────┘        └────────────────┘         └────────────────┘
                                  ▲
                     索引阶段子进程（用完即退，释放 NPU）
                     visual → face → asr → speaker → ocr
```

关键设计：**API 常驻进程不占 NPU**。每个索引阶段运行在独立子进程中，阶段结束
进程退出并释放显存；在线查询编码默认走 CPU，空闲时 NPU 占用为零。可选的
daemon 模式（`INDEXER_MODE=daemon`）用常驻 warm-pool 换取免模型重载。

## 3. 仓库目录

```text
├─ backend/
│  ├─ app/                  后端平台代码（见第 4 节）
│  ├─ requirements/         依赖锁：ascend.txt / ci.txt / dev.txt 等
│  └─ tests/                单元测试；tests/integration 为 Milvus 集成测试
├─ frontend/                React + TypeScript Web 前端
├─ compose/                 compose.yml / compose.ascend.yml / compose.milvus.yml
├─ docker/                  Dockerfile.ascend（应用镜像）
├─ deploy/
│  ├─ env/                  环境参数模板
│  ├─ models/               必需模型清单（ascend.models.json）
│  └─ orchestration/        可选 LLM 规划/重排的 provider 与 prompt 配置
├─ scripts/                 preflight.py / verify_models.py / smoke_check.py
├─ vendor-wheels/           Ascend ARM64 离线固定 wheels
└─ docs/                    平台文档（本文、部署、镜像制作、验证）
```

## 4. backend/app 模块分层

```text
backend/app/
├─ main.py                  FastAPI 组装入口：建 app、挂路由、生命周期管理
├─ api/                     API 接口层：路由 + 请求/响应模型(schemas.py)
│     system / video / job / entity / search / speaker / color_grading
├─ core/                    全局基础能力（叶子层，不依赖任何上层）
│     settings.py           全部环境配置（pydantic-settings，.env 驱动）
│     deployment.py         release/部署元信息（/api/health 返回）
│     model_pool.py         通用模型池（warm-pool 复用）
│     model_sources.py      模型来源与本地缓存守卫（禁止运行时隐式下载）
├─ catalog/                 资产元数据层
│     db.py                 SQLite：videos / jobs / entities / speakers / 仿色任务
├─ media/                   媒体处理层（ffmpeg 封装）
│     media.py              探测、抽帧、抽音频、缩略图、预览 clip
├─ indexing/                索引生产层（离线）
│     stage_executor.py     单阶段执行入口：锁 → 建索引 → 写 Milvus → 写 manifest
│     common / manifest / pipeline_manifest / text_semantic
│     └─ modalities/        每条检索通道一个子包（通道名与 API modalities 一致）
│         visual/  face/  asr/  ocr/  speaker/
├─ retrieval/               在线检索层
│     search.py             SearchEngine：五通道召回、阈值判定、时间段融合
│     retrieval_metrics.py  检索性能剖析（RetrievalProfiler）
├─ orchestration/           检索编排层（在 retrieval 之上）
│     retrieval_orchestration.py  LLM planner/reranker（OpenAI 兼容 provider）
├─ vector_store/            向量存储层（纯基础设施）
│     └─ milvus/            client / schema / indexer / search / flags / 锁 / 版本
├─ execution/               后台执行层
│     worker.py             每任务子进程编排（subprocess 模式）
│     stage_runner.py       阶段子进程 CLI 入口（python -m app.execution.stage_runner）
│     indexer_daemon.py     daemon 模式的常驻队列消费者
│     isolated_stage_workers.py  daemon 模式下按阶段隔离的常驻 worker 池
├─ identity/                实体身份层
│     speaker_service.py    说话人视图、声纹检索（voice search）
├─ integrations/            外部系统集成
│     color_grading.py      视频仿色服务（独立容器，经 APP_DATA_DIR 交换文件）
├─ maintenance/             离线维护任务
│     speaker_backfill.py   speaker 索引补建
├─ platform/                预留：运行时上下文（见 README，含 context 化改造计划）
├─ observability/           预留：日志/指标/追踪（见 README）
└─ evaluation/              预留：随平台发布的质量回归（见 README）
```

### 分层依赖规则（已按代码实测核验）

```text
api ──► main（运行时单例）
main ──► retrieval / orchestration / identity / integrations / catalog / core / media / execution
orchestration ──► retrieval / media / catalog / core
retrieval ──► catalog / core / indexing(共用编码器) / vector_store
execution ──► indexing.stage_executor / core / catalog
indexing ──► modalities / vector_store / core / media
identity ──► catalog / modalities.speaker / vector_store
vector_store ──► core.settings（叶子层）
core / catalog / media ──► 不依赖上层（叶子层）
```

约定：**下层永远不 import 上层**；同层之间除标注的共用外不互相依赖。

## 5. 数据流

### 索引（写路径）

```text
POST /api/videos 上传
  → catalog 记录 + runtime/uploads 存原片
POST /api/videos/{id}/index  (modalities 子集，可增量/重建)
  → catalog 创建 job
  → execution.worker（或 daemon）逐阶段派发子进程
  → indexing.stage_executor：
       media 解码抽帧/抽音频
       modalities/<通道> 编码
       vector_store.milvus 直写（P2 内存路径）+ NPZ 落盘
       pipeline_manifest 写通道 manifest
  → 阶段进程退出，释放 NPU
```

### 检索（读路径）

```text
POST /api/search (文字/图片, modalities, 可选 orchestration profile)
  → orchestration：可选 LLM planner 决定通道与参数（失败自动回退默认计划）
  → retrieval.SearchEngine：
       编码查询（CPU）→ 各通道独立召回（Milvus 优先，NPZ 兜底）
       → 阈值/证据判定 → 按时间邻近融合成片段
  → 可选 LLM reranker 重排
  → 返回 start/end、置信度、证据、缩略图/clip URL（实时抽帧，磁盘缓存）
```

## 6. 存储布局

```text
runtime/                     （容器内 APP_DATA_DIR，不进 Git）
  catalog.sqlite3            视频/任务/人物/声纹/仿色任务
  uploads/                   原始视频与字幕 sidecar
  indexes/{video_id}/        各通道 NPZ + index manifest（Milvus 的恢复兜底）
  frame_cache/  clips/       检索命中的实时抽帧与预览片段缓存
models/ → 容器内 /app/models  模型权重（部署前预缓存，运行时禁止下载）
Milvus                       五通道向量 collection（模型版本参与主键/schema）
```

## 7. API Surface（按路由模块）

| 路由模块 | 前缀/代表接口 | 职责 |
|---|---|---|
| `system_routes` | `GET /api/health` | 健康、设备状态、部署元信息 |
| `video_routes` | `POST/GET /api/videos*`、`/media`、`/clip`、`/frame`、`POST .../index` | 视频资产、播放、抽帧、索引任务创建 |
| `job_routes` | `GET /api/jobs*`、`POST .../cancel` | 索引任务查询与取消 |
| `entity_routes` | `/api/entities*`（含 voice-samples） | 人物库：参考脸、声纹样本 |
| `search_routes` | `POST /api/search`、`GET /api/orchestration/profiles` | 多模态检索与编排配置 |
| `speaker_routes` | `/api/videos/{id}/speakers*`、`POST /api/voice-search*` | 说话人视图、声音检索 |
| `color_grading_routes` | `/api/color-grading/*` | 视频仿色任务（可选能力） |

## 8. 扩展指南

### 新增一条检索通道（如新 embedding 模型通道）

1. `indexing/modalities/<name>/` 新建子包，实现 `build_<name>_index`（参照 visual）；
2. `indexing/stage_executor.py` 注册 `_run_<name>` 阶段；`execution/stage_runner.py` 的
   stage choices 加入通道名；
3. `vector_store/milvus/milvus_schema.py` 定义 collection schema 与主键，
   `milvus_indexer.py` 加对应 indexer；
4. `retrieval/search.py` 增加 `_<name>_candidates` 召回与融合权重；
5. `api/schemas.py` 的 modalities 校验、前端 `indexing.tsx` 通道选择同步扩展。

### 新增一个功能模块（如仿色这类外挂能力）

1. 服务逻辑放 `integrations/<name>.py`（外部系统）或独立顶层包（平台内能力）；
2. 路由放 `api/<name>_routes.py`，在 `main.py` 的路由列表注册；
3. 配置项进 `core/settings.py`（带 `<name>_enabled` 开关，默认关闭不影响主链路）；
4. 前端独立页面组件（参照 `ColorGradingPage.tsx`）。

## 9. 技术债登记（既定优化方向）

| 项 | 现状 | 计划 |
|---|---|---|
| 运行时单例 context 化 | 路由用函数内 `from app import main as runtime` 惰性取单例（约 38 处） | 建 `platform/context.py` 承载单例，main 只做组装；见 `app/platform/README.md` |
| 共享编码器抽层 | `retrieval/search.py` 从 `indexing.modalities` 借用 ClipEncoder 等 | 抽独立 encoders 层，检索/索引共同依赖 |
| 拆分 `retrieval/search.py` | 单文件约 77KB，五通道召回+融合混在一处 | 按通道拆分并抽出 Candidate/SearchResult 类型 |
| 拆分前端 `main.tsx` | 单文件约 34KB | 按 upload/search/assets/player 拆组件 |

以上均为结构优化，不改变行为；每项单独改造、单独回归验证。
