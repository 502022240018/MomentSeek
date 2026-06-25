# MomentSeek 视频片段检索 MVP

面向私有视频素材的三索引 baseline：

- `face_index`：InsightFace / ArcFace，同一人物与明星出现片段。
- `visual_index`：OpenCLIP，默认 5fps 抽帧、5 秒逻辑分段，文本或参考图检索场景、物体和视觉语义。
- `asr_index`：Whisper baseline（保留 FunASR 适配器），检索带时间戳的语音内容。

前端参考视频检索 Playground 的交互形态，后端提供 FastAPI 和 OpenAPI 文档。检索结果统一返回连续的 `start_time/end_time`、置信度、缩略图和命中证据。

## 关键设计

索引任务由常驻 API 启动，但每个模型阶段都运行在独立子进程中：

```text
API（不占 NPU）
  └─ 索引任务编排进程
       ├─ CLIP 阶段进程 → 退出并释放显存
       ├─ ArcFace 阶段进程 → 退出并释放显存
       └─ ASR 阶段进程 → 退出并释放显存
```

已有向量的召回不需要 NPU。在线查询编码默认在 CPU 完成，因此空闲时 NPU 显存占用为零。

## 目录

```text
backend/app/       FastAPI、索引编排、模型适配器、检索融合
frontend/          React + TypeScript Playground
runtime/           上传视频、SQLite 元数据、索引与缩略图（不进 Git）
models/            CLIP、InsightFace、ASR 权重（不进 Git）
compose.yml        CPU/无卡部署
compose.ascend.yml 昇腾单卡隔离覆盖配置
```

## 本地开发

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-cpu.txt
cd frontend && npm install && npm run build && cd ..
cp -r frontend/dist backend/app/static
cd backend && uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000`，API 文档位于 `/docs`。

## Docker 部署

CPU 模式（不会访问任何 NPU）：

```bash
cp .env.example .env
docker compose -f compose.yml up -d --build
```

当前 ARM64 昇腾服务器可先使用兼容运行时、但不映射 NPU：

```bash
docker compose -f compose.yml -f compose.server.yml up -d --build
```

昇腾模式必须在确认卡已释放后启动：

```bash
./scripts/check_resource.sh
NPU_DEVICE_ID=7 docker compose -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d --build
```

默认访问地址是 `http://SERVER_IP:8300`。除这个入口外，容器不向宿主机暴露数据库或内部端口。

更完整的 GitHub 迁移、clone 后部署、运行时数据迁移和共享服务器资源检查流程见 [DEPLOY.md](DEPLOY.md)。

共同开发交接、pipeline 细节、数据结构、已知问题和后续路线见 [docs/HANDOFF.md](docs/HANDOFF.md)。

## 模型准备

模型不提交到 Git。当前 baseline 约定：

1. 将 OpenCLIP 权重放到 `models/ViT-B-32.openai.bin`。
2. InsightFace `buffalo_l` 首次使用时下载到 `models/insightface/`。
3. MVP 当前使用 Whisper baseline，前端可选 `base / small / medium / large-v3` 和 `zh / en / auto`；也可以上传 JSON/SRT/VTT 字幕绕过 ASR 推理。FunASR 通过适配器保留为下一步的昇腾优化项。

在新的昇腾服务器上，通过 `ASCEND_RUNTIME_IMAGE` 指向与驱动匹配的 ARM64 CANN/torch_npu 基础镜像，不需要修改业务代码。
当前服务器镜像额外固定 NumPy 1.26，以兼容 InsightFace 的 ONNXRuntime；ARM64 wheel 放在不入 Git 的 `vendor-wheels/`，部署脚本可重新下载。

## 查询语义

- 参考图找人：参考图用于 ArcFace 身份向量，文字可作为场景约束。
- 文字找明星：先在“人物库”登记名称和参考正脸，之后文字中出现该名称即可调用 `face_index`。
- 文本/图片找场景物体：CLIP 支持纯文本、纯图片和加权组合；Visual 检索默认返回独立短片段，不再把连续 5 秒桶一路合并成整段视频。
- 文本找语音：ASR 首先支持关键词和近似字符匹配；语义文本 embedding 是后续升级项。

## MVP 验收

- 视频上传后可以分别建立 Face、Visual、ASR 索引。
- 三类查询都返回可播放的连续时间片段。
- 索引任务可查看阶段、进度和错误信息。
- 模型阶段结束后，`npu-smi info` 中不残留该任务的模型进程。
- 重启容器后资产、索引和人物库仍然存在。
- 在另一台服务器上可用 Git + Docker Compose 恢复部署。
