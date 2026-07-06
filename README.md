# MomentSeek 视频片段检索 MVP

面向私有视频素材的四通道检索 MVP：

- `visual_index`：SigLIP2 / CLIP 系列视觉 embedding，默认 5fps 抽帧、5 秒 bucket，按 bucket 内最大相似帧返回场景、物体和视觉语义命中。
- `face_index`：InsightFace / ArcFace，同一人物与明星出现片段。
- `asr_index`：Whisper baseline（保留 FunASR 适配器）+ 文本 embedding 语义索引，检索带时间戳的语音内容。
- `ocr_index`：RapidOCR + 文本 embedding 语义索引，检索画面中的字幕、招牌、标题和其他可读文字。

前端参考视频检索 Playground 的交互形态，后端提供 FastAPI 和 OpenAPI 文档。检索结果统一返回连续的 `start_time/end_time`、置信度、缩略图和命中证据。

## 关键设计

索引任务由常驻 API 启动，但每个模型阶段都运行在独立子进程中：

```text
API（不占 NPU）
  └─ 索引任务编排进程
       ├─ Visual 阶段进程 → 退出并释放显存
       ├─ Face 阶段进程 → 退出并释放显存
       ├─ ASR 阶段进程 → 退出并释放显存
       └─ OCR 阶段进程 → 退出并释放显存
```

已有向量的召回不需要 NPU。在线查询编码默认在 CPU 完成，因此空闲时 NPU 显存占用为零。

## 目录

```text
backend/app/       FastAPI、索引编排、模型适配器、检索融合
frontend/          React + TypeScript Playground
runtime/           上传视频、SQLite 元数据、索引与缩略图（不进 Git）
models/            SigLIP2/CLIP、InsightFace、ASR、OCR 权重（不进 Git）
compose.yml        CPU/无卡部署
compose.ascend.yml 昇腾单卡隔离覆盖配置
```

## 从 GitHub 拉取后快速验证

默认先用 `dev.cpu` 做 clean clone 验证；有 NVIDIA/CUDA 环境时再切换 `dev.cuda`。

Linux/macOS：

```bash
git clone https://github.com/502022240018/MomentSeek.git momentseek-mvp
cd momentseek-mvp
cp deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.sh dev.cpu --download
scripts/start_backend.sh
scripts/start_frontend.sh
```

Windows PowerShell：

```powershell
git clone https://github.com/502022240018/MomentSeek.git momentseek-mvp
cd momentseek-mvp
Copy-Item deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.ps1 -Profile dev.cpu -DownloadModels
scripts/start_backend.ps1
scripts/start_frontend.ps1
```

Windows、CUDA 开发和 smoke check 细节见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。打开 `http://127.0.0.1:8000`，API 文档位于 `/docs`。

## Docker 部署

CPU 模式（不会访问任何 NPU）：

```bash
cp deploy/env/dev.cpu.example .env
docker compose -f compose.yml up -d --build
```

当前 ARM64 昇腾服务器可先使用兼容运行时、但不映射 NPU：

```bash
docker compose -f compose.yml -f compose.server.yml up -d --build
```

昇腾模式必须在确认卡已释放后启动：

```bash
cp deploy/env/staging.ascend.example .env
# 编辑 .env：HOST_NPU_DEVICE_ID/ASCEND_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES 使用已获批的宿主物理卡号，NPU_DEVICE_ID 保持容器内逻辑 0。
./scripts/check_resource.sh
docker compose -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d --build
```

使用 `deploy/env/dev.cpu.example` 时默认访问地址是 `http://SERVER_IP:8000`；Ascend profile 当前示例端口是 `18300`。除 Web/API 入口外，容器不向宿主机暴露数据库或内部端口。

更完整的 GitHub 迁移、clone 后部署、运行时数据迁移和共享服务器资源检查流程见 [DEPLOY.md](DEPLOY.md) 和 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

最新共同开发入口和文档阅读顺序见 [docs/README.md](docs/README.md)。新会话启动 prompt 见 [docs/handoff/SESSION_BOOTSTRAP.md](docs/handoff/SESSION_BOOTSTRAP.md)。

历史交接记录保留在 [docs/archive/handoff/](docs/archive/handoff/)，只作为背景参考。

## 模型准备

模型不提交到 Git。当前约定：

1. 开发 profile 使用 `deploy/models/dev-full.models.json`，bootstrap 可自动下载校验脚本支持的 Hugging Face 小模型条目。
2. Ascend staging/prod 使用 `deploy/models/ascend-prod.models.json`，模型必须预缓存并挂载到容器内 `/app/models`。
3. visual / face / ASR / OCR 的模型、缓存和 lock 规则以 [docs/MODELS.md](docs/MODELS.md) 为准。

在新的昇腾服务器上，通过 `ASCEND_RUNTIME_IMAGE` 指向与驱动匹配的 ARM64 CANN/torch_npu 基础镜像，不需要修改业务代码。
当前服务器镜像额外固定 NumPy 1.26，以兼容 InsightFace 的 ONNXRuntime；ARM64 wheel 放在不入 Git 的 `vendor-wheels/`，部署前按 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 准备。

## 查询语义

- 参考图找人：参考图用于 ArcFace 身份向量，文字可作为场景约束。
- 文字找明星：先在“人物库”登记名称和参考正脸，之后文字中出现该名称即可调用 `face_index`。
- 文本/图片找场景物体：Visual 通道支持纯文本、纯图片和加权组合；检索默认返回独立短片段，不再把连续 5 秒桶一路合并成整段视频。
- 文本找语音：ASR 支持关键词/近似字符匹配 + 语义 embedding 检索；语义索引文件为 `asr_semantic.npz`。
- 文本找画面文字：OCR 支持画面文字语义检索；语义索引文件为 `ocr_semantic.npz`。

## MVP 验收

- 视频上传后可以分别建立 Visual、Face、ASR、OCR 索引。
- 四类查询都返回可播放的连续时间片段。
- 索引任务可查看阶段、进度和错误信息。
- 模型阶段结束后，`npu-smi info` 中不残留该任务的模型进程。
- 重启容器后资产、索引和人物库仍然存在。
- 在另一台服务器上可用 Git + Docker Compose 恢复部署。
