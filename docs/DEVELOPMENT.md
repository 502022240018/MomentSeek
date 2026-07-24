# 开发环境

## 目标

本文档说明 MomentSeek 本地开发环境的 profile 选择、依赖初始化、模型下载、前后端启动和基础验证流程。目标是让从 GitHub clone 下来的仓库可以在不接触线上服务器状态的前提下完成开发和 smoke check。

开发环境默认使用仓库内的示例 profile 生成 `.env`，并把运行时数据和模型缓存放在本地目录中。`.env` 是本机配置文件，不应提交；启动脚本不会主动覆盖已有 `.env`，如果需要切换 profile，应先人工确认差异后再替换。

## Profile 选择

当前开发 profile：

```text
dev.cpu：纯 CPU 开发，启动最稳，推荐作为 clean clone 默认入口。
dev.cuda：CUDA 开发，适合已经准备好 NVIDIA/CUDA/PyTorch CUDA 环境的本地机器。
```

示例文件位于：

```text
deploy/env/dev.cpu.example
deploy/env/dev.cuda.example
```

`dev.cpu` 和 `dev.cuda` 都使用 `deploy/models/dev-full.models.json`。`dev.cpu` 默认使用 `VISUAL_MODEL=chinese-clip-vit-b16`，用于 clean clone 最小验证；`dev.cuda` 默认使用 `VISUAL_MODEL=siglip2-so400m-384`，用于本地 GPU 演示或读取当前服务器迁移来的 SigLIP2 v3 索引。开发 profile 的必需校验项是 Hugging Face visual / semantic 模型；Face、SenseVoice/faster-whisper、RapidOCR 在 bootstrap 阶段不阻塞，但运行时只读取本地模型，缺失时会明确报错，不在索引任务中隐式下载。staging/prod 必须使用预缓存模型和锁文件校验，不能在运行时下载，也不能用 `scripts/bootstrap_dev.*` 准备环境。

`scripts/bootstrap_dev.*` 只接受 `dev.cpu` 和 `dev.cuda`。它们安装的是本地开发依赖，不安装 CANN/torch_npu，也不负责准备 Ascend staging/prod。`dev.cuda` 只表示运行配置允许 CUDA；如果需要 GPU 加速，先准备匹配本机驱动的 CUDA/PyTorch 环境，并用 `python -c "import torch; print(torch.cuda.is_available())"` 验证。

Docker GPU 演示或服务器 runtime 本地接管见 `docs/LOCAL_GPU_MIGRATION.md`。该流程把服务器数据同步到 `runtime-server/`，避免覆盖本地开发用 `runtime/`。

## 本地 Docker 开发模式

本地频繁调试时使用 `compose.dev.yml`，不要每次都运行发布式 `--build`。开发模式沿用已有镜像里的 Python/Node 依赖，但把宿主机源码挂进容器：

```text
backend/app -> /app/backend/app
frontend/dist -> /app/backend/app/static
```

后端 Python 修改会由 `uvicorn --reload` 自动重载；前端修改后在宿主机运行一次 `npm run build`，刷新浏览器即可看到新的静态文件。只有 requirements、Dockerfile、系统依赖或正式发布镜像变化时，才需要重新 `--build`。

CUDA 本地开发启动：

```powershell
docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml --env-file .env up -d --no-build app
```

CPU 本地开发启动：

```powershell
docker compose -f compose.yml -f compose.dev.yml --env-file .env up -d --no-build app
```

前端静态文件更新：

```powershell
Push-Location frontend; npm run build; Pop-Location
```

如果浏览器看起来还是旧界面，先确认本地容器是否使用了 `compose.dev.yml`，再确认 `frontend/dist` 的构建时间和浏览器缓存。

## Windows 快速启动

在仓库根目录运行。只有 `.env` 不存在时才复制示例文件；如果 `.env` 已存在，不要直接覆盖，先比较当前 `.env` 和目标 example，再手动合并需要的差异。

Bootstrap：

```powershell
if (-not (Test-Path .env)) { Copy-Item deploy/env/dev.cpu.example .env }
scripts/bootstrap_dev.ps1 -Profile dev.cpu -DownloadModels
```

如果本机已经准备好 CUDA 环境，可改用：

```powershell
if (-not (Test-Path .env)) { Copy-Item deploy/env/dev.cuda.example .env }
scripts/bootstrap_dev.ps1 -Profile dev.cuda -DownloadModels
```

终端 1 启动后端：

```powershell
scripts/start_backend.ps1
```

终端 2 启动前端：

```powershell
scripts/start_frontend.ps1
```

终端 3 等后端健康后运行 smoke check：

```powershell
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

需要手动创建 `.env` 时，保留的原始复制命令如下；仅在 `.env` 不存在或已经人工确认可替换时运行：

```powershell
Copy-Item deploy/env/dev.cpu.example .env
# 或者，在已经准备好 CUDA 环境时：
Copy-Item deploy/env/dev.cuda.example .env
```

## Linux 快速启动

在仓库根目录运行。只有 `.env` 不存在时才复制示例文件；如果 `.env` 已存在，不要直接覆盖，先比较当前 `.env` 和目标 example，再手动合并需要的差异。

Bootstrap：

```bash
[ -f .env ] || cp deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.sh dev.cpu --download
```

如果本机已经准备好 CUDA 环境，可改用：

```bash
[ -f .env ] || cp deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.sh dev.cuda --download
```

终端 1 启动后端：

```bash
scripts/start_backend.sh
```

终端 2 启动前端：

```bash
scripts/start_frontend.sh
```

终端 3 等后端健康后运行 smoke check：

```bash
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

需要手动创建 `.env` 时，保留的原始复制命令如下；仅在 `.env` 不存在或已经人工确认可替换时运行：

```bash
cp deploy/env/dev.cpu.example .env
# 或者，在已经准备好 CUDA 环境时：
cp deploy/env/dev.cuda.example .env
```

## 模型下载策略

开发 profile 的目标是降低新开发者启动成本，但当前 `scripts/verify_models.py --download` 只会下载 Hugging Face 模型条目。`dev-full.models.json` 因此只把 Hugging Face visual / semantic 条目标为必需；InsightFace、SenseVoice/faster-whisper 和 RapidOCR 在 bootstrap 阶段是可选校验项，缺失时不会阻塞 bootstrap。运行时只读取本地模型，缺失时会明确报错；不要依赖索引任务自动联网下载。显式下载入口是 bootstrap 脚本和 `scripts/verify_models.py --download`，模型清单来自 `MODEL_MANIFEST`：

```text
dev.cpu / dev.cuda -> deploy/models/dev-full.models.json
```

开发默认模型目录：

```text
models/
```

本地开发不要求每次启动都重新下载。bootstrap 会优先校验已有缓存，并写出 models lock；缺失 Hugging Face 模型时，只有显式传入 `-DownloadModels` 或 `--download` 才会尝试下载。ASR 默认是 `ASR_ENGINE=auto` + `ASR_MODEL=turbo` + `ASR_ZH_MODEL=iic/SenseVoiceSmall` + `ASR_VAD_STRATEGY=silero_12s`：先用 faster-whisper turbo 做轻量语言 probe，中文/方言走 SenseVoiceSmall，非中文走 faster-whisper turbo。SenseVoice 本地目录约定为 `models/funasr/iic/SenseVoiceSmall`，并需要 Python 依赖 `silero-vad`；faster-whisper turbo 本地 snapshot 目录约定为 `models/faster-whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`。`models/funasr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` 仍保留为 `ASR_VAD_STRATEGY=funasr_fsmn` fallback；`ct-punc` 用于非 SenseVoice FunASR 路径。其他模型条目按 `docs/MODELS.md` 的目录约定准备。

## 启动后端

Windows：

```powershell
scripts/start_backend.ps1
```

Linux：

```bash
scripts/start_backend.sh
```

默认后端地址：

```text
http://127.0.0.1:8000
```

开发启动脚本默认让后端绑定 `0.0.0.0`，因此同一网络中的其他机器可能访问到该端口。本地开发服务只应在可信网络中使用；如果需要更小暴露面，请调整 host 绑定或用防火墙限制访问。

启动前确认 `.env` 中的 `APP_PORT`、`APP_DATA_DIR`、`APP_MODEL_DIR` 和 `MODEL_MANIFEST` 符合当前开发 profile。

## 启动前端

Windows：

```powershell
scripts/start_frontend.ps1
```

Linux：

```bash
scripts/start_frontend.sh
```

前端开发服务器会使用本地后端 API。若前端出现 `failed to fetch`，先确认后端 `/api/health` 正常，再检查前端代理或 API base URL。

视频资产页的索引操作按通道执行：先勾选本次需要的 `Visual / Face / ASR / OCR`，再调整已选通道显示出的参数，最后点击对应视频的动作按钮。按钮会依据该视频的 `indexed_modalities` 显示“构建”“重建”或二者的组合；未选择通道的参数不会发送，已有索引也不会被删除。

## 验证命令

基础 smoke check：

```powershell
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

模型清单校验：

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json
```

后端测试：

```powershell
pytest
```

前端构建：

```powershell
Push-Location frontend; npm run build; Pop-Location
```

Linux/macOS：

```bash
(cd frontend && npm run build)
```

更多完成声明规则见 `docs/VALIDATION.md`。

## 常见问题

`.env` 已存在：不要直接覆盖。先比较 example 和当前 `.env`，尤其是端口、模型目录、runtime 目录和设备开关。

模型下载慢或失败：先确认当前 profile 是 `dev.cpu` 或 `dev.cuda`，并确认显式传入了 `-DownloadModels` 或 `--download`。当前自动下载只覆盖 Hugging Face 条目；InsightFace、SenseVoice/faster-whisper 和 RapidOCR 相关缓存不阻塞 bootstrap，但运行时只读取本地模型，缺失时应按 `docs/MODELS.md` 补齐缓存。线上 profile 不应打开运行时下载。

端口被占用：优先调整 `.env` 中的 `APP_PORT` 或前端开发端口。不要为了本地开发执行 broad kill。

smoke check 失败：先访问 `http://127.0.0.1:8000/api/health`，确认后端已启动且 profile、模型清单和设备状态符合预期。
