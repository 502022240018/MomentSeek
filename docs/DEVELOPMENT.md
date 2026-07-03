# 开发环境

## 目标

本文档说明 MomentSeek 本地开发环境的 profile 选择、依赖初始化、模型下载、前后端启动和基础验证流程。目标是让从 GitHub clone 下来的仓库可以在不接触线上服务器状态的前提下完成开发和 smoke check。

开发环境默认使用仓库内的示例 profile 生成 `.env`，并把运行时数据和模型缓存放在本地目录中。`.env` 是本机配置文件，不应提交；启动脚本不会主动覆盖已有 `.env`，如果需要切换 profile，应先人工确认差异后再替换。

## Profile 选择

当前开发 profile：

```text
dev.cpu：纯 CPU 开发，启动最稳，适合文档、API、前端和轻量 smoke check。
dev.cuda：CUDA 开发，适合有 NVIDIA GPU 的本地机器，默认开发推荐。
```

示例文件位于：

```text
deploy/env/dev.cpu.example
deploy/env/dev.cuda.example
```

`dev.cpu` 和 `dev.cuda` 都使用 `deploy/models/dev-full.models.json`，允许开发环境按需自动下载模型。自动下载只允许用于 dev profile；staging/prod 必须使用预缓存模型和锁文件校验，不能在运行时下载。

## Windows 快速启动

在仓库根目录运行：

```powershell
Copy-Item deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.ps1 -Profile dev.cuda -DownloadModels
scripts/start_backend.ps1
scripts/start_frontend.ps1
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

如果本机没有 CUDA，改用：

```powershell
Copy-Item deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.ps1 -Profile dev.cpu -DownloadModels
```

如果 `.env` 已存在，不要直接覆盖。先比较当前 `.env` 和目标 example，再决定是否手动合并。

## Linux 快速启动

在仓库根目录运行：

```bash
cp deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.sh dev.cuda --download
scripts/start_backend.sh
scripts/start_frontend.sh
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

如果本机没有 CUDA，改用：

```bash
cp deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.sh dev.cpu --download
```

如果 `.env` 已存在，不要直接覆盖。先比较当前 `.env` 和目标 example，再决定是否手动合并。

## 模型下载策略

开发 profile 可以自动下载模型，目的是降低新开发者启动成本。自动下载入口是 bootstrap 脚本和 `scripts/verify_models.py --download`，模型清单来自 `MODEL_MANIFEST`：

```text
dev.cpu / dev.cuda -> deploy/models/dev-full.models.json
```

开发默认模型目录：

```text
models/
```

本地开发不要求每次启动都重新下载。bootstrap 会优先校验已有缓存，并写出 models lock；缺失模型时，只有显式传入 `-DownloadModels` 或 `--download` 才会下载。

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
npm run build
```

更多完成声明规则见 `docs/VALIDATION.md`。

## 常见问题

`.env` 已存在：不要直接覆盖。先比较 example 和当前 `.env`，尤其是端口、模型目录、runtime 目录和设备开关。

模型下载慢或失败：先确认当前 profile 是 `dev.cpu` 或 `dev.cuda`，并确认显式传入了 `-DownloadModels` 或 `--download`。线上 profile 不应打开运行时下载。

端口被占用：优先调整 `.env` 中的 `APP_PORT` 或前端开发端口。不要为了本地开发执行 broad kill。

smoke check 失败：先访问 `http://127.0.0.1:8000/api/health`，确认后端已启动且 profile、模型清单和设备状态符合预期。
