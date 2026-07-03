# MomentSeek 多人开发与可复刻部署设计

日期：2026-07-03

## 背景

MomentSeek 当前已经具备 visual / face / ASR / OCR 四条检索通道，但开发和部署方式仍偏手工：

- 当前服务器 `momentseek-current-app` 使用 `momentseek-mvp:ascend` 镜像。
- 服务器通过 bind mount 挂载 backend、runtime、models。
- 后端核心运行文件与本地仓库基本匹配，但服务器不是 git checkout，前端源码与本地不完全一致。
- 模型目录和 runtime 目录位于服务器宿主机路径，不由 git 管理。
- 新人从 GitHub 拉取仓库后，缺少一条清晰的开发、验证、调试路径。

目标是把项目推进到适合多人协作的状态：GitHub 仓库本身包含足够说明、profile、脚本和 manifest，让其他开发者拉下来后可以启动完整功能进行开发验证，并让 staging / prod / 新服务器部署可复刻、可追溯。

## 目标

1. 新开发者从 GitHub clone 后，可以按文档完成开发环境 bootstrap。
2. Windows 和 Linux 都是一等开发入口。
3. 默认生产部署目标是 Ascend/NPU 服务器，同时支持 CUDA/GPU 开发和演示。
4. 开发环境允许自动下载小模型，尽量跑通完整 visual / face / ASR / OCR 功能。
5. staging / prod 不在运行时临时下载模型，必须提前准备并校验模型。
6. 每次部署都能回答：
   - 运行的是哪个 git commit。
   - 使用哪个前端构建产物。
   - 使用哪个 Docker image 或运行环境。
   - 使用哪个 env profile。
   - 使用哪个 model manifest。
   - 最近一次 health / smoke / resource 验证结果是什么。

## 非目标

第一阶段不强求完整 CI/CD、私有镜像仓库、自动 PR 部署或多机分布式索引。第一阶段先把开发入口、部署说明、manifest 和验证脚本打通；后续再把构建、发布和回滚自动化。

## 环境 Profile

项目维护四类 profile：

| Profile | 用途 | 默认硬件 | 模型策略 | 说明 |
|---|---|---|---|---|
| `dev.cpu` | 无 GPU 的本地开发和基础调试 | CPU | 可自动下载小模型 | 功能尽量完整，速度较慢 |
| `dev.cuda` | GPU 开发和演示 | NVIDIA CUDA | 可自动下载开发模型 | 开发者主要真实检索入口 |
| `staging.ascend` | 上线前验证 | Ascend NPU | 必须预缓存和校验 | 接近生产，先部署这里 |
| `prod.ascend` | 稳定演示/生产 | Ascend NPU | 必须预缓存和校验 | 严格禁止运行时下载 |

建议新增 env 示例：

```text
deploy/env/dev.cpu.example
deploy/env/dev.cuda.example
deploy/env/staging.ascend.example
deploy/env/prod.ascend.example
```

所有 profile 都统一容器内路径：

```text
APP_DATA_DIR=/app/runtime
APP_MODEL_DIR=/app/models
```

宿主机路径可按机器不同配置，但部署文档应建议使用统一布局：

```text
/opt/momentseek/
  releases/
  current -> releases/<release-id>
  runtime/
  models/
  env/
  logs/
  deployment-record.json
```

## 开发者体验

GitHub 新人入口应是：

```text
README.md
docs/DEVELOPMENT.md
docs/DEPLOYMENT.md
docs/MODELS.md
```

推荐启动流程：

```powershell
git clone <repo>
cd video_retrieval_mvp
Copy-Item deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.ps1
scripts/start_backend.ps1
scripts/start_frontend.ps1
```

Linux：

```bash
git clone <repo>
cd video_retrieval_mvp
cp deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.sh
scripts/start_backend.sh
scripts/start_frontend.sh
```

如果没有 GPU，则选择 `dev.cpu.example`。

`bootstrap_dev` 应完成：

1. 检查 Python、Node、ffmpeg 和可选 GPU runtime。
2. 创建 `runtime/` 和 `models/`。
3. 安装 backend 和 frontend 依赖。
4. 按 profile 准备模型；开发 profile 可自动下载。
5. 写入 `models/models.lock.json`。
6. 运行基础验证命令。
7. 输出下一步启动命令。

## 模型管理

模型不进入 git。git 只维护模型清单和校验脚本。

建议新增：

```text
deploy/models/dev-full.models.json
deploy/models/ascend-prod.models.json
scripts/verify_models.py
```

`dev-full.models.json` 面向开发者，允许自动下载，目标是完整功能可用：

```text
visual: 小型 OpenCLIP 或可选 SigLIP2
face: InsightFace buffalo_l
asr: Whisper base 或 small
ocr: RapidOCR PP-OCRv4 mobile
semantic: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

`ascend-prod.models.json` 面向 staging/prod，要求提前缓存并校验：

```text
visual: siglip2-so400m-384
face: InsightFace buffalo_l
asr: Whisper small
ocr: RapidOCR PP-OCRv4 mobile
semantic: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

运行后生成：

```text
models/models.lock.json
```

lock 文件记录：

```json
{
  "profile": "dev.cuda",
  "generated_at": "2026-07-03T00:00:00Z",
  "models": [
    {
      "name": "visual",
      "id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
      "source": "huggingface",
      "local_path": "models/openclip-vit-b32",
      "revision": "snapshot-or-version",
      "size_bytes": 0,
      "verified": true
    }
  ]
}
```

开发环境可以联网下载；staging/prod 的 `verify_models.py` 必须在模型缺失时失败，并提示预缓存步骤。

## Release Manifest

每次可部署版本生成一个 release manifest：

```text
deploy/releases/release.example.json
```

字段建议：

```json
{
  "release_id": "2026-07-03-ea4d676",
  "git_commit": "ea4d676d74ac657ea5f3df94f414d9ca6414cb3a",
  "branch": "main",
  "image": {
    "ascend": "momentseek-mvp:ascend-20260703-ea4d676",
    "cuda": "momentseek-mvp:cuda-20260703-ea4d676"
  },
  "frontend": {
    "build_command": "npm run build",
    "dist_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "mounted_to": "backend/app/static"
  },
  "models": {
    "manifest": "deploy/models/ascend-prod.models.json",
    "mount": "/app/models",
    "lock": "models/models.lock.json"
  },
  "runtime": {
    "mount": "/app/runtime",
    "migration": "none"
  },
  "env_profile": "staging.ascend",
  "verification": {
    "backend_tests": "required",
    "frontend_build": "required",
    "health": "required",
    "smoke_search": "required",
    "resource_check": "required"
  }
}
```

新增脚本：

```text
scripts/write_release_manifest.py
```

第一阶段可手动运行生成 manifest；第二阶段再接入 CI/CD。

## Deployment Record

每台服务器保存当前实际部署记录：

```text
/opt/momentseek/deployment-record.json
```

字段建议：

```json
{
  "server_id": "momentseek-prod-01",
  "environment": "prod.ascend",
  "release_id": "2026-07-03-ea4d676",
  "git_commit": "ea4d676d74ac657ea5f3df94f414d9ca6414cb3a",
  "image": "momentseek-mvp:ascend-20260703-ea4d676",
  "image_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "model_manifest": "deploy/models/ascend-prod.models.json",
  "model_lock": "models/models.lock.json",
  "env_profile": "prod.ascend",
  "deployed_at": "2026-07-03T00:00:00Z",
  "verified_at": "2026-07-03T00:00:00Z"
}
```

`/api/health` 应返回部署识别信息：

```text
app_version
git_commit
release_id
env_profile
image_tag
model_manifest
model_idle_policy
npu_enabled
npu_device_id
cuda_enabled
```

这样后续可以直接通过 health 判断服务器是否运行目标 release。

## 部署流程

### Staging / Prod

标准流程：

```text
feature branch
-> pull request / review
-> merge main
-> generate release manifest
-> deploy staging.ascend
-> run verification
-> promote prod.ascend
-> write deployment-record.json
```

staging 验证至少包括：

```text
backend tests
frontend build
model verification
/api/health
/api/jobs
small upload/index/search smoke
npu-smi resource check
```

prod 部署前必须确认 staging 使用同一 release manifest 通过验证。

### 新服务器复刻

新服务器不从现有服务器随意复制散乱目录开始，而是按 manifest 复刻：

```text
1. 准备 Docker、Ascend driver 或 CUDA runtime。
2. 创建 /opt/momentseek/ 标准目录。
3. 拉取 GitHub 仓库或 release artifact。
4. 准备 /opt/momentseek/models。
5. 运行 verify_models.py。
6. 使用指定 env profile 启动容器或服务。
7. 运行 smoke_check.py。
8. 写入 deployment-record.json。
```

## 脚本设计

第一阶段新增脚本：

```text
scripts/bootstrap_dev.ps1
scripts/bootstrap_dev.sh
scripts/start_backend.ps1
scripts/start_backend.sh
scripts/start_frontend.ps1
scripts/start_frontend.sh
scripts/verify_models.py
scripts/smoke_check.py
scripts/write_release_manifest.py
```

脚本职责：

| 脚本 | 职责 |
|---|---|
| `bootstrap_dev.*` | 开发环境准备、依赖安装、模型下载/校验、基础验证 |
| `start_backend.*` | 使用当前 `.env` 启动 FastAPI |
| `start_frontend.*` | 启动 Vite dev server |
| `verify_models.py` | 根据 model manifest 校验或下载模型 |
| `smoke_check.py` | health、jobs、可选上传/搜索 smoke |
| `write_release_manifest.py` | 生成 release manifest |

Windows 脚本应避免 PowerShell 路径和编码坑：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

Linux 脚本使用：

```bash
set -euo pipefail
```

## 文档设计

新增长期维护文档：

```text
docs/DEVELOPMENT.md
docs/DEPLOYMENT.md
docs/MODELS.md
```

同时更新：

```text
docs/README.md
docs/CURRENT.md
docs/ARCHITECTURE.md
docs/OPERATIONS.md
docs/VALIDATION.md
docs/ISSUES_AND_ROADMAP.md
```

信息归属：

| 内容 | 维护位置 |
|---|---|
| 新人开发入口 | `docs/DEVELOPMENT.md` |
| staging/prod/新服务器部署 | `docs/DEPLOYMENT.md` |
| 模型清单和缓存策略 | `docs/MODELS.md` |
| 当前服务器实际状态 | `docs/CURRENT.md` |
| 运维安全和共享服务器规则 | `docs/OPERATIONS.md` |
| 验证命令 | `docs/VALIDATION.md` |
| 未来 CI/CD、Docker 化和部署自动化问题 | `docs/ISSUES_AND_ROADMAP.md` |

## GitHub 协作流程

推荐多人流程：

```text
1. 每个人从 GitHub clone。
2. 本地选择 dev.cpu 或 dev.cuda。
3. 用 bootstrap 脚本准备完整开发环境。
4. 功能分支开发。
5. PR 前运行 backend tests、frontend build、smoke_check。
6. 合并 main 后生成 release manifest。
7. 部署 staging.ascend。
8. staging 验证通过后 promote prod.ascend。
```

分支建议：

```text
main: 可发布主线
feature/*: 功能开发
fix/*: 修复
docs/*: 文档
release/*: 可选，第二阶段再引入
```

## 第一阶段实施范围

第一阶段交付：

1. 新增 `docs/DEVELOPMENT.md`、`docs/DEPLOYMENT.md`、`docs/MODELS.md`。
2. 新增 `deploy/env/*.example`。
3. 新增 `deploy/models/*.json` 和 `deploy/releases/release.example.json`。
4. 新增开发 bootstrap / start 脚本的最小可用版本。
5. 新增 `verify_models.py`、`smoke_check.py`、`write_release_manifest.py`。
6. 扩展 `/api/health`，返回 release / git / env / model manifest 信息。
7. 更新 `docs/README.md` 的阅读顺序和更新规则。
8. 更新问题池，记录第二阶段 CI/CD、Dockerfile/compose、自动发布、回滚。

第一阶段完成后，其他人应能：

```text
git clone
选择 dev.cpu 或 dev.cuda
运行 bootstrap
启动 backend / frontend
上传小视频
建立索引
运行搜索
执行 smoke_check
```

## 第二阶段候选范围

第二阶段再考虑：

- Dockerfile 和 docker compose 标准化。
- Ascend / CUDA 镜像构建流程。
- GitHub Actions 构建和验证。
- Release artifact 打包。
- staging/prod 自动部署。
- 自动回滚和多版本保留。
- 模型下载镜像源或内部对象存储。

## 风险与约束

- Ascend 服务器是共享环境，部署或重启前必须遵循 `docs/OPERATIONS.md`。
- staging/prod 不允许运行时下载模型，否则索引任务会受网络问题影响并卡住。
- CUDA demo 不保证与 Ascend 的性能和 provider 行为完全一致。
- dev.cpu 能跑完整功能，但性能会明显变慢，不作为检索性能基准。
- 当前服务器前端源码与本地不完全一致，实施时需要先统一 GitHub 主线和服务器部署来源。

## 验收标准

第一阶段验收：

1. Windows 开发者能按 `docs/DEVELOPMENT.md` 完成 dev.cpu 或 dev.cuda bootstrap。
2. Linux 开发者能按 `docs/DEVELOPMENT.md` 完成 dev.cpu 或 dev.cuda bootstrap。
3. `scripts/verify_models.py` 能按 model manifest 校验或下载开发模型。
4. `scripts/smoke_check.py` 能检查 `/api/health` 和基础 API。
5. `/api/health` 能返回 release / git / env / model manifest 信息。
6. `docs/DEPLOYMENT.md` 能指导 staging.ascend 和新服务器复刻部署。
7. `docs/MODELS.md` 清楚说明模型目录、下载策略、预缓存策略和线上禁止运行时下载规则。
8. GitHub README 或 docs 入口能让新同学找到开发、部署、模型文档。
