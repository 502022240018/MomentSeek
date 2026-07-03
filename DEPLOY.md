# MomentSeek MVP 迁移与部署入口

本文件是兼容入口。后续权威规则维护在 `docs/`：

- 本地开发、clean clone、启动脚本：`docs/DEVELOPMENT.md`
- 模型 manifest、缓存、lock：`docs/MODELS.md`
- staging/prod/new-server 部署：`docs/DEPLOYMENT.md`
- 共享服务器安全操作：`docs/OPERATIONS.md`
- 验证和完成声明：`docs/VALIDATION.md`

## GitHub 仓库边界

应该提交代码、脚本、Docker/Compose 配置、`deploy/env/*.example`、`deploy/models/*.json`、`docs/` 和 `samples/`。

不应该提交：

```text
.env
runtime/
models/
vendor-wheels/
frontend/node_modules/
frontend/dist/
backend/app/static/
.pytest_cache/
__pycache__/
```

上传前检查：

```bash
git status --short
git status --ignored --short
```

## Clean Clone 本地验证

默认先走 CPU profile，目标是让新同学拉下仓库后能开发、调试、做最小真实检索验证。

Linux/macOS：

```bash
git clone git@github.com:YOUR_NAME/momentseek-mvp.git
cd momentseek-mvp
[ -f .env ] || cp deploy/env/dev.cpu.example .env
scripts/bootstrap_dev.sh dev.cpu --download
scripts/start_backend.sh
scripts/start_frontend.sh
```

Windows：

```powershell
git clone git@github.com:YOUR_NAME/momentseek-mvp.git
cd momentseek-mvp
if (-not (Test-Path .env)) { Copy-Item deploy/env/dev.cpu.example .env }
scripts/bootstrap_dev.ps1 -Profile dev.cpu -DownloadModels
scripts/start_backend.ps1
scripts/start_frontend.ps1
```

后端默认地址是 `http://127.0.0.1:8000`，API 文档在 `/docs`。更多 profile 选择和 smoke check 见 `docs/DEVELOPMENT.md`。

## Docker CPU / 无卡部署

CPU / 无卡模式不会映射 NPU/GPU 设备，适合新服务器冒烟或共享服务器资源未确认时使用。

```bash
cp deploy/env/dev.cpu.example .env
docker compose -f compose.yml up -d --build
curl http://127.0.0.1:8000/api/health
```

远程访问时按实际情况修改 `.env` 中的 `APP_PUBLIC_URL` 和 `APP_PORT`。

## Ascend / NPU 部署

不要用 `scripts/bootstrap_dev.*` 准备 staging/prod；这些脚本只接受 `dev.cpu` 和 `dev.cuda`。

Ascend staging/prod 以 release manifest 为中心：

1. 先按 `docs/OPERATIONS.md` 做只读资源检查，确认没有 active indexing jobs，且只操作 MomentSeek 容器/进程。
2. 准备 `deploy/env/staging.ascend.example` 或 `deploy/env/prod.ascend.example` 派生的 `.env`。
3. 设置 `HOST_RUNTIME_DIR=/opt/momentseek/runtime` 和 `HOST_MODEL_DIR=/opt/momentseek/models`，让 Compose 挂载共享 runtime / models。
4. 设置 `HOST_NPU_DEVICE_ID` 为已获批的宿主物理卡号，`ASCEND_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` 使用同一个物理卡号，`NPU_DEVICE_ID` 保持容器内逻辑 `0`。
5. 准备宿主机模型目录，并按容器内 `/app/models` 视角校验 `deploy/models/ascend-prod.models.json`。
6. 准备 `vendor-wheels/` 和匹配当前服务器驱动的 `ASCEND_RUNTIME_IMAGE`。
7. 构建并启动 Compose 前，再确认目标 NPU 卡无人使用。

完整流程见 `docs/DEPLOYMENT.md` 和 `docs/MODELS.md`。

## 运行时数据迁移

Git 不保存视频、索引、SQLite、缩略图或模型权重。迁移已有环境时，只在确认目标路径后复制：

```text
runtime/
models/
```

不要在共享服务器上删除 runtime 数据；任何替换或清理前都要确认没有 active indexing jobs。
