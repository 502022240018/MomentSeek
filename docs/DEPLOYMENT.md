# 部署流程

## 环境分层

MomentSeek 部署分为四类 profile：

```text
dev.cpu：本地 CPU 开发。
dev.cuda：本地 CUDA 开发。
staging.ascend：Ascend 服务器预发布验证。
prod.ascend：Ascend 生产或演示环境。
```

dev profile 可以自动下载模型；staging/prod profile 必须使用预缓存模型、model manifest 和 models lock 复现环境，禁止运行时下载。部署流程以 manifest 为准，不依赖操作者记忆。

## 标准目录

服务器标准目录：

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

`releases/` 保存不可变 release 内容，`current` 指向当前生效版本。`runtime/` 保存 catalog、uploads、indexes、thumbnails、clips 等运行时数据。`models/` 保存预缓存模型。`env/` 保存服务器实际 `.env` 或 profile 派生配置。`deployment-record.json` 记录最后一次部署的 release、profile、模型清单和验证结果。

当前共享服务器已有历史路径，迁移到标准目录前应先做只读盘点，不直接移动或清理现有 runtime。

## Release Manifest

Release manifest 描述一次可复现发布，示例见：

```text
deploy/releases/release.example.json
```

manifest 至少应记录：

```text
release_id
git_commit
branch
image
frontend build 信息
models.manifest
models.lock
runtime mount
env_profile
verification
```

生成入口：

```powershell
python scripts/write_release_manifest.py --model-manifest deploy/models/ascend-prod.models.json
```

staging、prod 和新服务器复制都应从 release manifest、env profile、model manifest 和 models lock 还原，而不是临时拼接命令。

## Deployment Record

`deployment-record.json` 是服务器当前状态记录，用于回答“当前跑的是哪个 release”。它应保存：

```text
release_id
git_commit
image_tag
env_profile
model_manifest
models_lock
deployed_at
deployed_by
verification_result
rollback_from
```

Deployment record 只记录事实，不替代 release manifest。release manifest 描述可发布内容，deployment record 描述某台服务器实际生效内容。

## Staging Ascend

staging 使用：

```text
deploy/env/staging.ascend.example
deploy/models/ascend-prod.models.json
```

staging 目标是验证 Ascend 设备、模型缓存、health metadata、smoke check 和资源占用。部署前先把模型预缓存到宿主机模型目录，并通过 `scripts/verify_models.py` 生成 lock。部署后检查 `/api/health` 中的 `env_profile`、`release_id`、`git_commit`、`image_tag` 和 `model_manifest`，确认与 release manifest 一致。

## Prod Ascend

prod 使用：

```text
deploy/env/prod.ascend.example
deploy/models/ascend-prod.models.json
```

prod 只接受已在 staging 验证过的 release manifest。上线前确认模型 lock、镜像 tag、git commit、runtime mount 和回滚目标。上线后用只读 health 和 smoke check 验证，不做临时下载、不临时改 profile。

## 新服务器复制

新服务器复制流程以 manifest 为中心：

1. clone 仓库或拉取指定 `git_commit`。
2. 准备标准目录 `/opt/momentseek/`。
3. 复制或预缓存 `deploy/models/ascend-prod.models.json` 中的模型到服务器模型目录。
4. 用 `scripts/verify_models.py` 校验模型并生成 models lock。
5. 根据 release manifest 选择 env profile、镜像 tag、runtime mount 和前端 build。
6. 启动后检查 `/api/health` 部署元信息和 smoke check。
7. 写入 `deployment-record.json`。

新服务器不应依赖旧服务器上的临时 shell 历史；所有可复现信息必须来自 release manifest、model manifest、models lock 和 env profile。

## 回滚原则

回滚优先切换 `current` 到上一个已验证 release，并保留 runtime 和 models 目录不动。回滚前后都要记录 deployment record，并用只读命令确认 health 和 smoke check。

如果回滚涉及数据库 schema 或 runtime 数据变化，先停止在该环境继续写入新任务，并单独评估数据兼容性。不要在未确认 active indexing jobs 的情况下替换 runtime。

## 共享服务器安全要求

任何服务器状态变更前，先执行 docs/OPERATIONS.md 的只读检查，并确认没有 active indexing jobs。

共享服务器只能操作明确归属 MomentSeek 的进程、容器、目录和端口。禁止 broad kill、禁止清理他人模型缓存、禁止重启不属于 MomentSeek 的服务。所有 staging/prod 操作都应先读 `docs/OPERATIONS.md`，再按 release manifest 执行。
