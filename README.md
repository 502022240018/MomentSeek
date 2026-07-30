# MomentSeek 平台交付版

这是从 `502022240018/MomentSeek` 的 `main` 分支整理出的平台交付仓库。目标是保留完整平台功能，并让服务器部署只依赖：

- 一份已经构建好的 ARM64 Ascend 应用镜像；
- 本仓库的 Compose、环境模板和检查脚本；
- 服务器上的模型目录和一张可用 Ascend NPU；
- Docker 与 Docker Compose。

本仓库不携带模型、视频、运行结果、实验记录和历史迁移脚本。Milvus 是平台默认依赖，但其维护脚本不是部署必需项；需要独立 Milvus 时使用 `compose.milvus.yml`。

具体来源提交和迁移边界见 [SOURCE.md](SOURCE.md)。

## 目录

```text
backend/app/                 后端平台代码
frontend/                    Web 前端源码
deploy/env/                  部署参数模板
deploy/models/               模型目录清单
deploy/orchestration/        可选的大模型查询规划/重排配置
scripts/preflight.py         启动前检查服务器、镜像、端口和 NPU
scripts/verify_models.py     检查模型文件是否齐全
scripts/smoke_check.py       启动后的接口冒烟检查
compose.yml                  平台通用容器定义
compose.ascend.yml           Ascend 设备和驱动映射
compose.milvus.yml           可选的独立 Milvus/etcd/MinIO
```

服务器部署先阅读 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 选择场景，再按
[docs/DEPLOYMENT_ASCEND.md](docs/DEPLOYMENT_ASCEND.md) 操作。
