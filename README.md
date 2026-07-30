# MomentSeek 平台交付版

这是从 `502022240018/MomentSeek` 的 `main` 分支整理出的 Ascend 平台交付仓库。目标是保留完整平台功能，同时让代码既能制作应用镜像，也能直接部署已经制作好的镜像。

当前只交付实际使用并经过服务器验证的 **ARM64 + Ascend 910B + Docker Compose** 路线，不放入尚未形成生产交付的 CPU/CUDA 和裸机部署文件。

本仓库不携带模型、视频、运行结果、实验记录和历史迁移脚本。Milvus 是平台默认依赖，但其历史维护脚本不是部署必需项；需要独立 Milvus 时使用 `compose/compose.milvus.yml`。

具体来源提交和迁移边界见 [SOURCE.md](SOURCE.md)。

## 目录

```text
backend/
├─ app/                      后端平台代码
└─ requirements/             Ascend实际依赖和版本约束
frontend/                    Web前端源码
docker/
└─ Dockerfile.ascend         制作完整Ascend应用镜像
compose/
├─ compose.yml               平台通用容器定义
├─ compose.ascend.yml        Ascend设备和驱动映射
└─ compose.milvus.yml        可选独立Milvus/etcd/MinIO
deploy/
├─ env/ascend.example        唯一环境参数模板
├─ models/ascend.models.json 必需模型目录清单
└─ orchestration/            可选查询规划/重排配置
scripts/
├─ preflight.py              部署前检查
├─ verify_models.py          模型完整性检查
└─ smoke_check.py            部署后接口检查
vendor-wheels/               Ascend ARM64必须固定的离线wheels
```

需要从源码制作镜像时阅读 [docs/IMAGE_BUILD.md](docs/IMAGE_BUILD.md)；拿到镜像后，按
[docs/DEPLOYMENT_ASCEND.md](docs/DEPLOYMENT_ASCEND.md) 部署。

## 不包含的内容

- 模型和OM文件：单独作为模型制品交付；
- CPU/CUDA部署：当前没有经过生产服务器验收；
- 本地开发环境：不是本仓交付目标；
- `LICENSE`：上游仓库没有许可证文件，需由代码所有者确认授权方式后补充，不能擅自指定。
