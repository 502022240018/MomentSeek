# MomentSeek 平台交付版

这是从 `502022240018/MomentSeek` 的 `main` 分支整理出的 Ascend 平台交付仓库。目标是保留完整平台功能，同时让代码既能制作应用镜像，也能直接部署已经制作好的镜像。

当前只交付实际使用并经过服务器验证的 **ARM64 + Ascend 910B + Docker Compose** 路线，不放入尚未形成生产交付的 CPU/CUDA 和裸机部署文件。

本仓库不携带模型、视频、运行结果和实验记录。Milvus 是唯一在线检索库；需要独立 Milvus 时使用 `compose/compose.milvus.yml`。部署本身不需要维护脚本，但已有运行数据切换到 Milvus-only 模式时，必须按 [Milvus-only 正式索引切换手册](docs/MILVUS_ONLY_MIGRATION.md) 执行。

具体来源提交和迁移边界见 [SOURCE.md](SOURCE.md)。

## 目录

```text
backend/
├─ app/                      后端平台代码（分层结构见 docs/ARCHITECTURE.md）
│  ├─ main.py                FastAPI 组装入口
│  ├─ api/                   路由 + 请求/响应模型
│  ├─ core/                  settings / 部署元信息 / 模型池与来源
│  ├─ catalog/               SQLite 资产元数据
│  ├─ media/                 ffmpeg 媒体处理
│  ├─ indexing/              索引生产；modalities/ 下每通道一个子包
│  ├─ retrieval/             在线检索引擎与融合
│  ├─ orchestration/         LLM 查询规划/重排
│  ├─ vector_store/milvus/   Milvus 客户端/schema/索引/检索
│  ├─ execution/             任务 worker、阶段子进程、daemon
│  ├─ identity/              说话人/声纹服务
│  ├─ integrations/          外部系统（视频仿色）
│  ├─ maintenance/           离线维护任务
│  └─ platform|observability|evaluation/  预留层（仅 README 约定）
├─ requirements/             依赖锁：ascend / ci / dev 等
└─ tests/                    单元测试；integration/ 为 Milvus 集成测试
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

平台架构、模块分层与扩展指南见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
SnapMind 风格的交互式检索实验见 [docs/PLANNER_LAB.md](docs/PLANNER_LAB.md)。
需要从源码制作镜像时阅读 [docs/IMAGE_BUILD.md](docs/IMAGE_BUILD.md)；拿到镜像后，按
[docs/DEPLOYMENT_ASCEND.md](docs/DEPLOYMENT_ASCEND.md) 部署。
已有视频从旧索引切换时，按 [docs/MILVUS_ONLY_MIGRATION.md](docs/MILVUS_ONLY_MIGRATION.md)
执行灰度、全量重建和版本行数核验。

## 测试

```bash
python -m pip install -r backend/requirements/ci.txt
cd backend && python -m pytest tests -m "not integration" -q   # 离线单测
```

Milvus 集成测试需要先起 `compose/compose.milvus.yml`，再运行
`python -m pytest tests/integration -m integration -q`；GitHub Actions 的
CI/Milvus workflow 会自动执行同样的步骤。

## 不包含的内容

- 模型和OM文件：单独作为模型制品交付；
- CPU/CUDA部署：当前没有经过生产服务器验收；
- 本地开发环境：不是本仓交付目标；
- `LICENSE`：上游仓库没有许可证文件，需由代码所有者确认授权方式后补充，不能擅自指定。
