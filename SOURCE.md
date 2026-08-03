# 迁移来源

- 上游仓库：`https://github.com/502022240018/MomentSeek`
- 来源分支：`main`
- 来源提交：`883fd3056ef47efd833fa10b16b661300e8a1548`
- 整理原则：保留平台运行代码，重新提供最小服务器部署层。

直接保留的主要代码：

- `backend/app/`
- `frontend/`
- `deploy/orchestration/`
- `vendor-wheels/`

重新整理的交付内容：

- Compose 与 Ascend 设备映射；
- Ascend Dockerfile、依赖锁和基础镜像约束；
- Milvus 独立部署；
- 环境参数模板；
- 模型目录清单与校验；
- 部署前预检和部署后冒烟；
- 通用服务器部署说明。

模型、运行数据、视频、实验记录、评测产物和历史运维脚本不进入本交付仓。

## 目录结构重构（agent/platform-restructure）

在上述内容范围之上，对 `backend/app` 做了一次**纯目录重组**（`git mv` +
import 路径更新，零逻辑改动），分层与依赖规则见 `docs/ARCHITECTURE.md`。
模块搬迁映射：

| 原路径（上游 main） | 新路径 |
|---|---|
| `app/settings.py` `deployment.py` `model_pool.py` `model_sources.py` | `app/core/` |
| `app/db.py` | `app/catalog/db.py` |
| `app/media.py` | `app/media/media.py` |
| `app/schemas.py` | `app/api/schemas.py` |
| `app/search.py` `retrieval_metrics.py` | `app/retrieval/` |
| `app/retrieval_orchestration.py` | `app/orchestration/` |
| `app/worker.py` `stage_runner.py` `indexer_daemon.py` `isolated_stage_workers.py` | `app/execution/` |
| `app/stage_executor.py` | `app/indexing/stage_executor.py` |
| `app/indexing/{visual,faces}.py` | `app/indexing/modalities/{visual,face}/` |
| `app/indexing/asr*.py`（6 个） | `app/indexing/modalities/asr/` |
| `app/indexing/{ocr,ocr_acl}.py` | `app/indexing/modalities/ocr/` |
| `app/indexing/{speaker,speaker_3dspeaker_runtime}.py` | `app/indexing/modalities/speaker/` |
| `app/indexing/milvus_*.py`（8 个） | `app/vector_store/milvus/` |
| `app/color_grading.py` | `app/integrations/` |
| `app/speaker_service.py` | `app/identity/` |
| `app/speaker_backfill.py` | `app/maintenance/` |

差异说明：

- 删除 `app/indexing/batch_buffer.py`：上游全仓无引用的死代码（被 Milvus
  P2 直写路径取代），及其对应测试类；
- 从上游迁入核心单元测试（35 个离线 + 1 个 Milvus 集成套件；上游
  `tests/integration/` 其余 4 个文件是无测试函数的诊断脚本，未迁入）、`pytest.ini`、
  `backend/requirements/{ci,dev}.txt` 与 GitHub Actions CI；排除依赖评测
  脚本/一次性验证的测试文件；
- 新增 `app/{platform,observability,evaluation}/README.md` 预留层约定，
  技术债与后续优化方向登记在 `docs/ARCHITECTURE.md` 第 9 节。
