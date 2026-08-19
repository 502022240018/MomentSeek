# Ascend 服务器部署验证记录

验证日期：2026-07-30

## 验证范围

- ARM64 openEuler 服务器；
- Docker 与独立版 Docker Compose v5.1.0；
- Ascend 910B4 单卡映射；
- 已构建的 MomentSeek Ascend 应用镜像；
- 本仓 `compose/compose.yml`、`compose/compose.ascend.yml`、`compose/compose.milvus.yml`；
- Milvus 2.6.20、etcd 3.5.16、MinIO；
- 服务器现有的五模态模型目录；
- 本仓 `docker/Dockerfile.ascend` 和5个依赖/约束文件；
- 前端首页和后端基础 API。

验证使用独立容器名、未占用端口、管理员确认可用的物理 NPU、独立运行目录和独立 Milvus。服务器原有平台端口、NPU、Milvus 和运行目录未修改。

## 验证结果

- 部署前预检通过；
- 9 项必需模型全部通过，包含 Hugging Face `snapshots/<revision>` 目录；
- Compose 三文件合并校验通过；
- 直接 `docker build` 和文档中的 `docker-compose build app` 均通过；
- 新构建镜像的核心依赖版本与已验证镜像一致；
- app、Milvus、etcd、MinIO 均为 healthy；
- `/api/health`、`/api/videos`、`/api/jobs` 全部通过；
- 前端首页 HTTP 200；
- 模型、Ascend 驱动和部署配置均为只读挂载；
- 运行目录为独立可写挂载。

## 实测发现并修正的问题

1. 服务器只有 `docker-compose`，没有 `docker compose`。预检和说明书已兼容两种命令。
2. 使用外部 Milvus 时不应强制要求 MinIO 凭据。预检新增 `--with-milvus` 开关。
3. `APP_PORT` 同时控制镜像内 Uvicorn 端口，原映射固定到容器 8000 会导致非 8000 端口部署失败。现已统一宿主机和容器端口，并让健康检查读取 `APP_PORT`。
4. 自建 Milvus 时应用可能先于 Milvus 启动。`compose/compose.milvus.yml` 已增加健康依赖。
5. Compose文件移入 `compose/` 后不会自动读取仓库根目录 `.env`。所有命令已明确增加 `--env-file .env`。
6. 第一次目录移动时 Milvus Compose 出现缩进错误；服务器 `config -q` 已发现并修正。
7. 把通用依赖中的 FastAPI、Uvicorn、Pillow重复安装到MindIE runtime会引入新版本冲突。最终方案改为由基础镜像契约固定这些版本，只安装Ascend平台增量依赖。

## 尚未覆盖

没有上传视频或执行完整五模态索引任务，因此没有触发模型常驻加载。业务级索引/检索和长时间稳定性仍应作为正式上线验收项。

## 2026-08-03 Milvus-only 追加验证

在同一服务器创建了独立的运行目录、Compose 项目、Milvus/etcd/MinIO 和应用容器；
没有修改正式实例的容器、端口、NPU、模型目录或运行数据。验证使用一张独占 NPU 卡和
只读共享模型目录。

- 两段测试视频完成 visual、face、ASR、speaker、OCR 的实际索引；
- 每个通道均写入独立 `asset_version`，直接查询 Milvus 的版本行数与 Catalog publication 一致；
- 同一视频二次 visual 重建后，新版本原子发布成功；旧版本按保留期由独立维护任务清理；
- 视觉、ASR、OCR、参考脸、speaker/声纹检索均通过真实 API 返回结果；
- 全部在线读接口只依赖 Milvus payload 与 Catalog publication；历史 NPZ 保持冷备状态且不进入运行路径；
- 使用更新后的镜像运行 `reindex_milvus_only --dry-run` 和 `--verify-only`：源文件检测、
  默认全量缺失通道识别、指定通道版本/行数核验均符合预期；
- 应用、Milvus、etcd、MinIO 均 healthy，应用日志未发现未处理异常。

故障注入“停止 Milvus 后提交索引任务”未在服务器执行：远程执行环境阻止了停容器操作，
以避免影响服务器。该 fail-closed 路径由单元测试覆盖，并由运行包零 NPZ reader 审计
验证不存在运行时降级读取。

## Milvus-only / Catalog publication 切换顺序

这次切换会把 `videos.indexed_modalities` 变成 Catalog publication 的派生状态。旧版应用仍按
旧字段和 manifest 工作，因此不能让旧、新应用在同一个 Catalog 上跨迁移阶段同时提供检索。
正式切换必须安排维护窗口，并按以下顺序执行；任一步失败都不启动新应用，也不删除旧数据。

1. 停止接收新索引任务和检索流量，等待正在运行的任务结束；备份 `catalog.sqlite3`，记录当前
   应用镜像 ID、Milvus collection 行数和旧容器启动参数。
2. 使用候选镜像先执行只读审计：

   ```bash
   python -m app.maintenance.migrate_index_publications
   python -m app.maintenance.migrate_visual_time_bounds
   python -m app.maintenance.migrate_entity_face_samples
   ```

   第一条命令在旧视觉边界无效时会报告 `requires_rebuild` 并返回非零，这是预期结果；必须确认
   `errors` 为空、每个非视觉模态只有一个明确的 Milvus 版本，并确认每个待修视觉视频都有
   `duration_ms` 和固定窗 `segment_ms`。
3. 写入已核验的 Catalog publication：

   ```bash
   python -m app.maintenance.migrate_index_publications --apply
   ```

   非视觉通道发布为 `ready`；旧视觉版本只登记为 `disabled` 和
   `migration_state=requires_rebuild`，不能进入在线检索。该命令此时仍可能因待修视觉项返回非零，
   应依据 JSON 报告而不是仅依据退出码继续判断。
4. 复用 Milvus 中的原始视觉 embedding，生成带显式时间边界的新 UUID 版本：

   ```bash
   python -m app.maintenance.migrate_visual_time_bounds --execute
   ```

   工具会全量校验源行、固定窗数学关系、复制后的行数与边界，并在发布前再次检查 Catalog
   指针没有发生竞态；只有全部通过的视频才原子切换为 `ready`。旧 Milvus 版本不删除。
5. 从仍保留的参考图重新生成注册人物向量：

   ```bash
   python -m app.maintenance.migrate_entity_face_samples --apply
   ```

   新样本写入并回读校验成功后才清空旧 `embedding_path`；参考图继续保留。不要默认使用
   `--replace`，只有人工确认既有 Milvus 样本损坏时才使用。
6. 再次执行三条只读命令并核对：所有在线模态均有 `ready` publication；视觉结果满足
   `0 <= start < end <= duration`；人物样本为 512 维有限、非零、归一向量。随后用固定查询集验证
   Visual、Face、ASR、OCR、Speaker 及 Planner 结果，再启动候选应用并恢复流量。
7. 回滚只恢复 Catalog 备份并启动旧镜像；迁移生成的新 Milvus UUID 版本可暂时作为孤儿保留。
   历史 NPZ、legacy manifest 和旧 Milvus 版本只做冷备，不得重新接入运行时 fallback。
