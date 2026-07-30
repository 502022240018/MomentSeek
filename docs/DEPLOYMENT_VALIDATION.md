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
