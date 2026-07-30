# Ascend 服务器部署验证记录

验证日期：2026-07-30

## 验证范围

- ARM64 openEuler 服务器；
- Docker 与独立版 Docker Compose v5.1.0；
- Ascend 910B4 单卡映射；
- 已构建的 MomentSeek Ascend 应用镜像；
- 本仓 `compose.yml`、`compose.ascend.yml`、`compose.milvus.yml`；
- Milvus 2.6.20、etcd 3.5.16、MinIO；
- 服务器现有的五模态模型目录；
- 前端首页和后端基础 API。

验证使用独立容器名、未占用端口、管理员确认可用的物理 NPU、独立运行目录和独立 Milvus。服务器原有平台端口、NPU、Milvus 和运行目录未修改。

## 验证结果

- 部署前预检通过；
- 9 项必需模型全部通过，包含 Hugging Face `snapshots/<revision>` 目录；
- Compose 三文件合并校验通过；
- app、Milvus、etcd、MinIO 均为 healthy；
- `/api/health`、`/api/videos`、`/api/jobs` 全部通过；
- 前端首页 HTTP 200；
- 模型、Ascend 驱动和部署配置均为只读挂载；
- 运行目录为独立可写挂载。

## 实测发现并修正的问题

1. 服务器只有 `docker-compose`，没有 `docker compose`。预检和说明书已兼容两种命令。
2. 使用外部 Milvus 时不应强制要求 MinIO 凭据。预检新增 `--with-milvus` 开关。
3. `APP_PORT` 同时控制镜像内 Uvicorn 端口，原映射固定到容器 8000 会导致非 8000 端口部署失败。现已统一宿主机和容器端口，并让健康检查读取 `APP_PORT`。
4. 自建 Milvus 时应用可能先于 Milvus 启动。`compose.milvus.yml` 已增加健康依赖。

## 尚未覆盖

本次采用服务器已有的完整应用镜像，验证的是交付部署链路，不是镜像构建链路。没有上传视频或执行完整五模态索引任务，因此没有触发模型常驻加载；镜像构建和业务级索引/检索验收应作为独立交付阶段执行。
