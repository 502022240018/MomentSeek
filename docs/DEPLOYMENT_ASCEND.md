# Ascend 服务器部署说明

本文只描述通用 Linux 服务器容器部署，不包含本地开发环境。先阅读
`docs/DEPLOYMENT.md` 选择镜像、Milvus 和更新场景。文中的域名、目录、端口、
镜像名和 NPU 卡号都是示例，必须替换为目标环境的实际值。

## 1. 交付物

部署前确认拿到四项内容：

1. 本仓库代码。
2. 已构建的 MomentSeek Ascend 应用镜像名称，例如 `momentseek/app:1.0.0-ascend`。
3. 完整模型目录；目录结构由 `deploy/models/ascend.models.json` 定义。
4. 一张经管理员确认可用的 Ascend NPU。

应用镜像已经包含 Python、CANN 用户态组件、后端依赖和前端静态文件。部署人员不需要现场安装 Python 包，也不需要额外构建“基础镜像”。`Dockerfile` 和基础镜像属于镜像制作流程，不属于服务器上线步骤。

镜像可以从镜像仓库拉取，也可以用 `docker load` 离线导入；两种方式完成后都必须能通过 `docker image inspect APP_IMAGE`。

## 2. 部署前只读检查

以下命令不会修改现有容器：

```bash
uname -m
docker version
docker compose version || docker-compose version
docker images --format '{{.Repository}}:{{.Tag}}'
npu-smi info
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
ss -lnt
```

确认：

- 架构为 `aarch64`；
- 应用镜像、Milvus、etcd、MinIO 镜像已经存在；若使用外部 Milvus，只要求应用镜像；
- 端口未占用；
- NPU 没有进程，并且管理员同意本部署使用；
- 模型目录存在且只读共享不会影响其他服务。

## 3. 创建独立部署目录

```bash
sudo mkdir -p /opt/momentseek/deploy
sudo mkdir -p /opt/momentseek/runtime
cd /opt/momentseek/deploy
```

把本仓库复制或克隆到该目录。每套环境必须使用独立的：

- `COMPOSE_PROJECT_NAME`
- `APP_CONTAINER_NAME`
- `APP_PORT`
- `HOST_RUNTIME_DIR`

不要复用其他实例的运行目录，也不要复用其他实例正在使用的 NPU。

## 4. 填写配置

```bash
cp deploy/env/ascend.example .env
vi .env
```

至少修改：

- `APP_IMAGE`：服务器上真实存在的完整应用镜像；
- `APP_PORT`、`APP_PUBLIC_URL`；
- `HOST_RUNTIME_DIR`、`HOST_MODEL_DIR`；
- `COMPOSE_PROJECT_NAME`、`APP_CONTAINER_NAME`、`MOMENTSEEK_NETWORK_NAME`；
- `HOST_NPU_DEVICE_ID`：宿主机物理 NPU 卡号；
- `MINIO_ROOT_USER`、`MINIO_ROOT_PASSWORD`。

`NPU_DEVICE_ID=0` 不要跟着物理卡号修改。容器只映射一张物理卡，应用在容器内按逻辑卡 0 使用。

同机部署多套环境时，项目名、容器名、网络名、端口、运行目录和物理 NPU
必须全部不同；模型目录可以只读共享。

如果连接已有 Milvus：

```dotenv
MILVUS_HOST=服务器可访问地址
MILVUS_PORT=19530
```

并且启动时不要加载 `compose/compose.milvus.yml`。确认该 Milvus 允许本环境读写；不同环境推荐使用独立 Milvus，避免集合和数据互相影响。

如果 Milvus 只监听宿主机 `127.0.0.1`，桥接网络中的应用容器无法用
`127.0.0.1` 连接它。应由管理员提供容器可达地址，或把 Milvus 加入同一
Docker 网络。

## 5. 自动预检

```bash
# 本仓 compose 同时启动独立 Milvus：
python3 scripts/preflight.py --env-file .env --with-milvus

# 若连接已有 Milvus，则不加 --with-milvus：
# python3 scripts/preflight.py --env-file .env
python3 scripts/verify_models.py \
  --manifest deploy/models/ascend.models.json \
  --model-dir "$(grep '^HOST_MODEL_DIR=' .env | cut -d= -f2-)" \
  --lock "$(grep '^HOST_RUNTIME_DIR=' .env | cut -d= -f2-)/models.lock.json"
```

任何 `[ERROR]` 都必须处理后再启动。预检不会创建、停止或删除容器。
升级已有实例时使用 `--upgrade`，否则同名容器和已占用端口会被当作误操作：

```bash
python3 scripts/preflight.py --env-file .env --with-milvus --upgrade
```

## 6. 启动

以下示例使用新版命令 `docker compose`。如果服务器只有独立版 Compose，
把所有 `docker compose` 原样替换为 `docker-compose`。

平台与独立 Milvus 一起启动：

```bash
docker compose \
  --env-file .env \
  -f compose/compose.yml \
  -f compose/compose.ascend.yml \
  -f compose/compose.milvus.yml \
  config

docker compose \
  --env-file .env \
  -f compose/compose.yml \
  -f compose/compose.ascend.yml \
  -f compose/compose.milvus.yml \
  up -d
```

连接已有 Milvus 时：

```bash
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml config
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml up -d
```

`config` 必须先成功；它能提前发现漏填参数、YAML 错误和挂载变量问题。

## 7. 验证

```bash
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml ps
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml logs --tail=200 app
python3 scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

若 `APP_PORT` 不是 8000，把最后一个地址改为实际端口。通过标准：

- app 为 `healthy`；
- `/api/health` 返回 `status=ok`；
- `/api/videos`、`/api/jobs` 可访问；
- 日志没有模型缺失、NPU 初始化失败或 Milvus 连接失败。

浏览器打开 `http://服务器IP:APP_PORT` 验证页面。

首次验证只证明服务与基础 API 可用。正式业务验收还应上传一段已授权的测试
视频，完成一次索引和一次检索，并确认任务结束后 NPU 资源符合配置预期。

## 8. 常用运维

```bash
# 查看状态
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml ps

# 查看日志
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml logs -f --tail=200 app

# 仅重启本环境应用
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml restart app

# 停止本环境；不会删除宿主机运行目录
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml down
```

如果本环境包含独立 Milvus，所有命令都追加 `-f compose/compose.milvus.yml`。不要用模糊匹配批量停止容器，不要删除模型目录和 `HOST_RUNTIME_DIR`。

## 9. 更新与回滚

更新前记录当前镜像：

```bash
docker inspect "$APP_CONTAINER_NAME" --format '{{.Config.Image}}'
```

把 `.env` 的 `APP_IMAGE` 改为新版本，然后执行相同的 `config`、`up -d` 和冒烟检查。回滚时把 `APP_IMAGE` 改回记录的旧镜像，再次 `up -d`。运行数据位于独立的 `HOST_RUNTIME_DIR`，镜像切换不会删除它。

## 10. 常见问题

- `port is already allocated`：更换 `APP_PORT`，不要停止未知容器。
- `/dev/davinciN` 不存在：物理卡号错误，重新执行 `npu-smi info`。
- `No space left on device`：检查 Docker 和运行目录磁盘，由管理员清理；不要擅自删除共享镜像。
- 模型检查失败：按输出补齐模型，不允许容器在线下载生产模型。
- Milvus 不健康：检查 etcd、MinIO、Milvus 三个服务日志和运行目录权限。
- 页面能开但索引失败：重点检查 NPU 卡、模型挂载、`MODEL_MANIFEST` 和 Milvus。
