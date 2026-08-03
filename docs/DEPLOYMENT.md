# 服务器部署总入口

本仓只讨论 Linux 服务器部署，不讨论开发机或本地调试。

## 1. 先确认交付路线

| 问题 | 场景 A | 场景 B | 选择结果 |
|---|---|---|---|
| 应用镜像从哪里来 | 镜像仓库可拉取 | 离线 `.tar` 导入 | 都支持 |
| Milvus 在哪里 | 本环境独立启动 | 连接已有 Milvus | 都支持 |
| 是否首次部署 | 新建实例 | 更新/回滚已有实例 | 都支持 |
| 同机实例数 | 单实例 | 多实例 | 都支持，参数必须隔离 |
| 模型是否联网下载 | 不允许，预置模型 | 运行时下载 | 只支持预置模型 |
| 硬件 | Ascend 910B | CUDA/CPU | 当前交付只承诺 Ascend |
| 运行方式 | Docker Compose | 裸机 Python/systemd | 当前交付只承诺容器 |

当前可以直接验收的是 **ARM64 + Ascend 910B + Docker Compose**。后端代码虽然包含 CPU/CUDA 兼容逻辑，但本精简交付仓没有提供并验证 CPU/CUDA 镜像、依赖锁和对应 Compose，因此不能把它们写成已交付能力。裸机部署同理，应另立交付项。

如果交付方需要从本仓源码制作完整应用镜像，先执行
`docs/IMAGE_BUILD.md`；部署人员已经拿到完整应用镜像时，不需要Dockerfile和基础镜像参与上线。

## 2. 镜像获取

### 2.1 从镜像仓库拉取

```bash
docker login REGISTRY.example.com
docker pull REGISTRY.example.com/team/momentseek:VERSION-ascend
docker image inspect REGISTRY.example.com/team/momentseek:VERSION-ascend
```

### 2.2 离线导入

交付方同时提供镜像 tar 和 SHA256：

```bash
sha256sum -c momentseek-VERSION-ascend.tar.sha256
docker load -i momentseek-VERSION-ascend.tar
docker image inspect IMAGE_NAME:TAG
```

部署服务器不负责从源码构建生产镜像。若服务器上没有应用镜像，必须从仓库拉取或离线导入；“服务器一定自带基础镜像”不是部署前提。

## 3. Milvus 选择

### 3.1 本环境独立 Milvus（推荐）

适合首次部署、正式环境隔离和同机多实例。启动时加载：

```text
compose/compose.yml + compose/compose.ascend.yml + compose/compose.milvus.yml
```

需要准备 Milvus、etcd、MinIO 镜像及独立运行目录。

### 3.2 连接已有 Milvus

适合已有受管 Milvus 服务的环境。启动时只加载：

```text
compose/compose.yml + compose/compose.ascend.yml
```

必须确认：

- 应用容器可以访问 `MILVUS_HOST:MILVUS_PORT`，不能误填只在宿主机有效的 `127.0.0.1`；
- 目标 Milvus 版本与应用客户端兼容；
- 该实例允许读写，且集合命名/数据隔离方案已经确定；
- Milvus 故障、备份和容量由谁负责。

## 4. 单实例与多实例

同一服务器每套环境必须拥有不同的：

- `COMPOSE_PROJECT_NAME`
- `APP_CONTAINER_NAME`
- `MOMENTSEEK_NETWORK_NAME`
- `APP_PORT`
- `HOST_RUNTIME_DIR`
- 物理 NPU 卡

`HOST_MODEL_DIR` 可以只读共享。Milvus 若独立部署，其数据位于各自的 `HOST_RUNTIME_DIR`，不可共享。

## 5. 首次部署、升级与回滚

- 首次部署：执行完整预检，确认容器名、端口和 NPU 都未占用。
- 升级：记录旧 `APP_IMAGE`，修改为新版本，使用 `preflight.py --upgrade`，再执行 `up -d`。
- 回滚：把 `APP_IMAGE` 改回旧版本并再次 `up -d`；不要删除运行目录。

涉及数据库结构变化时，必须先阅读对应版本的升级说明并备份；不能只靠切换镜像假设数据一定可逆。

## 6. 权限场景

- root：可直接访问 Docker 和 `/dev/davinci*`。
- 非 root：账号必须有 Docker 权限、部署目录写权限和设备访问权限；这些权限由服务器管理员配置。
- 无 `sudo` 且无 Docker 权限：不能部署，不应通过修改设备权限或开放 Docker socket 绕过管理。

## 7. 网络与安全

- 只需要本机访问：防火墙不开放 `APP_PORT`，通过反向代理或 SSH 隧道访问。
- 局域网/公网访问：由管理员开放端口或配置 HTTPS 反向代理。
- Milvus、etcd、MinIO 默认不映射宿主机端口，不应直接暴露公网。
- `.env` 含凭据，不提交 Git，不放入报告。
- 生产环境不要使用浮动镜像标签 `latest`，使用不可变版本或 digest。

## 8. 下一步

选好路线后，逐步执行 [Ascend 服务器部署说明](DEPLOYMENT_ASCEND.md)。模型目录规则见该说明的模型章节；部署验证记录只用于证明流程经过实测，不能作为配置模板。
