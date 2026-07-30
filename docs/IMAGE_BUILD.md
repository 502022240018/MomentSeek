# Ascend 应用镜像制作

这份说明只用于把本仓源码制作成完整应用镜像。已经拿到完整应用镜像的部署人员可以跳过，直接阅读 `DEPLOYMENT_ASCEND.md`。

## 1. 构建输入

必须准备：

1. ARM64 构建服务器；
2. Docker；
3. 与目标 Ascend 910B、驱动和 CANN 兼容的运行时基础镜像；
4. 本仓完整源码；
5. `vendor-wheels/` 中的6个 wheel和 `SHA256SUMS`；
6. Node镜像与Python依赖源，或对应的内部镜像仓库/PyPI镜像。

基础镜像只在“制作应用镜像”时使用。正式部署服务器拿到最终应用镜像后，不需要再单独准备基础镜像。

## 2. 为什么保留 vendor-wheels

`vendor-wheels/` 只保存 Ascend ARM64 构建中需要固定、并且容易受平台或可选依赖影响的文件：

- `grpcio`
- `orjson`
- `pymilvus`
- `python-dotenv`
- `cachetools`
- `insightface`

Dockerfile会先执行 `sha256sum -c SHA256SUMS`，校验通过才安装。其余依赖仍从配置的Python包源安装，因此这些wheel不代表“完全离线构建包”。

## 3. 确认基础镜像

```bash
docker image inspect YOUR_ASCEND_RUNTIME_IMAGE
```

基础镜像必须满足 `backend/requirements/constraints-ascend.txt` 中的运行时契约，重点包括：

```text
Python 3.11
FastAPI 0.115.11
Uvicorn 0.34.3
Pillow 11.2.1
torch 2.9.0
torch-npu 2.9.0.post1
torchaudio 2.9.0
OpenCV 4.11.0
```

不能只因为镜像里“有CANN”就认定兼容。

## 4. 构建

```bash
docker build \
  -f docker/Dockerfile.ascend \
  --build-arg ASCEND_RUNTIME_IMAGE=YOUR_ASCEND_RUNTIME_IMAGE \
  -t YOUR_REGISTRY/momentseek-platform:VERSION-ascend \
  .
```

也可以先把 `.env` 中的 `APP_IMAGE` 和 `ASCEND_RUNTIME_IMAGE` 填好，再使用Compose：

```bash
docker-compose \
  --env-file .env \
  -f compose/compose.yml \
  -f compose/compose.ascend.yml \
  build app
```

如果服务器使用新版插件，把 `docker-compose` 替换成 `docker compose`。

## 5. 构建结果检查

```bash
docker image inspect YOUR_REGISTRY/momentseek-platform:VERSION-ascend

docker run --rm \
  YOUR_REGISTRY/momentseek-platform:VERSION-ascend \
  python3 -c "from importlib.metadata import version; print(version('torch')); print(version('torch-npu')); print(version('pymilvus'))"
```

预期至少满足：

```text
torch=2.9.0
torch-npu=2.9.0.post1
pymilvus=2.6.16
```

MindIE基础镜像的 `pip check` 可能报告其内置组件元数据冲突，例如
`onnxruntime-cann`不能满足第三方包声明的普通`onnxruntime`名称，以及
`torchvision`声明的torch版本与厂商实际组合不同。本次验收要求：

- 新镜像的 `pip check` 结果不能比批准的基础/基线镜像增加问题；
- Dockerfile中的核心版本断言必须通过；
- 平台核心模块导入、健康检查和服务器冒烟必须通过。

不要为了清空 `pip check` 擅自安装普通ONNX Runtime或替换厂商torch栈。

## 6. 发布

在线环境：

```bash
docker push YOUR_REGISTRY/momentseek-platform:VERSION-ascend
```

离线交付：

```bash
docker save \
  -o momentseek-platform-VERSION-ascend.tar \
  YOUR_REGISTRY/momentseek-platform:VERSION-ascend

sha256sum momentseek-platform-VERSION-ascend.tar \
  > momentseek-platform-VERSION-ascend.tar.sha256
```

交付时同时提供镜像名、版本、SHA256、来源提交和适配的Ascend/CANN版本。不要使用无法追溯的 `latest`。

## 7. 不要放进镜像

- 模型和OM文件；
- 视频、索引和运行结果；
- `.env`；
- Milvus数据；
- 构建服务器上的缓存。

这些内容分别通过只读模型挂载、运行目录挂载和环境参数提供，避免每次更新代码都重新分发数GB模型。
