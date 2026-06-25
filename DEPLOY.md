# MomentSeek MVP 迁移与部署说明

这份文档面向“从 GitHub 拉下代码后，在本地或新服务器快速跑起来”的场景。默认策略是：Git 只保存代码和部署配置，不保存视频、索引、数据库、模型权重和密钥。

## 1. GitHub 仓库里应该包含什么

应该提交：

- `backend/`：FastAPI 后端、索引、检索逻辑。
- `frontend/`：React 前端。
- `docs/`、`samples/`、`scripts/`：说明、样例、资源检查脚本。
- `Dockerfile.cpu`、`Dockerfile.ascend`、`compose*.yml`、`Makefile`。
- `.env.example`、`.gitignore`、`.dockerignore`、`README.md`、`DEPLOY.md`。

不应该提交：

- `.env`
- `runtime/`
- `models/`
- `vendor-wheels/`
- `frontend/node_modules/`
- `frontend/dist/`
- `.pytest_cache/`、`__pycache__/`

当前 `.gitignore` 已经排除了这些目录。上传前仍建议执行：

```bash
git status --short
git status --ignored --short
```

确认没有 `.env`、视频、索引、模型权重进入待提交列表。

## 2. 首次上传到 GitHub

建议只把 `video_retrieval_mvp/` 作为独立仓库上传，不要从上一级 `prototype/` 整体上传，因为上一级目录里还有其他实验代码和数据。

```bash
cd video_retrieval_mvp

git init
git add .
git status
git commit -m "Initial MomentSeek MVP baseline"

git branch -M main
git remote add origin git@github.com:YOUR_NAME/momentseek-mvp.git
git push -u origin main
```

如果当前项目已经在另一个父级 Git 仓库里，最干净的做法是把 `video_retrieval_mvp/` 复制到一个新目录，再在新目录里执行上面的命令。

## 3. 从 GitHub 拉代码后怎么跑

### 3.1 CPU / 无卡模式，推荐默认方式

适合本地开发、新服务器冒烟、共享服务器没有确认空闲显卡时使用。这个模式不会映射 NPU/GPU 设备。

```bash
git clone git@github.com:YOUR_NAME/momentseek-mvp.git
cd momentseek-mvp

cp .env.example .env
# 按实际情况修改 .env 里的 APP_PORT 和 APP_PUBLIC_URL

docker compose -f compose.yml up -d --build
```

打开：

```text
http://127.0.0.1:8300
```

远程服务器则打开：

```text
http://SERVER_IP:8300
```

检查服务：

```bash
docker compose -f compose.yml ps
curl http://127.0.0.1:8300/api/health
docker logs -f momentseek-mvp-app
```

### 3.2 当前共享服务器兼容模式

当前服务器是共享环境，默认仍然应该先用不映射 NPU 的模式。这个模式使用服务器兼容的 Ascend 基础镜像，但 `NPU_ENABLED=false`，不会占用卡。

```bash
git clone git@github.com:YOUR_NAME/momentseek-mvp.git
cd momentseek-mvp

cp .env.example .env
sed -i 's#APP_PUBLIC_URL=.*#APP_PUBLIC_URL=http://SERVER_IP:8300#' .env

docker compose -f compose.yml -f compose.server.yml up -d --build
```

确认容器没有映射 NPU：

```bash
docker inspect momentseek-mvp-app \
  --format 'Privileged={{.HostConfig.Privileged}} Devices={{json .HostConfig.Devices}}'
```

期望看到：

```text
Privileged=false Devices=null
```

### 3.3 Ascend / NPU 模式，必须先确认资源

只有在明确确认某张卡无人使用，并且得到允许后，才使用 NPU 覆盖配置。

```bash
./scripts/check_resource.sh

NPU_DEVICE_ID=7 \
docker compose -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d --build
```

注意：

- `compose.ascend.yml` 会映射 `/dev/davinci*` 设备。
- 不确认资源时不要加这个文件。
- 索引阶段跑完后，模型子进程会退出；仍然建议用 `npu-smi info` 验证没有残留进程。

## 4. 模型和运行时数据

GitHub 不保存模型和运行时数据。新环境首次启动后会自动创建：

```text
runtime/
models/
```

### 4.1 只迁移代码

只需要 clone 代码并启动服务，然后重新上传视频、重新索引。

这种方式最干净，适合换服务器或给别人复现实验。

### 4.2 完整迁移已有视频、索引和人物库

如果要保留已有视频、索引、SQLite 元数据和人物库，需要额外迁移：

```text
runtime/
models/
```

示例：

```bash
rsync -avP old-server:/path/to/momentseek-mvp/runtime/ ./runtime/
rsync -avP old-server:/path/to/momentseek-mvp/models/ ./models/
```

或者打包：

```bash
tar -czf momentseek-runtime.tgz runtime models
scp momentseek-runtime.tgz new-server:/path/to/momentseek-mvp/
tar -xzf momentseek-runtime.tgz
```

迁移完成后重启：

```bash
docker compose -f compose.yml up -d
```

## 5. 本地非 Docker 开发

如果只是改代码、跑测试，可以不用 Docker。

后端：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-cpu.txt

cd backend
pytest -q
uvicorn app.main:app --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

生产构建：

```bash
cd frontend
npm ci
npm run build
cd ..
cp -r frontend/dist backend/app/static
```

## 6. 常用维护命令

查看服务：

```bash
docker compose -f compose.yml ps
docker logs -f momentseek-mvp-app
```

重启：

```bash
docker compose -f compose.yml restart
```

更新代码后重新构建：

```bash
git pull
docker compose -f compose.yml up -d --build
```

停止服务：

```bash
docker compose -f compose.yml down
```

停止服务但保留数据：

- 不要删除 `runtime/`
- 不要删除 `models/`

彻底清空测试数据时，才手动删除 `runtime/`。

## 7. 冒烟验证流程

启动后建议按这个顺序检查：

1. 打开首页：`http://SERVER_IP:8300`
2. 打开 API 文档：`http://SERVER_IP:8300/docs`
3. 调用健康检查：

   ```bash
   curl http://SERVER_IP:8300/api/health
   ```

4. 上传一段短视频。
5. 建立 `visual / face / asr` 索引。
6. 搜索：

   - 文本找场景：`a person speaking on a stage`
   - 参考图找人
   - 文本找语音：字幕或语音中出现过的关键词

7. 点击结果，确认能跳转播放对应片段。

## 8. 迁移时最容易踩的坑

- 忘记复制 `.env.example` 为 `.env`。
- `APP_PUBLIC_URL` 仍然写着旧服务器 IP。
- 端口 `8300` 被占用。
- 把 `runtime/` 或 `models/` 误提交到 GitHub。
- 在共享服务器上误加 `compose.ascend.yml`，导致映射 NPU。
- 新服务器没有合适的 Ascend/CANN/torch_npu 基础镜像，却直接启用 NPU 模式。

默认先用 CPU / 无卡模式跑通，是最稳的迁移方式。
