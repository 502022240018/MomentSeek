# 本地 GPU 接管服务器展示

本文档用于当前迁移场景：服务器 `momentseek-current-app` 暂时保持运行，先把 runtime 和必要模型同步到本地，再用本地 Docker GPU 后端接管公网展示。服务器确认不再需要后，才停止服务器容器释放 NPU 2。

## 当前切换结果

2026-07-07 已完成一次本地接管：

```text
本地容器：momentseek-mvp-app
本地端口：127.0.0.1:18301 -> container:8000
运行模式：dev.cuda, CUDA_ENABLED=true, NPU_ENABLED=false
runtime：./runtime-server -> /app/runtime
公网入口：https://entertainment-grocery-independently-generators.trycloudflare.com
服务器容器：momentseek-current-app 已停止
服务器 NPU 2：No running processes found
```

当前公网地址是 Cloudflare quick tunnel 临时地址，不保证长期有效。失效后重新启动：

```powershell
.\runtime\tools\cloudflared.exe tunnel --url http://127.0.0.1:18301 --no-autoupdate
```

本次验证命令覆盖：

```powershell
curl.exe http://127.0.0.1:18301/api/health
curl.exe http://127.0.0.1:18301/api/videos
curl.exe -X POST http://127.0.0.1:18301/api/search -F "query_text=新疆美食" -F "modalities=asr" -F "limit=3"
curl.exe -X POST http://127.0.0.1:18301/api/search -F "query_text=烤包子" -F "modalities=visual" -F "limit=3"
curl.exe https://entertainment-grocery-independently-generators.trycloudflare.com/api/health
curl.exe https://entertainment-grocery-independently-generators.trycloudflare.com/api/videos
```

## 前提

- 本机有 NVIDIA GPU，并能运行 `nvidia-smi`。
- Docker Desktop / WSL2 Docker CLI 可用，且容器能访问 GPU。
- 当前 PowerShell 中如果 `docker` 命令不可用，先安装或修复 Docker Desktop/WSL2，再继续。
- 服务器不要先停；迁移完成并验证本地可用后再停。

验证 Docker GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 同步 runtime

不要覆盖本地已有 `runtime/`。服务器数据同步到 `runtime-server/`，容器里仍挂载成 `/app/runtime`，这样 SQLite 中的 `/app/runtime/uploads/...` 路径可以继续工作。

服务器路径：

```text
/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime
```

PowerShell fallback：

```powershell
New-Item -ItemType Directory -Force runtime-server | Out-Null
$remote = "root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime"
scp "$remote/catalog.sqlite3" .\runtime-server\
scp -r "$remote/uploads" .\runtime-server\
scp -r "$remote/indexes" .\runtime-server\
scp -r "$remote/thumbnails" .\runtime-server\
scp -r "$remote/clips" .\runtime-server\
scp -r "$remote/entities" .\runtime-server\
scp -r "$remote/hf_cache" .\runtime-server\
```

如果有 `rsync`，优先使用增量同步：

```bash
mkdir -p runtime-server
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/catalog.sqlite3 runtime-server/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/uploads/ runtime-server/uploads/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/indexes/ runtime-server/indexes/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/thumbnails/ runtime-server/thumbnails/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/clips/ runtime-server/clips/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/entities/ runtime-server/entities/
rsync -a --info=progress2 -e ssh root@110.126.0.52:/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/hf_cache/ runtime-server/hf_cache/
```

## 模型缓存

当前服务器 visual 索引使用 `siglip2-so400m-384`。本地搜索已经有索引向量，但查询文本仍需要 SigLIP2 text encoder。

服务器当前可用的 SigLIP2 缓存在 `/app/runtime/hf_cache`，所以同步 runtime 时要包含 `hf_cache/`。本地 CUDA profile 默认 `VISUAL_HF_CACHE_DIR=/app/runtime/hf_cache`，容器会从 `/app/runtime/hf_cache` 读取它。

如果不同步 `runtime-server/hf_cache`，容器首次 visual 查询会因为缺少本地 SigLIP2 text encoder 而失败；不要依赖运行时从 Hugging Face 下载。ASR/OCR 语义检索使用 MiniLM，也需要提前放在 `models/text-embeddings` 或 `/app/models/text-embeddings` 对应缓存中。

## 启动本地 Docker GPU 后端

```powershell
Copy-Item deploy/env/dev.cuda.example .env
```

如果要继承当前 Cloudflare quick tunnel 的本地入口，把 `.env` 改成：

```text
APP_PORT=18301
APP_PUBLIC_URL=http://127.0.0.1:18301
HOST_RUNTIME_DIR=./runtime-server
HOST_MODEL_DIR=./models
APP_CPUS=12
APP_MEMORY_LIMIT=12g
NPU_ENABLED=false
CUDA_ENABLED=true
VISUAL_MODEL=siglip2-so400m-384
VISUAL_HF_CACHE_DIR=/app/runtime/hf_cache
```

如果 Docker 构建阶段在 `apt-get install` 遇到 Debian 源超时或 502，可以在本机 `.env` 追加：

```env
APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
APT_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
```

如果 `pip install` 从 PyPI 下载依赖时超时，可以追加：

```env
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

启动：

```powershell
docker compose -f compose.yml -f compose.cuda.yml up -d --build
```

验证：

```powershell
Invoke-RestMethod http://127.0.0.1:18301/api/health
Invoke-RestMethod http://127.0.0.1:18301/api/videos
```

搜索 smoke check：

```powershell
$body = @{
  query_text = "足球"
  modalities = @("visual", "asr")
  limit = 5
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:18301/api/search -Method Post -ContentType "application/json" -Body $body
```

## 公网切换

如果 Cloudflare quick tunnel 已经指向 PC 的 `127.0.0.1:18301`，则本地 Docker 后端监听 `18301` 后，可以停止原来的 SSH 转发到服务器，让 Cloudflare 直接进入本地后端。

切换前验证：

1. 本地 `/api/health` 正常。
2. 本地 `/api/videos` 能看到服务器同步过来的素材。
3. visual/asr 搜索有结果。
4. 原视频播放、缩略图和 clip 链接可用。

切换后再访问 trycloudflare 地址验证页面和搜索。只有这些都通过后，才停止服务器容器：

```powershell
ssh root@110.126.0.52 "docker stop momentseek-current-app"
```

不要 `pkill python`，不要 kill 任何未知进程，不要删除服务器 runtime。

## Troubleshooting

### visual 搜索报 `Expected a torch.device ... got:cuda`

原因是 CUDA 设备字符串只有 `cuda`，但 `torch.cuda.set_device(...)` 需要显式设备编号或整数。本地 CUDA 后端应使用 `cuda:0`。

已在 `backend/app/indexing/visual.py` 的 `resolve_device(..., cuda_enabled=True)` 中修复，并用 `backend/tests/test_visual_cache.py` 覆盖。

### Docker 构建很慢或下载失败

优先检查 `.env` 是否配置了：

```env
APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
APT_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

`Dockerfile.cuda` 已开启 apt retry 和 pip cache；重建时大部分依赖层应复用缓存。
