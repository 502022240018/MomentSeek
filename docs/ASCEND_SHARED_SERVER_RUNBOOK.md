# MomentSeek 共享 Ascend 服务器操作手册

本文档面向第一次接手服务器部署、更新和排障的同事，记录 `800IA2` 共享昇腾服务器上 MomentSeek 的实际约定。通用发布原则见 [DEPLOYMENT.md](DEPLOYMENT.md)，共享环境安全规则见 [OPERATIONS.md](OPERATIONS.md)。

日常查看/创建/取消任务、日志跟踪、资源监控和故障信息采集命令见 [ASCEND_OPERATIONS_COMMANDS.md](ASCEND_OPERATIONS_COMMANDS.md)。

> 状态快照：2026-07-20。IP、空闲 NPU、镜像版本和 Git 提交都可能变化，写操作前必须重新检查。

## 1. 当前服务器约定

| 项目 | 当前值 | 说明 |
| --- | --- | --- |
| 主机 | `800IA2` | openEuler 24.03 LTS，aarch64 |
| SSH 地址 | `100.199.4.24` | 账号和认证方式由管理员提供 |
| 公网入口 | `http://140.210.239.19:8000` | 当前防火墙已放行 `8000/tcp` |
| 工作根目录 | `/home/momentseek-29154` | 本项目专用目录 |
| 代码目录 | `/home/momentseek-29154/platform` | Git 仓库，`main` 分支 |
| 模型目录 | `/home/momentseek-29154/models/platform` | 挂载为 `/app/models` |
| 运行数据 | `/home/momentseek-29154/runtime` | 挂载为 `/app/runtime` |
| 日志目录 | `/home/momentseek-29154/logs` | 部署和专项测试日志 |
| 容器名 | `momentseek-29154-platform` | 只操作这个明确归属本项目的容器 |
| 镜像名 | `momentseek-29154-platform` | Git commit tag，另有 `current` tag |
| 服务端口 | `8000` | 脚本默认 18500，本机部署须显式覆盖 |
| 平台物理 NPU | `5` | 当前约定；每次部署仍须确认空闲 |
| 容器逻辑 NPU | `0` | 物理卡映射后应用看到 `npu:0` |
| 实验 NPU | `6` 等获准空闲卡 | 不得假设永久空闲 |
| Git 仓库 | `https://github.com/502022240018/MomentSeek.git` | 连接偶有超时，脚本有重试 |

不要把密码、Token、私钥写入仓库、脚本或日志。公网 Web IP 不一定是 SSH 地址。

## 2. 目录和数据

```text
/home/momentseek-29154/
├── platform/                 # Git 代码仓库
│   └── .server-build/        # 部署时生成，不提交 Git
├── models/platform/          # 模型，独立于镜像
├── runtime/
│   ├── catalog.sqlite3       # 视频、任务、人物等元数据
│   ├── uploads/              # 上传视频
│   └── indexes/              # 各通道索引
├── logs/                     # 部署、构建和实验日志
├── releases/                 # 预留发布产物
└── cache/                    # 项目专用缓存
```

- 更新代码不会自动更新模型，也不应删除运行数据。
- 重新构建镜像不复制大模型；模型由宿主目录挂载。
- `catalog.sqlite3`、`uploads/` 和 `indexes/` 是相关数据，迁移时一起考虑。
- 禁止未经确认删除 `runtime/`、`models/` 或清理全局 Docker 数据。

## 3. 登录和变更前检查

Windows PowerShell：

```powershell
ssh root@100.199.4.24
```

服务器上每次变更前执行只读检查：

```bash
cd /home/momentseek-29154/platform

docker ps --filter 'name=^/momentseek-29154-platform$' \
  --format 'name={{.Names}} image={{.Image}} status={{.Status}}'
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/jobs
npu-smi info -t proc-mem -i 5 -c 0
git status -sb
git log -1 --oneline
df -h /home/momentseek-29154
```

如有 `queued` 或 `running` 任务，先在页面取消，或调用：

```bash
curl -fsS -X POST http://127.0.0.1:8000/api/jobs/<JOB_ID>/cancel
```

确认状态为 `cancelled` 后再部署。禁止使用 `pkill python`、`killall` 或模糊匹配批量结束进程。

## 4. 首次部署

宿主机需要 Docker、Git、curl、`npu-smi`、`flock`、`ss`、`sha256sum`、Ascend 驱动和基础 MindIE 镜像。还需准备：

- `vendor-wheels/insightface-1.0.1-py3-none-any.whl`；该可信构建产物不进 Git。
- `/home/momentseek-29154/models/platform` 下的离线模型。
- 构建期间可访问 PyPI 和 npm 镜像；GitHub 仅用于获取代码。

首次获取源码：

```bash
mkdir -p /home/momentseek-29154
cd /home/momentseek-29154
git -c http.version=HTTP/1.1 clone \
  --depth 20 --filter=blob:none --no-tags \
  --branch main --single-branch \
  https://github.com/502022240018/MomentSeek.git platform
```

也可以从已有源码运行 `bash scripts/bootstrap_ascend_server_source.sh`。它只准备源码和目录，不构建镜像、不安装依赖、不启动服务。

## 5. 一条命令更新并部署

```bash
cd /home/momentseek-29154/platform
mkdir -p /home/momentseek-29154/logs
set -o pipefail

NPU_ID=5 APP_PORT=8000 \
  bash scripts/update_and_deploy_ascend_shared_server.sh \
  2>&1 | tee /home/momentseek-29154/logs/update-and-deploy-$(date +%F-%H%M%S).log
```

直接用 `bash` 不要求脚本具有可执行位。若要使用 `./scripts/update_and_deploy_ascend_shared_server.sh`，先运行：

```bash
chmod +x scripts/update_and_deploy_ascend_shared_server.sh
```

必须显式传 `NPU_ID=5 APP_PORT=8000`，否则脚本会用自身默认端口 18500，与当前公网入口不一致。

## 6. 两层脚本分别做什么

### 6.1 `update_and_deploy_ascend_shared_server.sh`

1. 检查 `git`、`curl`、`docker`。
2. 探测 GitHub；探测失败只警告，真正 Git 操作仍会继续。
3. 代码不存在则浅克隆；存在则校验 `origin`。
4. 若 tracked 文件有本地修改则停止，避免覆盖服务器改动。
5. 以 HTTP/1.1 最多重试三次 fetch，只允许 fast-forward 更新。
6. 校验最低部署提交、部署脚本和前端工具链。
7. 调用下层部署脚本。

正常情况下不需要事先手动 `git pull`。

### 6.2 `deploy_ascend_shared_server.sh`

1. 获取部署锁，防止并发部署。
2. 检查磁盘、NPU、端口、Git 和 InsightFace wheel 校验和。
3. 在 `.server-build/` 生成 openEuler/MindIE 依赖约束和 Dockerfile。
4. 构建前端、安装后端依赖，并保留基础镜像的 Torch/torch-npu 栈。
5. 生成以 Git commit 短 SHA 命名的镜像。
6. 查询旧服务任务，存在 queued/running 任务时拒绝部署。
7. 将旧容器改名为 `momentseek-29154-platform-rollback` 并停止，以释放常驻模型占用的 NPU。
8. 最多重试三次 NPU、关键 Python 包和 Silero ONNX smoke test，再用原 runtime/models 挂载启动新容器。
9. 轮询 health；失败自动删除新容器并恢复旧容器。
10. 成功后删除临时回滚容器，并标记镜像 `current`。
11. 输出模型清单；缺模型会报告，但不会让 API 部署整体失败。

Docker 会复用未变化的构建层。只改应用代码仍需生成镜像和替换容器，但通常不会重装全部依赖。

## 7. 当前运行配置

```text
ENV_PROFILE=prod.ascend
APP_PORT=8000
APP_DATA_DIR=/app/runtime
APP_MODEL_DIR=/app/models
NPU_ENABLED=true
NPU_DEVICE_ID=0
MODEL_IDLE_POLICY=resident
INDEXER_MODE=daemon
NPU_WORKER_MODE=isolated
INDEXER_IDLE_TIMEOUT_SECONDS=0
VISUAL_MODEL=siglip2-so400m-384
FACE_PROVIDER=cann
FACE_ORT_INTRA_OP_THREADS=8
FACE_ORT_INTER_OP_THREADS=1
ASR_ENGINE=funasr
ASR_DEVICE=auto
ASR_VAD_STRATEGY=silero_12s
OCR_ENGINE=rapidocr_acl
OCR_DEVICE=npu
OCR_ACL_MODEL_DIR=rapidocr/ascend/910b4-cann9-profile
```

索引由 daemon 串行调度；visual、face、ASR、OCR 各自在独立的常驻子进程中持有模型和 NPU context，禁止把 Torch-NPU、ORT CANN 和原生 ACL 合回同一进程。正式 OCR 使用已验证的 PP-OCRv6 OM + ACL 路径，不使用结果不正确的 ONNX Runtime CANN OCR 路径。

`CPU_THREAD_LIMIT=8` 约束 BLAS、OpenMP 和 CPU ORT，但不能约束 CANN EP 与 ffmpeg 自己的线程。2026-07-21 回归峰值约 810 PID、19.7 GiB 宿主内存；部署脚本因此默认增加 24 CPU 配额和 2048 PID 上限。不要把 PID 上限直接压到 1024，初始化峰值余量太小。

容器使用 `--network host`，Uvicorn 直接监听宿主机 `0.0.0.0:8000`，不是 Docker `-p` 映射。

## 8. 模型路径与校验

生产需求由 `deploy/models/ascend-prod.models.json` 定义。清单 target 是容器路径，例如：

```text
/app/models/hf-cache
/app/models/insightface/models/buffalo_l
/app/models/funasr/models/iic--SenseVoiceSmall/snapshots/master
/app/models/text-embeddings
/app/models/rapidocr
```

对应宿主机根目录为 `/home/momentseek-29154/models/platform`。检查模型：

```bash
docker exec momentseek-29154-platform \
  python3 /app/scripts/verify_models.py \
  --manifest /app/deploy/models/ascend-prod.models.json \
  --lock /app/models/models.lock.json
```

缺模型时先核对实际目录、manifest target 和代码加载路径。不要随意复制第二套大模型；布局不同可评估移动、软链接或统一修改清单。

## 9. 部署验收

```bash
curl -fsS http://127.0.0.1:8000/api/health
docker ps --filter 'name=^/momentseek-29154-platform$' \
  --format 'name={{.Names}} image={{.Image}} status={{.Status}}'
docker logs --tail 100 momentseek-29154-platform
ss -lntp | grep ':8000'
npu-smi info -t proc-mem -i 5 -c 0
```

预期 health 为 `status: ok`、`env_profile: prod.ascend`、`npu_enabled: true`，容器最终为 healthy，`0.0.0.0:8000` 正在监听，公网入口可打开。

服务刚启动且尚未执行 NPU 索引时，`npu-smi` 显示无进程不一定是故障。

## 10. 任务、日志和资源排查

```bash
# 所有任务及单个任务
curl -fsS http://127.0.0.1:8000/api/jobs | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/api/jobs/<JOB_ID> | python3 -m json.tool

# 服务日志；Ctrl+C 只退出查看，不停止容器
docker logs --tail 200 momentseek-29154-platform
docker logs -f --since 10m momentseek-29154-platform

# 最近生成的索引文件
find /home/momentseek-29154/runtime/indexes \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %10s %p\n' 2>/dev/null \
  | sort -r | head -30

# 容器进程、NPU 和 CPU/内存
docker exec momentseek-29154-platform \
  ps -ef | grep -E 'uvicorn|indexer_daemon|stage_runner|worker' | grep -v grep
npu-smi info -t proc-mem -i 5 -c 0
docker stats --no-stream momentseek-29154-platform
```

任务状态包括 `queued`、`running`、`completed`、`failed`、`cancelled`；`stage` 是当前通道，`metrics` 是阶段计时。任务初期没有索引文件不能单独证明卡死，要结合 API、日志和资源判断。

## 11. 常见问题

### GitHub 或 fetch 超时

入口脚本的网络探测只作参考，fetch 会重试三次。可稍后重跑整个部署命令，不要并发运行。单独验证：

```bash
git -C /home/momentseek-29154/platform \
  -c http.version=HTTP/1.1 \
  -c http.lowSpeedLimit=1 \
  -c http.lowSpeedTime=60 \
  fetch --deepen=20 --prune origin main
```

持续失败时确认网络策略，不要删除现有仓库重新 clone。

### `tracked source files have local modifications`

```bash
cd /home/momentseek-29154/platform
git status --short
git diff -- <被修改文件>
```

确认修改无价值且目标明确后才恢复单个文件。禁止 `git reset --hard`。脚本显示 `M` 常表示服务器上曾临时修改，必须先看 diff。

### 本机 health 正常，公网打不开

```bash
curl -fsS http://127.0.0.1:8000/api/health
ss -lntp | grep ':8000'
firewall-cmd --list-all
```

本机正常并监听 `0.0.0.0:8000`，但客户端端口测试失败，通常是主机防火墙或上游安全组。换一个未放行端口不会自动解决。防火墙变更需确认授权和端口归属。

临时 SSH 隧道必须在 Windows 本机执行：

```powershell
ssh -N -L 18500:127.0.0.1:8000 root@100.199.4.24
```

随后访问 `http://127.0.0.1:18500`。左侧本地端口占用时可换其他端口。

### 新容器健康失败

脚本会自动恢复旧容器。检查：

```bash
docker ps -a --filter name=momentseek-29154-platform
docker logs --tail 300 momentseek-29154-platform
docker images 'momentseek-29154-platform*'
```

不要在部署过程中删除 `-rollback` 容器，它可能是唯一自动回滚目标。

### 构建很慢

```bash
tail -f /home/momentseek-29154/logs/platform-image-build.log
```

先辨别是 Git、npm/pip、前端还是 Docker 层。`Using cache` 表示复用了缓存；依赖、Dockerfile、基础镜像或 lock 变化会使相应层重建。

## 12. 回滚、备份和边界

正常部署内置健康失败自动回滚。上线后发现功能问题时，先停止新任务并取消活动任务，再查看旧镜像：

```bash
docker images momentseek-29154-platform \
  --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}'
```

手动回滚是有影响操作。必须确认旧 tag、端口、NPU、挂载和无活动任务。优先选择对应 Git 提交后重跑版本化部署脚本，不要凭记忆拼缺参数的 `docker run`。

重要演示前应一致性备份 `catalog.sqlite3`、`uploads/`、`indexes/` 和 `models.lock.json`。SQLite 正在写入时不能只复制单个数据库文件，应先停止写入或使用 SQLite 一致性备份。

未经确认禁止：

- 删除 runtime、models、uploads、indexes。
- 清理所有 Docker 镜像、volume 或全局缓存。
- 重启 Docker、服务器或 Ascend 驱动。
- 结束未知 Python、VLLM、ComfyUI 或其他用户进程。
- 抢占未明确获准的 NPU。

每次重要变更记录时间、Git commit、镜像 tag、NPU、端口、执行命令、health、模型校验和是否回滚。

## 13. 本机已知环境和专项实验

### 13.1 环境快照

当前基础 MindIE 容器曾验证：Python 3.11.6、Torch 2.9.0、torch-npu 2.9.0.post1、Transformers 4.51.0、MindIE 3.0 开发构建、ffmpeg 可用，主机可见 8 张 NPU。NPU 6 的实验板卡识别为 Ascend 910B4。当前平台容器中的 ATC、pyACL、OPP 和运行库均来自 CANN 8.5.1；早期宿主审计曾出现 9.0 beta 信息，不能用于描述当前容器。

这些是审计快照，不是代码依赖声明。应以当前镜像构建输出和下面命令重新确认：

```bash
docker exec -i momentseek-29154-platform python3 - <<'PY'
import torch, torch_npu, transformers, onnxruntime
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('transformers=', transformers.__version__)
print('ort_providers=', onnxruntime.get_available_providers())
PY
```

服务器对 GitHub、Hugging Face、ModelScope 和 Docker Hub 的连接可能间歇失败。正式运行不依赖外网下载；大模型应在其他可联网环境准备、校验后传到模型挂载目录。超过 1 GB 的文件优先用支持断点续传的 `rsync --partial --append-verify`；不能使用 rsync/scp 时再分卷传输并在服务器合并，合并后必须核对 SHA-256。

### 13.2 OCR OM + ACL 实验（不属于正式服务）

目标是保留 PP-OCRv6 Small 模型和前后处理，只把不正确的 ONNX Runtime CANN 执行路径替换为提前编译的 OM + ACL。实验使用获准的 NPU 6，不能占用平台 NPU 5。

当前真实运行模型为：

```text
det: /app/models/rapidocr/PP-OCRv6_det_small.onnx
cls: /app/models/rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx
rec: /app/models/rapidocr/PP-OCRv6_rec_small.onnx
```

先把脚本复制进当前容器，再对上传目录记录索引分辨率下的真实张量形状：

```bash
cd /home/momentseek-29154/platform
docker cp scripts/ocr_shape_profile.py \
  momentseek-29154-platform:/tmp/ocr_shape_profile.py

docker exec -e PYTHONPATH=/app/backend momentseek-29154-platform \
  python3 /tmp/ocr_shape_profile.py \
  --video-root /app/runtime/uploads \
  --model-root /app/models/rapidocr \
  --output /app/runtime/ocr-shape-profile.json \
  --decode-height 720 \
  --frames-per-video 12 \
  --max-videos 12
```

查看概要：

```bash
python3 - <<'PY'
import json
p='/home/momentseek-29154/runtime/ocr-shape-profile.json'
d=json.load(open(p, encoding='utf-8'))
print('videos =', len(d['videos']))
print(json.dumps(d['tensor_shapes'], ensure_ascii=False, indent=2))
PY
```

在具备 ATC 且挂载实验 NPU 的隔离容器中编译精确形状 OM：

```bash
docker cp scripts/build_ppocr_om_from_profile.py \
  momentseek-29154-platform:/tmp/build_ppocr_om_from_profile.py

docker exec -e PYTHONPATH=/app/backend momentseek-29154-platform \
  python3 /tmp/build_ppocr_om_from_profile.py \
  --profile /app/runtime/ocr-shape-profile.json \
  --model-root /app/models/rapidocr \
  --output-dir /app/models/rapidocr/ascend/910b4-cann9-profile \
  --soc-version Ascend910B4 \
  --precision-mode must_keep_origin_dtype \
  2>&1 | tee /home/momentseek-29154/logs/ppocr-om-build.log
```

上述命令展示参数和产物约定，但当前正式平台容器只挂载物理 NPU 5；不要为了实验直接修改或重启正式容器。实际编译应使用单独命名、显式挂载 NPU 6 的实验容器。当前 exact-shape manifest 的 `product_ready=false`，只用于可行性验证。

生成的 OM 与 SoC、shape 策略及 CANN/Runtime 版本相关。当前实验 OM 由 CANN 8.5.1 生成并在同版本 pyACL 上验证，不能未经验证复制到其他 CANN 版本。接入生产前至少完成：CPU ONNX 与 OM 原始输出对齐、端到端文字结果对齐、冷启动/稳态速度、不同分辨率和异常输入测试。

## 14. 常用命令速查

```bash
# 更新、构建、部署
cd /home/momentseek-29154/platform
set -o pipefail
NPU_ID=5 APP_PORT=8000 \
  bash scripts/update_and_deploy_ascend_shared_server.sh \
  2>&1 | tee /home/momentseek-29154/logs/update-and-deploy-$(date +%F-%H%M%S).log

# 健康、任务、取消
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/jobs | python3 -m json.tool
curl -fsS -X POST http://127.0.0.1:8000/api/jobs/<JOB_ID>/cancel

# 日志、资源、版本
docker logs -f --since 10m momentseek-29154-platform
npu-smi info -t proc-mem -i 5 -c 0
docker stats --no-stream momentseek-29154-platform
git -C /home/momentseek-29154/platform log -1 --oneline
docker inspect momentseek-29154-platform \
  --format 'image={{.Config.Image}} status={{.State.Status}} health={{.State.Health.Status}}'
```

如本文档与脚本冲突，以当前 Git 提交中的脚本为准并同步修正文档；地址、NPU 和端口以部署前的现场只读检查为准。
