# MomentSeek 日常运维与故障排查命令手册

本文档用于 `800IA2` 共享 Ascend 服务器上的日常值守，覆盖服务、视频、索引任务、日志、资源、故障定位和恢复。服务器目录、部署原理和首次部署见 [ASCEND_SHARED_SERVER_RUNBOOK.md](ASCEND_SHARED_SERVER_RUNBOOK.md)。

> 当前约定：代码 `/home/momentseek-29154/platform`，运行数据 `/home/momentseek-29154/runtime`，容器 `momentseek-29154-platform`，宿主端口 `8000`，物理 NPU `5`。这些值变化时先修改本节变量，不要盲目复制命令。

## 1. 建议先设置的变量

登录服务器后执行：

```bash
export MS_ROOT=/home/momentseek-29154
export MS_REPO=/home/momentseek-29154/platform
export MS_RUNTIME=/home/momentseek-29154/runtime
export MS_CONTAINER=momentseek-29154-platform
export MS_PORT=8000
export MS_NPU=5
export MS_API=http://127.0.0.1:${MS_PORT}

cd "$MS_REPO"
```

这些变量只在当前 shell 会话有效，不会修改平台配置。

## 2. 一分钟巡检

每次接手、部署前或发现页面异常时，依次执行：

```bash
date
uptime

docker ps --filter "name=^/${MS_CONTAINER}$" \
  --format 'name={{.Names}} image={{.Image}} status={{.Status}}'

curl -fsS "${MS_API}/api/health" | python3 -m json.tool
curl -fsS "${MS_API}/api/jobs" | python3 -m json.tool

npu-smi info -t proc-mem -i "$MS_NPU" -c 0
docker stats --no-stream "$MS_CONTAINER"
df -h "$MS_ROOT"
df -i "$MS_ROOT"
```

判断顺序：

1. 容器是否 Up/healthy。
2. health API 是否返回 `status=ok`。
3. 是否有 queued/running/failed 任务。
4. NPU 5 是否由本项目使用，显存是否异常。
5. CPU、内存、磁盘空间和 inode 是否耗尽。

## 3. 服务查看、启动、停止和重启

### 3.1 查看服务

```bash
docker ps -a --filter "name=^/${MS_CONTAINER}$"
docker inspect "$MS_CONTAINER" \
  --format 'image={{.Config.Image}} running={{.State.Running}} status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} started={{.State.StartedAt}}'
curl -fsS "${MS_API}/api/health" | python3 -m json.tool
ss -lntp | grep ":${MS_PORT}"
```

### 3.2 启动已存在的容器

仅在容器状态为 Exited 时使用：

```bash
docker start "$MS_CONTAINER"

for i in $(seq 1 30); do
  curl -fsS --max-time 3 "${MS_API}/api/health" && break
  sleep 2
done
```

### 3.3 安全停止服务

先确认没有活动任务：

```bash
curl -fsS "${MS_API}/api/jobs" | python3 -c '
import json, sys
jobs=json.load(sys.stdin)
active=[j for j in jobs if j.get("status") in {"queued", "running"}]
print(json.dumps(active, ensure_ascii=False, indent=2))
raise SystemExit(1 if active else 0)
'
```

输出 `[]` 且退出码为 0 后：

```bash
docker stop --time 30 "$MS_CONTAINER"
```

停止容器会释放常驻模型和 NPU，但不会删除宿主机的模型、上传视频或索引。

### 3.4 安全重启服务

不要对有活动任务的容器直接 `docker restart`。确认任务为空后：

```bash
docker restart --time 30 "$MS_CONTAINER"

for i in $(seq 1 30); do
  if curl -fsS --max-time 3 "${MS_API}/api/health"; then
    echo
    break
  fi
  sleep 2
done
```

重启后 queued 任务仍保存在 SQLite，daemon 会继续消费；但运维上仍应先弄清任务为何未运行，避免用重启掩盖问题。

### 3.5 更新并重新部署

```bash
cd "$MS_REPO"
mkdir -p "$MS_ROOT/logs"
set -o pipefail

NPU_ID="$MS_NPU" APP_PORT="$MS_PORT" \
  bash scripts/update_and_deploy_ascend_shared_server.sh \
  2>&1 | tee "$MS_ROOT/logs/update-and-deploy-$(date +%F-%H%M%S).log"
```

部署脚本会拒绝在存在 queued/running 任务时停止旧服务。

## 4. 视频管理

### 4.1 列出视频

```bash
curl -fsS "${MS_API}/api/videos" | python3 -m json.tool
```

只显示常用字段：

```bash
curl -fsS "${MS_API}/api/videos" | python3 -c '
import json, sys
for v in json.load(sys.stdin):
    print(v["id"], v.get("status"), v.get("duration"), v.get("indexed_modalities"), v.get("name"))
'
```

### 4.2 查看一个视频

```bash
VIDEO_ID=<视频ID>
curl -fsS "${MS_API}/api/videos/${VIDEO_ID}" | python3 -m json.tool
```

### 4.3 上传视频

```bash
VIDEO_FILE='/绝对路径/示例.mp4'
curl --fail-with-body --progress-bar \
  -F "video=@${VIDEO_FILE}" \
  "${MS_API}/api/videos"
```

带已有转写 sidecar：

```bash
TRANSCRIPT_FILE='/绝对路径/示例.json'
curl --fail-with-body --progress-bar \
  -F "video=@${VIDEO_FILE}" \
  -F "transcript=@${TRANSCRIPT_FILE}" \
  "${MS_API}/api/videos"
```

响应中的 `id` 就是后续使用的 `VIDEO_ID`。

### 4.4 重命名视频

```bash
curl --fail-with-body -X PATCH \
  -H 'Content-Type: application/json' \
  -d '{"name":"新的展示名称.mp4"}' \
  "${MS_API}/api/videos/${VIDEO_ID}" | python3 -m json.tool
```

### 4.5 删除视频

```bash
curl --fail-with-body -X DELETE \
  "${MS_API}/api/videos/${VIDEO_ID}" | python3 -m json.tool
```

删除会同时移除该视频上传文件、索引和缓存，是不可逆操作。有 queued/running 任务时 API 会拒绝。执行前再次核对 `VIDEO_ID` 和名称。

## 5. 创建索引任务

可用通道是 `visual`、`face`、`asr`、`ocr`。Speaker 不是独立 modality，而是 ASR 的可选子功能 `asr_speaker_enabled`。

### 5.1 默认索引（Visual + Face + ASR）

```bash
curl --fail-with-body -X POST \
  -H 'Content-Type: application/json' \
  -d '{}' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool
```

### 5.2 指定通道

只建 Visual：

```bash
curl --fail-with-body -X POST \
  -H 'Content-Type: application/json' \
  -d '{"modalities":["visual"]}' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool
```

建 Visual、Face、ASR、OCR：

```bash
curl --fail-with-body -X POST \
  -H 'Content-Type: application/json' \
  -d '{"modalities":["visual","face","asr","ocr"]}' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool
```

ASR 并启用 Speaker：

```bash
curl --fail-with-body -X POST \
  -H 'Content-Type: application/json' \
  -d '{"modalities":["asr"],"asr_speaker_enabled":true}' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool
```

OCR 当前正式配置仍在 CPU 上，长视频会明显慢于 NPU 通道；不要因任务慢就直接结束容器。

### 5.3 使用明确的索引参数

仅在实验方案已经定义参数时使用，不要日常随意调参：

```bash
curl --fail-with-body -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "modalities":["visual","face","asr"],
    "visual_model":"siglip2-so400m-384",
    "visual_sample_fps":5.0,
    "face_sample_fps":1.0,
    "asr_engine":"funasr",
    "asr_language":"auto",
    "asr_speaker_enabled":true
  }' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool
```

接口响应中的 `id` 是 `JOB_ID`。同一视频已有 queued/running 任务时，API 返回 HTTP 409。

## 6. 查看、等待和取消任务

### 6.1 列出所有任务

```bash
curl -fsS "${MS_API}/api/jobs" | python3 -m json.tool
```

按视频过滤：

```bash
curl -fsS "${MS_API}/api/jobs?video_id=${VIDEO_ID}" | python3 -m json.tool
```

只看活动任务：

```bash
curl -fsS "${MS_API}/api/jobs" | python3 -c '
import json, sys
for j in json.load(sys.stdin):
    if j.get("status") in {"queued", "running"}:
        print(j["id"], j["status"], j.get("stage"), j.get("progress"), j.get("video_id"), j.get("modalities"))
'
```

### 6.2 查看一个任务

```bash
JOB_ID=<任务ID>
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
```

重点字段：

- `status`：queued、running、completed、failed、cancelled。
- `stage`：正在运行的通道或 starting/completed/failed/cancelled。
- `progress`：总体进度，不代表通道内部逐帧进度。
- `metrics`：已经完成阶段的耗时等指标。
- `error`：失败或取消原因。
- `updated_at`：任务记录最近更新时间。

### 6.3 持续观察任务

若安装了 `watch`：

```bash
watch -n 3 "curl -fsS '${MS_API}/api/jobs/${JOB_ID}' | python3 -m json.tool"
```

没有 `watch`：

```bash
while true; do
  clear
  date
  curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
  sleep 3
done
```

按 `Ctrl+C` 只停止监看，不会取消任务。

### 6.4 取消一个任务

```bash
curl --fail-with-body -X POST \
  "${MS_API}/api/jobs/${JOB_ID}/cancel" | python3 -m json.tool
```

- queued：从逻辑上取消，不影响正在运行的其他任务。
- running：标为 cancelled，并重启索引 daemon 来中断当前工作；其他 queued 任务仍保留。
- completed/failed：不能取消，API 返回 HTTP 409。

取消后确认：

```bash
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
tail -n 100 "$MS_RUNTIME/indexer-daemon.log"
```

不要为了取消单个任务执行 `docker stop`、`pkill python` 或 `killall python`。

## 7. 日志在哪里、分别看什么

### 7.1 API、容器启动和崩溃日志

```bash
docker logs --tail 200 "$MS_CONTAINER"
docker logs --since 30m --timestamps "$MS_CONTAINER"
docker logs -f --since 10m --timestamps "$MS_CONTAINER"
```

用于查看：Uvicorn 启动、HTTP 请求、应用启动失败、容器退出前异常。`Ctrl+C` 只退出日志跟随。

### 7.2 索引 daemon 日志

daemon 模式的主要索引日志：

```bash
ls -lh "$MS_RUNTIME/indexer-daemon.log"
tail -n 200 "$MS_RUNTIME/indexer-daemon.log"
tail -F "$MS_RUNTIME/indexer-daemon.log"
```

按任务 ID 搜索：

```bash
grep -n -C 5 "$JOB_ID" "$MS_RUNTIME/indexer-daemon.log" | tail -100
```

用于查看：daemon 是否启动、领取哪个 job、通道顺序、模型加载、阶段异常和 traceback。

### 7.3 单任务日志

`job-<id>.log` 主要用于 subprocess 模式；当前正式环境是 daemon 模式，不一定生成它：

```bash
JOB_LOG="$MS_RUNTIME/job-${JOB_ID}.log"
if [[ -f "$JOB_LOG" ]]; then
  tail -n 200 "$JOB_LOG"
else
  echo "No per-job log; inspect indexer-daemon.log and docker logs"
fi
```

### 7.4 部署和镜像构建日志

```bash
ls -lht "$MS_ROOT/logs" | head -20
tail -n 200 "$MS_ROOT/logs/platform-image-build.log"
tail -F "$MS_ROOT/logs/platform-image-build.log"
```

日志中可能包含服务器路径和模型名，上传外部系统前先脱敏；不得上传 Token、密码、私钥或完整环境变量。

## 8. 查看资源和产物增长

### 8.1 NPU

```bash
npu-smi info
npu-smi info -t proc-mem -i "$MS_NPU" -c 0
```

平台只应使用获准的物理 NPU。容器内显示 `npu:0` 是映射后的逻辑编号，不代表宿主物理卡 0。

### 8.2 CPU、内存和容器进程

```bash
docker stats --no-stream "$MS_CONTAINER"
docker top "$MS_CONTAINER" -eo pid,ppid,pcpu,pmem,etime,args
docker exec "$MS_CONTAINER" ps -ef
```

### 8.3 磁盘和目录大小

```bash
df -h "$MS_ROOT"
df -i "$MS_ROOT"
du -sh "$MS_RUNTIME" "$MS_ROOT/models/platform" "$MS_ROOT/logs"
du -sh "$MS_RUNTIME"/* 2>/dev/null | sort -h
```

### 8.4 最近产生的索引文件

```bash
find "$MS_RUNTIME/indexes" \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %p\n' 2>/dev/null \
  | sort -r | head -40
```

指定视频：

```bash
find "$MS_RUNTIME/indexes/${VIDEO_ID}" \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %p\n' 2>/dev/null \
  | sort -r
```

任务早期或某个阶段内部可能长时间不落最终文件，因此“文件没变化”只能作为一个信号，不能单独判定卡死。

## 9. 常见问题排查

### 9.1 页面打不开

```bash
docker ps --filter "name=^/${MS_CONTAINER}$"
curl -v --max-time 5 "${MS_API}/api/health"
ss -lntp | grep ":${MS_PORT}"
firewall-cmd --list-all 2>/dev/null || true
docker logs --tail 200 "$MS_CONTAINER"
```

- 本机 health 失败：先查容器和应用日志。
- 本机成功但公网失败：检查服务是否监听 `0.0.0.0`、防火墙和上游安全组。
- 换到一个没放行的端口不会自动解决。

### 9.2 任务一直 queued

```bash
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
docker exec "$MS_CONTAINER" \
  ps -ef | grep '[a]pp.indexer_daemon'
tail -n 200 "$MS_RUNTIME/indexer-daemon.log"
```

可能原因：前一个任务仍在运行；daemon 没启动/已退出；容器刚重启；SQLite 或运行目录异常。当前设计串行处理任务，前面有 running 时后面的 queued 是正常现象。

若没有任何 running、daemon 进程不存在，先保留日志并确认无活动写入，再安全重启容器。不要直接把 SQLite 中的状态手工改成 running。

### 9.3 任务 running 但进度长时间不变

同时采集：

```bash
date
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
tail -n 200 "$MS_RUNTIME/indexer-daemon.log"
npu-smi info -t proc-mem -i "$MS_NPU" -c 0
docker stats --no-stream "$MS_CONTAINER"
docker top "$MS_CONTAINER" -eo pid,ppid,pcpu,pmem,etime,args
find "$MS_RUNTIME/indexes/${VIDEO_ID}" -type f -ls 2>/dev/null | tail -30
```

注意：`progress` 主要在通道切换时更新，Visual/Face/OCR 等单一阶段内部可能不连续增长。应比较日志时间戳、CPU/NPU 活动和产物，而不是只看百分比。

确认异常后使用取消 API；不要先杀进程。取消也无响应时再保存上述证据，检查 API/container 状态。

### 9.4 任务 failed

```bash
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool
grep -n -C 20 "$JOB_ID" "$MS_RUNTIME/indexer-daemon.log" | tail -200
tail -n 300 "$MS_RUNTIME/indexer-daemon.log"
docker logs --since 30m --timestamps "$MS_CONTAINER" | tail -300
```

优先记录第一个 traceback 和其前面的错误，不要只截取最后一行。常见方向：模型路径缺失、模型加载失败、NPU 资源忙、视频解码失败、磁盘满、CANN/算子错误、内存不足。

若出现 `107003`、`stream is not in current context`，先确认正式容器为隔离模式并查看各阶段记录的 worker PID：

```bash
docker inspect "$MS_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^NPU_WORKER_MODE='
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool \
  | grep -E 'isolated_worker_pid|isolated_worker_attempts|107003|current context'
```

期望 `NPU_WORKER_MODE=isolated`，不同通道 PID 不同，同一通道跨连续任务 PID 相同。不要仅重启后继续使用 `legacy`；那只会暂时清空 context，下一次混用运行时仍可能复现。

### 9.5 NPU 无进程

- 无任务时：正常，模型可能尚未加载或已随容器重启释放。
- NPU 阶段 running 时：结合日志检查模型是否仍在 CPU 加载/预处理，或配置没有启用 NPU。
- Face/Visual 完成后进入 CPU OCR：NPU 低占用可能正常。

检查容器配置：

```bash
docker inspect "$MS_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'NPU|FACE_PROVIDER|OCR_DEVICE|ASR_DEVICE|INDEXER_MODE|INDEXER_IDLE_TIMEOUT'
```

### 9.6 NPU `Resource_Busy / 507899`

```bash
npu-smi info -t proc-mem -i "$MS_NPU" -c 0
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

不要结束未知占用者。部署脚本已处理旧平台常驻模型与新 smoke 容器的冲突：它会先审计任务、停止旧平台释放 NPU，再测试新镜像，失败则恢复旧平台。

### 9.7 磁盘满或 inode 满

```bash
df -h "$MS_ROOT"
df -i "$MS_ROOT"
du -sh "$MS_RUNTIME"/* "$MS_ROOT/logs"/* 2>/dev/null | sort -h | tail -30
```

不要执行 `docker system prune -a`，也不要批量删除 uploads/indexes/models。先定位归属和保留策略，再由负责人批准具体目标。

### 9.8 模型缺失

```bash
docker exec "$MS_CONTAINER" \
  python3 /app/scripts/verify_models.py \
  --manifest /app/deploy/models/ascend-prod.models.json \
  --lock /app/models/models.lock.json
```

清单使用容器路径 `/app/models/...`，宿主机根目录是 `$MS_ROOT/models/platform`。优先核对挂载和目录布局，不要让正式服务临时联网下载。

### 9.9 数据库只读检查

严禁直接修改任务状态。可执行 SQLite 一致性检查：

```bash
python3 - <<'PY'
import sqlite3
p='/home/momentseek-29154/runtime/catalog.sqlite3'
con=sqlite3.connect(f'file:{p}?mode=ro', uri=True)
print(con.execute('PRAGMA integrity_check').fetchone()[0])
print('videos=', con.execute('SELECT count(*) FROM videos').fetchone()[0])
print('jobs=', con.execute('SELECT count(*) FROM jobs').fetchone()[0])
con.close()
PY
```

预期 `integrity_check` 输出 `ok`。异常时停止写入并先备份，不要尝试在线修表。

## 10. 标准故障信息采集

出现问题时一次性执行以下只读脚本块，便于他人分析：

```bash
OUT="$MS_ROOT/logs/ops-diagnostic-$(date +%F-%H%M%S).txt"
{
  echo '===== time ====='
  date
  uptime
  echo '===== git ====='
  git -C "$MS_REPO" status -sb
  git -C "$MS_REPO" log -1 --oneline
  echo '===== container ====='
  docker ps -a --filter "name=^/${MS_CONTAINER}$"
  docker inspect "$MS_CONTAINER" --format 'image={{.Config.Image}} state={{json .State}}' 2>&1
  echo '===== health ====='
  curl -sS --max-time 5 "${MS_API}/api/health" 2>&1
  echo
  echo '===== jobs ====='
  curl -sS --max-time 10 "${MS_API}/api/jobs" 2>&1
  echo
  echo '===== processes ====='
  docker top "$MS_CONTAINER" -eo pid,ppid,pcpu,pmem,etime,args 2>&1
  echo '===== npu ====='
  npu-smi info -t proc-mem -i "$MS_NPU" -c 0 2>&1
  echo '===== disk ====='
  df -h "$MS_ROOT"
  df -i "$MS_ROOT"
  echo '===== daemon log ====='
  tail -n 300 "$MS_RUNTIME/indexer-daemon.log" 2>&1
  echo '===== container log ====='
  docker logs --tail 300 --timestamps "$MS_CONTAINER" 2>&1
} | tee "$OUT"

echo "diagnostic=$OUT"
```

分享前检查脱敏。不要附加 `env`、SSH 配置、shell history、Token、密码或私钥。

## 11. 禁止操作和升级条件

禁止：

- `pkill python`、`killall python`、模糊 `grep | xargs kill`。
- 未检查任务就重启/删除容器或部署。
- 手工修改 `catalog.sqlite3` 的 jobs/videos 状态。
- 删除整个 runtime、models、uploads 或 indexes。
- `docker system prune -a`、重启 Docker daemon、重启主机。
- 操作未知容器、未知 NPU 进程或其他团队目录。

出现以下情况应停止写操作并升级给负责人：

- SQLite integrity check 不是 `ok`。
- 同一故障在安全重试后重复出现。
- NPU 被未知进程占用。
- 需要修改防火墙、安全组、驱动、CANN 或 Docker 服务。
- 需要删除或迁移运行数据。
- 自动回滚后旧服务也无法恢复健康。

## 12. 最常用命令速查

```bash
# 健康
curl -fsS "${MS_API}/api/health" | python3 -m json.tool

# 视频
curl -fsS "${MS_API}/api/videos" | python3 -m json.tool

# 创建 Visual + Face + ASR 索引
curl --fail-with-body -X POST -H 'Content-Type: application/json' -d '{}' \
  "${MS_API}/api/videos/${VIDEO_ID}/index" | python3 -m json.tool

# 任务
curl -fsS "${MS_API}/api/jobs" | python3 -m json.tool
curl -fsS "${MS_API}/api/jobs/${JOB_ID}" | python3 -m json.tool

# 取消
curl --fail-with-body -X POST "${MS_API}/api/jobs/${JOB_ID}/cancel" \
  | python3 -m json.tool

# 日志
tail -F "$MS_RUNTIME/indexer-daemon.log"
docker logs -f --since 10m --timestamps "$MS_CONTAINER"

# 资源
npu-smi info -t proc-mem -i "$MS_NPU" -c 0
docker stats --no-stream "$MS_CONTAINER"
df -h "$MS_ROOT"
```
