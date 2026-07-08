# 运维与共享服务器操作规范

本文档记录 MomentSeek 在共享服务器和公网访问上的操作规范。

## 黄金规则

- 任何服务器状态变更前，先做只读检查。
- 禁止 broad kill。
- 未确认没有 active indexing jobs 前，不要重启容器。
- 不要触碰 ComfyUI、VLLM 或其他人的进程。
- 只操作明确属于 MomentSeek 的容器/进程/文件。
- 不要删除 runtime 数据，除非用户明确要求并且目标路径已核对。

## 当前服务器

```text
服务器：root@110.126.0.52
当前容器：momentseek-current-app
当前端口：宿主机 18300 -> 容器 8000
旧 baseline 容器：momentseek-mvp-app，端口 8300
```

旧 baseline 容器是历史参考，不要随手删除。

## 必须先做的只读检查

任何服务器状态变更前，先运行：

```bash
ssh root@110.126.0.52 "docker ps --filter name=momentseek-current-app"
ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/health"
ssh root@110.126.0.52 "npu-smi info"
ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/jobs"
```

如需检查容器内部进程：

```bash
ssh root@110.126.0.52 "docker exec momentseek-current-app ps -ef | grep -E 'app.worker|app.stage_runner|indexer_daemon' | grep -v grep"
```

只有确认没有 active indexing jobs 后，才可以考虑重启。

## 健康检查

后端 health：

```bash
ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/health"
```

容器状态：

```bash
ssh root@110.126.0.52 "docker ps --filter name=momentseek-current-app --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

NPU 状态：

```bash
ssh root@110.126.0.52 "npu-smi info"
```

## 公网访问

当前短期方案可能是：

```text
Cloudflare quick tunnel -> PC 127.0.0.1:18301 -> local Docker backend
```

前端 `failed to fetch` 时按顺序检查：

1. PC 本地后端 `127.0.0.1:18301/api/health` 是否健康。
2. Docker 容器 `momentseek-mvp-app` 是否 healthy。
3. cloudflared 进程是否还在。
4. 当前 trycloudflare 域名是否已经失效。

本地接管时常用检查：

```powershell
docker ps --filter name=momentseek-mvp-app
curl.exe http://127.0.0.1:18301/api/health
Get-CimInstance Win32_Process -Filter "name='cloudflared.exe'" | Select-Object ProcessId,CommandLine
```

重新创建 quick tunnel：

```powershell
.\runtime\tools\cloudflared.exe tunnel --url http://127.0.0.1:18301 --no-autoupdate
```

当前项目只面向自己和少数同学测试，暂时继续使用临时公网方案。稳定公网入口记录在 `docs/ISSUES_AND_ROADMAP.md`。

## 允许操作

只读检查后可以做：

- 查询 health、jobs、容器状态、NPU 状态。
- 查看 `momentseek-current-app` 日志。
- 在用户确认后操作 `momentseek-current-app`。
- 在用户要求时启动或停止本机 SSH 转发 / cloudflared。

## 禁止操作

除非用户明确批准具体目标，并且已经确认 active jobs，否则禁止：

- `pkill python`、`killall python`、宽泛的 `grep | xargs kill`。
- 重启或删除无关容器。
- 删除 `runtime/`、`models/`、uploads、indexes、thumbnails、clips。
- 仅因为某进程占用 NPU 就 kill。
- 触碰 VLLM、ComfyUI 或未知 python 进程。

## 服务器操作记录格式

任何服务器状态变更都要记录：

```text
时间：
目的：
执行前检查：
执行命令：
影响范围：
验证结果：
资源释放状态：
相关 issue：
```

长期经验和事故复盘写入 `docs/LESSONS_LEARNED.md`。
