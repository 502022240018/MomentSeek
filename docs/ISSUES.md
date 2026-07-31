# 问题池（平台交付仓）

本仓唯一的已知问题与待办清单。来自 2026-07-31 目录重构后的全仓体检。
状态：`open / planned / in_progress / done`。

## 高风险（功能性，待改）

### SEC-001 API 无鉴权 + CORS 全开

```text
优先级：P0（公网部署时）
状态：open
位置：backend/app/main.py（CORSMiddleware allow_origins=["*"]；全部路由无认证）
问题：上传/删除视频/删人物库/建索引均匿名可调；公网入口暴露时任何人可清空素材库
      或用索引任务占满 NPU。
方案：APP_API_TOKEN 环境变量 + 简单 Bearer 校验中间件（为空则不启用，不影响内网），
      CORS 来源改为可配置。鉴权放应用层还是隧道层由部署 owner 决定。
```

### SEC-002 上传无大小限制

```text
优先级：P0（共享服务器）
状态：open
位置：backend/app/api/video_routes.py::upload_video（流式写盘无上限）
问题：单个超大/恶意上传可打满共享服务器磁盘，影响机上所有服务。
方案：MAX_UPLOAD_BYTES 设置项；_save_upload 累计字节数超限即中断并删除半成品。
```

### OPS-001 frame_cache / clips 缓存无淘汰

```text
优先级：P1
状态：open
位置：runtime/frame_cache、runtime/clips（仅随视频删除清理）
问题：按毫秒时间戳粒度缓存，检索越多增长越快，无上限。
方案：目录容量上限 + LRU 清理（启动时或定期任务）。
```

## 中风险（功能性，待改）

### OPS-002 SQLite 未开 WAL

```text
优先级：P1
状态：open
位置：backend/app/catalog/db.py::connect（每次新建连接，默认 rollback journal）
问题：API + worker 子进程 + daemon 多进程写同库，靠 busy_timeout=30s 硬扛，
      并发高时可能锁库或卡 30 秒。
方案：连接时 PRAGMA journal_mode=WAL + synchronous=NORMAL；注意 WAL 文件
      随 runtime 目录迁移。
```

### OPS-003 任务日志无轮转

```text
优先级：P2
状态：open
位置：runtime/job-*.log、runtime/indexer-daemon.log（append-only）
问题：长期运行无限增长；job 日志仅在删除视频时清理。
方案：daemon 日志按大小截断；job 日志随任务完成保留 tail 或定期清理。
```

### OPS-004 取消任务依赖 /proc（仅 Linux）

```text
优先级：P3
状态：open（文档标注即可）
位置：backend/app/main.py::_terminate_process_group
说明：平台仓目标运行环境是 Linux 容器；Windows 裸机开发时任务取消不可用。
```

## 结构优化（既定技术债，见 ARCHITECTURE.md 第 9 节）

- ~~context 化：消除路由对 app.main 的 38 处惰性互引~~ → 已完成
  （2026-07-31，`platform/context.py`）
- ~~共享编码器抽层~~ → 已完成（2026-07-31，`app/encoders/{visual,face,text}.py`）
- `settings.resolve_path` 多候选猜测逻辑待固定语义
- 拆分 `retrieval/search.py`（约 77KB）与前端 `main.tsx`（约 34KB）
- face/asr/ocr/speaker 通道升级 Milvus P2 内存直写（visual 已完成）

## 非技术项

- LICENSE 缺失：上游无许可证文件，需代码 owner 确认授权方式后补充。

## 已修复

- 2026-07-31 `worker.py` 搬迁后 backend 目录定位错误（parents[1]→[2]）
- 2026-07-31 compose app 服务加 `init: true`，修复 detached worker 僵尸进程堆积
- 2026-07-31 删除 `main.py`/`stage_runner.py` 无使用者的兼容 re-export
- 2026-07-31 speaker_service Milvus 回退、model_pool 后台清理线程补告警日志
- 2026-07-31 `ASCEND_RT_VISIBLE_DEVICES` 必须为容器内逻辑 0（env 模板已注明）
