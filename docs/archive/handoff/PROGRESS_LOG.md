> Archived reference. Current documentation starts at `docs/README.md`.

# 项目进度日志

本文件记录阶段性进展、关键决策和后续待办。详细实验数据仍放在专题文档中。

## 2026-07-03

### 建立交接目录

新增：

```text
video_retrieval_mvp/docs/handoff/
```

包含：

- `README.md`
- `SESSION_BOOTSTRAP.md`
- `CURRENT_STATUS.md`
- `PROGRESS_LOG.md`
- `MAINTENANCE_GUIDE.md`

目的：

- 新会话快速恢复上下文。
- 给同事交接时有固定入口。
- 避免进度散落在多个聊天和文档里。

### 输出四条检索通道说明

新增：

```text
video_retrieval_mvp/docs/current_retrieval_channels.md
```

内容包括：

- visual / face / asr / ocr 的索引频率。
- `.npz` / `.json` 字段格式。
- 向量维度和存储大小。
- 召回粒度。
- 当前服务器实际配置。

### 安装 Superpowers

已通过 Codex CLI 安装：

```text
superpowers@openai-curated
```

验证：

```text
installed = true
enabled = true
```

需要新开 Codex 会话以刷新 skills 列表。

### 恢复公网访问链路

现象：

```text
前端 failed to fetch
```

排查结论：

- 服务器后端 `18300` 健康。
- 旧 Cloudflare quick tunnel 域名失效。
- 本机 `127.0.0.1:18301` SSH 转发断开。

处理：

- 恢复 SSH 转发：

```text
PC:127.0.0.1:18301 → drama-server:127.0.0.1:18300
```

- 新建 Cloudflare quick tunnel：

```text
https://cathedral-advertising-provincial-allocated.trycloudflare.com
```

注意：该链接可能随时失效。

### visual 索引与召回状态确认

服务器 4 个视频 visual 已全部重索引为：

```text
siglip2-so400m-384
```

召回逻辑已改为：

```text
raw_score = visual_top1
```

即按 5s bucket 内最大相似帧排序，不再用 top1/top3/mean 混合平均。

相关 commit：

```text
3d8933b fix: rank visual recall by best frame
```

## 后续待办

### 文档类

- 将 `docs/HANDOFF_CURRENT.md` 中仍然有效的内容逐步拆分进专题文档，减少单文件过长。
- 将服务器操作 SOP 扩充到 `docs/server-operations.md`。
- 每次变更索引格式时同步更新 `docs/current_retrieval_channels.md`。

### 工程类

- Visual 多视频检索需要优化跨视频排序校准：当前 SigLIP2 + 5s bucket MaxSim 能显著提升局部目标召回，但当搜索范围包含大量无关视频时，per-video percentile 会让每个视频的“本视频内部最佳片段”排得过高，可能导致综艺等无关视频在“绿茵足球场有人在踢球”等查询中误召靠前。后续应集中诊断 `raw_score/visual_top1`、`visual_top3`、`visual_mean`、`percentile`、`robust_z` 的关系，并考虑用 raw/global calibration 作为跨视频排序主信号，per-video percentile 作为辅助召回信号。
- 前端增加当前公网/后端健康状态提示，减少 `failed to fetch` 不透明问题。
- 增加“索引文件完整性检查”脚本。
- 增加“当前视频索引状态导出”脚本。
- 对 OCR 做更合理的 chunk 合并策略，而不是仅每 20s 一帧。
- 对 ASR chunk 做后处理：过短合并、过长拆分、可选滑窗语义块。

### 运维类

- 研究固定 Cloudflare Tunnel 或其他稳定公网入口，避免 quick tunnel 域名频繁变化。
- 建立服务器操作前检查清单：
  - active jobs
  - docker ps
  - npu-smi
  - 端口健康
  - 只操作本项目进程
