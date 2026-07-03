> Archived reference. Current documentation starts at `docs/README.md`.

# 当前状态快照

更新时间：2026-07-03

## 1. 项目位置

本地工作目录：

```text
C:\Users\29154\Projects\video-removal-system\prototype
```

核心代码目录：

```text
video_retrieval_mvp/
```

GitHub 仓库：

```text
https://github.com/502022240018/MomentSeek
```

当前主要工作分支：

```text
feat/asr-search-asset-improvements
```

## 2. 当前系统能力

MomentSeek 当前 MVP 已有四条检索通道：

| 通道 | 当前能力 |
|---|---|
| `visual` | 文本/图片搜画面，当前服务器索引为 SigLIP2，按最大相似帧召回 5s bucket |
| `face` | 参考图或人物库 entity 搜人脸出现片段 |
| `asr` | 搜语音转写文本，支持 lexical + 可选 semantic |
| `ocr` | 搜画面文字，支持 lexical + 可选 semantic |

详细字段、频率和存储格式见：

```text
video_retrieval_mvp/docs/current_retrieval_channels.md
```

## 3. 服务器状态

服务器：

```text
root@110.126.0.52
```

当前项目容器：

```text
momentseek-current-app
```

端口：

```text
宿主机 18300 → 容器 8000
```

典型健康检查：

```bash
ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/health"
```

重要约束：

- 服务器是共用环境。
- 禁止 broad kill、批量杀进程、误杀 ComfyUI / VLLM / 他人任务。
- 只允许操作明确属于 MomentSeek 的容器/进程。
- 重启 `momentseek-current-app` 前必须确认没有 active indexing jobs。

## 4. 当前 visual 状态

服务器 4 个视频的 visual 索引已重跑为：

```text
siglip2-so400m-384
```

当前 visual 召回逻辑：

```text
query → SigLIP2 text/image embedding
→ 与 frame_embeddings 做 cosine
→ 每个 5s bucket 取最大帧相似度 top1
→ raw_score = visual_top1
→ 返回 5s bucket
```

也就是说，现在 visual 排序不再用 5s 平均向量分数。

## 5. 当前四条索引频率

| 通道 | 当前服务器实际频率 |
|---|---:|
| visual | 5fps，5s bucket |
| face | 1fps |
| ASR | 由 Whisper segment/chunk 决定 |
| OCR | 0.05fps，即每 20s 一帧 |

## 6. 前端/公网访问状态

当前系统可能有两种访问方式：

1. 服务器直连：

```text
http://110.126.0.52:18300
```

2. 通过本机中转 + Cloudflare quick tunnel：

```text
Cloudflare 公网链接 → PC:127.0.0.1:18301 → drama-server:127.0.0.1:18300
```

注意：Cloudflare quick tunnel 域名不是固定的。前端出现 `failed to fetch` 时，优先检查：

- 服务器后端是否健康。
- 本机 `127.0.0.1:18301` SSH 转发是否还在。
- cloudflared 进程是否还在。
- 当前使用的 trycloudflare 域名是否已经失效。

最近一次可用过的 quick tunnel：

```text
https://cathedral-advertising-provincial-allocated.trycloudflare.com
```

它可能随时失效，不能作为长期固定入口。

## 7. Superpowers 状态

已安装并启用：

```text
superpowers@openai-curated
```

验证命令：

```powershell
codex plugin list --json
```

当前已知 Superpowers skills 包括：

```text
brainstorming
systematic-debugging
test-driven-development
verification-before-completion
writing-plans
executing-plans
requesting-code-review
receiving-code-review
finishing-a-development-branch
using-git-worktrees
dispatching-parallel-agents
subagent-driven-development
writing-skills
using-superpowers
```

当前旧会话里 skills 列表可能不会热更新。新开 Codex 会话后再确认。

## 8. 近期需要注意的问题

- OCR 目前只有部分视频建立了索引，不是全部视频都有 `ocr.json`。
- ASR semantic 是可选增强；部分视频可能缺少 `asr_semantic.npz`。
- Visual 当前使用 MaxSim + per-video percentile 排序辅助。MaxSim 提高短瞬间目标召回，但多视频搜索包含大量无关视频时，每个无关视频的“本视频内部最佳片段”也可能获得很高 percentile，后续需要集中优化跨视频排序校准和单帧尖峰误召抑制。
- Cloudflare quick tunnel 易失效，前端 `failed to fetch` 不一定是后端坏了。
- 搜索首次加载 SigLIP2 时可能较慢，因为 API 进程需要加载模型。
- 服务器 NPU 2 是共用资源，索引/搜索后要注意显存释放策略。
