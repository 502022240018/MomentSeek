# 新 Codex 会话启动 Prompt

把下面这段复制到新的 Codex 会话中，用于恢复 MomentSeek 上下文。

```text
我们继续 MomentSeek / video_retrieval_mvp 项目。

工作目录：
C:\Users\29154\Projects\video-removal-system\prototype

项目目录：
video_retrieval_mvp/

行动前请按顺序读取：

1. video_retrieval_mvp/docs/README.md
2. video_retrieval_mvp/docs/CURRENT.md
3. video_retrieval_mvp/docs/ISSUES_AND_ROADMAP.md
4. video_retrieval_mvp/docs/RETRIEVAL_CHANNELS.md
5. video_retrieval_mvp/docs/ARCHITECTURE.md
6. video_retrieval_mvp/docs/OPERATIONS.md
7. video_retrieval_mvp/docs/VALIDATION.md
8. video_retrieval_mvp/docs/LESSONS_LEARNED.md

重要上下文：

- MomentSeek 是一个视频检索 MVP，核心通道包括 visual / face / ASR / OCR。
- 当前服务器：root@110.126.0.52。
- 当前容器：momentseek-current-app。
- 端口：宿主机 18300 -> 容器 8000。
- 服务器是共享环境。不要 broad kill，不要未经确认重启容器，不要碰 ComfyUI / VLLM / 未知进程。
- 只允许操作明确属于 MomentSeek 的资源。
- 任何服务器状态变更前，先执行 docs/OPERATIONS.md 中的只读检查。
- 当前 visual 索引为 SigLIP2 siglip2-so400m-384。
- Visual 召回使用 5s bucket MaxSim：raw_score = visual_top1。
- 当前 ASR 索引使用分层 pipeline：raw transcript parser -> retrieval_chunk_builder -> MiniLM semantic embedding；默认 `SenseVoiceSmall + Silero external VAD 12s`，可选 `faster-whisper turbo + builtin VAD`；默认不保存 raw transcript，debug 开关开启后才写 `runtime/indexes/{video_id}/debug/`。
- ASR/OCR semantic embedding 是 MiniLM 文本向量，不能和 visual embedding 混用。
- 如果要声明完成/修复/通过，必须先按 docs/VALIDATION.md 运行验证命令并读取输出。

初始只读检查：

1. git -C video_retrieval_mvp status --short
2. codex plugin list --json
3. 读取上述 docs
4. 如需连接服务器，只先做只读检查：
   - ssh root@110.126.0.52 "docker ps --filter name=momentseek-current-app"
   - ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/health"
   - ssh root@110.126.0.52 "npu-smi info"
```
