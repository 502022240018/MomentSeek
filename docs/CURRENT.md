# MomentSeek 当前状态

更新时间：2026-07-09

## 项目位置

```text
工作目录：C:\Users\29154\Projects\video-removal-system\prototype
项目目录：video_retrieval_mvp/
当前分支：feat/asr-search-asset-improvements
仓库：    https://github.com/502022240018/MomentSeek
```

MomentSeek 是一个多模态视频检索 MVP，当前有四条检索通道：

| 通道 | 当前能力 |
|---|---|
| `visual` | 文本/图片搜画面，当前服务器使用 SigLIP2，在 5s bucket 内按帧级 MaxSim 召回 |
| `face` | 参考图或人物库 entity 搜人脸出现片段 |
| `asr` | 搜语音转写文本，支持 lexical 和可选 semantic |
| `ocr` | 搜画面文字，支持 lexical 和可选 semantic |

详细通道协议见 `docs/RETRIEVAL_CHANNELS.md`。

## 当前运行入口

当前公网展示已从共享服务器 NPU 切到本机 Docker GPU 后端：

```text
Cloudflare quick tunnel
-> PC 127.0.0.1:18301
-> local Docker container momentseek-mvp-app:8000
```

当前本地容器配置摘要：

```text
容器：momentseek-mvp-app
端口：宿主机 18301 -> 容器 8000
runtime：./runtime-server -> /app/runtime
CUDA_ENABLED=true
NPU_ENABLED=false
VISUAL_HF_CACHE_DIR=/app/runtime/hf_cache
```

2026-07-07 验证结果：

```text
本地 /api/health：status=ok, env_profile=dev.cuda, cuda_enabled=true
本地 /api/videos：8 个视频
本地 ASR 搜索“新疆美食”：返回命中
本地 visual 搜索“烤包子”：返回命中
公网 /api/health：status=ok
公网 /api/videos：8 个视频
```

当前 quick tunnel 地址是临时地址，失效后需要重新启动 cloudflared：

```text
https://entertainment-grocery-independently-generators.trycloudflare.com
```

## 共享服务器状态

```text
服务器：root@110.126.0.52
当前容器：momentseek-current-app（已停止，仅保留容器和 runtime，不删除）
端口：宿主机 18300 -> 容器 8000
runtime：/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime
代码：/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/backend
模型：/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp/models
```

服务器是共享环境。任何 kill、restart、docker 操作前必须看 `docs/OPERATIONS.md`，不要影响 ComfyUI、VLLM 或其他人的任务。

最近一次停服前检查和停服后结果：

```text
停服前 /api/jobs：active_count=0
执行：docker stop momentseek-current-app
momentseek-current-app：Exited (0)
服务器 18300 /api/health：不可访问
NPU 2：No running processes found
```

## 当前模型和索引状态

当前已迁移到本地的索引配置摘要：

```text
visual_model = siglip2-so400m-384
visual_sample_fps = 5.0
visual_segment_seconds = 5.0
visual_decode_height = 256

face_model = buffalo_l
face_provider = cann
face_sample_fps = 1.0
face_decode_height = 720

asr_engine = funasr
asr_zh_model = iic/SenseVoiceSmall
asr_model = turbo
asr_language = auto
asr_vad_strategy = silero_12s
asr_semantic_model = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
asr_semantic_device = cpu

ocr_engine = rapidocr
ocr_sample_fps = 0.05
ocr_version = PP-OCRv4
ocr_semantic_enabled = true
```

当前代码的索引格式已切换到 schema v3：

```text
runtime/indexes/{video_id}/index_manifest.json
runtime/indexes/{video_id}/visual.npz
runtime/indexes/{video_id}/face.npz
runtime/indexes/{video_id}/asr.npz
runtime/indexes/{video_id}/ocr.npz
```

v3 不兼容旧索引。部署新代码后，旧的 visual v2、`faces.npz`、`asr.json`、`ocr.json` 需要重跑对应通道索引；新查询层只读取 v3 manifest 和 v3 npz。

当前 visual 召回逻辑：

```text
query -> SigLIP2 query embedding
-> 与 frame_embeddings 做 cosine
-> 每个 visual segment 取最大相似帧 top1
-> raw_score = visual_top1
-> Candidate.score = visual_rank_score = clip((raw_score + 1) / 2, 0, 1)
-> percentile / robust_z 只用于视频内判定和 evidence 诊断
-> 返回固定 bucket 或 shot-aware segment
```

当前 ASR pipeline：

```text
audio_extract
-> SenseVoiceSmall + Silero 12s external VAD by default
   or faster-whisper turbo + builtin VAD when explicitly selected
-> model_transcribe
-> raw_transcript parser
-> safe raw split for timestamp/punctuation or long raw items
-> retrieval_chunk_builder
   final merge window = 8/12/15 seconds
-> MiniLM semantic embedding
-> asr.npz
```

默认 `asr.npz` 不保存 raw transcript，只保留检索需要字段。需要排查 ASR 切分问题时，开启 `ASR_DEBUG_ARTIFACTS=true` 和 `ASR_SAVE_RAW_TRANSCRIPT=true`，debug 文件写入 `runtime/indexes/{video_id}/debug/`。

## 公网访问

当前短期公网访问使用 Cloudflare quick tunnel，并通过用户 PC 中转：

```text
Cloudflare quick tunnel -> PC 127.0.0.1:18301 -> local Docker backend
```

Quick tunnel 域名不是固定的。前端出现 `failed to fetch` 时，先检查：

1. PC 本地后端 `127.0.0.1:18301/api/health` 是否健康。
2. Docker 容器 `momentseek-mvp-app` 是否 healthy。
3. cloudflared 是否还在。
4. 当前 trycloudflare 域名是否过期。

短期决定：继续使用当前临时方案，仅用于自己和少数同学测试。更稳定公网入口记录在 `docs/ISSUES_AND_ROADMAP.md`。

## 当前注意事项

- 多人开发与可复制部署方案已设计，第一阶段新增 dev.cpu/dev.cuda/staging.ascend/prod.ascend profile 和 manifest。
- 当前 metadata/schema 正在切到 v3；部署到服务器后必须安排重跑索引。
- ASR/OCR semantic 索引是可选增强；缺失时会退回 lexical。
- Visual MaxSim 提高短瞬间召回，但多视频搜索时可能增加误召，见 `docs/ISSUES_AND_ROADMAP.md` 的 `RQ-001`。
- 首次搜索加载 SigLIP2 可能较慢。
- NPU 2 是共享资源，当前 MomentSeek 已释放；如未来要恢复服务器后端，必须遵循 `docs/OPERATIONS.md`。
