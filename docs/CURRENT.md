# MomentSeek 当前状态

更新时间：2026-07-13

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

2026-07-13 验证结果：

```text
本地 /api/health：status=ok, env_profile=dev.cuda, cuda_enabled=true
本地 /api/videos：9 个视频，全部 ready，全部有 ASR v3 索引
本地 /api/jobs：active_count=0
ASR NPZ：9/9 可读取 chunk_times_ms、texts、384-d embeddings、embedding_chunk_indices
ASR API 搜索：6 条中文、西语和英文查询的目标视频均为 Top-1
后端测试：149 passed（已删除 6 个仅覆盖旧 asr_postprocess/旧报告的死路径测试）
最终回归：runtime-server/analysis/asr_formal_regression_20260713_final/
最终听查/检索评估：runtime-server/analysis/asr_formal_review_eval_20260713_final_v2/
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

asr_engine = auto
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
-> auto language probe with faster-whisper turbo
   短视频单窗口；长视频在开头/中段/后段做 3 窗口投票
   -> zh/yue/cmn: SenseVoiceSmall + Silero 12s external VAD
   -> en/es/pt/etc: faster-whisper turbo + 24s contiguous original-audio windows
-> model_transcribe
   SenseVoice：原始文本优先，timestamp 只负责选择安全边界
   faster-whisper：异常无句末窗口才局部 builtin-VAD fallback
-> unit-aware retrieval_chunk_builder
   不跨 decode window 合并，合并后最长 12s，不硬切模型完整长句
-> semantic eligibility
   纯语气/连接词、过短项、明显不可信文字/时长比不生成 embedding
-> MiniLM semantic embedding
-> asr.npz
```

2026-07-13 已使用上述正式路径重建本地全部 9 个视频的 ASR 索引。最终路由：5 个中文长视频和 1 个方言视频走 SenseVoice，世界杯广告与比赛集锦走英文 faster-whisper，球星牛奶广告走西语 faster-whisper。书籍纪录片曾因只探测片头 30s 被误判为英文，改成多位置投票后已重建为 `funasr / zh`。

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
- 本地 9 个视频的 ASR 已全部重跑为 schema v3；其他机器部署后仍需对各自旧索引重跑对应通道。
- ASR/OCR semantic 索引是可选增强；缺失时会退回 lexical。
- Visual MaxSim 提高短瞬间召回，但多视频搜索时可能增加误召，见 `docs/ISSUES_AND_ROADMAP.md` 的 `RQ-001`。
- 首次搜索加载 SigLIP2 可能较慢。
- NPU 2 是共享资源，当前 MomentSeek 已释放；如未来要恢复服务器后端，必须遵循 `docs/OPERATIONS.md`。
