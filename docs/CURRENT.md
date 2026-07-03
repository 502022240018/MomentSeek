# MomentSeek 当前状态

更新时间：2026-07-03

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

## 服务器状态

```text
服务器：root@110.126.0.52
当前容器：momentseek-current-app
端口：宿主机 18300 -> 容器 8000
runtime：/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime
代码：/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/backend
模型：/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp/models
```

服务器是共享环境。任何 kill、restart、docker 操作前必须看 `docs/OPERATIONS.md`，不要影响 ComfyUI、VLLM 或其他人的任务。

最近一次只读检查结果：

```text
momentseek-current-app：Up, healthy, 0.0.0.0:18300->8000/tcp
/api/health：status=ok, npu_enabled=true, npu_device_id=0, model_idle_policy=process_exit
NPU 2：有 uvicorn 进程，占用符合 MomentSeek API 预期
```

## 当前模型和索引状态

当前服务器配置摘要：

```text
visual_model = siglip2-so400m-384
visual_sample_fps = 5.0
visual_segment_seconds = 5.0
visual_decode_height = 256

face_model = buffalo_l
face_provider = cann
face_sample_fps = 1.0
face_decode_height = 720

asr_engine = whisper
asr_model = small
asr_semantic_model = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
asr_semantic_device = cpu

ocr_engine = rapidocr
ocr_sample_fps = 0.05
ocr_version = PP-OCRv4
ocr_semantic_enabled = true
```

当前服务器 4 个视频的 visual 索引已重跑为：

```text
siglip2-so400m-384
```

当前 visual 召回逻辑：

```text
query -> SigLIP2 query embedding
-> 与 frame_embeddings 做 cosine
-> 每个 5s bucket 取最大相似帧 top1
-> raw_score = visual_top1
-> 返回 5s bucket
```

## 公网访问

当前短期公网访问可能使用 Cloudflare quick tunnel，并通过用户 PC 中转：

```text
Cloudflare quick tunnel -> PC 127.0.0.1:18301 -> drama-server 127.0.0.1:18300
```

Quick tunnel 域名不是固定的。前端出现 `failed to fetch` 时，先检查：

1. 服务器后端 `127.0.0.1:18300` 是否健康。
2. PC 本地 SSH 转发 `127.0.0.1:18301` 是否还在。
3. cloudflared 是否还在。
4. 当前 trycloudflare 域名是否过期。

短期决定：继续使用当前临时方案，仅用于自己和少数同学测试。更稳定公网入口记录在 `docs/ISSUES_AND_ROADMAP.md`。

## 当前注意事项

- 部分视频没有 OCR 索引。
- ASR/OCR semantic 索引是可选增强；缺失时会退回 lexical。
- Visual MaxSim 提高短瞬间召回，但多视频搜索时可能增加误召，见 `docs/ISSUES_AND_ROADMAP.md` 的 `RQ-001`。
- 首次搜索加载 SigLIP2 可能较慢。
- NPU 2 是共享资源，服务器操作必须遵循 `docs/OPERATIONS.md`。
