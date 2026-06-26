# MomentSeek 当前交接文档（2026-06-26）

> 这份文档是当前开发/部署状态的交接版，覆盖本地与服务器两套服务、三路索引 pipeline、模型与设备配置、数据管理、Cloudflare 公网转发、已知问题和下一步建议。  
> 旧版较长背景文档见 `docs/HANDOFF.md`；本文件以”现在实际能用的系统”为准。

## 0. TL;DR

当前 MomentSeek 已经是一套可跑通的多模态视频片段检索 MVP：

- Visual：OpenCLIP 做场景/物体/视觉语义索引和检索；per-query/per-video robust z-score + percentile 自适应阈值。
- Face：InsightFace / ArcFace 做人物出现片段索引和检索；logistic confidence 校准与 visual 分数同量纲。
- ASR：代码支持 `engine=auto`（中文优先 FunASR/Paraformer，fallback Whisper），但服务器镜像未装 funasr，当前实跑 Whisper small；ASR 检索是关键词/近似字符匹配，还不是语义检索。
- 前端：React + Vite，风格参考 TwelveLabs Playground。
- 后端：FastAPI + SQLite + 本地文件索引。
- 当前有本地和服务器两套前后端并行运行。
- 搜索支持全召回 + 阈值标记（低于阈值的结果单独展示），播放弹窗只循环命中时间段。
- 素材管理：可重命名、删除视频及其所有索引。

当前公网入口：

```text
https://jobs-qualified-eligibility-soft.trycloudflare.com
```

公网链路依赖用户 PC 作为中转：

```text
浏览器
  → Cloudflare Tunnel
  → 用户 PC:127.0.0.1:18301
  → SSH local forward
  → drama-server:127.0.0.1:18300
  → 服务器容器 momentseek-current-app
```

所以：用户 PC、SSH 转发、cloudflared 任一断开，公网链接都会失效。

## 1. 当前运行拓扑

### 1.1 本地服务

本地项目目录：

```text
C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp
```

本地虚拟环境：

```text
C:\Users\29154\Projects\video-removal-system\prototype\.venv
```

本地常用端口：

```text
后端 API: http://127.0.0.1:8000
前端 dev: http://127.0.0.1:5173
```

本地 GPU：

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 6GB
```

本地当前适合做：

- 代码开发。
- 快速验证前端。
- 小样本索引。
- CUDA 路径测试。

### 1.2 服务器服务

SSH alias：

```text
drama-server
```

实际服务器：

```text
110.126.0.52
```

服务器 current 版容器：

```text
momentseek-current-app
```

服务器 current 版容器端口：

```text
http://110.126.0.52:18300
```

这个直连地址通常需要 VPN/内网环境；给同事演示建议用 Cloudflare 链接。

服务器 current 版代码和数据目录：

```text
/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app
```

服务器复用的模型目录：

```text
/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp/models
```

旧版服务器容器仍存在，没有覆盖：

```text
momentseek-mvp-app
端口: 8300
```

旧版服务不要随手删，它是之前部署留下的 baseline。

## 2. 公网访问方案

当前公网 quick tunnel：

```text
https://jobs-qualified-eligibility-soft.trycloudflare.com
```

API docs：

```text
https://jobs-qualified-eligibility-soft.trycloudflare.com/docs
```

Health：

```text
https://jobs-qualified-eligibility-soft.trycloudflare.com/api/health
```

### 2.1 PC 上的 SSH 转发

PC 上开了本地端口转发：

```powershell
ssh.exe -N -L 127.0.0.1:18301:127.0.0.1:18300 drama-server
```

含义：

```text
PC:127.0.0.1:18301 → drama-server:127.0.0.1:18300
```

如果需要重新启动：

```powershell
Start-Process -FilePath "ssh.exe" `
  -ArgumentList @("-N","-L","127.0.0.1:18301:127.0.0.1:18300","drama-server") `
  -WindowStyle Hidden
```

测试：

```powershell
Invoke-RestMethod http://127.0.0.1:18301/api/health
```

### 2.2 PC 上的 Cloudflare Tunnel

cloudflared 路径：

```text
runtime\tools\cloudflared.exe
```

当前 cloudflared 指向：

```powershell
runtime\tools\cloudflared.exe tunnel --url http://127.0.0.1:18301 --no-autoupdate --protocol http2
```

重新启动示例：

```powershell
cd C:\Users\29154\Projects\video-removal-system\prototype\video_retrieval_mvp
runtime\tools\cloudflared.exe tunnel --url http://127.0.0.1:18301 --no-autoupdate --protocol http2
```

Cloudflare 会生成新的 `https://*.trycloudflare.com` 链接。Quick tunnel 不是固定域名。

### 2.3 查看当前 PC 转发进程

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match '127.0.0.1:18301:127.0.0.1:18300|cloudflared.exe' } |
  Select-Object ProcessId,Name,CommandLine
```

当前观察到：

- 一个 SSH forward 进程。
- 一个指向服务器的 cloudflared 进程。
- 可能还有一个旧 cloudflared 进程指向本地前端；不要误关。

## 3. 服务器 current 容器配置

容器名：

```text
momentseek-current-app
```

端口：

```text
18300:8000
```

当前启动策略：

- 不重建镜像。
- 使用已有镜像 `momentseek-mvp:ascend`。
- 挂载当前代码目录到 `/app/backend`。
- 挂载当前 runtime 到 `/app/runtime`。
- 挂载旧模型目录到 `/app/models`。
- 使用物理 2 号 NPU。

容器实际环境变量（`docker run -e` 注入，2026-06-26 实测）：

```text
APP_DATA_DIR=/app/runtime
APP_MODEL_DIR=/app/models
APP_PUBLIC_URL=http://110.126.0.52:18300

NPU_ENABLED=true
ASCEND_RT_VISIBLE_DEVICES=2
ASCEND_VISIBLE_DEVICES=2
NPU_DEVICE_ID=0
TORCH_DEVICE_BACKEND_AUTOLOAD=0

CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=/app/models/ViT-B-32.openai.bin

FACE_MODEL=buffalo_l
FACE_PROVIDER=cann
FACE_SAMPLE_FPS=1.0

ASR_ENGINE=whisper        # funasr 未装，即便改 auto 也会 fallback 回 whisper
ASR_MODEL=small
ASR_LANGUAGE=en           # 仅默认值，前端建索引时按视频实际语言覆盖
ASR_DEVICE=auto
```

**关于环境变量优先级（重要）**：后端用 pydantic-settings，优先级是
**OS 环境变量（`docker run -e`） > `.env` 文件**。容器是 `docker run` 直接起的（非 compose），
这些 `-e` 值在创建容器时就固定了。所以：

- 写 `backend/.env` 再 `docker restart` **无效** —— `-e` 注入的值会覆盖 `.env`。
- 要改默认值必须 **重建容器**（`docker rm` + `docker run`，且必须带 `--privileged`，
  NPU 靠特权模式访问 `/dev/davinci*`，`HostConfig.Devices` 为空）。
- 但 `asr_language` / `asr_model` 可由前端**按任务覆盖**（写入 `jobs.options`），
  所以日常不需要改容器默认值，建索引时在前端选对语言即可。

`ASR_ENGINE` 不能按任务覆盖（只从 settings 读）。要真正启用 FunASR，需要：
先在镜像里装 `funasr`，再重建容器把 `ASR_ENGINE` 设为 `auto`。

注意 NPU 编号：

- 宿主机物理卡：2 号卡。
- 容器里通过 `ASCEND_RT_VISIBLE_DEVICES=2` 只暴露一张卡。
- 容器内逻辑编号是 `npu:0`。
- 所以 `NPU_DEVICE_ID=0` 是正确的。

检查容器：

```bash
ssh drama-server "docker ps --filter name=momentseek-current-app --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

检查 health：

```bash
ssh drama-server "curl -s http://127.0.0.1:18300/api/health"
```

## 4. 当前模型与版本

服务器 current 容器内包版本：

```text
torch              2.9.0+cpu
torch_npu          2.9.0.post1+gitee7ba04
open_clip_torch    3.3.0
openai-whisper     20250625
insightface        1.0.1
numpy              1.26.4
pydantic-settings  2.12.0
onnxruntime        CANNExecutionProvider + CPUExecutionProvider
```

模型文件：

```text
CLIP:
  ViT-B-32.openai.bin                     577MB

InsightFace buffalo_l:
  insightface/models/buffalo_l/det_10g.onnx
  insightface/models/buffalo_l/w600k_r50.onnx
  insightface/models/buffalo_l/1k3d68.onnx
  insightface/models/buffalo_l/2d106det.onnx
  insightface/models/buffalo_l/genderage.onnx

Whisper:
  tiny.pt                                  72MB
  small.pt                                461MB
  medium.pt                               1.5GB
```

三路当前设备策略：

| 模块 | 当前模型 | 设备策略 | 当前结论 |
| --- | --- | --- | --- |
| Visual | OpenCLIP ViT-B-32 + OpenAI 权重 | NPU，物理 2 号卡 | 已验证可用 |
| Face | InsightFace buffalo_l | CANNExecutionProvider / NPU | smoke test 可初始化；索引任务也跑通过 |
| ASR | OpenAI Whisper | `ASR_DEVICE=auto -> npu:0` | `small` 跑通；`medium` 在共享卡上 OOM |

## 5. 当前数据状态

服务器 current runtime：

```text
/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime
```

当前视频素材包括至少：

| 视频 | 状态 | 当前索引 |
| --- | --- | --- |
| 世界杯广告.mp4 | ready | visual + asr |
| 球星牛奶广告 | ready | visual + face + asr |
| 给阿嬷的情书预告片 | ready | visual + face + asr |
| 6223_bP5KfdFJzC4_360.0_510.0.mp4 | ready | visual + face + asr |

当前人物库包括：

- 亚马尔
- 哈兰德
- 姆巴佩
- 梅西
- 郑木生

路径迁移注意：

从本地拷贝 runtime 到服务器后，SQLite 里的 `file_path`、`reference_path`、`embedding_path` 曾经还是 Windows 路径。已在服务器 current DB 中修成：

```text
/app/runtime/uploads/...
/app/runtime/entities/...
```

后续如果再次从本地拷贝 SQLite 到服务器，必须重新修路径。

## 6. 代码结构

核心目录：

```text
backend/app/
├─ main.py              # FastAPI 路由、静态前端挂载
├─ settings.py          # 配置项
├─ schemas.py           # 请求校验
├─ db.py                # SQLite Catalog
├─ worker.py            # 索引任务编排
├─ stage_runner.py      # visual/face/asr 阶段子进程入口
├─ search.py            # 三路召回与结果融合
├─ media.py             # 视频探测、抽帧、抽音频、缩略图
└─ indexing/
   ├─ visual.py         # OpenCLIP 索引
   ├─ faces.py          # InsightFace 索引
   ├─ asr.py            # Whisper/FunASR/字幕索引
   └─ common.py

frontend/src/
├─ main.tsx             # 当前前端主要页面都在这里
├─ api.ts               # API 类型和请求函数
└─ styles.css
```

当前前端已经做了：

- 上传视频/字幕。
- 建立三路索引。
- Visual fps 可配置。
- Visual 分段秒数可配置。
- Face fps 可配置。
- ASR 模型可选：`base / small / medium / large-v3`。
- ASR 语言可选：`zh / en / auto`。
- 当前默认 ASR 模型已改为 `small`，避免 medium 在共享 NPU 上 OOM。
- 索引阶段耗时展示。
- 搜索耗时展示。
- 播放弹窗：只循环命中的 `[start_time, end_time]` 片段，不播放全片。
- 搜索全召回 + 阈值标记：阈值以上/以下分组，低分卡片 dimming + "低于阈值" chip，结果头部显示命中数/低于阈值数。
- 素材管理（Assets 页）：支持重命名、删除视频（含索引/缩略图/任务清理）。

## 7. Pipeline 详解

### 7.1 上传

前端上传视频到：

```text
POST /api/videos
```

后端：

1. 生成 video_id。
2. 保存到 `runtime/uploads/{video_id}.{suffix}`。
3. 用 OpenCV 探测视频信息。
4. 写入 SQLite `videos`。
5. 可选字幕保存为 `runtime/uploads/{video_id}.transcript.srt/json/vtt`。

### 7.2 建索引

前端调用：

```text
POST /api/videos/{video_id}/index
```

请求示例：

```json
{
  "modalities": ["visual", "face", "asr"],
  "visual_sample_fps": 5.0,
  "visual_segment_seconds": 5.0,
  "face_sample_fps": 2.0,
  "asr_model": "small",
  "asr_language": "zh"
}
```

后端流程：

```text
main.py create_index_job
  → Catalog.create_job
  → worker.launch_job
  → app.worker 子进程
  → 每个 stage 再启动 app.stage_runner 子进程
  → stage 完成后写 metrics
  → video.indexed_modalities 更新
  → 全部完成后 video.status=ready
```

设计原因：

- 模型只在子进程里加载。
- 阶段完成后进程退出，释放显存。
- 共享服务器更安全。
- `index-worker.lock` 保证单实例内一次只跑一个索引任务。

### 7.3 Visual 索引

代码：

```text
backend/app/indexing/visual.py
```

步骤：

1. 按 fps 抽帧，当前默认 5fps。
2. 按 `timestamp // segment_seconds` 分桶，当前默认 5 秒。
3. 每个桶保存一张缩略图。
4. CLIP encode_image。
5. 每个桶向量求平均并 normalize。
6. 保存 `visual.npz`。

`visual.npz`：

```text
embeddings
start_times
end_times
thumbnails
model
```

当前服务器：

```text
device = npu:0
实际物理卡 = 2
```

### 7.4 Face 索引

代码：

```text
backend/app/indexing/faces.py
```

步骤：

1. 按 `face_sample_fps` 抽帧。
2. InsightFace 检测人脸。
3. 提取 ArcFace embedding。
4. 通过 embedding cosine + bbox IoU 做简易 track。
5. 每个 track 聚合 embedding，保存最佳 crop。
6. 保存 `faces.npz`。

`faces.npz`：

```text
embeddings
start_times
end_times
thumbnails
qualities
model
```

当前服务器：

```text
FACE_PROVIDER=cann
onnxruntime providers = CANNExecutionProvider + CPUExecutionProvider
```

注意：如果 CANN 子图不支持某些 op，onnxruntime 可能局部 fallback 到 CPU，但 provider 已经是 CANN 优先。

### 7.5 ASR 索引

代码：

```text
backend/app/indexing/asr.py
```

步骤：

1. 如果有 sidecar 字幕，直接 parse。
2. 否则抽取 16k mono wav。
3. 调 Whisper 或 FunASR 适配器。
4. 保存 `asr.json`。

`asr.json`：

```json
{
  "engine": "whisper",
  "model": "small",
  "language": "zh",
  "chunks": [
    {
      "start_time": 0.0,
      "end_time": 2.0,
      "text": "..."
    }
  ]
}
```

当前服务器容器实际：

```text
ASR_ENGINE=whisper     # funasr 未装；代码支持 auto，装了 funasr 后重建容器即可启用
ASR_MODEL=small
ASR_DEVICE=auto        # resolve_asr_device(...) -> npu:0
ASR_LANGUAGE=en        # 仅默认值，前端按任务覆盖
```

`resolve_asr_device` 优先级：CUDA（本地 3060）→ NPU（torch_npu）→ CPU。

代码层面 `ASR_ENGINE=auto` / `ASR_ZH_MODEL=paraformer-zh` 已就绪，但服务器镜像未装 funasr，
所以即便设 auto 也会 fallback 回 whisper。要真正用 Paraformer 需先装 funasr 再重建容器。

重要经验：

- `Whisper tiny + npu:0` smoke test 成功。
- `Whisper small + zh + NPU` 已成功补跑一个 150 秒视频，耗时约 54.5 秒。
- `Whisper medium + zh + NPU` 在共享 2 号卡上 OOM，暂时不建议默认使用。
- **现有 4 个视频是多语种**（中文/英文/西班牙语），每个都已用对的语言索引：
  - 6223_bP5KfdFJzC4：中文 `zh` ✅
  - 世界杯广告：英文 `en` ✅（本来就是英文视频，en 正确）
  - 球星牛奶广告：西班牙语 `auto`→es ✅
  - 给阿嬷的情书预告片：中文 `zh`，台语腔偏糙（口音+small 模型限制，非配置问题）

## 8. 搜索 Pipeline

搜索接口：

```text
POST /api/search
```

字段：

```text
query_text
query_image
modalities=visual,face,asr
video_ids
alpha
limit
```

### 8.1 Visual 检索

1. CLIP encode_text / encode_image。
2. 与 `visual.npz` 做 cosine。
3. 对每个视频内部分数做 robust distribution。
4. 返回 strong/fuzzy/weak/fallback 候选。

当前已经不用固定 `cos >= 0.12`，而是结合：

```text
raw cosine
percentile
robust_z
median
MAD
```

### 8.2 Face 检索

两种方式：

1. 上传参考图，提取脸向量。
2. 查询文本中包含人物库名称，使用 entity embedding。

与 `faces.npz` 做 cosine，当前阈值约 0.35。

ArcFace cosine 经 logistic 函数校准为置信度，与 Visual percentile 同量纲：

```python
confidence = 1 / (1 + exp(-12 * (cosine - 0.45)))
# cosine=0.45 → 50%，cosine=0.65 → 92%
```

### 8.3 ASR 检索

当前 ASR 检索不是语义检索，是 lexical：

- 子串命中得分最高。
- 否则用字符 n-gram coverage。
- 支持部分简繁折叠。

所以：

```text
How many times → 可以搜到 "How many times?"
someone complains → 不一定能搜到 "How many times?"
```

后续要加真正 ASR 语义检索，需要给 ASR chunk 建文本 embedding 索引，例如 `asr_semantic.npz`。

### 8.4 片段合并

检索返回的是原视频时间段，不是物理切片。

结果字段：

```text
media_url
start_time
end_time
score
modalities
evidence
thumbnail_url
above_threshold   # 是否高于阈值
decision          # strong / fuzzy / weak / lexical_hit
```

搜索返回全召回结果，`above_threshold=false` 的候选排在后面，前端用分隔线和 dimming 区分。
`above_count` 字段表示高于阈值的命中数。

Visual-only 相邻桶不会无限合并成整段视频；这是已修过的重要问题。

## 9. 目前重要事件记录

### 9.0 2026-06-26 本次开发内容（feat/asr-search-asset-improvements）

本次开发在 `feat/asr-search-asset-improvements` 分支提交了 4 个 commit，已 push 到 GitHub：

**后端改动：**

- `worker.py`：Windows 文件锁 EDEADLOCK 修复，`LK_LOCK` 改为轮询 `LK_NBLCK`，长任务队列不再死锁。
- `settings.py`：`asr_engine` 默认改为 `auto`，新增 `asr_zh_model=paraformer-zh`，`asr_device` 默认改为 `auto`。
- `indexing/asr.py`：新增 `resolve_asr_device()`（CUDA → NPU → CPU），`_funasr()` 和 `_whisper()` 都支持 npu 设备；`engine=auto` 优先 FunASR，不可用时 fallback Whisper（不再默默降级到 tiny/cpu）。
- `search.py`：新增 `face_confidence()` logistic 校准；全召回 + `above_threshold` 标记；最终排序 `(above_threshold, score)` 降序；返回 `above_count`。
- `main.py`：新增 `PATCH /api/videos/{id}`（重命名）和 `DELETE /api/videos/{id}`（删除视频+索引+缩略图+任务日志）。
- `db.py`：新增 `delete_video()`，`update_video()` 支持 name 字段。
- `schemas.py`：新增 `VideoRenameRequest`。

**前端改动：**

- `main.tsx`：播放弹窗只循环 `[start_time, end_time]`；结果卡片低于阈值显示 "低于阈值" chip + dimming；搜索结果头部展示命中数/低于阈值数，插入分隔线；Assets 页增加重命名/删除按钮。
- `api.ts`：`SearchResult` 加 `above_threshold`、`decision`；`SearchResponse` 加 `above_count`；新增 `renameVideo()`、`deleteVideo()`。
- `styles.css`：低分卡样式、阈值分隔线、危险按钮等。

### 9.1 ASR 卡住/搜不到

早期 ASR 问题包括：

- 没安装 `openai-whisper`。
- Whisper 内部找不到 ffmpeg。
- 视频无音轨导致任务失败。
- Whisper tiny 中文质量差。
- ASR 输出繁体，简体搜索搜不到。

已处理：

- 安装/使用 `openai-whisper`。
- 用 `imageio-ffmpeg` fallback。
- 先抽 wav，再用 Python `wave + numpy` 加载音频，规避 Whisper 内部 ffmpeg。
- 无音轨写空 ASR，不再失败。
- 前端可选 ASR 模型和语言。
- 搜索端增加部分简繁折叠。

### 9.2 Face 任务“卡住”

一次 Face job 显示 running，但 worker 进程已经不存在，日志显示 `KeyboardInterrupt`。已手动清理 stale job，并恢复视频状态。

排查方法：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'app\.worker|app\.stage_runner' }
```

服务器：

```bash
docker exec momentseek-current-app ps -ef | grep -E "worker|stage_runner"
```

### 9.3 服务器部署

服务器恢复连接后，做了：

1. 检查 `npu-smi info`。
2. 检查 Docker 容器和端口。
3. 确认已有旧 MomentSeek 服务在 `8300`，不覆盖。
4. 新建 current 服务目录：

```text
/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app
```

5. 同步本地代码和 runtime。
6. 修 SQLite 中 Windows 路径为容器路径。
7. 启动 `momentseek-current-app`，端口 `18300`。
8. 用 PC 做 SSH 转发和 Cloudflare 公网。

### 9.4 三路尽量上 NPU

已完成：

- Visual：NPU。
- Face：CANN provider。
- ASR：`ASR_DEVICE=auto -> npu:0`。

验证结果：

- Face CANN 初始化成功。
- Whisper tiny NPU smoke test 成功。
- Whisper small NPU 实际任务成功。
- Whisper medium NPU 实际任务 OOM。

因此当前推荐：

```text
ASR 默认 small
不要在共享 2 号卡上默认 medium/large-v3
```

## 9.5 910B 性能基准（2026-06-26 多视频实测）

在物理 2 号卡上对 4 个视频（150s/360p、360s/544p、64s/720p、117s/720p）做了分环节计时（visual 5fps / face 2fps，warm NPU）。三路都确认在 NPU：visual=`npu:0`、face=`cann`、asr=`npu:0`。

**分环节耗时（150s/360p 视频示例）：**

```text
VISUAL  total 23.7s | load 8.6  decode 4.2(750帧)  clip_encode 10.8
FACE    total 10.7s | load 1.9  decode 4.0(300帧)  detect+embed 4.9(320检出)
ASR     total 12.2s | audio 0.6  load 3.5  transcribe 8.2(14句)
```

**每秒视频处理耗时 + 1 小时串行外推：**

| 视频 | 分辨率 | Visual | Face | ASR转写 | 串行1h |
| --- | --- | --- | --- | --- | --- |
| 6223 | 640×360 | 0.10 s/s | 0.06 | 0.06(稀疏) | 13.6min |
| worldcup | 960×544 | 0.09 s/s | 0.06 | 0.095(密集) | 15.0min |
| grandma | 1280×720 | 0.16 s/s | 0.10 | 0.11(中) | 22.8min |
| milk | 1280×720 | 0.19 s/s | 0.09 | 0.27(极密) | 33.5min¹ |

¹ milk 仅 64s，×56 外推噪声大；worldcup/grandma 更可信。

**结论：1 小时视频真实区间 13–34min（串行），两个主因：**

1. **分辨率（卡 Visual）**：720p visual 比 360p 慢约 2 倍。关键——这个慢**不在 NPU**：`clip_encode` 大头是 CPU 把每帧 resize 到 224（PIL 预处理）+ 解码，分辨率越高越重。**Visual 这条路 NPU 几乎没干活，瓶颈是 CPU**。
2. **语音密度（卡 ASR）**：转写速率跨视频差 5 倍（稀疏 0.06 → 极密 0.27 s/s）。

**固定成本**：每路模型加载基本恒定 ≈ 14.5s/job（visual 8.6 + face 2 + asr 4），与分辨率无关。warm vs cold 差 ~2 倍（首次跑要编译 NPU kernel）。

**优化验证（做了又回滚）**：试过"一次解码喂 visual+face"合并 + 解码线程流水线。**合并反而慢 43%**（19.1s→27.4s）：①解码非瓶颈（`decode_wait`=0.09s，流水线全盖住）；②合并把 torch_npu(CLIP)+onnxruntime-CANN(face) 塞一个进程**同卡互抢**，推理 9s→15.6s。**"每阶段独立子进程"还隔离了两个 NPU 运行时**，已回滚。

> 部署教训：同步代码后没立刻重启 → 线上 uvicorn 内存里旧 media + 磁盘新 faces，人物库导入懒加载新 faces 时报 `cannot import threaded_frame_reader`。**改服务器代码必须立刻 `docker restart`**。

**真正该优化的杠杆（按收益）：**

1. **干掉每 job 模型重载**（process_exit 每阶段重载 14.5s + kernel 编译）→ 常驻模型服务最值钱。
2. **优化 visual 的 CPU 解码/预处理**（cv2.resize 替 PIL、GPU/NPU 上做 resize），而非堆 NPU 算力。
3. **ASR 换 faster-whisper / 分段并行**（长视频头号瓶颈）。
4. 三路并行（暂缓）。**解码合并已被证伪，别再走。**

### 9.5.1 已实现（本地仓库，待部署审阅）

- **cv2.resize 预处理**（杠杆 2）：`ClipEncoder` 从模型 transform 提取 resize/crop/mean/std，用 cv2 替 PIL，提取失败回退 PIL。NPU 实测预处理快 36%(360p)–42%(720p)，嵌入 cosine 0.996（检索无感）。改动在 `backend/app/indexing/visual.py`。
- **warm pool + 空闲超时**（杠杆 1）：`backend/app/model_pool.py`（模型缓存，空闲 `indexer_idle_timeout_seconds` 后释放并清 NPU 缓存）+ `backend/app/indexer_daemon.py`（长驻轮询 DB 队列，进程内跑各阶段，CLIP/InsightFace 池化）。常驻显存实测 ~2.3GB。`build_visual_index`/`build_face_index` 加了可选 `encoder` 参数。用 `python -m app.indexer_daemon` 替代 per-job 子进程 worker；API 路径不变。
- 测试：`backend/tests/test_model_pool.py`（缓存/空闲驱逐/队列取最旧），全套 16 项通过。
- **部署注意**：①cv2 部署后新建的 visual 索引用 cv2，旧索引用 PIL，两者 cosine 0.996 可混用，无需重建；②warm pool 守护进程会和 `launch_job` 抢同一队列，启用时需二选一（要么跑 daemon、要么走子进程）；③改服务器代码后必须 `docker restart`（见上方部署教训）。

## 10. 当前已知问题

### 10.1 公网链接依赖 PC

当前 Cloudflare tunnel 不是跑在服务器，而是跑在用户 PC 上。PC 断网/睡眠/VPN/SSH 断开都会导致公网链接不可用。

长期方案：

- 在服务器直接跑 cloudflared。
- 或使用正式 Cloudflare named tunnel。
- 或部署带鉴权的公网服务。

### 10.2 无鉴权

当前公网链接任何人拿到都能访问、上传、检索。不适合敏感视频。

建议尽快加：

- Basic Auth。
- 简单访问密码。
- 或 Cloudflare Access。

### 10.3 ASR 不是语义检索

当前 ASR 只能关键词/近似字符匹配。后续要做语义，需要：

```text
asr chunks → text embedding → asr_semantic.npz
query text → same embedding model → cosine search
```

可选模型：

- bge-small-zh/en
- multilingual-e5
- text2vec
- 服务器上可考虑轻量 embedding 模型。

### 10.4 medium OOM

共享 2 号卡上曾出现：

```text
Whisper medium + zh + NPU
RuntimeError: NPU out of memory
```

处理方式：

- 前端默认改为 `small`。
- 失败视频只补跑 ASR small，不重跑 Visual/Face。
- 保留 failed job 作为历史记录，不影响 ready 视频。

### 10.5 Face CANN 仍需性能验证

Face provider 已切 CANN，初始化和任务跑通。但实际是否所有子图都在 NPU 上，需要进一步 profile。onnxruntime 有可能对不支持的 op fallback 到 CPU。

## 11. 常用运维命令

### 11.1 服务器 health

```bash
ssh drama-server "curl -s http://127.0.0.1:18300/api/health"
```

### 11.2 公网 health

```powershell
curl.exe -s https://jobs-qualified-eligibility-soft.trycloudflare.com/api/health
```

### 11.3 查看容器

```bash
ssh drama-server "docker ps --filter name=momentseek-current-app --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

### 11.4 查看日志

```bash
ssh drama-server "docker logs --tail=200 momentseek-current-app 2>&1"
```

### 11.5 查看 NPU

```bash
ssh drama-server "npu-smi info"
```

只看进程：

```bash
ssh drama-server "npu-smi info | grep -A30 'Process id'"
```

### 11.6 查看任务

```powershell
curl.exe -s https://jobs-qualified-eligibility-soft.trycloudflare.com/api/jobs
```

或者服务器：

```bash
ssh drama-server "curl -s http://127.0.0.1:18300/api/jobs | python -m json.tool | head -120"
```

### 11.7 单独补跑 ASR

PowerShell：

```powershell
$body = @{ modalities = @('asr'); asr_model = 'small'; asr_language = 'zh' } | ConvertTo-Json -Compress
Invoke-RestMethod `
  -Uri 'https://jobs-qualified-eligibility-soft.trycloudflare.com/api/videos/{video_id}/index' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body
```

### 11.8 搜索测试

ASR：

```powershell
curl.exe -s -X POST https://jobs-qualified-eligibility-soft.trycloudflare.com/api/search `
  -F "query_text=How many times" `
  -F "modalities=asr" `
  -F "limit=1"
```

Visual：

```powershell
curl.exe -s -X POST https://jobs-qualified-eligibility-soft.trycloudflare.com/api/search `
  -F "query_text=football player" `
  -F "modalities=visual" `
  -F "limit=1"
```

### 11.9 清理 stale running job

如果容器重启后任务仍显示 running，但没有 worker/stage_runner 进程，需要手动清理。建议写脚本操作 SQLite，不要手敲复杂 SQL。

思路：

```python
update jobs set status='failed', stage='failed', error='stale running job cleaned'
where status in ('running','queued')

update videos set status='ready'
where status='indexing'
```

注意：如果某视频确实没有任何索引，应设回 `uploaded`；如果已有部分索引，可设 `ready`。

## 12. 当前 Git 状态

2026-06-26 已在 `feat/asr-search-asset-improvements` 分支提交 4 个 commit 并 push 到 GitHub：

```text
feat: add asset management (rename, delete video) with file cleanup
feat: full-recall search with above_threshold marking and segment player loop
feat: face confidence calibration and asr device resolution (CUDA/NPU/CPU)
fix: replace msvcrt LK_LOCK with polling LK_NBLCK to prevent EDEADLOCK
```

PR 需在 GitHub 手动创建（`gh` CLI 未安装）：

```text
https://github.com/502022240018/MomentSeek/pull/new/feat/asr-search-asset-improvements
```

服务器 `momentseek-current-app` 容器已挂载最新代码（`/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/backend`），代码已是最新；但环境变量需通过 `.env` 修正后重启容器生效。

如果 GitHub 走代理：

```bash
git config http.proxy http://127.0.0.1:7890
git config https.proxy http://127.0.0.1:7890
```

## 13. 下一步建议

优先级建议：

1. **立即**：在 GitHub 手动创建 PR（`feat/asr-search-asset-improvements → main`）。
2. 加鉴权，至少 Basic Auth 或 Cloudflare Access。
3. 把公网 tunnel 从 PC quick tunnel 改成服务器直接跑的 named tunnel（现在 PC 断开公网就断）。
4. 在服务器镜像里装 funasr，重建容器把 `ASR_ENGINE=auto`，改善中文（尤其台语腔）ASR 质量。
5. 前端增加”ASR 文本预览”，快速判断搜不到是 ASR 问题还是搜索问题。
6. ASR 加语义检索（chunk → text embedding → cosine）。
7. 给 job 加 cancel 功能。
8. Face CANN 做 profile，确认实际 NPU 加速效果。
9. 把前端 `main.tsx` 拆组件，方便多人协作。
10. 搜索 profile 控件（recall / balanced / precision）暴露到前端。

> 注：服务器容器当前工作正常，不需要为了改 `ASR_LANGUAGE` 默认值去重建（前端按任务覆盖即可）。
> 只有在”装 funasr + 启用 auto”时才值得重建容器，且必须带 `--privileged`。

## 14. 给接手同事的一句话

MomentSeek 当前已经能演示“上传视频、建立 visual/face/asr 三路索引、按文本/图像/人物/语音检索出视频时间片段”。  
当前最重要的工程状态是：服务器 current 版跑在 `momentseek-current-app`，通过用户 PC 的 SSH + Cloudflare 暴露公网；三路尽量走 2 号 NPU，但 Whisper medium 在共享卡上会 OOM，所以默认必须用 small。后续重点不是把 demo 再堆功能，而是做稳定性、鉴权、ASR 语义检索、任务取消和多人协作代码整理。
