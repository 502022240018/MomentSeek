# 模型管理

## 总原则

MomentSeek 的模型管理以 model manifest 和 models lock 为准。manifest 声明需要哪些模型、来源、目标目录和是否允许下载；lock 记录一次校验后的实际结果，用于复现和审计。

开发环境可以显式下载校验脚本支持的模型条目，方便新同学启动。当前 `scripts/verify_models.py --download` 只下载 Hugging Face 条目；dev manifest 中的 InsightFace、FunASR/SenseVoice、faster-whisper 和 RapidOCR 条目是 bootstrap 阶段的可选校验项。运行时默认只读取 `models/` 或 `/app/models` 下的本地模型，缺失时应明确报错，而不是在索引任务中隐式联网下载。线上环境必须预缓存模型并校验，禁止运行时下载，避免部署时依赖外网、污染共享缓存或在索引任务中出现不可控延迟。

## 开发模型

开发默认使用：

```text
deploy/models/dev-full.models.json
```

该 manifest 的 `allow_download` 为 `true`，覆盖本地开发需要的 visual、face、asr、ocr 和 semantic 模型。`dev.cpu` 与 `dev.cuda` 都使用这份清单。开发 visual 默认使用 `VISUAL_MODEL=chinese-clip-vit-b16`，对应 Hugging Face 模型 `OFA-Sys/chinese-clip-vit-base-patch16`。`allow_download` 表示允许开发校验入口尝试下载其支持的条目，不表示所有模型都会由 `verify_models.py --download` 直接拉取。当前 dev manifest 将 Hugging Face visual / semantic 设为必需，把 Face、SenseVoice/faster-whisper、RapidOCR 设为可选，以保证 clean clone 的 bootstrap 不被非 HF 缓存阻塞。

开发启动时可以运行：

```powershell
scripts/bootstrap_dev.ps1 -Profile dev.cpu -DownloadModels
```

或：

```bash
scripts/bootstrap_dev.sh dev.cpu --download
```

没有 `-DownloadModels` 或 `--download` 时，校验脚本只检查已有缓存并写 lock，不主动下载。即使传入下载开关，非 Hugging Face 条目也可能需要先按库要求准备依赖或缓存。ASR 默认使用 `ASR_ENGINE=auto` + `ASR_MODEL=turbo` + `ASR_ZH_MODEL=iic/SenseVoiceSmall` + `ASR_VAD_STRATEGY=silero_12s`：先用 faster-whisper turbo 做轻量语言 probe，中文/方言走 SenseVoiceSmall，非中文走 faster-whisper turbo。SenseVoice 本地目录约定是 `models/funasr/iic/SenseVoiceSmall`，并通过 `silero-vad` 做外置 VAD；faster-whisper turbo 本地 snapshot 目录约定是 `models/faster-whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`。`fsmn-vad` 保留为 `ASR_VAD_STRATEGY=funasr_fsmn` fallback；`ct-punc` 用于非 SenseVoice FunASR 路径。

## Ascend Staging/Prod 模型

Ascend staging/prod 使用：

```text
deploy/models/ascend-prod.models.json
```

该 manifest 的 `allow_download` 为 `false`。模型必须提前放入服务器宿主机模型目录，并挂载到容器内 `/app/models`。staging/prod 启动前先运行校验并生成 lock；校验失败时修复模型缓存，而不是让服务在运行时下载。

注意路径视角：`ascend-prod.models.json` 的 target 是容器内 `/app/models/...`，不是宿主机 `/opt/momentseek/models/...`。线上校验应在容器内执行，或在一个把宿主机模型目录挂载成 `/app/models` 的等价环境中执行。不要把 manifest 中的 target 临时改成宿主机路径；路径变化应通过挂载关系解决，否则 release manifest 和 runtime 配置会失去一致性。

## 模型目录

当前约定路径：

```text
开发默认：models/
Ascend 标准宿主机：/opt/momentseek/models
容器内：/app/models
当前服务器宿主机：/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp/models
```

开发 profile 的 `APP_MODEL_DIR=models`。Ascend profile 的 `APP_MODEL_DIR=/app/models`，由 Compose 通过 `HOST_MODEL_DIR=/opt/momentseek/models` 挂载到宿主机模型目录。

## Model Manifest

model manifest 是模型需求声明，字段包括：

```text
schema_version
name
allow_download
models[].name
models[].kind
models[].id
models[].target
models[].required
```

当前模型类别：

```text
visual：SigLIP2、ChineseCLIP 或 OpenCLIP，用于文本/图片搜画面。
face：InsightFace buffalo_l / ArcFace，用于参考图和人物库检索。
asr：默认 auto 路由；中文/方言使用 SenseVoiceSmall/FunASR，英文/西语/葡语等使用 faster-whisper turbo。
ocr：RapidOCR PP-OCRv4，用于画面文字检索。
semantic：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2，用于 ASR/OCR 语义检索。
```

manifest 只描述模型需求，不描述具体部署 release；release manifest 会引用 model manifest。

## Models Lock

models lock 是校验结果快照，用于确认某台机器上的模型缓存满足 manifest。推荐路径：

```text
runtime/dev-models.lock.json
models/models.lock.json
```

lock 应包含 profile、manifest 名称、模型条目、目标路径、是否存在和校验状态。staging/prod 的 lock 应和 release manifest 一起保存，作为新服务器复制和回滚的依据。

## 下载和校验

开发环境下载并校验 Hugging Face 条目，同时校验其他条目是否已由对应库或缓存准备好：

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json --download
```

开发环境只校验：

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json
```

Ascend staging/prod 只校验。以下命令必须在容器内或等价 `/app/models` 挂载环境中运行：

```powershell
python scripts/verify_models.py --manifest deploy/models/ascend-prod.models.json --lock /app/models/models.lock.json
```

若线上校验失败，应补齐预缓存模型后重新校验。不要通过修改线上 manifest 的 `allow_download` 绕过校验。

## 运行时禁止隐式下载

MomentSeek 的后端运行时禁止隐式下载模型。开发环境可以通过 bootstrap 或 `scripts/verify_models.py --download` 显式准备脚本支持的模型；服务启动、搜索和索引阶段只读取本地模型。staging/prod 进一步要求所有必需模型提前预缓存并通过 models lock 校验。原因：

```text
外网可用性不可控。
下载会拉长启动或索引耗时。
共享服务器缓存可能被污染。
模型版本漂移会破坏可复现性。
```

线上只允许使用 release manifest 指定的 model manifest 和已生成的 models lock。任何模型清单、缓存目录或下载策略变化，都应更新 `docs/MODELS.md`，并在 `docs/ISSUES_AND_ROADMAP.md` 记录需要追踪的问题或后续工作。
