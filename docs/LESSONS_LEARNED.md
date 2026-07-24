# 经验与踩坑记录

本文档记录未来新会话必须继承的操作经验、工具坑和事故教训。

## 记录格式

```text
日期：
类别：
现象：
根因：
影响：
经验：
以后规则：
相关文档：
```

## 2026-07-03 PowerShell UTF-8 输出

```text
类别：PowerShell
现象：
  中文 Markdown 用默认 PowerShell 输出时出现乱码。
根因：
  控制台输出编码没有设为 UTF-8。
影响：
  如果直接相信乱码输出，可能误读交接文档。
经验：
  读取中文文档时必须显式使用 UTF-8。
以后规则：
  使用：
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Get-Content -Raw -Encoding UTF8 <path>
相关文档：
  docs/README.md
```

## 2026-07-03 PowerShell 通配符和 `rg`

```text
类别：PowerShell
现象：
  类似 `rg ... docs/*.md` 的命令在 Windows 下报路径语法错误。
根因：
  PowerShell/Windows 路径和 Unix 风格 glob 行为不同。
影响：
  搜索可能失败，或者没有覆盖目标文件。
经验：
  优先使用 `rg -g "*.md" <root>` 或 `rg --files`。
以后规则：
  使用：
    rg -n "pattern" docs -g "*.md"
    rg --files docs
相关文档：
  docs/VALIDATION.md
```

## 2026-07-03 Git diff 看不到未跟踪文件

```text
类别：Git
现象：
  编辑 handoff 文档后，`git diff -- docs/handoff/...` 没有输出。
根因：
  handoff 目录当时是未跟踪目录，普通 `git diff` 不展示未跟踪文件内容。
影响：
  容易误以为没有改动，或者漏看未跟踪文档。
经验：
  `git diff` 要配合 `git status --short` 和直接内容搜索。
以后规则：
  使用：
    git status --short
    rg -n "expected text" <untracked-file-or-directory>
相关文档：
  docs/VALIDATION.md
```

## 共享服务器误杀进程事故

```text
日期：2026-07-02
类别：Server
现象：
  一次服务器操作误杀了其他人的进程。
根因：
  终止进程前没有把归属缩小到足够明确；在共享服务器上用宽泛进程匹配非常危险。
影响：
  可能中断其他人的 ComfyUI、VLLM 或其他 python/NPU 任务。
经验：
  服务器是共享环境。NPU 占用或进程名叫 python，都不能证明它属于 MomentSeek。
以后规则：
  禁止 broad kill。任何 kill/restart 前：
    1. 精确确认 MomentSeek 容器/进程。
    2. 检查 active jobs。
    3. 检查 docker ps 和 npu-smi。
    4. 确认目标归属 MomentSeek。
    5. 优先只在明确批准后操作 `momentseek-current-app`。
相关文档：
  docs/OPERATIONS.md
```

## Windows 文件名大小写

```text
日期：2026-07-03
类别：Git / Windows
现象：
  想创建 `ARCHITECTURE.md` 时，仓库里已有 `architecture.md`，Windows 下两者不能安全共存。
根因：
  当前文件系统对路径大小写不敏感。
影响：
  如果直接创建大写新文件，Git 状态可能混乱。
经验：
  先归档或改名旧文件，再创建大小写不同的新文件。
以后规则：
  仅改变文件名大小写时，使用两步 rename：
    git mv old.md temp.md
    git mv temp.md NEW.md
相关文档：
  docs/README.md
```

## 2026-07-07 本地 CUDA visual 设备字符串

```text
类别：Docker / CUDA / PyTorch
现象：
  本地 Docker GPU 后端 health 和 ASR 正常，但 visual 搜索返回：
    Expected a torch.device with a specified index or an integer, but got:cuda
根因：
  visual 通道的 resolve_device 在 CUDA 可用时返回 "cuda"，
  后续 torch.cuda.set_device(device) 需要 "cuda:0" 或整数设备号。
影响：
  SigLIP2 visual query encoder 无法在本地 CUDA 后端初始化，visual 搜索失败。
经验：
  NPU 的 "npu:0" 和 CUDA 的 "cuda:0" 都要显式带设备编号；
  不要把 torch 的通用 device 字符串和 set_device 接口要求混为一谈。
以后规则：
  本地 CUDA profile 修改后至少验证：
    1. /api/health
    2. /api/videos
    3. ASR 搜索
    4. visual 搜索
  visual 设备选择逻辑必须有单元测试覆盖。
相关文档：
  docs/LOCAL_GPU_MIGRATION.md
```

## 2026-07-07 Cloudflare quick tunnel 临时域名

```text
类别：公网入口
现象：
  旧 trycloudflare 域名 DNS 解析失败，前端无法访问。
根因：
  Cloudflare quick tunnel 是临时地址；进程退出或时间变化后，旧域名可能失效。
影响：
  这类失败不是后端 502，也不是 Docker API 崩溃，而是公网入口已经不存在。
经验：
  先检查本地 /api/health，再检查 cloudflared 进程和日志。
以后规则：
  quick tunnel 只适合自己和少数同学临时测试；
  失效后用 runtime/tools/cloudflared.exe 重新创建，并把新地址写入 docs/CURRENT.md。
相关文档：
  docs/OPERATIONS.md
  docs/LOCAL_GPU_MIGRATION.md
```

## 2026-07-07 ASR auto 语言与翻译型输出

```text
类别：ASR / indexing quality
现象：
  某些旧 ASR 索引里，中文台词被记录成等价英文文本，导致用户看到“原视频中文、索引英文”的错位。
根因：
  旧索引没有记录 Whisper task、requested_language、detected_language 等证据；
  如果 ASR 调用链曾使用 translate 或旧路径产生翻译文本，后续无法从 manifest 追溯。
影响：
  中文 query 可能仍能被 multilingual embedding 找到一部分结果，但 lexical 匹配、证据展示和用户信任都会变差。
经验：
  ASR 文本语言异常时，先看 index_manifest.json 的 task/requested_language/detected_language；
  再抽样打开 asr.npz 的 texts，而不是只看搜索结果。
以后规则：
  Whisper 必须显式 task="transcribe"；
  asr_engine=auto 且 asr_language=auto 时应先走 faster-whisper turbo 轻量语言 probe，再按语言路由，不能无条件先走中文 FunASR；
  ASR manifest 必须记录 task、requested_language、detected_language、chunk_builder_stats、text_profile；
  发现旧索引疑似翻译输出时，优先 ASR-only 重跑，不要直接重跑全部通道。
相关文档：
  docs/RETRIEVAL_CHANNELS.md
  docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md
```

## 2026-07-08 本地容器模型缓存位置

```text
类别：Docker / model cache
现象：
  本地 ASR 实验跑过 SenseVoice/FunASR，但按新默认配置重建 ASR 时，
  /app/models/funasr 下找不到 iic/SenseVoiceSmall、fsmn-vad、ct-punc。
根因：
  实验阶段模型可能下载到了容器内部 /root/.cache/modelscope；
  该目录不是宿主机挂载目录，docker compose --force-recreate 后容器可写层会被替换。
影响：
  新代码的 local-only 模型解析会正确报缺模型；如果没有提前迁移缓存，就需要重新显式下载模型。
经验：
  能跑过实验不等于模型已经进入项目可复用目录。必须检查宿主机挂载目录，而不是只看实验是否成功。
以后规则：
  所有要长期复用的模型必须放在项目 models/ 或部署约定的 /app/models 挂载目录；
  recreate 容器前，如果怀疑模型在容器内部 cache，先复制 /root/.cache/modelscope 到宿主机 models/funasr；
  运行时索引不得隐式下载模型，只能由 bootstrap、verify_models.py --download 或人工显式模型准备步骤下载。
相关文档：
  docs/MODELS.md
  docs/DEVELOPMENT.md
```

## 2026-07-10 本地 Docker 调试不要默认发布式重建

```text
类别：Docker / local development
现象：
  本地只改 ASR 后端和前端默认选项时，运行 docker compose up -d --build 触发 CUDA requirements 重新安装，
  开始重新下载 700MB+ 的 PyTorch wheel。
根因：
  当前 Dockerfile 把依赖安装和源码烘进镜像；requirements 层一旦缓存失效，发布式 build 会重装重依赖。
影响：
  普通代码调试被拖成很长的镜像构建，也容易让人误以为“每次改代码都要重新安装所有东西”。
经验：
  本地频繁调试应该使用源码挂载的 compose.dev.yml；发布式 build 只用于依赖、Dockerfile 或正式交付镜像变化。
以后规则：
  本地开发优先启动：
    docker compose -f compose.yml -f compose.cuda.yml -f compose.dev.yml --env-file .env up -d --no-build app
  前端改动先在宿主机 npm run build，让挂载的 frontend/dist 被 FastAPI 读取。
  只有 requirements、Dockerfile、系统依赖或正式发布镜像变化时，才使用 --build。
相关文档：
  docs/DEVELOPMENT.md
```

## 2026-07-13 开发容器挂载和临时 pip 安装不是一回事

```text
类别：Docker / local development
现象：
  宿主机 ASR 新代码的 149 个测试通过，但容器正式回归仍执行旧 asr.py；
  使用 compose.dev.yml 重建后，faster-whisper 又从容器中消失。
根因：
  原容器没有挂载 backend/app，运行的是旧镜像源码；
  faster-whisper 曾只临时安装在容器可写层，镜像本身没有该依赖，recreate 后自然丢失。
影响：
  宿主机测试和容器真实行为分叉；临时安装造成“之前能跑、重建后不能跑”。
经验：
  调试前必须用 docker inspect 检查 backend/app mount；
  pip show 只能证明当前容器有包，不能证明镜像含有该包。
以后规则：
  本地开发固定使用 compose.yml + compose.cuda.yml + compose.dev.yml；
  Python 依赖必须写入 requirements 并至少构建一次镜像，不能把临时 pip install 当部署结果；
  代码改动用 reload，只有依赖清单变化才重建镜像。
相关文档：
  docs/DEVELOPMENT.md
  compose.dev.yml
```

## 2026-07-13 长视频语言不能只探测片头

```text
类别：ASR / language routing
现象：
  全量重建时，中文书籍纪录片被片头 30s 探测成 en，错误路由到 faster-whisper；
  同一视频主体片段和人工听查都表明主要语言是中文。
根因：
  auto route 只采样视频开头，默认假设片头语言等于整片主语言；长视频的片头、预告、音乐或外语引语会破坏该假设。
影响：
  可能再次生成与主体语言不一致的文本，损害 lexical 检索、证据展示和用户信任。
经验：
  语言识别正确不代表采样策略正确；必须区分“模型判错”和“样本不代表主体”。
以后规则：
  短视频可单窗口探测；长视频固定从开头、主体中段和后段取 3 个短窗口投票；
  票数优先，置信度和只用于同票决胜；manifest 继续记录 detected_language/language_route/route_reason；
  全量重建时逐条检查最终 engine 和 detected_language，发现异常立即停止后续自动提交。
相关文档：
  docs/RETRIEVAL_CHANNELS.md
  docs/experiments/asr/2026-07-13-asr-production-chunk-pipeline.md
```
