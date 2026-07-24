# 正式试用版依赖基线

更新时间：2026-07-24

## 目标

正式试用版必须满足四个条件：

1. 构建输入可复现，运行时不隐式下载模型或源码。
2. CPU、CUDA、Ascend 的设备栈彼此隔离，业务依赖保持同源。
3. 平台、Milvus、Planner/Reranker 是独立服务，可分别升级和回滚。
4. 所有启用的检索模态（visual、face、asr、ocr、speaker）均有依赖、
   模型资产和启动前校验；缺少必需资产时部署失败，而不是任务运行后才失败。

## 唯一来源

| 层 | 来源 | 说明 |
|---|---|---|
| 通用 Python | `backend/requirements.txt` | CPU/CUDA 完整功能入口 |
| Speaker | `backend/requirements-speaker.txt` | 只包含生产 CAM++ 路径的直接依赖 |
| Ascend 可解析层 | `backend/requirements-ascend.txt` | 可由 pip 正常解析，不允许改写设备 ABI |
| Ascend 设备保持层 | `backend/requirements-ascend-preserve.txt` | 使用 `--no-deps`，保护 MindIE Torch 栈 |
| Ascend ABI 合同 | `backend/constraints-ascend.txt` | 基础镜像必须提供并保持的版本 |
| Milvus ARM64 客户端 | `backend/requirements-milvus-client.txt` + `vendor-wheels/` | 远程服务模式，不安装 milvus-lite |
| 前端 | `frontend/package-lock.json` | `npm ci`，禁止无 lock 构建 |
| 模型/源码资产 | `deploy/models/ascend-prod.models.json` | 必需资产及固定版本 |
| 服务镜像 | Compose 文件 | 平台、Milvus、Planner/Reranker 分离 |

`scripts/deploy_ascend_shared_server.sh` 不再生成第二份 requirements 或
Dockerfile，只使用仓库内的 `Dockerfile.ascend` 和上述清单。

## 正式试用版矩阵

### Ascend 平台

- 基础镜像：
  `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts`
- Python 3.11、openEuler 24.03、Torch 2.9.0、torch_npu 2.9.0.post1。
- NumPy 1.26.4、SciPy 1.14.1、Transformers 4.51.0。
- `torch/torch_npu/torchvision/torchaudio` 作为一个 ABI 单元升级，禁止单包升级。
- `pip check` 对厂商镜像已有的元数据误报不作为单独放行依据；必须同时通过
  固定版本断言、关键 import、CANN EP、NPU tensor 和真实模型 smoke。

### Milvus

- Milvus Server 2.6.20。
- PyMilvus 2.6.16（官方 2.6.20 对应 SDK）。
- 试用版从空的 2.6 collection 建库，再从保留的 NPZ 全量 backfill。
- 2.4 到 2.6 不做原地数据目录升级；需要回滚时回到独立 2.4 数据目录和旧应用镜像。
- SQLite 继续保存业务元数据，NPZ 在试用阶段继续作为恢复和检索回退资产。
- Milvus 请求先做 0.5 秒 TCP 可达性预检，SDK 查询失败预算固定为 3 秒；
  单次请求第一次失败后，其余视频直接走 NPZ，避免 SDK 默认重连按
  “视频 × 模态”串行累积。

### Speaker

- 固定 `modelscope/3D-Speaker` 源码 revision `065629c313ea`。
- 固定 CAM++：
  `iic/speech_campplus_sv_zh_en_16k-common_advanced@v1.0.0`。
- 固定 FSMN VAD：
  `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@v2.0.4`。
- VAD 和 Kaldi FBank 固定在 CPU；CAM++ 主网络在 Ascend 使用 NPU，在 CUDA
  使用 CUDA，其余使用 CPU。该边界避开 torch_npu 不支持的复数 FFT abs 算子。
- 音频入口统一经 ffmpeg 转为 16 kHz 单声道 PCM WAV，Speaker 适配层使用
  SoundFile 读取。不要仅为 `torchaudio.load()` 安装 TorchCodec；这会重复
  引入一套 FFmpeg ABI，增加离线镜像体积和动态库兼容面。
- 不安装官方研究环境的 NumPy `<1.24` / scikit-learn `1.0.2`。
- 不启用重叠说话检测，因此不安装 `pyannote.audio`。
- 聚类使用现有 SciPy + scikit-learn，Speaker 不再直接依赖 ARM64 需现场
  编译的 `fastcluster` 和外部 `hdbscan`；FunASR 自身声明的 `umap-learn`
  仍由 FunASR 依赖闭包管理。

### Planner / Reranker

- 平台只依赖 OpenAI-compatible HTTP 协议，不在平台镜像安装 vLLM。
- Qwen3.5-4B vLLM Ascend 服务保持独立容器、独立 NPU、独立镜像。
- Planner 与 Reranker 的 provider、模型、prompt 和超时由 profile 配置；
  允许共用一个服务，也允许分别指向两个模型。
- 平台升级不能隐式重启推理服务，推理服务升级也不能改写平台依赖。

### CPU / CUDA

- CPU 和 CUDA 文件继承通用依赖，设备栈在各自文件中固定。
- 当前 Torch 2.6 CPU/CUDA 线继续作为开发基线；升级到 2.9 需要单独验证
  OpenCLIP、InsightFace、Whisper 和显存占用，不能跟随 Ascend 自动升级。
- `faster-whisper` 和 `openai-whisper` 是 CPU/CUDA 完整回退能力；Ascend 正式
  profile 使用 FunASR，不把缺少 ARM64 CTranslate2 的 faster-whisper 宣称为可用能力。

### 前端

- React 19、TypeScript 5.9.3、Vite 6.4.3，按 lockfile 构建。
- Node 只存在于 frontend build stage，不进入运行镜像。
- 前端依赖升级必须同时通过 TypeScript build、静态资源打包和真实 API smoke。

## 已知厂商元数据例外

- `insightface`、`silero-vad` 的 PyPI 元数据声明 `onnxruntime`，Ascend 镜像实际
  使用 ABI 兼容的 `onnxruntime-cann`。
- 厂商 `torchvision 0.16` 元数据仍声明 Torch 2.1，但 MindIE 镜像提供并验证
  的设备栈是 Torch 2.9；平台不调用加载失败的可选 `torchvision.io` 扩展。
- CANN 的 `op-compile-tool` 把 Python 标准库模块列成外部包，属于厂商元数据误报。
- FunASR/Hydra/OmegaConf 固定 ANTLR 4.9.3；`latex2sympy2-extended` 的 4.13.2
  声明不在平台执行路径。上述例外不能仅凭 `pip check` 放行，必须通过实际
  import、CANN EP、NPU tensor 和模型 smoke。
- PyMilvus 2.6 的 ORM API 已标记将在 3.1 移除。正式试用版继续使用已验证的
  2.6 ORM；迁移到新 `MilvusClient` API 作为下一条 P1 技术债，不能与本次
  2.4→2.6 数据迁移同时改写。

## 升级准则

直接纳入基线：

- 能删除重复依赖或编译链、降低 ARM64 构建风险的替代。
- 有明确性能/可靠性收益，且数据可从 NPZ 重建的存储升级。
- 不改变模型输出空间或有完整回归集覆盖的补丁升级。

独立 A/B 镜像后再切换：

- MindIE、CANN、Torch、torch_npu、ONNX Runtime CANN 任一项变化。
- 会改变 embedding space、切分时间轴、speaker track 或检索分数的模型升级。
- Milvus 跨存储格式升级。

## 发布门禁

1. requirements 引用闭包、JSON/YAML/Compose 配置和 frontend lock 校验。
2. 后端全量单元测试。
3. ARM64 镜像 build；固定版本断言和关键模块 import。
4. Milvus schema dry-run、建库、每模态写入/检索/删除 smoke。
5. 五模态各一条真实索引；Speaker 额外验证 VAD、CAM++ NPU embedding 和聚类。
6. 历史 NPZ 全量 backfill，按视频/模态核对行数。
7. Planner + Reranker 请求 smoke；失败开放策略和 trace 均验证。
8. 前端上传、检索、人物登记/删除、speaker 页面真实操作。

任何一项失败都不替换现有共享服务器容器。
