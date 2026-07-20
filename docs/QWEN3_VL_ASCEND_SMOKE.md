# Qwen3-VL Ascend 单图 Smoke

本流程在独立临时容器中验证 Qwen3-VL 的兼容性、基础能力、热请求延迟和峰值 NPU HBM，不修改或停止正式 MomentSeek 容器。

## 前置条件

- 服务器为 Atlas 800I A2 / Ascend 910B4，现场 CANN 8.5.1。
- 已确认并获准使用一张空闲实验卡；默认物理 NPU 6。
- 正式容器 `momentseek-29154-platform` 正常运行，用于解析匹配驱动的基础镜像。
- 模型已离线传到 `/home/momentseek-29154/vlm-exp/models/Qwen3-VL-2B-Instruct`。
- 测试图位于 `/home/momentseek-29154/vlm-exp/input/test.jpg`。

模型目录至少应包含 `config.json`、processor/tokenizer 文件及完整 safetensors 权重。不要在运行时依赖 Hugging Face 或 ModelScope。

## 执行

```bash
cd /home/momentseek-29154/platform
git pull --ff-only

npu-smi info -t proc-mem -i 6 -c 0

bash scripts/run_qwen3_vl_ascend_smoke.sh
```

首次执行会在 `/home/momentseek-29154/vlm-exp/venv` 安装独立的 Transformers 4.57+ 环境。它复用基础镜像中的 Torch/torch-npu，不更改正式容器。后续执行复用该 venv。

结果写入：

```text
/home/momentseek-29154/vlm-exp/output/<model>-<timestamp>.json
/home/momentseek-29154/vlm-exp/output/<model>-<timestamp>.log
```

常用覆盖参数：

```bash
PHYSICAL_NPU=6 \
MODEL_NAME=Qwen3-VL-4B-Instruct \
RUNS=20 WARMUP_RUNS=3 MAX_NEW_TOKENS=96 \
  bash scripts/run_qwen3_vl_ascend_smoke.sh
```

模型或图片不在默认位置时：

```bash
MODEL_HOST=/absolute/path/to/model \
IMAGE_HOST=/home/momentseek-29154/vlm-exp/input/test.jpg \
  bash scripts/run_qwen3_vl_ascend_smoke.sh
```

模型和图片会以只读方式单独挂载，因此可以使用实验目录外的绝对路径。脚本拒绝占用存在进程的 NPU，临时推理容器退出后会自动删除。

## 成功标准

- 输出包含合理的中文画面描述。
- JSON 包含加载时间、p50/p95、token/s、峰值 allocated/reserved HBM。
- 运行结束后 `npu-smi info -t proc-mem -i 6 -c 0` 不残留实验进程。

如果出现兼容性错误，保留完整 log，并记录基础镜像、`pip freeze`、`npu-smi info`。不要直接升级宿主 CANN、驱动或正式平台容器的 Python 包。
