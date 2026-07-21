# Qwen3-VL Ascend 单图 Smoke

本流程在独立临时容器中验证 Qwen3-VL 的兼容性、基础能力、热请求延迟和峰值 NPU HBM，不修改或停止正式 MomentSeek 容器。

## 前置条件

- 服务器为 Atlas 800I A2 / Ascend 910B4，现场 CANN 8.5.1。
- 已确认并获准使用一张空闲实验卡；默认物理 NPU 7。
- 正式容器 `momentseek-29154-platform` 正常运行，用于解析匹配驱动的基础镜像。
- 服务器可以稳定访问 ModelScope；模型缺失或不完整时会自动从官方 `Qwen/Qwen3-VL-2B-Instruct` 下载到 `/home/momentseek-29154/vlm-exp/models/Qwen3-VL-2B-Instruct`。
- 可选：测试图位于 `/home/momentseek-29154/vlm-exp/input/test.jpg`。如果不存在，脚本会从 `runtime/uploads` 第一条视频的第 10 秒自动抽帧。

模型目录完整时会直接复用；至少应包含 `config.json`、processor/tokenizer 文件及完整 safetensors 权重。下载中断后可重新执行脚本继续准备。设置 `DOWNLOAD_MODEL=false` 可以禁止自动联网下载并在模型缺失时直接失败。

## 执行

模型准备好后，推荐使用一键入口。它会检查实验分支、正式服务 health、模型文件、NPU 7 和 runtime 视频，运行 smoke 后展示最新 JSON 并检查 NPU 是否释放：

```bash
cd /home/momentseek-29154/platform
git fetch origin agent/ascend-qwen3-vl-smoke
git switch agent/ascend-qwen3-vl-smoke
git pull --ff-only
bash scripts/run_qwen3_vl_ascend_experiment.sh
```

前三条 Git 命令只在首次进入或更新实验分支时需要。之后重复测试只需执行最后一条命令。脚本不会自动切换分支或修改 tracked 文件。

测试图有三种来源，优先级如下：

1. 手动上传默认图片：

   ```bash
   mkdir -p /home/momentseek-29154/vlm-exp/input
   # 在本地执行：
   scp test.jpg root@100.199.4.24:/home/momentseek-29154/vlm-exp/input/test.jpg
   ```

2. 指定服务器上的图片：

   ```bash
   IMAGE_HOST=/absolute/path/to/test.jpg \
     bash scripts/run_qwen3_vl_ascend_smoke.sh
   ```

3. 从 runtime 视频抽帧。默认自动选择 `runtime/uploads` 第一条视频，也可以明确指定视频和时间点：

   ```bash
   VIDEO_HOST=/home/momentseek-29154/runtime/uploads/example.mp4 \
   FRAME_TIMESTAMP=25 \
     bash scripts/run_qwen3_vl_ascend_smoke.sh
   ```

抽出的图片写入实验目录，不修改 runtime 中的原视频。首次自动抽帧后，后续执行会复用生成的 `test.jpg`；要换帧时删除该测试图，或指定新的 `IMAGE_HOST`。

首次执行会在 `/home/momentseek-29154/vlm-exp/venv` 安装独立的 Transformers 4.57+、ModelScope 等依赖，并在需要时下载模型。它复用基础镜像中的 Torch/torch-npu，不更改正式容器。后续执行复用 venv 和模型文件。

依赖安装和 ModelScope 下载容器使用宿主网络，以复用共享服务器已验证的 DNS 和软件源连通性；不开放新的监听端口。

结果写入：

```text
/home/momentseek-29154/vlm-exp/output/<model>-<timestamp>.json
/home/momentseek-29154/vlm-exp/output/<model>-<timestamp>.log
```

常用覆盖参数：

```bash
PHYSICAL_NPU=7 \
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
- 运行结束后 `npu-smi info -t proc-mem -i 7 -c 0` 不残留实验进程。

如果出现兼容性错误，保留完整 log，并记录基础镜像、`pip freeze`、`npu-smi info`。不要直接升级宿主 CANN、驱动或正式平台容器的 Python 包。
