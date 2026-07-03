> Archived reference. Current documentation starts at `docs/README.md`.

# Visual CLIP 910B 评测记录

更新时间：2026-07-01
目的：评估 MomentSeek visual 索引候选方案在昇腾 910B 上的检索效果与处理速度，重点比较不同 CLIP 模型、center crop / padding / sliding window 等预处理方案。

## 1. 实验环境

- 服务器：`drama-server` / `cluster-worker-poeub`
- 容器：`momentseek-current-app`
- NPU 映射：容器内 `npu:0` = 宿主机物理 2 号卡，来自 `ASCEND_VISIBLE_DEVICES=2`
- NPU 型号：Ascend 910B3
- Python / 主要包：
  - Python 3.11.14
  - torch 2.9.0+cpu
  - torch_npu 2.9.0.post1
  - open_clip_torch 3.3.0
- 评测 batch：
  - image/video embedding batch size = 128
  - text batch size = 256
- 速度数据口径：只使用服务器 910B 实测；本地 RTX 3060 速度结果不作为结论。

## 2. 数据集与输出位置

评测集来自 `C:\Users\29154\Videos\视频检索测试` 下的视频抽样，并用 Qwen 自动标注构建初版 visual eval set。

- Image retrieval：
  - 300 张平衡抽样图片
  - 每张最多 3 条 query
  - 实际 query 数：889
- Sequence retrieval：
  - 200 个 5s 片段 contact sheet
  - 每段约 2fps 抽帧生成高清 contact sheet
  - 实际 query 数：1198

本地结果：

```text
eval/visual/outputs/visual_clip_910b_report.md
eval/visual/outputs/visual_clip_910b_effect_summary.csv
eval/visual/outputs/visual_clip_910b_speed_summary.csv
eval/visual/outputs/visual_clip_910b_best_summary.csv
eval/visual/outputs/charts_910b/
```

服务器结果：

```text
/mnt/mog2/wyl/comfyui-wxy/momentseek-current/app/runtime/momentseek-eval-910b/eval/visual/outputs/
```

评测脚本：

```text
scripts/visual_clip_eval.py
scripts/visual_clip_speed_benchmark.py
scripts/visual_clip_910b_report.py
```

## 3. 对比模型

| 模型 | 权重位置 | 说明 |
| --- | --- | --- |
| ViT-B-32 | `/app/models/ViT-B-32.openai.bin` | 当前 MVP baseline |
| ViT-B-16 | `/app/models/ViT-B-16.openai.safetensors` | patch 更细，效果通常优于 B32 |
| ViT-L-14 | `/app/models/ViT-L-14.openai.safetensors` | 更大模型，本轮效果最好 |

## 4. 对比预处理策略

Image/frame 侧：

- `center_crop`：OpenCLIP 默认 resize + center crop。
- `letterbox` / padding：保持原始长宽比缩放到 CLIP 输入方图内，用均值背景 padding，不拉伸画面。
- `sliding_*`：沿长边做多个方形滑窗 crop，再分别送入 CLIP。索引编码阶段 `sliding_max` / `sliding_top3` / `sliding_mvp_mix` 的计算量基本一致，区别主要在检索聚合/打分方式。

Sequence 侧：

- `sheet_whole_*`：把 5s contact sheet 作为一张图送入 CLIP。
- `cells_*`：把 contact sheet 拆回多张 sampled frames/cells 后编码并聚合。
- `cells_sliding_*`：每个 cell 再做 spatial sliding crop 后编码并聚合。

## 5. 效果结果摘要

完整指标见 `visual_clip_910b_effect_summary.csv`。这里列每个模型/任务的最佳策略。

| 模型 | 任务 | 最佳策略 | R@1 | R@5 | R@10 | MRR | Median rank |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ViT-B-32 | image | sliding_max_center_crop | 24.3% | 51.4% | 63.1% | 0.376 | 5.0 |
| ViT-B-32 | sequence | cells_sliding_top3_center_crop | 26.5% | 55.8% | 71.8% | 0.406 | 4.0 |
| ViT-B-16 | image | sliding_mvp_mix_center_crop | 28.8% | 53.8% | 64.1% | 0.407 | 4.0 |
| ViT-B-16 | sequence | cells_sliding_top3_center_crop | 30.9% | 65.9% | 78.3% | 0.462 | 3.0 |
| ViT-L-14 | image | sliding_mvp_mix_center_crop | 34.8% | 57.4% | 67.6% | 0.456 | 4.0 |
| ViT-L-14 | sequence | cells_sliding_top3_center_crop | 41.1% | 70.6% | 83.1% | 0.543 | 2.0 |

阶段性判断：

- ViT-L-14 效果最好，尤其 sequence 检索提升明显。
- ViT-B-16 相比当前 B32 baseline 效果有稳定提升，是一个性价比较高的候选。
- sliding window 对 image 和 sequence 都有帮助，说明单纯把整帧或整张 contact sheet 平均表达会丢失局部目标。

## 6. Image/frame 速度：预处理 vs encoder

用户当前关注 image/frame 索引耗时，因此本节不考虑 sequence。

换算假设：

```text
1 小时视频 × 3 fps = 10800 帧
```

口径：

- “预处理”包含：读图/裁剪或滑窗、CLIP CPU preprocess、CPU/NPU 传输、回 CPU 等非 encoder 部分。
- “encoder”只统计 NPU 上 CLIP image encoder 前向。
- 不包含：视频解码抽帧、数据库写入、检索时相似度打分。

| 模型 | 方案 | 1帧预处理 | 1帧encoder | 1帧合计 | 10800帧预处理 | 10800帧encoder | 10800帧合计 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B-32 | center_crop | 0.0731s | 0.0006s | 0.0737s | 13.16min | 0.11min | 13.27min |
| ViT-B-32 | letterbox/padding | 0.0706s | 0.0006s | 0.0712s | 12.70min | 0.11min | 12.81min |
| ViT-B-32 | sliding window | 0.1282s | 0.0022s | 0.1304s | 23.08min | 0.39min | 23.47min |
| ViT-B-16 | center_crop | 0.0708s | 0.0009s | 0.0718s | 12.75min | 0.16min | 12.92min |
| ViT-B-16 | letterbox/padding | 0.0716s | 0.0009s | 0.0725s | 12.89min | 0.16min | 13.06min |
| ViT-B-16 | sliding window | 0.1970s | 0.0034s | 0.2004s | 35.46min | 0.61min | 36.08min |
| ViT-L-14 | center_crop | 0.0726s | 0.0042s | 0.0768s | 13.07min | 0.75min | 13.82min |
| ViT-L-14 | letterbox/padding | 0.0739s | 0.0042s | 0.0781s | 13.31min | 0.75min | 14.06min |
| ViT-L-14 | sliding window | 0.1450s | 0.0152s | 0.1602s | 26.10min | 2.73min | 28.83min |

## 7. 速度结论

1. 当前 image/frame 索引瓶颈主要在预处理，不在 NPU encoder。
   - 例如 ViT-L-14 + sliding window：10800 帧里 encoder 约 2.73min，预处理约 26.10min。
   - 例如 ViT-B-32 + sliding window：encoder 约 0.39min，预处理约 23.08min。

2. 普通 `center_crop` / `letterbox` 方案整体非常接近。
   - 1 小时视频 3fps 抽帧约 13–14min。
   - 这说明如果只做单视图 CLIP，模型大小对总耗时影响不大，因为 CPU 预处理占主导。

3. `sliding window` 明显增加总耗时，但带来更好的局部目标召回。
   - B32 sliding：约 23.47min / 10800 帧。
   - B16 sliding：约 36.08min / 10800 帧。
   - L14 sliding：约 28.83min / 10800 帧。

4. 910B encoder 吞吐不是主要矛盾。
   - B32/B16 单视图 encoder 对 10800 帧只需约 0.1–0.2min。
   - L14 单视图 encoder 约 0.75min。
   - 即使用 sliding，encoder 时间也远小于预处理时间。

## 8. 当前建议

短期可选方案：

- 如果优先 demo 速度：`ViT-B-32` 或 `ViT-B-16` + `letterbox/padding`，一小时 3fps 抽帧约 13min。
- 如果优先 image 检索效果：`ViT-L-14` + `sliding_mvp_mix`，一小时 3fps 抽帧约 29min。
- 如果要平衡效果和耗时：优先继续比较 `ViT-B-16 + sliding_mvp_mix` 与 `ViT-L-14 + letterbox/padding` 在真实查询上的体验。

下一步优化重点：

1. 把预处理从 PIL/CPU 进一步优化到更高效路径，例如 cv2 批处理、解码即缩放、NPU/GPU 侧 resize/crop。
2. 对 sliding window 做自适应触发：不是每帧都滑窗，只对宽高比极端、目标较小或粗排不确定的帧启用。
3. 在索引结构中区分“全帧 embedding”和“局部 crop embedding”，避免所有查询都承担 sliding 成本。
4. 把视频解码抽帧、数据库写入也纳入下一轮端到端耗时统计；本次表格只衡量已经抽出的 image/frame 进入 CLIP 这段。
