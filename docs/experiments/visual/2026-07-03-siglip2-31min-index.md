# SigLIP2 31min Visual 索引耗时

原始实验日期：2026-07-02
整理到当前文档体系：2026-07-03

## 目的

测量当前 SigLIP2 visual 模型在 31 分钟 1080p 视频上的索引耗时，配置为 5fps 抽帧和 5s bucket。

## 环境

```text
服务器：110.126.0.52
容器：momentseek-current-app
设备：容器 npu:0 = 宿主机物理 NPU 2
Visual model：siglip2-so400m-384
```

## 输入

```text
视频：五哈团美食速度挑战纯享_31min_1080p.mp4
时长：1873.44s
分辨率：1920x1080
抽帧：5.0fps
分段：5.0s
Batch size：32
Decode height：256
```

服务器输出路径：

```text
/app/runtime/bench/visual_siglip2_31min_20260702-084826
```

## 指标

| 指标 | 值 |
|---|---:|
| Model load | 13.018s |
| Indexing excluding model load | 298.394s |
| Total including model load | 311.413s |
| Frames indexed | 9367 |
| Segments indexed | 375 |

## 备注

- 实验写入单独 bench 目录。
- 没有覆盖生产索引或数据库行。
- 子进程退出后 NPU 显存回到 baseline。
- Hugging Face 缓存模型会解析到本地 snapshot 路径，避免对已缓存 SigLIP2/ChineseCLIP 反复做在线 metadata 检查。

## 建议

这个结果可以作为当前 SigLIP2 visual 索引 31 分钟 1080p 视频、5fps 设置下的粗略成本基线。后续修改 decode height、batch size、预处理路径或模型生命周期时，优先和这个结果对比。
