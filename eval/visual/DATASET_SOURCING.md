# Visual evaluation dataset sourcing plan

我们把 visual 评估集拆成两部分：

```text
1. image_retrieval
   测 CLIP 对单张图片/单帧的图文匹配能力。

2. sequence_retrieval
   测我们如何把一段图片序列/视频片段表达成可检索对象。
```

这两个任务必须分开看。否则如果检索失败，我们不知道是：

- CLIP 单张图就看不懂；
- 预处理把目标裁掉/压小了；
- 抽帧没抽到关键瞬间；
- segment 平均把局部信号冲淡了；
- 时间片段合并策略有问题。

## 开源数据集复用策略

开源数据集适合做“标准能力对比”，但不完全覆盖我们的产品问题，尤其是：

- 真实 4K/1080p 与低分辨率的同内容对比；
- 小物体、边缘目标、字幕/屏幕文字；
- center crop、padding、tile 这类预处理策略对召回的影响；
- 片段级返回是否精准。

因此建议采用：

```text
开源标准集小样本
  + 自建分辨率/裁剪压力集
```

### 第一优先级：MSR-VTT，用于 image/text-video sanity check

用途：

- 快速比较不同 CLIP/SigLIP 模型；
- 快速比较 center crop / padding / tile 等预处理；
- 主要做视频级或抽帧后的 image-level 检索 sanity check。

优点：

- 常见、轻量、社区 baseline 多；
- caption 多，适合快速检验文本-视觉匹配。

局限：

- 不是专门做片段定位；
- 不专门测 4K、小物体、边缘裁剪。

### 第二优先级：QVHighlights，用于 moment/sequence retrieval

用途：

- 文本查询找视频中相关 moment；
- 更接近 MomentSeek 的“返回时间片段”目标。

优点：

- 有 query 和相关片段时间标注；
- 可以直接评估 Recall@K、MRR、时间 overlap/IoU。

局限：

- 视频来源和可下载性可能受 YouTube 状态影响；
- 分辨率不一定稳定。

### 第三优先级：ActivityNet Captions，用于长视频多事件

用途：

- 评估长视频中多个事件片段的检索；
- 测 segment 长度、shot/segment 合并策略。

优点：

- 每个视频有多个 temporally annotated sentence descriptions；
- 适合 sequence-level 评估。

局限：

- 数据更大；
- 视频源可用性、下载速度、授权要单独处理。

### 后续：DiDeMo / Charades-STA / ActivityNet-Entities

- DiDeMo：适合 moment retrieval，但视频数据较大；
- Charades-STA：适合室内动作/行为定位；
- ActivityNet-Entities：适合后续做 object index / object crop，因为它包含实体和框标注。

## 我们当前初版怎么做

初版不追求大，而追求“能暴露关键问题”。建议：

```text
自有视频：
  8-15 个高质量视频

开源小样本：
  MSR-VTT 200-1000 个样本
  QVHighlights 100-300 个样本

标注：
  每个自有视频 10-20 条 query
  每条 query 至少 1 个 positive 时间段
```

## 数据不要放 GitHub

GitHub 里只放：

- manifest schema；
- query/label 模板；
- 下载/转换脚本；
- 评估脚本；
- 小型示例文本。

不要放：

- 原始视频；
- 大量抽帧图片；
- 商业/版权素材；
- 本机绝对路径 manifest。

本机生成的文件请命名为：

```text
*.local.json
*.local.jsonl
```

这些已在 `.gitignore` 中忽略。

