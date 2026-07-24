# VLM 精排第一阶段测试集

目标是从 MomentSeek 本地真实召回结果构造“小而难”的 Top-20 候选池，验证 VLM 能否把每条 query 的少数强相关镜头推到 Top-1/Top-3。

首批 seed 包含 20 条复杂查询：10 条 `visual_only`，10 条 `evidence_fusion`。每条查询包含至少三个约束，并划分为 `tune/dev/holdout`。Seed 中的 `expected_video` 只用于人工核查候选池覆盖情况，不作为相关性标签，也不应提供给 reranker。

`visual_only` 使用 Visual Top-K。`evidence_fusion` 分别召回 Visual Top-K 和 ASR Top-K，再对同视频、时间重叠的候选去重合并。候选保留各通道独立的来源排名；不会在候选生成阶段用跨通道加权分数提前淘汰正例。

## 1. 启动本地平台

```powershell
./scripts/start_backend.ps1
```

确认 `http://127.0.0.1:8000/docs` 可访问，并确保目标视频已经完成所需通道的索引。合成 Evidence 查询依赖 Visual 和 ASR；缺少对应索引时 API 会明确报错，应先在平台补索引，不能静默降级成纯视觉实验。

## 2. 采集 Top-20 与有序帧

```powershell
python scripts/build_vlm_rerank_phase1.py collect --base-url http://127.0.0.1:18301
```

默认输出到 `runtime/eval/vlm_rerank_phase1/`。调试不同候选池策略时应通过 `--output` 写入新目录，避免混用帧：

- `manifest.json`：采集参数；
- `candidates/candidate_sets.jsonl`：query、原始排序、Evidence 与候选元数据；
- `frames/<query_id>/`：每个候选的四张有序帧；
- `annotations.jsonl`：待人工填写的逐候选标注。

要采集 Top-30：

```powershell
python scripts/build_vlm_rerank_phase1.py collect --base-url http://127.0.0.1:18301 --limit 30
```

## 3. 人工标注

启动本地标注网页：

```powershell
python scripts/vlm_rerank_annotation_server.py
```

浏览器会打开 `http://127.0.0.1:18765/`。网页支持 Query/模式/完成状态筛选、四级相关度、逐条查询约束判断、备注、键盘快捷键和自动写回。快捷键：`0`--`3` 设置相关度，左右方向键切换候选，`S` 保存并进入下一个候选。

网页直接原子更新 `runtime/eval/vlm_rerank_phase1_channel_union/annotations.jsonl`，不要同时开启两个标注服务编辑同一文件。

填写 `annotations.jsonl`：

- `relevance=3`：完整满足查询；
- `relevance=2`：大部分满足，但缺少一个关键约束或时序不完整；
- `relevance=1`：只满足少数条件，是困难负例；
- `relevance=0`：不相关；
- `constraint_labels`：只对确实适用的维度填写 `true/false`，不适用项保留 `null`；
- `reason`：简短记录判定依据。

每条 query 目标是保留 1--3 个三级相关候选和若干一/二级困难候选。如果 Top-20 没有三级相关候选，应记录为“召回未覆盖”，不能把精排失败与召回失败混为一谈。

检查文件完整性和标注取值：

```powershell
python scripts/build_vlm_rerank_phase1.py validate
```

## 4. 两种输入视图

- `visual_only`：直接使用候选的 `model_input`，其中只有 query 和 `frame_paths`。
- `evidence_fusion`：直接使用候选的 `model_input`，其中额外包含经过清洗的 ASR/OCR/Face/Visual evidence。清洗会移除召回分数、阈值和决策字段，避免排序泄漏。

所有模型必须使用同一候选池、相同帧数和相同视觉 token 上限。原始 `rank`/`baseline_score` 仅用于基线评估，不提供给模型，避免排序泄漏。

第一轮完成后再扩展到 40 条查询和 Top-30；不要在人工确认本批数据前批量生成更多近似 query。

## 已标注视觉 Smoke v1

`smoke_v1/` 保存前六条纯视觉 Query 的完整人工标注与候选元数据：120 个候选、480 张有序帧。图片集中存放在 `vlm_rerank_visual_smoke_v1.tar.gz` 中，避免 Git 同时维护大量松散二进制文件。

在服务器解压：

```bash
cd /home/momentseek-29154/platform/eval/vlm_rerank_phase1/smoke_v1
sha256sum -c vlm_rerank_visual_smoke_v1.tar.gz.sha256
mkdir -p /home/momentseek-29154/vlm-exp/eval
tar -xzf vlm_rerank_visual_smoke_v1.tar.gz \
  -C /home/momentseek-29154/vlm-exp/eval
```

校验值：

```text
971bc6af3c776d2add41c079adc6a7d764ce2e8417211ad0c78e1919211119b2
```
