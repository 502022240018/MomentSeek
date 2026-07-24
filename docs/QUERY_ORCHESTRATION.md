# Query Planner 与 Reranker 编排

## 目标

编排层位于 `/api/search` 与既有 `SearchEngine` 之间。`SearchEngine` 仍是确定性的
visual / face / asr / ocr 召回与融合实现；编排层只负责：

1. Planner 根据 query、请求允许的通道和实际已建索引，选择检索通路与参数。
2. 调用第一阶段检索。
3. 按计划决定是否对 Top-N 结果做 text 或 multimodal rerank。
4. 返回并持久化完整 execution trace，供 query 最佳实践分析和后续 skill 生成。

Planner 与 Reranker 没有共享模型假设。一个 profile 可以把两者指向同一个
Qwen3.5-4B，也可以分别指向小型 Planner、专用 VLM Reranker 或其他
OpenAI-compatible 服务。

## 配置

默认 registry：

```text
deploy/orchestration/qwen35-vllm.json
```

结构：

```json
{
  "providers": {
    "planner-provider": {
      "type": "openai_compatible",
      "base_url": "http://127.0.0.1:18081/v1",
      "model": "qwen3.5-4b"
    }
  },
  "profiles": {
    "experiment-name": {
      "planner": {
        "provider": "planner-provider",
        "prompt_path": "prompts/planner-v1.txt",
        "prompt_version": "planner-v1"
      },
      "reranker": {
        "provider": "reranker-provider",
        "prompt_path": "prompts/reranker-v1.txt",
        "prompt_version": "reranker-v1"
      }
    }
  }
}
```

`base_url_env`、`model_env` 和 `api_key_env` 可以让部署环境覆盖 endpoint、模型名和
密钥，而不修改已版本化的 profile。任何新模型实验应新增 profile 或 provider，不要覆盖
已有 prompt version。

当前 Qwen3.5 profile 使用：

```text
QWEN35_VLLM_BASE_URL
QWEN35_PLANNER_MODEL
QWEN35_RERANKER_MODEL
```

即使当前两个模型名相同，也保留两个独立变量。

## 启用与降级

默认关闭，避免没有 vLLM 服务时改变既有检索：

```text
ORCHESTRATION_ENABLED=true
ORCHESTRATION_CONFIG_PATH=deploy/orchestration/qwen35-vllm.json
ORCHESTRATION_PROFILE=qwen35-unified
ORCHESTRATION_FAIL_OPEN=true
```

`ORCHESTRATION_FAIL_OPEN=true` 时：

- Planner 失败：使用请求中的 modalities / alpha / limit 做确定性召回。
- 某个精排候选失败：保留该候选原始召回分数。
- Reranker 整体失败：保留第一阶段顺序。

请求可以分别指定：

```text
planner_mode=auto|off|force
reranker_mode=auto|off|force
orchestration_profile=<profile name>
```

`force` 用于实验和故障发现；生产交互请求通常用 `auto`。

## Planner 输出协议

Planner 必须只输出 JSON。核心字段：

```text
query_intent
modalities
alpha
visual_profile
visual_subqueries
candidate_limit
result_limit
merge_gap
max_result_seconds
channel_limits
rerank.enabled
rerank.strategy
rerank.top_n
rerank.frame_count
rerank.window_seconds
rerank.score_weight
rationale
```

服务端使用 Pydantic 再校验，并强制：

- modalities 只能来自请求允许通道与实际已建索引的交集。
- result_limit 不超过 candidate_limit。
- rerank.top_n 不超过 candidate_limit。
- channel_limits 只能配置 visual / face / asr / ocr，单通道范围为 1--300。
- visual_subqueries 最多保留 4 个去重后的视觉子查询；未选择 visual 时清空。
- text rerank 的 frame_count 固定为 0。
- 9--11 秒时序窗口至少使用 8 帧。

因此模型输出不能绕过请求范围或调用不存在的索引。

组合或时序视觉 query 会被 Planner 拆为 2--4 个英文视觉子查询。Visual 首阶段按每个
子查询在片段内的最佳帧计算覆盖度，并综合平均覆盖与最弱约束分数，降低只匹配人物、
场景或物体名词的静态片段排名。

## Reranker

multimodal rerank 会在结果时间段内均匀采样帧，写入现有 `frame_cache`，并发调用
OpenAI-compatible `/chat/completions`。当前 Qwen3.5 profile 使用 Yes/No token
logprobs 形成连续相关性分数：

时序 query 会在精排前把短候选以中心点扩展为 Planner 指定的 9--11 秒窗口。trace
同时保留原始和扩展后的起止时间，便于判断窗口扩展是否改善动作顺序覆盖。

```text
final_score =
  rerank.score_weight * rerank_score
  + (1 - rerank.score_weight) * retrieval_score
```

响应结果保留：

```text
retrieval_score
rerank_score
original_rank
score
```

默认建议从 Top-20、4 frames、concurrency=4 开始；时间关系强的 query 使用 9--11
秒窗口和至少 8 frames。模型或 prompt 变化后必须重新验证排序质量，不能只比较延迟。

## Execution trace 与 skill 数据

每次查询响应包含 `execution`，同时按 JSONL 追加到：

```text
ORCHESTRATION_TRACE_PATH=/app/runtime/orchestration-traces.jsonl
```

trace schema version 当前为 1，记录：

```text
request_id / query / profile
Planner provider、model、prompt_version、raw_output、validated plan、耗时
实际 retrieval 通道与参数、候选数量、耗时
Reranker provider、model、prompt_version、候选原始分数与精排分数、耗时
最终结果数量与总耗时
```

后续生成 skill 时，应在人工或离线指标确认 query 结果后，再把 trace 标注为正例；
不能把未评审的 Planner 输出直接当成最佳实践。

## vLLM Ascend 运行注意事项

Qwen3.5 vLLM 服务必须常驻并在开放流量前预热：

1. 一次关闭 thinking 的 Planner JSON 请求。
2. 一次目标 frame_count 的单候选精排。
3. 一次目标 concurrency 的并发精排。

当前实测编译模式首次 Planner 和首批多图请求存在明显按需编译成本。健康检查只能证明
HTTP server ready，不能代替上述形状预热。
