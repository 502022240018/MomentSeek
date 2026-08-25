# Planner 优化交接文档（0829 环境）

> 目标：能完整重建 vLLM 服务与 Planner Lab 容器，理解 Planner 全部相关代码、各版本 prompt 的差异，以及本次"方案 D（Token-Aware Adaptive Planning）"延迟优化的原理与部署要求。
>
> 配套文档：`docs/Planner_Latency_Optimization_Record.md`（优化记录以及延迟/Token 实测分解）、`docs/PLANNER_LAB.md`（Planner Lab 功能界面）。

---

## 0. 注意（最容易踩的坑）

- **使用 `ORCHESTRATION_PROFILE=qwen35-adaptive`**，最新的延迟优化才会生效。
- vLLM 侧必须 `--enable-prefix-caching`，当前端口 **18084**，NPU **4**（容器 `qwen35-v023-0829`，镜像 `vllm-ascend:v0.23.0rc1-openeuler`）。

---

## 1. Planner 相关代码位置与作用

| 路径 | 作用 |
|------|------|
| `backend/app/orchestration/snapmind_lab.py` | **核心文件**。Planner 主逻辑：`propose()` 调 LLM 生成 balanced 方案 + optimization_hints；`_derive_fast_plan()` / `_derive_deep_plan()` 在 Python 侧派生 fast/deep（0 LLM token）；`_sanitize_plan_set()` 校验；`HeuristicPlanGenerator` 兜底；`CAPABILITIES`（7 项，含 `voice.search`）；数据模型 `OptimizationHints` / `PlanSet` / `VoiceReference`。 |
| `backend/app/orchestration/retrieval_orchestration.py` | 编排执行层：读取 profile 配置、组织 LLM 调用、执行检索计划。 |
| `backend/app/orchestration/__init__.py` | 包导出。 |
| `backend/app/api/planner_lab_routes.py` | Planner Lab 的 HTTP 路由：`/api/planner-lab/capabilities`、propose 接口等。 |
| `deploy/orchestration/qwen35-vllm.json` | **Profile 定义**。每个 profile 指定 planner 的 `prompt_path` / `max_tokens` / `temperature` 及 reranker 配置。 |
| `deploy/orchestration/prompts/*.txt` | 各版本 prompt（见第 2 节）。 |
| `docker/Dockerfile.planner-lab-overlay` | Planner Lab overlay 镜像构建（在 platform 基础镜像上替换应用代码）。 |
| `scripts/test_planner_optimize_0829.sh` | 优化验证测试脚本（capabilities=7、prefix cache 增量、hints_guided 记录等）。 |
| `backend/tests/test_planner_d_optimization.py` | 方案 D 派生逻辑单测。 |
| `backend/tests/test_snapmind_planner_lab.py` | Planner Lab 单测。 |

### 1.1 三个 profile（来自 `qwen35-vllm.json`）

| profile | prompt | max_tokens | 说明 |
|---------|--------|-----------|------|
| `qwen35-adaptive` | `snapmind-planner-v2-adaptive.txt` | **800** | **方案 D，本次优化使用的 profile** |
| `qwen35-unified` | `snapmind-planner-v2-role-aware.txt` | 1200 | 旧版：LLM 直接产出 fast/balanced/deep 三方案 |
| `qwen35-temporal-efficient` | `snapmind-planner-v2-role-aware.txt` | 1200 | 旧版变体 |

---

## 2. Prompt 版本详解（`deploy/orchestration/prompts/`）

### 2.1 所有 prompt 文件

| 文件名 | 版本/用途 | 活跃状态 |
|--------|----------|---------|
| `snapmind-planner-v2-adaptive.txt` | **v2-adaptive（方案 D）** | ✅ **当前使用**（profile: `qwen35-adaptive`） |
| `snapmind-planner-v2-role-aware.txt` | v2-role-aware（原版） | ✅ 活跃（profile: `qwen35-unified` / `qwen35-temporal-efficient`） |
| `snapmind-planner-v2-role-aware.txt.backup` | 同事合并前的备份 | 归档（8月20日） |
| `snapmind-planner-v1.txt` | v1 | 历史版本 |
| `planner-v1.txt` | v1 变体 | 历史版本 |
| `planner-v2-temporal.txt` | v2 时序变体 | 历史版本 |
| `planner-v3-temporal-efficient.txt` | v3 时序优化 | 历史版本 |
| `reranker-v1.txt` | Reranker prompt | ✅ 活跃 |

### 2.2 **核心差异：`snapmind-planner-v2-adaptive.txt` vs `snapmind-planner-v2-role-aware.txt`**

这是**本次优化的核心**：从"LLM 产出三方案"切换到"LLM 产出 hints + balanced，后端派生 fast/deep"。

#### 2.2.1 **role-aware（旧版，38 行，max_tokens=1200）**

**输出结构**：
```json
{
  "query_intent": "...",
  "constraints": [...],
  "negative_constraints": [...],
  "identity_mentions": [...],
  "plans": [
    {"plan_id": "fast", "steps": [...]},      // LLM 完整生成
    {"plan_id": "balanced", "steps": [...]},  // LLM 完整生成
    {"plan_id": "deep", "steps": [...]}       // LLM 完整生成
  ]
}
```

**特点**：
- LLM 直接产出 **3 个完整的检索方案**（fast/balanced/deep）。
- 每个方案的 steps、query、top_k、weight 全部由 LLM 生成。
- 平均输出 **~570 tokens**（三方案约占 69%）。
- 明确规定 `"Use plan IDs exactly fast, balanced, deep"`。
- 包含 `voice.search` 规则（新增的 capability）。

**适用场景**：需要 LLM 完全控制三个方案的差异化策略时。

---

#### 2.2.2 **adaptive（新版，152 行，max_tokens=800）**

**输出结构（TWO-STAGE）**：
```json
{
  "optimization_hints": {                    // 新增！派生指导
    "fast_strategy": "primary_only | keep_asr_support | keep_all_support",
    "fast_top_k_ratio": 0.4,                 // fast 模式 top_k 缩减比例
    "deep_needs_rerank": true,               // deep 模式是否加 vlm.rerank
    "deep_extra_modality": ["ocr"],          // deep 模式新增的模态
    "deep_enhance_primary": false,           // deep 模式是否扩大 primary top_k
    "query_complexity": "simple | moderate | complex",
    "rationale": "单一视觉"                   // 80 字符内
  },
  "query_intent": "...",
  "constraints": [...],
  "negative_constraints": [...],
  "identity_mentions": [...],
  "plans": [
    {"plan_id": "balanced", "steps": [...]}  // 仅 1 个方案！
  ]
}
```

**派生逻辑（Python 侧，`snapmind_lab.py`）**：
- **Fast 派生**（`_derive_fast_plan()`）：
  - 根据 `fast_strategy` 过滤 steps（保留 primary、可选保留 ASR support、可选保留全部 support）。
  - 按 `fast_top_k_ratio` 缩减各 step 的 `top_k`。
  - `result_limit` 减半。
- **Deep 派生**（`_derive_deep_plan()`）：
  - 按 `deep_enhance_primary=true` 扩大 primary step 的 `top_k` 至 1.5x。
  - 按 `deep_extra_modality` 列表增补 support steps（如 ["ocr"] → 插入 ocr.search support step）。
  - 按 `deep_needs_rerank=true` 追加 vlm.rerank verifier step（`step_id: "rerank"`，`top_k: 20`）。
  - `result_limit` 不变或略增。

**特点**：
- LLM 只生成 **1 个 balanced 方案 + hints**，平均输出 **~410 tokens**（balanced 约占 69%，hints 约占 23%，envelope 约占 9%）。
- **Fast/deep 派生耗时 <1ms**（纯 Python dict 操作）→ **0 LLM tokens**。
- decode 时间为 **~10s**（LLM 产出降至 410 tokens  decode 占延迟 97.7%）。
- **总延迟 ~10.25s**，相比旧版 **节约 ~160 输出 tokens**。
- **优势**：可快速迭代派生策略（修改 Python 代码）而无需调整 prompt / 重新微调 LLM。
- Prefix caching 优化：将 `capability_registry` 移至 system prompt（静态内容），用户 context 从 431 字符降至 93 字符，prefix cache hit 从 0 → ~270 tokens。

**COMPACT OUTPUT RULES**：
- 省略空字段（`depends_on: []`、`operation: "search"`、`target_id: "main"`）。
- 短 ID（`s1`/`s2`/`s3`）。
- 文本限长（description <40 字符，rationale <50 字符，hints rationale <80 字符）。

---

### 2.3 选择指南

| 需求 | 推荐 profile | prompt |
|------|-------------|--------|
| 生产环境（方案 D 优化） | `qwen35-adaptive` | `snapmind-planner-v2-adaptive.txt` |
| LLM 完全控制三方案差异 | `qwen35-unified` | `snapmind-planner-v2-role-aware.txt` |
| 兼容旧版测试/对比 | `qwen35-unified` | `snapmind-planner-v2-role-aware.txt` |

---

## 3. 方案 D（Token-Aware Adaptive Planning）原理

### 3.1 问题背景

**旧版瓶颈**（role-aware）：
- LLM 产出 3 个完整方案（~570 tokens），fast/deep 方案大量逻辑可从 balanced 机械派生（如 fast = balanced - support steps，deep = balanced + rerank）。
- 输出 token 多 → decode 时间长 → 总延迟高。
- Prefix caching 收益低（用户 context 包含 `capability_registry`，431 字符，每次查询都变）。

### 3.2 方案 D 设计

**核心思想**：LLM 只做"需要语义理解的决策"，机械派生交给 Python。

**两阶段输出**：
1. **Stage 1: Query Analysis**（`optimization_hints`）
   - LLM 分析查询复杂度（simple/moderate/complex）。
   - 决策 fast 模式策略（primary_only / keep_asr_support / keep_all_support）、top_k 缩减比例。
   - 决策 deep 模式是否需要 rerank、新增模态、扩大 primary top_k。
2. **Stage 2: Balanced Plan**（`plans`）
   - LLM 产出 **唯一 1 个** balanced 方案（2-4 steps）。
   - Fast/deep 方案由后端 `_derive_*_plan()` 根据 hints 自动派生。

### 3.3 派生算法伪代码

```python
# Fast 派生（snapmind_lab.py:1262-1325）
def _derive_fast_plan(balanced, hints):
    steps = []
    if hints.fast_strategy == "primary_only":
        steps = [s for s in balanced.steps if s.role == "primary"]
    elif hints.fast_strategy == "keep_asr_support":
        steps = [s for s in balanced.steps 
                 if s.role in ("primary", "constraint") or 
                    (s.role == "support" and "asr" in s.tool_id)]
    else:  # keep_all_support
        steps = [s for s in balanced.steps 
                 if s.role in ("primary", "support", "constraint")]
    
    for step in steps:
        step.top_k = int(step.top_k * hints.fast_top_k_ratio)
    
    return Plan(
        plan_id="fast",
        result_limit=balanced.result_limit // 2,
        steps=steps
    )

# Deep 派生（snapmind_lab.py:1327-1450）
def _derive_deep_plan(balanced, hints, has_query_image):
    steps = copy.deepcopy(balanced.steps)
    
    # 扩大 primary top_k
    if hints.deep_enhance_primary:
        for s in steps:
            if s.role == "primary":
                s.top_k = int(s.top_k * 1.5)
    
    # 新增模态 support steps
    for modality in hints.deep_extra_modality:
        if modality not in [s.tool_id for s in steps]:
            steps.append(make_support_step(modality, query=...))
    
    # 追加 vlm.rerank
    if hints.deep_needs_rerank and has_query_image:
        steps.append({
            "step_id": "rerank",
            "role": "verifier",
            "tool_id": "vlm.rerank",
            "operation": "rerank",
            "top_k": 20,
            "depends_on": [s.step_id for s in steps if s.role == "primary"]
        })
    
    return Plan(plan_id="deep", steps=steps)
```
---

## 4. vLLM 服务部署（Qwen3.5-4B + v0.23.0rc1）

### 4.1 部署信息

**镜像**：`m.daocloud.io/quay.io/ascend/vllm-ascend:v0.23.0rc1-openeuler`
**模型路径**：`/home/momentseek-29154/vlm-exp/models/Qwen3.5-4B`（只读挂载至容器 `/model`）  
**缓存路径**：`/home/momentseek-29154/experiments/cache/vllm-qwen35-v023`（挂载至容器 `/root/.cache`）

### 4.2 vLLM 启动参数（关键）

```bash
vllm serve /model \
  --host 0.0.0.0 \
  --port 18084 \
  --trust-remote-code \
  --dtype bfloat16 \                    # 910B4 必须使用 bfloat16
  --max-model-len 4096 \
  --served-model-name qwen3.5-4b \      # API 调用时使用此模型名
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 32 \
  --enable-prefix-caching \             # ⚠️ 必须启用！方案 D 的 Phase 1 优化依赖此特性
  --mamba-ssm-cache-dtype bfloat16      # Qwen3.5 是 Mamba hybrid 架构
```

**关键点**：
- `--enable-prefix-caching`：**必须启用**。若缺失，Phase 1 优化（capability_registry 移至 system prompt）失效，prefix cache hit = 0。
- `--served-model-name qwen3.5-4b`：后端配置（`qwen35-vllm.json`）中的 `planner.provider` 和 `reranker.provider` 引用此名称。
- `--dtype bfloat16`：910B4 NPU 限制，不可改为 float16。
- `--mamba-ssm-cache-dtype bfloat16`：Qwen3.5-4B 使用 Mamba 混合架构，需显式指定 SSM cache dtype。

---


## 5. 完整环境配置

### 5.1 示例

```bash
# MomentSeek 0829 Environment Configuration
# 此环境与29154环境完全隔离

# ════════════════════════════════════════════════════════════════════
# Docker Compose Configuration
# ════════════════════════════════════════════════════════════════════
COMPOSE_PROJECT_NAME=momentseek-0829
MOMENTSEEK_NETWORK_NAME=momentseek-0829-net

# ════════════════════════════════════════════════════════════════════
# Application Ports (避开29154环境)
# ════════════════════════════════════════════════════════════════════
APP_PORT=8100

# ════════════════════════════════════════════════════════════════════
# Milvus Configuration
# ════════════════════════════════════════════════════════════════════
MILVUS_ENABLED=true
MILVUS_HOST=localhost
MILVUS_PORT=19531
MILVUS_GRPC_PORT=19531
MILVUS_HEALTH_PORT=9092
MINIO_CONSOLE_PORT=9002

# Milvus resource limits
MILVUS_CPUS=4
MILVUS_MEMORY_LIMIT=4g

# MinIO Configuration (Milvus 依赖的对象存储)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Visual ANN Search Configuration
VISUAL_USE_DISKANN=true
VISUAL_ANN_TOP_K=500
VISUAL_ANN_SEGMENT_TOP_N=3

# ════════════════════════════════════════════════════════════════════
# NPU Configuration (使用NPU 0，空闲设备)
# 注意：NPU 1-3 已被 29154 环境占用，NPU 5 也被占用
# ════════════════════════════════════════════════════════════════════
NPU_ENABLED=true
HOST_NPU_DEVICE_ID=0
ASCEND_VISIBLE_DEVICES=0
ASCEND_RT_VISIBLE_DEVICES=0

# ════════════════════════════════════════════════════════════════════
# Model Directories (共享29154的模型目录，只读)
# ════════════════════════════════════════════════════════════════════
HOST_MODEL_DIR=/home/momentseek-29154/models/platform

# ════════════════════════════════════════════════════════════════════
# Runtime Directory (0829专用)
# ════════════════════════════════════════════════════════════════════
HOST_RUNTIME_DIR=/home/momentseek_0829_develop/workplace/MomentSeek_planner/runtime

# ════════════════════════════════════════════════════════════════════
# Logging Configuration
# ════════════════════════════════════════════════════════════════════
LOG_LEVEL=INFO

# OCR Hybrid Search Configuration (DiskANN + BM25)
OCR_HYBRID_RECALL_SIZE=200
OCR_LEXICAL_WEIGHT=0.7
OCR_DISKANN_SEARCH_LIST=200

# ASR Hybrid Search Configuration (DiskANN + BM25)
# ASR is semantic-first (longer transcripts, richer semantics) vs OCR's lexical-first.
ASR_HYBRID_RECALL_SIZE=100
ASR_SEMANTIC_WEIGHT=0.65
ASR_DISKANN_SEARCH_LIST=100

# Speaker Retrieval Configuration (DiskANN + COSINE, single-phase, no re-score)
# SPEAKER_IDENTITY_THRESHOLD only drives the above_threshold display flag;
# voice-search passes -1.0 to keep every candidate.
# SPEAKER_DISKANN_SEARCH_LIST is dynamically raised to >= ann_limit at query time.
SPEAKER_IDENTITY_THRESHOLD=0.50
SPEAKER_DISKANN_SEARCH_LIST=128
SPEAKER_RECALL_MULTIPLIER=1

# Face Retrieval Configuration (DiskANN + COSINE, single-phase, no re-score)
# FACE_IDENTITY_THRESHOLD only drives the above_threshold/decision display flag;
# the cross-modal fusion score is face_confidence(cosine).
# FACE_DISKANN_SEARCH_LIST is dynamically raised to >= ann_limit at query time.
FACE_IDENTITY_THRESHOLD=0.35
FACE_DISKANN_SEARCH_LIST=128
FACE_RECALL_MULTIPLIER=1

# ════════════════════════════════════════════════════════════════════
# Planner Lab Configuration (Experimental - codex/snapmind-planner-lab)
# ════════════════════════════════════════════════════════════════════
# 启用 Planner Lab 功能和界面
PLANNER_LAB_ENABLED=true

# Planner Lab 独立端口和 NPU 配置
PLANNER_LAB_PORT=8101
PLANNER_LAB_NPU_ID=1

# 启用 Qwen3.5 LLM 编排（使用29154的vLLM服务）
ORCHESTRATION_ENABLED=true

# vLLM 连接配置
# 使用 v0.23.0rc1 服务 (NPU 4, 端口 18084, 启用 prefix caching)
# 容器: qwen35-v023-0829
QWEN35_VLLM_BASE_URL=http://127.0.0.1:18084/v1
QWEN35_PLANNER_MODEL=qwen3.5-4b
QWEN35_RERANKER_MODEL=qwen3.5-4b

# 编排框架配置（默认值，通常无需修改）
ORCHESTRATION_CONFIG_PATH=deploy/orchestration/qwen35-vllm.json
ORCHESTRATION_PROFILE=qwen35-adaptive                                          # ⚠️ 方案 D 关键！
ORCHESTRATION_FAIL_OPEN=true
ORCHESTRATION_TRACE_ENABLED=true
ORCHESTRATION_TRACE_PATH=runtime/orchestration-traces.jsonl
PLANNER_LAB_PROMPT_PATH=deploy/orchestration/prompts/snapmind-planner-v2-adaptive.txt
```

### 5.2 关键配置项说明

| 配置项 | 值 | 说明 | 修改风险                                                    |
|--------|----|----|-------------------------------------------------------------|
| `PLANNER_LAB_PORT` | **8101** | Planner Lab 容器端口 | ⚠️ 需同步修改部署脚本                                       |
| `PLANNER_LAB_NPU_ID` | **1** | Planner Lab 使用的宿主机 NPU 设备号 | ⚠️ 需确认 NPU 空闲                                          |
| `QWEN35_VLLM_BASE_URL` | **http://127.0.0.1:18084/v1** | vLLM 服务地址 | 🔴 **必须与实际Qwen容器端口一致**                           |
| `ORCHESTRATION_PROFILE` | **qwen35-adaptive** | 方案 D profile | 🔴 **修改此项会切换到旧版（qwen35-unified）或其他 profile** |
| `PLANNER_LAB_PROMPT_PATH` | **deploy/orchestration/prompts/snapmind-planner-v2-adaptive.txt** | Prompt 文件路径 | ⚠️ 与 profile 配置应对应                                    |
| `HOST_MODEL_DIR` | `/home/momentseek-29154/models/platform` | 模型文件宿主机路径 | ⚠️ 路径不存在会导致容器启动失败                             |
| `MILVUS_PORT` | **19531** | Milvus 向量数据库端口 | ⚠️ 需同步修改 Milvus 启动脚本                               |

---

## 6. 验证测试（`test_planner_optimize_0829.sh`）

### 6.1 测试脚本功能

**文件**：`scripts/test_planner_optimize_0829.sh`  
**测试项**：
1. **Capabilities 数量**：验证 7 个 capabilities（含 `voice.search`）。
2. **方案数量**：验证每个请求返回 3 个方案（fast/balanced/deep）。
3. **Prefix cache 增量**：验证第 2+ 次请求的 `prefix_cache_hit_tokens` > 0（Phase 1 优化）。
4. **Optimization hints**：验证响应包含 `optimization_hints` 字段（方案 D 特征）。
5. **派生标记**：验证 fast/deep 方案包含 `hints_guided: true` 元数据（Python 派生标记）。
6. **端口/URL 正确性**：验证 vLLM URL 为 `http://127.0.0.1:18084/v1`。

### 6.2 运行完整测试

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek_planner


# 运行测试
bash scripts/test_planner_optimize_0829.sh

# 期望输出（最后几行）：
# ========================================
# 测试结果汇总
# ========================================
# 通过: 28
# 失败: 0
# 警告: 0
# 
# ✅ 所有测试通过！
```

### 6.3 测试失败诊断

| 失败项 | 可能原因 | 解决方法 |
|--------|---------|---------|
| Capabilities 数量 ≠ 7 | 代码未包含 `voice.search` | 检查 `CAPABILITIES` 元组（应 7 项），重新构建 overlay 镜像 |
| 所有请求超时 | vLLM 服务未启动或端口错误 | `curl http://localhost:18084/v1/models`，检查 `.env.0829` 中 `QWEN35_VLLM_BASE_URL` |
| Prefix cache hit = 0 | vLLM 未启用 prefix caching | 检查 `start_qwen_v023_0829.sh` 是否包含 `--enable-prefix-caching` |
| 无 `optimization_hints` | Profile 非 `qwen35-adaptive` | 检查 `.env.0829` 中 `ORCHESTRATION_PROFILE` 和容器环境变量 |
| 无 `hints_guided` 标记 | 派生逻辑未执行 | 检查 `snapmind_lab.py` 的 `_derive_*_plan()` 函数是否正常 |
| 502 错误"无可用索引" | `catalog.db` 为空文件 | `rm -f runtime/catalog.db && docker restart momentseek-0829-planner-lab`（见 5.5 Q1） |

### 6.4 单个测试查询示例

```bash
# 手动测试单个查询
curl -X POST http://localhost:8101/api/planner-lab/propose \
  -H "Content-Type: application/json" \
  -d '{
    "query": "找到演讲者展示产品的镜头",
    "mode": "balanced",
    "video_ids": ["test_video_001"],
    "has_query_image": false
  }' | jq '.plans | length'

# 期望输出：3（fast/balanced/deep）

# 检查 optimization_hints
curl -s -X POST http://localhost:8101/api/planner-lab/propose \
  -H "Content-Type: application/json" \
  -d '{
    "query": "找到鼓掌的镜头",
    "mode": "balanced",
    "video_ids": ["test_video_001"],
    "has_query_image": false
  }' | jq '.optimization_hints'

# 期望输出：包含 fast_strategy, deep_needs_rerank 等字段的对象
```

---

## 7. 相关文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| **延迟优化记录** | `docs/Planner_Latency_Optimization_Record.md` | 方案 A-D 对比、实测延迟/Token 分解、Phase 1-4 优化细节 |
| **Planner Lab 功能** | `docs/PLANNER_LAB.md` | Planner Lab 界面、API 文档、使用示例 | |
| **测试脚本** | `scripts/test_planner_optimize_0829.sh` | 自动化验证脚本（capabilities、prefix cache、hints_guided） |
---




**补充说明（2026-08-25）**：
- ✅ 已修复  缺失 voice.search 的问题
- 新增内容：Planning Rules 中 3 条 voice.search 规则 + Registered Capabilities 中 voice.search 定义
- 现在 adaptive prompt 和 role-aware prompt 对 voice.search 的支持完全一致
- Capabilities 总数：7 个（visual, face, asr, ocr, voice, vlm.rerank, confidence.filter）

