# Face 模态优化实施计划

**版本**: 1.0
**日期**: 2026-08-07
**分支**: `feature/Face_optimize`
**作者**: Claude Code (Kiro)
**前置参考**: `Milvus_optimization_plan.md`（方案3）、`Visual_record.md`、`OCR_record.md`、`ASR_IMPLEMENTATION_RECORD.md`、`Speaker_IMPLEMENTATION_RECORD.md`、`SPEAKER_OPTIMIZATION_PLAN.md`、`OCR_ASR_PLAN_CRITICAL_ERRORS.md`

---

## 0. 本计划的定位与"防错"声明

前四个模态优化最大的教训来自 `OCR_ASR_PLAN_CRITICAL_ERRORS.md`：**初版方案凭想象写字段名/文件路径，导致 35+ 处 `dense_embedding`、错误的 `processors/` 路径、误删 `has_embedding`**。本计划中所有文件路径、函数名、字段名、行为描述，均已逐一对照当前 backend 真实代码核实。凡本计划提及的符号，均可在下述文件中直接找到。

Face 与 Speaker 在检索本质上高度相似（都是**纯向量检索、无文本**、写入前已归一化），因此 Face 优化可**几乎完整套用刚完成的 Speaker 优化路径**：
- ✅ **借鉴**：消除两阶段重打分（方案3）、IVF_FLAT/HNSW → DiskANN 迁移（面向千万级）、检索参数配置化、legacy 清理、`search_list` 动态取值的硬约束。
- ❌ **不套用**：BM25 / `sparse_embedding` / analyzer / hybrid_search —— Face 查询是**人脸 ArcFace 向量**而非文本，词面检索无意义（见 §2）。

**Face 是五个模态中最后一个仍使用 IVF_FLAT + L2 的模态**，其余四个（visual/asr/ocr/speaker）均已是 DiskANN。本计划的核心即补齐这最后一块。

---

## 1. Face 模态现状（已核实）

### 1.1 涉及文件清单（真实路径）

| 职责 | 文件 | 关键符号 |
|------|------|---------|
| 索引构建（追踪+写 Milvus） | `backend/app/indexing/modalities/face/faces.py` | 跨帧追踪、`track_embedding = normalize(np.mean(track.embeddings, axis=0))`（L55/L85）、`normalize(face.normed_embedding)`（L134） |
| 编码器（索引+查询共用） | `backend/app/encoders/face.py` | `FaceEncoder`、`encode_reference()` |
| 置信度映射函数 | `backend/app/retrieval/search.py` | `face_confidence()`（**L101**，`1/(1+exp(-12*(cosine-0.45)))`）—— 注意**不在** `face.py`；`milvus_search.py` 由 `from app.retrieval.search import face_confidence` 引入 |
| Schema | `backend/app/vector_store/milvus/milvus_schema.py` | `create_face_schema()`（L256-264） |
| Collection/索引配置 | `backend/app/vector_store/milvus/milvus_client.py` | `_STATIC_INDEX_CONFIGS["face_embeddings"]`（L80-84） |
| **在线检索** | `backend/app/vector_store/milvus/milvus_search.py` | `milvus_face_candidates()`（L828-906）、`_ann_search()`（L350-413）、`_MODALITY_METRIC`（L59-65）、`_STATIC_INDEX_TYPES`（L68-73）、`_diskann_search_list_for()`（L333-347） |
| 检索路由/融合 | `backend/app/retrieval/search.py` | 生产路径：`_milvus_candidates_for_video()`（L1106，face 调用 L1160-1170）、融合权重（L694）、`_face()` 编码器（L806-818）、`_resolve_face_query()`（L1028）。死代码：孤儿函数 `_face_candidates()`（NPZ，L349-385，无调用点）、已死方法 `_candidates_for_video()`（L1077-1104，无 dispatch，内含对**不存在**的 `_face_for_video()` 的悬空调用 L1100） |
| API 入口 | `backend/app/api/entity_routes.py` | 参考图编码入口 |
| 设置 | `backend/app/core/settings.py` | `face_model`/`face_sample_fps`/`face_provider`/`face_ort_*`（L66-88）—— **无任何检索调优项** |

### 1.2 Schema（真实字段，勿臆造）

`create_face_schema()`（`milvus_schema.py` L256-264）= `_common_fields()` + face 专有字段：

```python
# _common_fields() 提供: pk, video_id, asset_version, model_version
FieldSchema("track_idx", DataType.INT64),
FieldSchema("start_ms",  DataType.INT64),
FieldSchema("end_ms",    DataType.INT64),
FieldSchema("best_ms",   DataType.INT64),
FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["face"]),  # 512, InsightFace buffalo_l
```

> **注意**：真实 schema 用的是常量 `EMBEDDING_DIMS["face"]`（`milvus_schema.py` L32-38 定义 = `512`），**非字面量 512**。改动/照抄时勿把常量写死为字面值。

- **512 维**（InsightFace buffalo_l `normed_embedding`），是所有模态里维度最大的（Speaker 192、Visual/CLIP 通常 512、文本 1024 视模型）。**维度越大，IVF_FLAT 全内存代价越高，DiskANN 迁移收益越大。**
- **没有 `text`、`sparse_embedding`、`has_embedding` 字段** —— 纯 ANN 检索，正确。
- 写入 Milvus 前 embedding 已 `normalize()` 为单位向量：track embedding 是各帧嵌入的归一化均值（`faces.py` L55/L85）。**这是 DiskANN + COSINE 迁移可行的关键前提**：归一化向量下 COSINE 与 L2 单调等价，Milvus 的 DiskANN 在本项目已被 visual/speaker 证明支持 COSINE。

### 1.3 索引配置（真实）

`face_embeddings`（`milvus_client.py` L80-84）：
```python
{"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
```
检索期 `_ann_search`（`milvus_search.py` L373-374）走 IVF_FLAT 分支：`{"metric_type": "L2", "params": {"nprobe": _IVF_NPROBE}}`，`_IVF_NPROBE = 64`（L55，硬编码）。

同步映射（`milvus_search.py`）：`_MODALITY_METRIC["face"] = "L2"`（L63）、`_STATIC_INDEX_TYPES["face"] = "IVF_FLAT"`（L71）。

### 1.4 在线检索（核心优化对象）

`milvus_face_candidates()`（`milvus_search.py` L828-906）当前逻辑：

```python
query_norm = normalize(query)
ann_limit  = min(limit * 2, 16_384)                       # ① 2倍扩召回
hits = _ann_search(..., ann_limit,
    ["track_idx", "start_ms", "end_ms", "best_ms", "embedding"])  # ② 拉回 embedding
for hit in hits:
    raw_emb = hit.get("embedding")
    if raw_emb is None:                                    # ← 该分支永不执行（见下）
        squared_l2 = float(hit["_distance"])
        cosine = max(-1.0, min(1.0, 1.0 - squared_l2 / 2.0))
    else:
        track_vec = normalize(raw_emb)
        cosine = float(np.dot(query_norm, track_vec))      # ③ Python 侧重算 cosine
    scored.append((cosine, hit))
scored.sort(...); scored[:limit]                           # ④ 重排后截断
```

**这正是 `Milvus_optimization_plan.md` 方案3 要消除的"两阶段重打分"**，且与 Speaker 优化前完全同构。关键事实：
- 因 `"embedding"` **始终**在 `output_fields`，**生产环境**中 `raw_emb is None` 分支**永远不会执行** —— 生产侧是死代码，实际总是走 ③ 的 Python 重算。
- **重要例外（影响测试改动排期）**：`raw_emb is None` 的 L2→cosine 公式分支（L867-868）**并非完全无人执行** —— 现有单测 `test_face_candidates_l2_to_cosine_conversion`（`test_milvus_search_metric.py` L151-198）的 mock **不返回** `"embedding"` 字段，因此该单测恰恰走的是这条 L2 公式分支。这解释了两个排期事实：(a) **P0-a 采用 L2→cosine 公式后，该单测无需改动即保持绿色**（P0-a 的输出逻辑与该分支一致）；(b) **测试的实质性重写集中在 P1 度量迁移**（COSINE 后 `cosine=float(_distance)`，L2 公式被彻底移除），而非 P0-a。详见 §6 步骤3 与 §7。
- Face embedding 写入前已归一化，Schema 为 `FLOAT_VECTOR`（float32）。单位向量下 L2 与 cosine 有精确解析关系：`cosine = 1 - squared_l2 / 2`；迁移到 COSINE 后，Milvus `_distance` 直接就是精确 cosine。
- IVF_FLAT 是**精确重排索引**（FLAT），`_distance` 本就精确；ANN 近似只影响"召回哪些邻居"，不影响"已返回邻居的距离精度"。
- 因此 ②③④ 都是冗余开销：每次多传 `512×4 = 2048 B/条 × ann_limit` 向量（limit=20、ann_limit=40 时约 80KB），并在 Python 侧做无谓 normalize + 点积。

**阈值/融合**：`threshold=0.35`（函数默认参数，硬编码 L834）；`face_confidence(cosine) = 1/(1+exp(-12*(cosine-0.45)))`（`search.py` L101，**非 `face.py`**）；跨模态融合权重 `{"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}`（`search.py` L694）—— **face 权重最高**，检索质量对最终排序影响最大。

### 1.5 NPZ 在线路径现状（已重新核实 —— 修正初版调用链描述）

> **初版本节把调用链写错了，此处按真实代码更正**（对齐 §0 防错声明）：
> - `_face_candidates()`（`search.py` L349-385）是 NPZ 全量扫描实现：`scores = embeddings @ normalize(query)`（NPZ 中 embeddings 也已归一化），`argsort` 取 top-k。
> - **`_face_candidates()` 没有任何调用点**（全仓 `grep` 确认：定义在 L349，无生产/测试调用者）—— 是孤儿函数。
> - 唯一"引用" face NPZ 的是 `_candidates_for_video()`（L1077-1104）内的 `self._face_for_video(video, face_query, ...)`（L1100）。但 **`_face_for_video()` 在整个 backend 中根本没有定义**（`_visual_for_video`/`_asr_for_video` 同样不存在）—— 这是一处**悬空调用**，一旦执行会 `AttributeError`。
> - 而 **`_candidates_for_video()` 本身也没有任何调用点**：检索主循环只走 `_milvus_candidates_for_video()`（`search.py` L1415）。它整段是**死代码**（推测是 ASR NPZ 路径移除时遗留的非 Milvus 分支残骸）。
> - 结论：Face 已**没有可执行的 NPZ 在线路径**。生产走 `_milvus_candidates_for_video()` → `milvus_face_candidates()`。需清理的是**两个死符号**：孤儿 `_face_candidates()`（L349）+ 死方法 `_candidates_for_video()`（L1077-1104，含悬空调用）。详见 §4.7。
> - **索引侧 NPZ 写出（`faces.py` L169 `atomic_save_npz`，离线恢复制品）应保留**，与 OCR `rebuild_ocr_from_npz` / Speaker 一致。本计划**不动索引构建链路**。

### 1.6 查询编码器 provider（已核实的小缺陷）

`search.py` `_face()`（L810-817）初始化 `FaceEncoder` 时第 2 个参数（provider）**硬编码为 `"cpu"`**（L812），而 `settings.face_provider`（L83，默认 `"cpu"`）**未被此处使用**。索引侧用 `face_provider`，查询侧不用 —— 若要 NPU/GPU 加速查询端参考图编码，需接通此处（见 §4.8）。

---

## 2. 借鉴什么、不借鉴什么（关键判断）

| 维度 | ASR/OCR | Speaker（刚优化） | Face | 结论 |
|------|---------|------|------|------|
| 查询输入 | 文本 | 音频声纹向量 | **人脸 ArcFace 向量** | ❌ 无 BM25/hybrid：无 text 字段，查询非文本 |
| 相似度 | dense(IP)+sparse(BM25) | 纯向量 cosine | **纯向量 cosine** | 只需单路向量检索 |
| 写入前归一化 | — | 是 | **是**（L55/L85） | ✅ DiskANN+COSINE 可行 |
| 目标规模 | 亿级 | 千万级 | **千万级 track** | ✅ 需 DiskANN 降内存 |
| 现用索引 | DISKANN | DISKANN（已迁） | **IVF_FLAT（内存）** | ✅ **迁移 DISKANN** |
| 现用度量 | IP | COSINE | **L2** | ✅ **迁移 COSINE** |
| 重打分 | 已消除 | 已消除 | **仍两阶段重打分** | ✅ **消除** |
| 参数配置化 | settings | settings | **全硬编码** | ✅ **配置化** |
| NPZ 在线 fallback | 已删 | 已是 Milvus-only | **仍存在** | ✅ 清理 |

**DiskANN + COSINE 在本项目已被两次证明可行**：visual（`_get_visual_index_config`）与 speaker（`_STATIC_INDEX_CONFIGS["speaker_embeddings"]`，`milvus_client.py` L85-97）均用 `DISKANN` + `COSINE` + `search_list_size: 128`。Face 迁移可**直接沿用同一参数体例**。

---

## 3. 已从四个模态吸取的教训 × Face 现状

| 教训（来源） | Face 现状 |
|------|------|
| 字段名/路径必须核对真实代码（`OCR_ASR_PLAN_CRITICAL_ERRORS`） | 本计划已逐一核实 |
| 每个 `search()` 必须传 `timeout`（`ASR_IMPLEMENTATION_RECORD` 决策5） | ✅ `_ann_search` L398 **已传** `timeout=milvus_query_timeout_seconds` |
| 查询按 `asset_version` 隔离 | ✅ `_ann_search` expr L393-396 **已含** `video_id` + `asset_version` |
| DiskANN `search_list >= limit` 硬约束，须动态取 `max(limit, setting)`（Speaker v2.1 纠错） | ⚠️ **Face 必遵守**：`ann_limit = min(limit*2, 16_384)` 最大可达 16384，静态 128 会违反约束 → 检索失败/截断（见 §4.5） |
| HNSW/IVF_FLAT→DiskANN 非 in-place，`_init_collections` 只 `load()` 不重建（Speaker §4.2/`milvus_client` 校验钩子） | ⚠️ **Face 须显式重灌数据**；已存在 fail-fast 校验（见下） |
| 索引类型 fail-fast 校验（Speaker 迁移副效果） | ✅ `_verify_ann_index_type_once` **已覆盖 face**（现校验 IVF_FLAT，L141-142）；迁移后须同步改为校验 DISKANN，否则旧 collection 检索直接抛错 |
| `above_threshold` 只影响显示不影响聚合（`OCR_record` 问题3） | Face `above_threshold` 仅用于 evidence 文案（L890），参与跨模态融合的是 `score=conf`，无此风险 |
| 删 collection 前检查存在性（`OCR_record` 问题4） | 迁移涉及 drop/rebuild，须复用 `utility.has_collection` |
| `_diskann_search_list_for()` 按模态键控，防止跨模态误用 tuning（Speaker 设计） | ⚠️ 当前仅有 speaker 分支，face 命中会 **raise**（L345-347）→ 迁移须新增 face 分支（见 §4.5） |
| 用 mock 断言防回归 | 本计划为每项改动配 mock 单测 |

---

## 4. 优化项（按优先级）

### 4.1 【P0-a】消除 `milvus_face_candidates` 两阶段重打分 + 清理死代码

**目标**：信任 Milvus 距离，不再拉回 `embedding`、不再 Python 重算；删掉永不执行的 `raw_emb is None` 死分支。与 Speaker P0-a 完全同构。

**改动**（仅 `milvus_search.py` `milvus_face_candidates`）：

1. `output_fields` 去掉 `"embedding"`：`["track_idx", "start_ms", "end_ms", "best_ms"]`。
2. 直接由距离得 cosine，**删除** `raw_emb`/`normalize`/`np.dot` 双分支：
   - **P0-a 阶段（仍 IVF_FLAT + L2）**：`cosine = max(-1.0, min(1.0, 1.0 - float(hit["_distance"]) / 2.0))`（L2→cosine 精确公式，即原死分支的逻辑）。
   - **P1 迁移 COSINE 后**：简化为 `cosine = float(hit["_distance"])`（见 §4.4）。
3. `ann_limit`：P0-a 阶段**保留 `min(limit*2, 16_384)`** 确保行为等价；配置化（§4.2）后收窄为 `limit * face_recall_multiplier`（默认 1）。
4. 其余（threshold、Candidate 构造、evidence、features、排序截断）**保持不变**。

**行为等价性**：IVF_FLAT 是精确重排索引，`_distance` 精确；L2→cosine 是精确解析变换。输出 cosine 与旧路径在 float32 精度内一致 → top-k 结果集完全一致。

**收益**：单次省 `2048 B × ann_limit` 传输 + Python normalize+点积循环。因 face 权重最高且检索走 per-video fan-out，收益按视频数放大。

**风险**：极低。转换公式本就存在于代码（死分支），IVF_FLAT 精确重排佐证等价。

---

### 4.2 【P0-b】Face 检索参数配置化（消除硬编码）

**问题**：`threshold=0.35`（函数默认 L834）、`_IVF_NPROBE=64`（L55）、`ann_limit` 倍数全硬编码；`settings.py` 无任何 face 检索项。四个模态优化均把关键参数下沉 settings。

**改动**（`backend/app/core/settings.py`，附 validator，参考 `validate_asr_positive` 与 Speaker 的区间校验）：
```python
face_identity_threshold: float = 0.35   # ArcFace 同人判断阈值（仅影响 above_threshold 显示）
face_recall_multiplier: int = 1         # ann_limit = limit * 该值（重排取消后默认 1）
face_diskann_search_list: int = 128     # DiskANN 检索期 search_list 基线（须 >= limit，见 §4.5）
```
> **说明**：P1 迁移 DiskANN 后 IVF_FLAT 弃用，故**不新增 `face_ivf_nprobe`**——`_IVF_NPROBE` 常量随 IVF_FLAT 分支的死代码化一并处理（§4.6）。若担心 P0/P1 之间存在过渡窗口需要调 nprobe，可临时加 `face_ivf_nprobe`，但因 P0-b 与 P1 建议一并落地，通常不必。

**接线**：
- `milvus_face_candidates(... threshold: float | None = None ...)`：为 None 时取 `get_settings().face_identity_threshold`；`search.py` 调用侧的硬编码 `0.35` 同步改为传 None 或读 setting。
- `ann_limit = limit * settings.face_recall_multiplier`（默认 1）。
- `face_diskann_search_list` 供 §4.5 的 `_diskann_search_list_for("face")` 使用。

---

### 4.3 【P1】IVF_FLAT → DiskANN 迁移（补齐最后一个模态）

**依据**：业务最终态需千万级向量检索。IVF_FLAT 全内存，512 维千万级约 `1e7 × 512 × 4B ≈ 20 GB` 原始向量 + IVF 聚类结构常驻内存（维度最大，代价最重）；DiskANN 将向量与图落盘、仅 PQ 常驻，内存降 ~90%（与 visual/asr/ocr/speaker 一致）。**Face 是唯一未迁的模态**，迁移后五模态索引栈统一。

**改动**：

1. **索引配置**（`milvus_client.py` L80-84）：`face_embeddings` 由 IVF_FLAT → DISKANN，度量 L2 → COSINE：
   ```python
   "face_embeddings": {
       "index_type": "DISKANN",
       "metric_type": "COSINE",          # visual/speaker 已证明 DiskANN 支持 COSINE
       "params": {
           "max_degree": 56,
           "search_list_size": 128,
           "pq_code_budget_gb": 0.125,
           "build_dram_budget_gb": 32.0,
       },
   }
   ```
   （参数体例对齐 speaker/asr/ocr。）

2. **检索期映射**（`milvus_search.py`）：
   - `_STATIC_INDEX_TYPES["face"]`：`"IVF_FLAT"` → `"DISKANN"`（L71）。
   - `_MODALITY_METRIC["face"]`：`"L2"` → `"COSINE"`（L63）——见 §4.4。
   - `_ann_search` 已有 DISKANN 分支（L365-372），face 迁移后自动命中，无需改分支逻辑；但**必须**在 `_diskann_search_list_for()` 新增 face 分支（见 §4.5），否则 raise。

3. **fail-fast 校验同步**：`_verify_ann_index_type_once` 现对 face 校验 IVF_FLAT（L141-142 注释）。迁移后它会自动按新的 `get_modality_index_type("face")="DISKANN"` 校验。**部署顺序含义**：一旦配置改为 DISKANN，若 collection 仍是旧 IVF_FLAT，face 检索会 fail-fast 抛错。因此**改配置 + 重灌数据必须先于服务启用**。

4. **迁移执行（开发阶段，直接重建数据）**：
   - 本项目处于开发早期，**不提供独立迁移脚本、不做存量原地迁移**。改配置后直接重灌 face 索引数据（drop collection → `_init_collections` 以新 DISKANN 配置重建，或重跑 face 索引）。
   - 原因：`_init_collections` 对已存在的 collection 只 `load()` 不重建索引；仅改配置只对全新 collection 生效。schema 字段不变 → 重建时向量无需重新嵌入。
   - 开发环境先验证**可建成 + 可检索**（DiskANN 对极小数据集可能有构建下限，见 §8）。

**与 4.1 的关系**：互补，应一起落地。4.1 去掉 embedding 回传后，DiskANN 完全不必从盘读完整向量（只用 PQ 距离 + 元数据），叠加收益最大。

---

### 4.4 【P1】L2 → COSINE 度量迁移（随 DiskANN 一并完成）

**依据**：Face embedding 是单位向量，L2 = COSINE 的单调变换，但 L2 需 `1 - squared_l2/2` 转换公式。visual/speaker 都直接用 COSINE。迁移到 COSINE 后：
- `milvus_face_candidates` 中 §4.1 的 `cosine = 1 - squared_l2/2` 简化为 `cosine = float(hit["_distance"])`（Milvus COSINE 对归一化 float32 返回精确 cosine）。
- `_MODALITY_METRIC["face"] = "COSINE"`，`_ann_search` 的 `metric` 自动变为 COSINE。

**注意**：`face_confidence()`（`search.py` L101，**非 `face.py`**）以 cosine 为输入，中心 0.45、斜率 12，**保持不变**——它作用于最终 cosine，与底层度量无关。COSINE 迁移不改变传入 `face_confidence` 的 cosine 数值（float32 精度内等价）。

---

### 4.5 【P1 · 必改】`_diskann_search_list_for()` 新增 face 分支 + `search_list` 动态取值

**这是 Speaker v2.1 纠错的同类硬约束，Face 迁移必须一并处理，否则检索直接失败。**

**问题1（会 raise）**：`_diskann_search_list_for()`（`milvus_search.py` L333-347）当前**只有 speaker 分支**，其余模态 `raise MilvusServiceError`。face 迁移 DISKANN 后，`_ann_search` 的 DISKANN 分支（L371）会调用 `_diskann_search_list_for("face")` → 直接抛错。

**改动**：
```python
def _diskann_search_list_for(modality: str) -> int:
    settings = get_settings()
    if modality == "speaker":
        return settings.speaker_diskann_search_list
    if modality == "face":
        return settings.face_diskann_search_list      # 新增
    raise MilvusServiceError(...)
```

**问题2（会截断/失败）**：DiskANN 硬约束 `search_list >= limit`。`_ann_search` 收到的 `limit` 是 `milvus_face_candidates` 传入的 `ann_limit = min(limit*2, 16_384)`。face 的上层 `limit = channel_limits["face"]`，默认 `limit * 3`（`search.py` L1311，`limit` 为请求 top-k），可较大。固定 `search_list=128` 在 `ann_limit>128` 时违反约束。

**已内建的正确处理**：`_ann_search` DISKANN 分支已用 `search_list = max(limit, _diskann_search_list_for(modality))`（L371）——与 visual v2 的 `max(top_k, 100)` 同构。face 迁移后**自动获得**该动态取值，无需额外改 `_ann_search`；只需保证 `_diskann_search_list_for("face")` 返回配置基线（问题1）。

---

### 4.6 【P1】Legacy 残余清理（与 Visual/OCR/ASR/Speaker 对齐）

**随本次迁移一并处理**：
1. **重打分死代码**：`milvus_face_candidates` 的 `raw_emb is None` 双分支、`normalize`+`np.dot` 块 —— 随 4.1 删除。
2. **`milvus_face_candidates` docstring**：L837-849 描述"two-phase / retrieve embedding / L2 metric"，须改写为"single-phase trusted COSINE distance"（对齐 Speaker 迁移后的 docstring）。
3. **`_IVF_NPROBE` 常量 + `_ann_search` IVF_FLAT 分支**（L55、L373-374）：face→DISKANN、speaker=DISKANN 后，**再无任何模态使用 IVF_FLAT**，该分支与常量变为死代码。**保守处理**：可保留 + 加注"当前无模态使用，保留备用"，或一并删除。建议**删除**（比 Speaker 保留 HNSW 分支更彻底，因为 IVF_FLAT 确无未来模态计划；若保守则加注）。
4. **`_HNSW_EF` 与 HNSW 分支**（L54、L375-379）：Speaker 计划已判定为死代码并建议保留+加注。本轮 face 迁移不新增其使用者，**沿用 Speaker 的决定（保留+加注）**，不重复处理。
5. **fail-fast 校验注释更新**（`_verify_ann_index_type_once` L140-145、`_diskann_search_list_for` L336-340、`_ann_search` L376-377）：多处注释写死"face=IVF_FLAT"，迁移后须全部更新为"face=DISKANN"，否则误导后续维护。

**P2 单列**：NPZ 在线路径 `_face_candidates()` 与死方法 `_candidates_for_video()` 清理见 §4.7。

**不做**：不删除索引侧 NPZ 写出（离线恢复制品，与 OCR/Speaker 一致保留）；不动索引构建链路（`faces.py`/schema 字段/追踪逻辑）。

---

### 4.7 【P2】清理 NPZ 在线孤儿函数 `_face_candidates()` + 已死方法 `_candidates_for_video()`

**问题（已按真实代码重新核实，纠正初版对调用链的臆测）**：
- `_face_candidates()`（`search.py` L349-385）是 NPZ 全量扫描实现（`scores = embeddings @ normalize(query)` → `argsort` 取 top-k），**但在整个 backend 无任何调用点**（`grep` 仅命中定义处与 `milvus_face_candidates` 的同名前缀），是孤儿函数。
- 承载其"调用"的 `_candidates_for_video()`（L1077-1104）**同样无任何调用点**——检索主循环只调 `_milvus_candidates_for_video()`（L1415），且无 `getattr`/字符串动态派发。该方法整段是死代码。
- 更关键：`_candidates_for_video()` 体内 L1100 调用的 `self._face_for_video(...)` **在整个 backend 中没有任何定义**（`_visual_for_video`/`_asr_for_video` 亦不存在）。这条 NPZ 分支应是随 ASR NPZ 路径删除时未清理干净的残留——一旦被执行会直接 `AttributeError`。因此初版"由 `_face_for_video()` 在非 Milvus 路由时调用"的描述不成立：不存在可运行的非 Milvus 路由，`_face_candidates` 亦无人调用。

**改动**（对齐 ASR/OCR 已删 NPZ 在线路径的决策）：
1. 删除孤儿函数 `_face_candidates()`（L349-385）。
2. 删除已死方法 `_candidates_for_video()`（L1077-1104，含对不存在的 `_face_for_video` 的悬空调用）。
- 仅保留 `milvus_face_candidates()`（经 `_milvus_candidates_for_video()`）作为唯一在线 face 检索路径。

**排序**：P2，可在 P0/P1 稳定后单独落地。落地前用 `grep` 复核 `_face_candidates` / `_candidates_for_video` / `_face_for_video` 三者的引用点确认均无生产/测试依赖（当前核实：均无）。**索引侧 NPZ 写出不受影响，保留。**

---

### 4.8 【P2】查询编码器 provider 修复

**问题**：`search.py` `_face()`（L812）硬编码 `FaceEncoder(..., "cpu", ...)`，忽略 `settings.face_provider`。查询端参考图编码无法用 NPU/GPU 加速。

**改动**：将第 2 个参数改为 `self.settings.face_provider`（与索引侧一致）。

**排序**：P2，独立小改。**风险**：低——需确认 `FaceEncoder` 对非 cpu provider 的行为（模型下载/provider 可用性），开发环境先验证。默认值仍是 `"cpu"`，不改配置时行为不变。

---

## 5. `_ann_search` 共用路径评定（face 与 speaker 共用，防误伤）

`_ann_search` 被 **face 与 speaker 共用**。Speaker 已迁 DISKANN，本次 face 也迁 DISKANN，两者将走**同一** DISKANN 分支。关键防误伤点：

- **metric 按模态独立**：`_MODALITY_METRIC` 中 speaker=COSINE、face=COSINE（迁移后），各取各的，`_ann_search` 的 `metric = _MODALITY_METRIC[modality]` 天然隔离。
- **search_list 按模态独立**：`_diskann_search_list_for()` 按 modality 键控（Speaker 刻意设计），face 新增分支取 `face_diskann_search_list`，**不会**继承 speaker 的 `speaker_diskann_search_list`。这正是 Speaker 计划强调的"防跨模态 tuning 误用"设计的价值兑现。
- **fail-fast 校验按模态独立**：`_verify_ann_index_type_once` 按 modality 校验各自 `get_modality_index_type`，face/speaker 互不干扰。

**结论**：face 迁移完全复用 speaker 已铺好的 DISKANN 基础设施，仅需新增 face 的 metric/index_type 映射值 + `_diskann_search_list_for` face 分支 + 配置项。**这是本次优化风险可控的根本原因**——基础设施已由 Speaker 优化验证过。

---

## 6. 实施步骤与顺序

1. **P0-a 重打分消除**（`milvus_search.py`）
   - 改 `milvus_face_candidates`：去 `"embedding"` 输出、用 L2→cosine 公式（暂）、删死分支；先保 `limit*2`。
   - 单测：mock `_ann_search`，断言 (a) `output_fields` **不含** `"embedding"`；(b) Candidate `raw_score`/cosine == `1 - _distance/2`；(c) 排序/截断/阈值不变。
2. **P0-b 配置化**（`settings.py` + `milvus_search.py` + `search.py`）
   - 新增 3 项 settings + validator；接线 threshold（None→setting）、`ann_limit = limit * face_recall_multiplier`。
   - 单测：默认值加载；threshold 生效。
3. **P1 DiskANN + COSINE 迁移**（`milvus_client.py` + `milvus_search.py`）
   - 配置 IVF_FLAT→DISKANN + L2→COSINE；`_STATIC_INDEX_TYPES["face"]="DISKANN"`；`_MODALITY_METRIC["face"]="COSINE"`。
   - `_diskann_search_list_for()` 新增 face 分支（§4.5 问题1）。
   - `milvus_face_candidates` 的 cosine 简化为 `float(hit["_distance"])`（§4.4）。
   - **更新既有断言（`test_milvus_search_metric.py`，改动面比"改一行"大，须整块处理，详见 §7 回归小节）**：
     - (a) 参数化元组 L77 `("face","L2","IVF_FLAT")` → `("face","COSINE","DISKANN")`（否则 `test_per_modality_metric_and_index` 回归失败；`test_modality_metric_matches_collection_configs`/`test_modality_index_type_matches_collection_configs` 会随配置自动跟随，无需手改）。
     - (b) **L136-198 整块"L2 → cosine 转换"测试须重写/移除**：`test_face_candidates_l2_to_cosine_conversion`（L151-198）当前用 `l2_dist = 2*(1-cosine_expected)` 构造 `fake_hit.distance`，断言 `raw_score ≈ cosine_expected`。P1 后 `milvus_face_candidates` 改为 `cosine = float(hit["_distance"])`（不再做 L2→cosine 变换），该 mock 会令 `raw_score == 0.56 ≠ 0.72` → **断言失败**。须改为让 `fake_hit.distance` 直接返回 cosine（如 `0.72`），断言 `raw_score ≈ 0.72`；`fake_hit.entity` 保持不含 `"embedding"`（对齐 §4.1 去 embedding 回传）。
     - (c) `test_l2_cosine_round_trip`（L140-148）测的是 L2↔cosine 解析变换，P1 后该逻辑已从生产路径移除 → **移除该测试**，或降级为纯数学注释（不再代表 face 检索行为）。
   - 迁移（开发阶段）：改配置后**直接重灌 face 索引数据**（drop collection → `_init_collections` 重建，或重跑索引）；开发环境先验证可建成 + 可检索。
   - 单测：face 走 DISKANN 分支且 `search_list == max(limit, face_diskann_search_list)`；timeout 仍转发；fail-fast 校验对 face 期望 DISKANN。
4. **P1 legacy 清理**（`milvus_search.py`）
   - 删/加注 `_IVF_NPROBE` + IVF_FLAT 分支（§4.6.3）；改写 `milvus_face_candidates` docstring；更新所有"face=IVF_FLAT"注释为 DISKANN。
5. **P2 死代码清理**（`search.py`，独立落地）
   - 删孤儿函数 `_face_candidates()`（L349-385，无调用点）+ 删已死方法 `_candidates_for_video()`（L1077-1104，无 dispatch，且含对不存在的 `_face_for_video`/`_visual_for_video` 的悬空调用）；索引侧 NPZ 写出保留。
6. **P2 查询 provider 修复**（`search.py` L812，独立落地）。
7. **验证与文档**：`docs/Face_IMPLEMENTATION_RECORD.md`（对齐 Speaker/ASR/OCR record 体例）。

> **构建链路（`faces.py`/schema 字段/追踪逻辑/索引侧 NPZ 写出）本轮不改动**——已满足 timeout/asset_version 隔离，DiskANN 迁移不涉及 schema 字段变更。

---

## 7. 验证方案

**正确性（P0 重点，开发阶段以功能正确为主）**：
- **重打分消除（4.1）等价性——索引类型须保持不变**：同为 IVF_FLAT 下，对比"重打分开/关"两条路径，断言 `np.allclose(old_cosines, new_cosines, atol=1e-4)` 且 top-k `track_idx` 一致。IVF_FLAT 是精确重排索引，这是最干净的等价性证明场景（比 Speaker 的 HNSW 更严格可复现）。
- **不要**把"IVF_FLAT+重算" vs "DiskANN+信任距离"直接对比要求 top-k 恒等：DiskANN 是近似索引，召回邻居集本就可能与 IVF_FLAT（精确）不同，恒等断言会产生"非 bug 的失败"。DiskANN 召回质量属效果测评，本阶段不做。
- mock 单测防回归：`output_fields` 不含 `embedding`；timeout 仍转发；face 走 DISKANN 分支且 `search_list == max(limit, setting)`；`_diskann_search_list_for("face")` 返回配置值不 raise。

**DiskANN 迁移专项**：
- fail-fast 校验通过（实际索引 == DISKANN）。
- 开发环境小数据可建成、可检索（验证无"段太小"构建问题）。
- COSINE 迁移后 `_distance` ≈ 旧 L2→cosine 值（同 IVF_FLAT 基线对比，float32 精度内）。

**性能**：
- 单次 face 检索：RPC 输出字节数（去 embedding 前后）、P50/P95；内存占用（IVF_FLAT vs DiskANN，512 维收益应大于 speaker 192 维）。

**回归**：
- face 相关测试全绿（`grep` 定位 `test_*face*`、`test_milvus_search_metric.py`）。
- **`test_milvus_search_metric.py` 的 face 耦合点须全部处理（P1）**，不止参数元组：
  - L73-79 参数化元组 `("face","L2","IVF_FLAT")` → `("face","COSINE","DISKANN")`。
  - `test_modality_metric_matches_collection_configs`（L39-51）与 `test_modality_index_type_matches_collection_configs`（L54-66）**无需手改**——它们从 `get_collection_index_config` 动态取值，随 `milvus_client.py` 配置迁移自动通过；但须确认 `_MODALITY_METRIC`/`_STATIC_INDEX_TYPES` 与 `_STATIC_INDEX_CONFIGS` 同步改完，否则这两个一致性测试会红。
  - `test_modality_index_type_covers_all_modalities`（L29-36）断言允许集含 `IVF_FLAT`，迁移后 face 返回 DISKANN 仍在集合内，**不会红**；可保留。
  - **L136-198 整块"L2→cosine 转换"测试须重写/删除**：`test_l2_cosine_round_trip`（L140-148）测的是 L2→cosine 解析式，COSINE 迁移后该式从生产代码移除 → **删除或改标注为历史**；`test_face_candidates_l2_to_cosine_conversion`（L151-198）的 mock 用 `fake_hit.distance = 2*(1-cosine)` 喂 L2 值并断言 `raw_score≈cosine`，COSINE 后代码变为 `cosine=float(hit["_distance"])`，会读到 L2 数值导致断言失败 → **必须改为让 `.distance` 直接返回 cosine**（并同步去掉 `embedding` 依赖、对齐新 `output_fields`）。
- 参考图检索端到端冒烟（`entity_routes.py` 入口）；确认 face 权重 0.55 下融合排序正常。

---

## 8. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 信任距离导致分数细微漂移 | 极低 | 低 | IVF_FLAT 精确重排 + 归一化 float32；同索引类型 `allclose` 验证（§7） |
| **`_diskann_search_list_for("face")` 未加分支 → 检索 raise** | **高（若漏改）** | **高** | §4.5 必改项；单测断言不 raise |
| **DiskANN `search_list < limit` 致检索失败/截断** | **中** | **高** | `search_list = max(limit, setting)` 已内建（L371）；单测断言 |
| DiskANN 对小段构建失败/慢 | 中 | 中 | 开发环境先验证可建成；必要时大数据集验证 |
| 未重灌数据即启用 → fail-fast 校验致 face 检索全挂 | 中 | 高 | 改配置+重灌数据须先于服务启用（§4.3 部署顺序） |
| **`test_milvus_search_metric.py` L136-198 L2→cosine 转换测试未随 COSINE 迁移重写** | **中** | **中** | 不止改 L73-79 元组；`test_face_candidates_l2_to_cosine_conversion` 须让 mock `.distance` 直接返回 cosine，`test_l2_cosine_round_trip` 删除/标注历史（§6 步骤3、§7） |
| 误删 IVF_FLAT 分支影响未来模态 | 低 | 低 | 保守可保留+加注；确无 IVF_FLAT 模态计划 |
| 查询 provider 改非 cpu 引发 provider 不可用 | 低 | 中 | P2 独立；默认仍 cpu；开发环境验证后再改配置 |
| 误动构建链路破坏追踪/schema | 低 | 高 | 本轮不改 `faces.py`/schema 字段/索引侧 NPZ |

---

## 9. 与既有约定的一致性检查清单

- [ ] 所有 `search()`/`query()` 调用转发 `timeout`（`_ann_search` 现已满足）
- [ ] 所有查询表达式含 `video_id` + `asset_version` 隔离（现已满足）
- [ ] 不新增 BM25/analyzer/sparse 字段（人脸向量查询无意义）
- [ ] 消除 `milvus_face_candidates` 两阶段重打分并删 `raw_emb is None` 死分支
- [ ] `output_fields` 去掉 `"embedding"`（去向量回传）
- [ ] `face_embeddings` 迁移为 **DISKANN + COSINE**（对齐 visual/speaker 证明的组合）
- [ ] `_STATIC_INDEX_TYPES["face"]` = `"DISKANN"`、`_MODALITY_METRIC["face"]` = `"COSINE"`
- [ ] **`_diskann_search_list_for()` 新增 face 分支**（否则 raise）—— §4.5 必改
- [ ] **DiskANN 分支 `search_list = max(limit, setting)`**（已内建；ann_limit 最大 16384，静态值会违约）
- [ ] `milvus_face_candidates` cosine 简化为 `float(hit["_distance"])`（COSINE 迁移后）
- [ ] 关键检索参数下沉 settings（`face_identity_threshold`/`face_recall_multiplier`/`face_diskann_search_list`）
- [ ] `search.py` 调用侧硬编码 `0.35` 改为读 setting
- [ ] fail-fast 校验迁移后对 face 期望 DISKANN；所有"face=IVF_FLAT"注释更新为 DISKANN
- [ ] 清理/加注 `_IVF_NPROBE` + IVF_FLAT 分支死代码；改写 `milvus_face_candidates` docstring
- [ ] 更新 `test_milvus_search_metric.py`：(a) L77 参数元组 `("face","L2","IVF_FLAT")` → `("face","COSINE","DISKANN")`；(b) **重写/删除 L136-198 的 L2→cosine 转换测试块**（`test_l2_cosine_round_trip` + `test_face_candidates_l2_to_cosine_conversion`），mock `.distance` 直接返回 cosine
- [ ] （P2）清理孤儿函数 `_face_candidates()`（L349，无调用点）+ 死方法 `_candidates_for_video()`（L1077-1104，含对不存在的 `_face_for_video` 的悬空调用）；索引侧 NPZ 写出保留
- [ ] （P2）查询编码器 provider 接通 `settings.face_provider`
- [ ] 不改动索引构建链路（`faces.py`/schema 字段/追踪/索引侧 NPZ）
- [ ] 每项改动配 mock 单测防回归

---

**结论**：Face 是五模态中最后一个 IVF_FLAT + L2 的模态，其检索本质与 Speaker 高度同构（纯向量、写入前归一化、无文本）。因此本优化**几乎完整复用刚验证过的 Speaker 路径**：**P0 双改**——(a) 消除两阶段重打分（方案3，低风险高确定性，IVF_FLAT 精确重排使等价性证明比 Speaker 更干净）、(b) 检索参数配置化；**P1** 迁移 **DiskANN + COSINE**（512 维内存收益为五模态最大，且补齐索引栈统一），其中 **§4.5 `_diskann_search_list_for()` 新增 face 分支为必改硬约束**，漏改即检索 raise。辅以 legacy 死代码清理、NPZ 在线路径清理（P2）与查询 provider 修复（P2）。**不**引入 BM25/hybrid。风险可控的根本原因是 DiskANN 基础设施已由 Speaker 优化铺好并按模态隔离，face 只需填入自己的映射值与配置项。
