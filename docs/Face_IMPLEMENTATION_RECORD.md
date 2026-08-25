# Face 模态优化实施记录

**分支**: `feature/Face_optimize`
**日期**: 2026-08-07
**前置参考**: `FACE_OPTIMIZATION_PLAN.md`、`Milvus_optimization_plan.md`(方案3)、`SPEAKER_IMPLEMENTATION_RECORD.md`、`Visual_record.md`、`OCR_record.md`、`ASR_IMPLEMENTATION_RECORD.md`、`OCR_ASR_PLAN_CRITICAL_ERRORS.md`

---

## 📋 概述

### 目标
Face 是五个模态中**最后一个仍使用 IVF_FLAT + L2** 的模态,其余四个(visual/asr/ocr/speaker)均已 DiskANN。承接 Speaker 优化经验(检索本质高度同构:纯向量、写入前归一化、无文本),对 Face 人脸检索做:

1. **消除两阶段重打分**(`Milvus_optimization_plan.md` 方案3):信任 Milvus 距离,不再拉回 `embedding` 到客户端做 Python 侧 `normalize`+`np.dot` 重算,删除永不执行的 `raw_emb is None` 死分支。
2. **IVF_FLAT → DiskANN + L2 → COSINE 迁移**:面向千万级 track 规模;512 维是五模态最大,IVF_FLAT 全内存代价最重,DiskANN 落盘 + PQ 常驻内存降 ~90%,迁移后五模态索引栈统一。

辅以 **检索参数配置化(P0-b)**、**legacy 死代码清理(P1)**、**NPZ 在线孤儿/死代码清理(P2)**、**查询编码器 provider 修复(P2)**。

### 关键判断:借鉴什么、不借鉴什么
- ✅ **借鉴**:DiskANN + COSINE(visual/speaker 已两次证明可行)、消除重打分、参数下沉 settings、timeout/asset_version 隔离(已满足)、`_diskann_search_list_for` 按模态键控、`search_list = max(limit, setting)` 硬约束、mock 单测防回归。
- ❌ **不套用**:BM25 / `sparse_embedding` / analyzer / `hybrid_search`。Face 查询是**人脸 ArcFace 向量**而非文本,词面检索无意义;schema 无 `text` 字段。

### 风险可控的根本原因
DiskANN 基础设施已由 Speaker 优化铺好并**按模态隔离**(metric / index_type / search_list / fail-fast 校验各自独立)。Face 迁移只需填入自己的映射值 + 配置项 + `_diskann_search_list_for` 新增 face 分支,复用同一 `_ann_search` DISKANN 分支,无需改分支逻辑。

---

## 🏗️ 实施详情

### 1. 消除两阶段重打分(P0-a,`milvus_search.py`)

**改动函数**:`milvus_face_candidates()`

**改动前**(两阶段):
```python
ann_limit = min(limit * 2, 16_384)                    # 2倍扩召回(为重排服务)
hits = _ann_search(..., ann_limit,
    ["track_idx","start_ms","end_ms","best_ms","embedding"])  # 拉回 embedding
for hit in hits:
    raw_emb = hit.get("embedding")
    if raw_emb is None:                               # ← 生产永不执行的死分支
        squared_l2 = float(hit["_distance"])
        cosine = max(-1.0, min(1.0, 1.0 - squared_l2 / 2.0))
    else:
        track_vec = normalize(raw_emb)
        cosine = float(np.dot(query_norm, track_vec)) # Python 侧重算
```

**改动后**(单阶段,最终 COSINE 形态):
```python
ann_limit = min(limit * settings.face_recall_multiplier, 16_384)  # 默认 multiplier=1
hits = _ann_search(..., ann_limit,
    ["track_idx","start_ms","end_ms","best_ms"])       # 不含 embedding
scored = [(float(hit["_distance"]), hit) for hit in hits]  # COSINE 距离即精确 cosine
scored.sort(key=lambda x: x[0], reverse=True)
```

**等价性依据**:
- Face embedding 写入前已 `normalize` 为单位向量(track embedding 是各帧嵌入的归一化均值,`faces.py` L55/L85;单帧 `normalize(face.normed_embedding)` L134)。
- 因 `"embedding"` **始终**在旧 `output_fields`,生产环境 `raw_emb is None` 分支**永不执行**——实际总走 Python 重算,是纯冗余。
- 归一化 float32 + COSINE 度量下,Milvus 返回的 `_distance` **就是精确 cosine**;DiskANN 近似性只影响"召回哪些邻居",不影响"已返回邻居的距离精度"。

**收益**:单次省 `512×4B = 2048 B × ann_limit` 向量传输 + Python normalize+点积循环。因 face 跨模态融合权重最高(0.55)且检索走 per-video fan-out,收益按视频数放大。

---

### 2. 检索参数配置化(P0-b,`settings.py` + `milvus_search.py` + `search.py`)

新增 3 项 face 检索 settings + validator:
```python
face_identity_threshold: float = 0.35   # ArcFace 同人判断阈值(仅影响 above_threshold/decision 显示)
face_recall_multiplier: int = 1         # ann_limit = limit * 该值(重排取消后默认 1)
face_diskann_search_list: int = 128     # DiskANN 检索期 search_list 基线(动态升至 >= ann_limit)
```
validator:`face_recall_multiplier`/`face_diskann_search_list` 须 > 0;`face_identity_threshold` 须 ∈ [-1.0, 1.0]。

**接线**:
- `milvus_face_candidates(threshold: float | None = None)`:为 None 时取 `face_identity_threshold`。
- `search.py` `_milvus_candidates_for_video` 调用侧硬编码 `0.35` 改为传 `None`(读 setting)。
- `ann_limit = limit * face_recall_multiplier`。
- `face_diskann_search_list` 供 `_diskann_search_list_for("face")` 使用。

`.env.0829` 同步新增示例(值 = settings 默认,不改运行时行为):
```bash
FACE_IDENTITY_THRESHOLD=0.35
FACE_DISKANN_SEARCH_LIST=128
FACE_RECALL_MULTIPLIER=1
```

---

### 3. IVF_FLAT → DiskANN + L2 → COSINE 迁移(P1,`milvus_client.py` + `milvus_search.py`)

**3.1 索引配置**(`milvus_client.py` `_STATIC_INDEX_CONFIGS["face_embeddings"]`):
```python
"face_embeddings": {
    # Migrated IVF_FLAT → DISKANN for 千万级 scale (disk-resident vectors +
    # PQ in memory). L2 → COSINE: face embeddings are normalised unit vectors,
    # and visual/speaker proved DiskANN supports COSINE in this stack.
    "index_type": "DISKANN",
    "metric_type": "COSINE",
    "params": {
        "max_degree": 56,
        "search_list_size": 128,
        "pq_code_budget_gb": 0.125,
        "build_dram_budget_gb": 32.0,
    },
}
```
参数体例对齐 speaker/asr/ocr。

**3.2 检索期映射**(`milvus_search.py`):
- `_STATIC_INDEX_TYPES["face"]`:`"IVF_FLAT"` → `"DISKANN"`。
- `_MODALITY_METRIC["face"]`:`"L2"` → `"COSINE"`。
- `_ann_search` 已有 DISKANN 分支,face 迁移后自动命中,无需改分支逻辑。

**3.3 `_diskann_search_list_for()` 新增 face 分支(§4.5 必改硬约束)**:
```python
if modality == "speaker":
    return settings.speaker_diskann_search_list
if modality == "face":
    return settings.face_diskann_search_list   # 新增,否则 face 命中 → raise
raise MilvusServiceError(...)
```
**为何必改**:该 helper 早前仅 speaker 分支,其余模态 raise。face 迁 DISKANN 后 `_ann_search` DISKANN 分支会调 `_diskann_search_list_for("face")`,漏改即检索直接抛错。

**3.4 `milvus_face_candidates` cosine 简化**:`cosine = 1 - squared_l2/2` → `cosine = float(hit["_distance"])`(COSINE 对归一化 float32 返回精确 cosine)。docstring 改写为 "single-phase ANN with trusted COSINE distance"。

**⚠️ `search_list >= limit` 硬约束(已内建)**:`_ann_search` DISKANN 分支已用 `search_list = max(limit, _diskann_search_list_for(modality))`。face 上层 `channel_limits["face"]` 默认 `limit*3`(`search.py` L1311)可较大,固定 128 在 `ann_limit>128` 时违约;`max(limit, setting)` 兜住。face 迁移**自动获得**该动态取值。

**3.5 索引配置 fail-fast 校验**:`_verify_ann_index_type_once` 按模态校验 index type 与 metric type，Face 必须为 `DISKANN/COSINE`。若 collection 仍是旧 `IVF_FLAT/L2` 或迁移成 `DISKANN/L2`，首次 Face 检索会 fail-fast 抛 `MilvusServiceError`。因此必须先执行 §5 的一次性迁移，再启用新服务。

---

### 4. Legacy 死代码清理(P1,`milvus_search.py`)

- **随 §1 删除**:`milvus_face_candidates` 的 `raw_emb is None` 双分支、`normalize`+`np.dot` 重算块、`limit*2` 扩召回、L2→cosine 转换公式。
- **`_IVF_NPROBE` 常量 + IVF_FLAT 分支删除**:face→DISKANN、speaker=DISKANN 后**再无任何模态使用 IVF_FLAT**,该分支与常量成死代码。**已删除**(比 Speaker 保留 HNSW 更彻底,因 IVF_FLAT 确无未来模态计划)。`_ann_search` else 错误信息同步改为 "only DISKANN and HNSW are supported"。
- **`_HNSW_EF` + HNSW 分支**:沿用 Speaker 的决定(**保留 + 加注**"当前无模态使用,保留以备将来")。
- **注释更新**:多处 "face=IVF_FLAT" 注释(`_verify_ann_index_type_once` 作用域说明、`_diskann_search_list_for` docstring、`_ann_search` HNSW 分支注释、模块头 ANN 参数注释)全部更新为 "face=DISKANN"。
- **docstring**:`milvus_face_candidates` 由 "two-phase / retrieve embedding / L2 metric" 改写为 "single-phase trusted COSINE distance"。

---

### 5. NPZ 在线孤儿/死代码清理(P2,`search.py`)

> **审计发现(纠正初版计划对调用链的臆测)**:初版计划称 `_face_candidates()` 由 `_face_for_video()` 在非 Milvus 路由时调用。逐行核对真实代码后发现:

- `_face_candidates()`(旧 L349-385)是 NPZ 全量扫描实现,**全仓无任何调用点**——是孤儿函数。
- `_candidates_for_video()`(旧 L1077-1104)**同样无任何调用点**(检索主循环只调 `_milvus_candidates_for_video()`,无 `getattr`/字符串动态派发)——整段死代码。
- 其体内 L1100 调用的 `self._face_for_video(...)` **在整个 backend 中根本没有定义**(`_visual_for_video`/`_asr_for_video` 亦不存在)——悬空调用,一旦执行会 `AttributeError`。推测是 ASR NPZ 路径移除时遗留的非 Milvus 分支残骸。

**改动**:删除孤儿函数 `_face_candidates()` + 删除已死方法 `_candidates_for_video()`(含悬空调用)。仅保留 `milvus_face_candidates()`(经 `_milvus_candidates_for_video()`)作为唯一在线 face 检索路径。删除后确认:`normalize` 仍在 `search.py` 其它处使用(L259,不构成未用 import);`face_confidence` 仍定义(L101)并被 `milvus_search.py` import。

**后续平台级收口**：Face builder 已改为内存直写 Milvus，不再生成 NPZ；写入行数校验通过后，由 Catalog publication 原子发布在线版本。历史 NPZ 只作为冷备，不存在运行时读取或应用内恢复入口。

---

### 6. 查询编码器 provider 修复(P2,`search.py` `_face()`)

**问题**:`_face()` 初始化 `FaceEncoder` 时 provider 硬编码 `"cpu"`、device_id 硬编码 `0`,忽略 `settings.face_provider`。查询端参考图编码无法用 NPU 加速。

**改动**(对齐索引侧 `stage_executor.py` L221-222 的 `face_provider` + `npu_device_id` 组合):
```python
self._face_encoder = FaceEncoder(
    self.settings.face_model,
    self.settings.face_provider,   # was "cpu"
    self.settings.npu_device_id,   # was 0
    str(self.settings.app_model_dir / "insightface"),
    self.settings.face_ort_intra_op_threads,
    self.settings.face_ort_inter_op_threads,
)
```
> **比计划更进一步**:计划仅提改 provider;但索引侧 provider 与 device_id 成对传入,若 provider=`cann` 而 device_id 仍 0 会用错 NPU。故一并接通 `npu_device_id` 保持查询/索引一致。默认 `face_provider="cpu"` 时行为不变。`FaceEncoder` 已对 `cann` 但 ORT 无 `CANNExecutionProvider` 时 fail-fast(`face.py` L46-50),不会静默回落 CPU。

---

## ⚖️ 关键决策与权衡

### 1. 信任 COSINE 距离(不再重打分)
归一化 float32 + COSINE 下重算是纯冗余。IVF_FLAT 是精确重排索引,P0-a 阶段(仍 IVF_FLAT)的 L2→cosine 是精确解析变换,输出与旧路径在 float32 精度内一致 → top-k 结果集完全相同。**低风险高确定性**(比 Speaker 的 HNSW 更干净可复现)。

### 2. DiskANN 面向千万级 · 512 维收益最大
IVF_FLAT 全内存,512 维千万级约 `1e7 × 512 × 4B ≈ 20 GB` 原始向量 + IVF 聚类结构常驻内存(五模态维度最大,代价最重)。DiskANN 落盘 + PQ 常驻,内存降 ~90%。**权衡**:小数据量下单查询延迟略高于内存 IVF_FLAT(多一次盘 IO),用可配 `search_list` 兜住;业务目标是千万级,内存是硬约束。

### 3. 不引入 BM25/hybrid
Face 查询是人脸 ArcFace 向量,无文本、无词面语义,BM25 无意义;schema 无 `text` 字段。

### 4. `ann_limit` 从 `limit*2` 收窄为 `limit*1`
重排取消后宽召回无意义。P0-b 的 `face_recall_multiplier` 默认 1,与 P0-a 的 `limit*2` 存在行为差异,故 **P0-b 与 P1 一并落地**规避过渡窗口(本次即一并落地)。

### 5. 迁移:一次性原地替换向量索引
正式环境使用 `scripts/migrate_face_diskann_index.py`，只将 `face_embeddings.embedding` 的旧索引原地替换为 `DISKANN/COSINE`。schema 与向量数据均未改变，因此不读取 NPZ、不删除 collection、不重新生成人脸向量；脚本会校验迁移前后行数一致。所有正式环境迁移完成后可删除该一次性脚本。

---

## ✅ 验证

### 代码修改清单
| 文件 | 改动 |
|------|------|
| `backend/app/vector_store/milvus/milvus_search.py` | 消除重打分(去 embedding 回传 + 删死分支);`_MODALITY_METRIC["face"]=COSINE`;`_STATIC_INDEX_TYPES["face"]=DISKANN`;cosine 简化为 `float(_distance)`;`_diskann_search_list_for` 新增 face 分支;threshold/multiplier 接线;删 `_IVF_NPROBE`+IVF_FLAT 分支;docstring + 全部 "face=IVF_FLAT" 注释更新 |
| `backend/app/vector_store/milvus/milvus_client.py` | `face_embeddings` IVF_FLAT/L2 → DISKANN/COSINE |
| `backend/app/core/settings.py` | 新增 3 项 face settings + 2 个 validator |
| `backend/app/retrieval/search.py` | 调用侧硬编码 `0.35`→`None`;删孤儿 `_face_candidates()`;删死方法 `_candidates_for_video()`;`_face()` provider `"cpu"`→`face_provider`、device_id `0`→`npu_device_id` |
| `backend/scripts/migrate_face_diskann_index.py` | 一次性将既有 Face 向量索引原地迁移为 `DISKANN/COSINE`，保留 collection、行数据与 Catalog publication |
| `.env.0829` | 新增 face 配置示例(= 默认值) |
| `backend/tests/test_milvus_search_metric.py` | 参数元组 `("face","L2","IVF_FLAT")` → `("face","COSINE","DISKANN")`;删 `test_l2_cosine_round_trip`;`test_face_candidates_l2_to_cosine_conversion` → `test_face_candidates_trusts_cosine_distance`(mock `.distance` 直接返回 cosine,无 embedding 依赖);模块 docstring 更新 |
| `backend/tests/test_speaker_index_verify.py` | `test_face_ivf_flat_not_judged_against_speaker_type` → 拆为 `test_face_diskann_collection_passes`(DISKANN 通过)+ `test_face_stale_ivf_flat_collection_fails_fast`(旧 IVF_FLAT fail-fast);模块 docstring 更新 |

### 原 PR 容器内实测(0829 镜像,本次轻量补丁前已通过)
在 `momentseek-0829-platform` 容器内(Python 3.11.6 + 真实 `pymilvus`/`pydantic`/`numpy`)执行,edited 源文件经 `docker cp` 注入后运行:

```bash
# 核心 face 迁移 + fail-fast 校验
python -m pytest tests/test_milvus_search_metric.py tests/test_speaker_index_verify.py -v
# → 16 passed(含 test_per_modality_metric_and_index[face-COSINE-DISKANN]、
#   test_face_candidates_trusts_cosine_distance、test_face_diskann_collection_passes、
#   test_face_stale_ivf_flat_collection_fails_fast)

# 回归:search / settings / speaker / batch / orchestration / timeout
python -m pytest tests/test_search.py tests/test_settings.py tests/test_speaker_candidates.py \
  tests/test_milvus_batch_search.py tests/test_retrieval_orchestration.py tests/test_milvus_query_timeout.py -q
# 原有 NPZ fallback 用例已删除；在线检索由 Milvus 契约测试覆盖
```
- 三个 validator 在容器内正确加载并拒绝非法值(经 settings 测试覆盖)。
- `search.py` 删除死代码后 `ast.parse` 语法检查通过。

### 单测覆盖(mock,无需运行 Milvus)
- `output_fields` 不含 `embedding`(重打分已消除)。
- face `Candidate.raw_score` == 信任的 Milvus COSINE `_distance`(无 L2→cosine 转换)。
- `_MODALITY_METRIC["face"]==COSINE`、`get_modality_index_type("face")==DISKANN`,与 `_COLLECTION_CONFIGS` 一致(动态一致性测试自动跟随)。
- face 走 DISKANN 分支;`_diskann_search_list_for("face")` 返回配置值不 raise。
- 索引漂移 fail-fast:Face 只接受 `DISKANN/COSINE`;旧 `IVF_FLAT/L2` 或 `DISKANN/L2` 均抛 `MilvusServiceError`。

### 等价性验证方法(落地环境执行,本阶段以功能正确为主)
- **重打分消除等价性——索引类型须保持不变**:同为 IVF_FLAT 下,对比"重打分开/关"两条路径,断言 `np.allclose(old_cosines, new_cosines, atol=1e-4)` 且 top-k `track_idx` 一致(IVF_FLAT 精确重排,最干净的等价性证明场景)。
- **不要**把 "IVF_FLAT+重算" vs "DiskANN+信任距离" 直接对比要求 top-k 恒等:DiskANN 近似索引召回邻居集本就可能与 IVF_FLAT(精确)不同,恒等断言会产生非 bug 的失败。DiskANN 召回质量属效果测评,本阶段不做。

---

## 🩹 后续修复:Face 聚合分数兼容性(2026-08-11)

> **前端验证暴露的聚合 bug,非本次索引/度量迁移引入,但由迁移后召回分布变化放大。**

### 问题现象
前端人脸检索时:
1. 召回多为 10–20s 长片段,且片段内混入非目标人脸;
2. 片段显示分(如 99%)与其 evidence 明细不符——明细里出现 `[milvus] face cosine=-0.010 · confidence=0.4% · 低于阈值`,但该片段仍停留在高分区、未被划入"低于阈值"展示。

### 根因(单一根因,两处表现)
`search.py` `_should_merge()` 对 **face-only** 分组走到 `return near`,**只判定时间相邻(默认 gap≤2s),完全不校验分数**。而 OCR 有 `_ocr_scores_compatible`(比例/绝对差门槛)、visual 只并 overlap,唯独 face 无任何分数约束。于是时间上恰好邻近但 cosine 差距极大(如 0.72 vs −0.01)的**不同人脸 track 被并入同一组**。随后 `_fuse_candidate_groups()`:
- 显示分 `score = max(组内各 track)` → 取到最高分 track(99%);
- `above_threshold = any(item.above_threshold ...)` → 只要组内一条超阈值,整段判为超阈值,低分片段无法进入"低于阈值"区;
- `evidence = [组内每条 track]` → 明细混入低分非目标项。

三者叠加即为"分数 99% 却带 0.4% 低于阈值明细、且片段过长"。

> **为何迁移后才显现**:P0/P1 前 `ann_limit = limit*2` 宽召回 + IVF_FLAT 精确重排,截断后进入聚合的低分项较少;迁移为 `multiplier=1` + DiskANN 近似召回后,进入聚合的低分邻居分布变宽,把这个**既存**的聚合缺陷放大到可见。

### 修复(`backend/app/retrieval/search.py`)
1. **新增 `_face_scores_compatible(group, candidate)`** + 常量 `_FACE_MERGE_MAX_COSINE_DROP = 0.15`:
   - 以 face 的 `raw_score`(即 cosine)为准,做**对称带宽**判定:候选 cosine 须满足 `>= 组内最强 − 0.15` **且** `<= 组内最弱 + 0.15`,保证同组 track 属同一相似度层级——既防低分锚点吸入高分 track,也防高分组吸入低分 track。
   - `raw_score` 缺失时退化为 True(仅按时间),防御性兜底(face 理论恒有 cosine)。
   - 阈值取 cosine(非 confidence):`face_confidence` sigmoid 陡峭且非线性,同样 0.15 的 drop 在中心区/边缘区对应的 confidence 跨度悬殊,cosine 带宽更均匀可解释。
2. **`_should_merge` 新增 face-only 分支**:
   ```python
   if candidate.modality == "face" and group_modalities == {"face"}:
       return near and _face_scores_compatible(group, candidate)
   ```
   仅作用于纯 face 分组;混合模态(face+ocr/asr/visual)不走此分支,face 作为其他模态锚定时刻的辅助证据,行为不变。

### 设计权衡
- **只治 face-only,不动混合模态**:混合分组由文本/OCR 等锚定时刻,face 是补充证据,无需独立分数门槛;贸然加约束会削弱跨模态融合。
- **对齐 OCR 策略但参数独立**:OCR 用 confidence 比例(0.90)/绝对差(0.10)下界;face 用 cosine 对称带宽(0.15)。face 带宽略宽,因 face track 本身已是"同人连续出现"的聚合单元,且 cosine 分布比 OCR confidence 宽。
- **显示更诚实而非召回变差**:修复后低分非目标 track 独立成段并正确落入"低于阈值"区,不再"藏"在高分片段的 evidence 里。

### 验证
- **容器内逻辑验证**(`momentseek-0829-platform`,真实依赖,6 场景全绿):用户 bug 场景(0.72 vs −0.01)拒绝合并、同人相近(0.72 vs 0.68)合并、大跌拒绝/小跌接受、混合模态不校验 cosine、`raw_score=None` 兜底。
- **单测**:新增 `backend/tests/test_face_merge_fix.py`(11 例,含用户场景回归 `test_regression_user_bug_scenario` 与最近 OCR 组选择回归)。
- 已 `docker cp` 注入容器并 `docker restart momentseek-0829-platform`,`/api/health` 正常。

### 遗留与调优建议
- `_FACE_MERGE_MAX_COSINE_DROP = 0.15` 为初值,建议按真实数据统计"同人跨 track cosine 分布" vs "异人 cosine 分布"再定标(过小→同人片段碎裂;过大→异人误并)。
- 本修复假设**上游 track 本身可信**(同一 track 内均为同人,`faces.py` 追踪逻辑正确);若 track 构建把异人并入同一 track,则超出本修复范围。
- 相关文档:`FACE_BUG_DIAGNOSIS.md`(详细诊断)、`FACE_MERGE_FIX_SUMMARY.md`(修复总结)。

---

## 🔧 深度代码审核与修复(2026-08-11)

> **审核范围**: `backend/app/retrieval/search.py` 聚合与融合逻辑全面审核,基于前端测试反馈与独立 agent 代码审查。

### 审核发现

#### 问题1: 混合模态下非OCR候选只比对`groups[-1]`导致合并到错误的OCR组 【Critical】

**问题描述**:
- `_groups()` L721-724 对非OCR候选只与 `groups[-1]` 比对,但此时 `groups` 已被 `_groups_ocr_score_first` 按**分数**(非时间)顺序预填入OCR组
- Sweep-line算法的正确性前提(组按时间递增、候选按时间递增)被破坏

**实际场景**:
```
OCR组1: 5-7s  (seed分数0.95)
OCR组2: 10-12s (seed分数0.90)
Face候选: 6-7s

groups = [组1, 组2]  ← 按分数顺序,非时间顺序
_should_merge(groups[-1]=组2(10-12s), face(6-7s)):
  - near = 6 <= 12+2 = True  ← 单向判断,候选在组"之前"也判True
  → face被并入组2(10-12s),而非真正匹配的组1(5-7s)
```

**后果**:
1. 生成横跨6-12s的"虚长片段",中间7-10s完全无命中
2. 组1从此无法再获得任何非OCR候选(被永久锁死)
3. 融合后 `start_time=min(6,10)=6`、`end_time=max(7,12)=12`,时间边界错误

**修复** (L702-742):
1. **非OCR候选遍历所有组并选择时间间隔最小的组**:
   ```python
   for candidate in sorted(non_ocr_candidates, ...):
       target_group = min(
           (g for g in groups if _should_merge(g, candidate, gap, max_duration)),
           key=lambda g: _temporal_gap(g, candidate),
           default=None,
       )
       if target_group is not None:
           target_group.append(candidate)
       else:
           groups.append([candidate])
   ```

2. **双向间隙判断** (L569-573):
   ```python
   # 旧: near = candidate.start_time <= group_end + gap  ← 单向,候选在组前也为True
   # 新:
   gap_between = max(
       candidate.start_time - group_end,  # 候选在组后的间隙
       group_start - candidate.end_time,  # 候选在组前的间隙
       0.0,                                # 重叠时间隙为0
   )
   near = gap_between <= gap
   ```

3. **最终按时间排序** (L742):
   ```python
   return sorted(groups, key=lambda g: (g[0].video_id, min(item.start_time for item in g)))
   ```

#### 问题2: `_face_scores_compatible` 对 `raw_score=None` 的处理不对称 【Medium】

**问题描述**:
- L437-440 当**组内锚点** face 的 `raw_score=None` 时,`group_cosines=[]`,无论候选有没有 `raw_score` 都返回 False
- 与 docstring 承诺 "raw_score 缺失时退化为仅按时间合并,返回 True" 矛盾

**修复** (L418-455):
```python
# 检查组内是否有 face 命中
if not any(item.modality == "face" for item in group):
    return False

group_cosines = [...]
# 组内无可用 cosine 或候选无 cosine → 退化为纯时间合并
if not group_cosines or candidate.raw_score is None:
    return True
```

**权衡**: 组内锚点缺 cosine 时放开会丢掉"挡住低 cosine 非目标 face"的保护,但这只在理论边界发生(生产环境 face 恒有 cosine),且比当前"永远拒绝、与文档矛盾"更合理。

#### 问题3: `_should_merge_ocr_only` 与 `_ocr_scores_compatible` 是死代码 【Medium】

**问题描述**:
- `_should_merge_ocr_only` 仅在 `_should_merge` L557 被调用,条件是 `candidate.modality == "ocr" and group_modalities == {"ocr"}`
- 但 `_should_merge` 只在 `_groups` L722 的**非OCR候选**循环里调用,`candidate.modality` 永不为 `"ocr"`
- 该分支恒不触发,两个函数都是死代码。OCR合并已完全由 `_groups_ocr_score_first` 接管

**修复** (L470-546):
- 在两个函数的 docstring 添加标记: `【已被 _groups_ocr_score_first 替代,保留以备将来独立 OCR 合并场景】`

#### 问题4: 代码规范与可维护性问题 【Low】

**4a. 注释与常量值不符** (L408):
```python
# 修复前: _OCR_MERGE_MIN_SCORE_RATIO = 0.90  # 收紧至80%
# 修复后: _OCR_MERGE_MIN_SCORE_RATIO = 0.90  # 至少保留 90% 的最佳分数
```

**4b. 重复逻辑** - 抽取全局阈值 helper (L454-469):
```python
def _apply_global_threshold(candidates: list[Candidate], modality: str) -> None:
    """对指定模态的候选应用全局动态阈值。

    规则：
    - 阈值 = max(0.10, 全局最高分 * 0.3)
    - 低于阈值的候选标记 above_threshold=False 并在 evidence 添加 "· 低于阈值"
    """
    modality_candidates = [c for c in candidates if c.modality == modality]
    if not modality_candidates:
        return
    global_top_score = max(float(c.score) for c in modality_candidates)
    global_threshold = max(0.10, global_top_score * 0.3)
    for candidate in modality_candidates:
        candidate.above_threshold = float(candidate.score) >= global_threshold
        if not candidate.above_threshold and " · 低于阈值" not in (candidate.evidence or ""):
            candidate.evidence = (candidate.evidence or "") + " · 低于阈值"
```

调用侧 (L1466-1467):
```python
# 替代原 L1424-1448 的两个重复块
_apply_global_threshold(candidates, "ocr")
_apply_global_threshold(candidates, "asr")
```

**4c. 性能优化** - O(n²) → O(n) (L631-651):
```python
# 修复前: seed_idx = time_sorted.index(seed)  ← O(n) 查找在 O(n) 循环内 = O(n²)
# 修复后:
candidate_to_idx = {id(c): i for i, c in enumerate(time_sorted)}
seed_idx = candidate_to_idx[seed_id]
```

### 验证

**容器内逻辑验证** (2026-08-11):
```bash
# 双向间隙判断
✓ 候选在组后(gap=1s < 2s): True
✓ 候选在组前(gap=3s > 2s): False
✓ 候选在组前(gap=0.5s < 2s): True

# raw_score=None 兜底
✓ 组内无raw_score兜底: True

# 混合模态择优合并
测试场景: OCR组1(5-7s,分数0.95) OCR组2(10-12s,分数0.90) Face(6-7s)
结果:
  组1: ['ocr', 'face'] @ [(5.0, 7.0), (6.0, 7.0)]  ← Face正确合并到时间匹配的组1
  组2: ['ocr'] @ [(10.0, 12.0)]
✅ 混合模态择优合并验证通过
```

**部署状态**:
- 修改文件已复制到容器 `/app/backend/app/retrieval/search.py`
- 容器已重启 (`docker restart momentseek-0829-platform`)
- 健康检查通过: `http://127.0.0.1:8100/api/health` → `{"status":"ok"}`

### 代码修改清单

| 文件 | 改动 |
|------|------|
| `backend/app/retrieval/search.py` | 1. 新增 `_apply_global_threshold` helper 消除重复逻辑<br>2. `_face_scores_compatible` 修复 raw_score=None 不对称处理<br>3. `_ocr_scores_compatible` / `_should_merge_ocr_only` 标记为死代码<br>4. `_should_merge` 改用双向间隙判断<br>5. `_groups` 非OCR候选改为遍历所有组择优 + 最终按时间排序<br>6. `_groups_ocr_score_first` 预建索引映射避免 O(n²)<br>7. L408 注释修正为 "至少保留 90%"<br>8. L1466-1467 用 `_apply_global_threshold` 替代重复块 |

### 预期效果

1. **混合模态场景**: face/visual/asr 候选不再错误合并到时间不匹配的 OCR 组,片段时间边界正确
2. **边界防御性**: 组内/候选 raw_score=None 时行为与 docstring 一致,不再"永远拒绝"
3. **代码可维护性**: 死代码明确标记、重复逻辑抽取、O(n²) 优化、注释与常量值对齐

### 遗留建议

- 本次修复针对**聚合逻辑错误**。`_FACE_MERGE_MAX_COSINE_DROP = 0.15` 阈值调优仍需按真实数据统计。
- 依赖上游 `faces.py` track 构建质量(同一 track 内均为同人)。

---

## 📚 参考
- `docs/FACE_OPTIMIZATION_PLAN.md`(实施方案,已按真实代码逐项核实并修正 2 处事实错误)
- `docs/Milvus_optimization_plan.md`(方案3:消除重打分)
- `docs/SPEAKER_IMPLEMENTATION_RECORD.md`(同构前置模态,基础设施来源)
- `backend/app/vector_store/milvus/milvus_search.py` / `milvus_client.py`
- `backend/app/retrieval/search.py`
- `backend/app/indexing/modalities/face/faces.py`(构建链路,本轮未改)
- `backend/app/encoders/face.py`(编码器,查询侧 provider 已接通)
