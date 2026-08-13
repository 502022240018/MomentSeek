# Speaker 模态优化实施计划

**版本**: 2.2
**日期**: 2026-08-06
**分支**: `feature/Speaker_optimize`
**作者**: Claude (Kiro)
**前置参考**: `Milvus_optimization_plan.md`(方案3)、`Visual_record.md`、`OCR_record.md`、`ASR_IMPLEMENTATION_RECORD.md`、`OCR_ASR_PLAN_CRITICAL_ERRORS.md`

> **v2.2 变更(依用户追问:路径 B 是否与千万级初衷相悖、检索侧能否迁进 Milvus)**:
> 1. **§4.6 重写为正式优化方案**:路径 B 每次请求成本按"单视频 utterance 数"有界、不随 collection 千万级增长,故本轮延后是安全的(理由改为"计算按单视频有界",非"冷路径可忽略")。但明确其正确的规模化设计——**构建期已算好的 track 聚合(`save_speaker_index`)持久化到 track 级存储,读时叠加 overlay**,把面板读/算从 O(utterances) 降到 O(tracks);触碰构建链路,独立于 P0/P1 择期落地。
> 2. **§4.7 新增·跨模态千万级真正依赖**:`video_id` 是无 partition key / 无标量索引的普通 VARCHAR,路径 A、B 都靠它过滤;千万级下这才是真正瓶颈,远大于路径 B 的 Python 重算。点名为跨模态 schema 决策,应上升到 `Milvus_optimization_plan.md`,本计划不落地。
> 3. 澄清"检索侧迁进 Milvus":取数已在 Milvus(`query()`);留在 Python 的是聚合(GROUP-BY/质心/argmax),Milvus 无服务端聚合,不能下推——正解是持久化预计算而非"塞进 Milvus"。同步更新 §5.2/§6/§9/结论。

> **v2.1 变更(计划复核后修正,均已对照真实代码核实)**:
> 1. **【纠错·必改】DiskANN `search_list` 必须动态取 `max(limit, setting)`**:`VoiceSearchRequest.limit` 上限 200(`schemas.py` L210)、上传路径 `min(200, limit)`(`speaker_routes.py` L108),经 `ann_limit=min(limit*2,16_384)` 后最大达 400;固定 `search_list=128` 违反 DiskANN `search_list>=limit` 硬约束,`limit>64` 时报错/截断。已改 §4.2/§4.3(对齐 visual `max(top_k,100)`)。
> 2. **【纠错】更新既有回归用例**:`test_milvus_search_metric.py` L81 硬编码 `("speaker","COSINE","HNSW")`,迁移后须改为 `DISKANN`,否则失败。已入 §6。
> 3. **【纠错】§4.5 fan-out 键冲突**:`best_by_utterance` 现按 `unit_id` 键控,仅在 per-video 循环内安全;批量跨视频合并须改用 `(video_id, unit_id)` 复合键并靠 `video_id` 输出字段解复用,`milvus_speaker_candidates` 不能原样复用。已入 §4.5。
> 4. **【纠错】路径 B 与 DiskANN 正交**:`video_speakers` 走 `query()`(标量取字段),**不经 ANN 索引**,DiskANN 迁移对其既无收益也无代价。已修正 §5.2/§7/§8/结论中"DiskANN 拖慢路径 B"的错误表述。
> 5. **【补充】死代码清理**:迁移后 `_HNSW_EF` 与 `_ann_search` 的 HNSW 分支不再可达(face=IVF_FLAT、speaker=DISKANN),列入 §4.4。
> 6. **【决策·开发阶段】** 不提供独立重建脚本、不做存量原地迁移,改配置后直接重灌索引数据;效果测评(DiskANN 召回质量 A/B)留待后续,本阶段验证只保证功能正确(§6/§7)。

> **v2.0 变更(依据用户补充要点)**:
> 1. Face 未迁 DiskANN 仅因尚未优化,**不是**架构策略。最终业务需千万级向量检索,**Speaker 应迁移 DiskANN**(Face 同理,但不在本计划范围)。已删除 v1.0 中"Speaker 应保留 HNSW"的错误论证。
> 2. 新增 legacy 残余代码清理项,与优化后的 Visual/OCR/ASR 保持一致。
> 3. 新增 §5 专章:两条特殊检索路径的**优化前/后差异**与**思路适宜性评定**。

---

## 0. 本计划的定位与"防错"声明

前三个模态优化最大的教训来自 `OCR_ASR_PLAN_CRITICAL_ERRORS.md`:**初版方案凭想象写字段名/文件路径,导致 35+ 处 `dense_embedding`、错误的 `processors/` 路径、误删 `has_embedding`**。本计划中所有文件路径、函数名、字段名、行为描述,均已逐一对照当前 `feature/Speaker_optimize` 分支的真实代码核实。凡本计划提及的符号,均可在下述文件中直接找到。

Speaker 与文本模态(ASR/OCR)在检索本质上不同,因此**部分借鉴、部分不套用**:
- ✅ **借鉴**:DiskANN 磁盘索引(面向千万级规模)、消除重打分(方案3)、参数配置化、legacy 清理、timeout/asset_version 隔离。
- ❌ **不套用**:BM25 / `sparse_embedding` / analyzer / hybrid_search —— Speaker 查询是**音频声纹向量**而非文本,词面检索无意义(见 §2)。

---

## 1. Speaker 模态现状(已核实)

### 1.1 涉及文件清单(真实路径)

| 职责 | 文件 | 关键符号 |
|------|------|---------|
| 索引调度 | `backend/app/indexing/stage_executor.py` | `_run_speaker()` (L291-317),依附 ASR |
| 索引构建 | `backend/app/indexing/modalities/speaker/speaker.py` | `build_speaker_index()`、`_adaptive_turn_units()`、`save_speaker_index()`、`load_speaker_index()`、`encode_voice_query()` |
| CAM++ 运行时 | `backend/app/indexing/modalities/speaker/speaker_3dspeaker_runtime.py` | 3D-Speaker 封装 |
| Schema | `backend/app/vector_store/milvus/milvus_schema.py` | `create_speaker_schema()` (L267-276)、`speaker_pk()` (L97) |
| 索引写入 | `backend/app/vector_store/milvus/milvus_indexer.py` | `SpeakerMilvusIndexer` (L506-552)、`upsert_from_npz` (L541,恢复路径) |
| Collection 配置 | `backend/app/vector_store/milvus/milvus_client.py` | `speaker_embeddings` 配置 (L85-89, L165-168) |
| **在线检索** | `backend/app/vector_store/milvus/milvus_search.py` | `milvus_speaker_candidates()` (L765-833)、`_ann_search()` (L234-283) |
| 查询/服务层 | `backend/app/identity/speaker_service.py` | `voice_search_vectors()`、`_voice_search_vectors_milvus()`、`video_speakers()`、`_speaker_data_from_milvus()` |
| API | `backend/app/api/speaker_routes.py` | `/api/voice-search`、`/api/voice-search/upload`、`/api/videos/{id}/speakers` |
| 设置 | `backend/app/core/settings.py` | `speaker_device`/`speaker_model_repo`/`speaker_model_cache_dir` (L111-113) — **无任何检索调优项** |
| 测试 | `backend/tests/test_speaker_index.py`、`test_speaker_no_speech.py`、`test_speaker_service.py` | |

> `backend/app/retrieval/search.py` **完全不引用 speaker**。Speaker 不走通用跨模态融合路径,只经 `speaker_service` 的声纹检索对外服务。这决定了本优化的**影响面小**,风险可控。

### 1.2 Schema(真实字段,勿臆造)

`create_speaker_schema()` = `_common_fields()` + speaker 专有字段:

```python
# _common_fields() 提供: pk, video_id, asset_version, model_version
FieldSchema("utterance_idx",  DataType.INT64),
FieldSchema("start_ms",       DataType.INT64),
FieldSchema("end_ms",         DataType.INT64),
FieldSchema("asr_chunk_idx",  DataType.INT64),   # 关联 ASR 分段
FieldSchema("track_id",       DataType.INT64),   # 自动聚类得到的说话人轨道
FieldSchema("embedding",      DataType.FLOAT_VECTOR, dim=192),  # 3D-Speaker CAM++
```

- **192 维**(CAM++,`3dspeaker-campplus-zh-en-192-v1`),是所有模态里维度最小的。
- **没有 `text`、`sparse_embedding`、`has_embedding` 字段。**
- 写入 Milvus 前 embedding 已 `_normalize()` 为单位向量(`speaker.py` L327、L143)。**这是 DiskANN 迁移可行的关键前提**:归一化向量下 COSINE 与 IP 等价,且 Milvus 的 DiskANN 在本项目已被 visual 证明支持 COSINE。

### 1.3 索引配置(真实)

`speaker_embeddings`(`milvus_client.py` L85-89):
```python
{"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
```
检索期 `_ann_search` 用 `{"metric_type": "COSINE", "params": {"ef": 128}}`(`_HNSW_EF = 128`,硬编码)。
**且 `_ann_search` 当前显式拒绝 DISKANN**(L252-257,只支持 HNSW/IVF_FLAT)—— 这是 DiskANN 迁移必须一并修改的点。

### 1.4 在线检索(核心优化对象)

`milvus_speaker_candidates()`(`milvus_search.py` L765-833)当前逻辑:

```python
query_norm = normalize(query)
ann_limit  = min(limit * 2, 16_384)                    # ① 2倍扩召回
hits = _ann_search(..., ann_limit,
    ["utterance_idx","start_ms","end_ms","track_id","asr_chunk_idx","embedding"])  # ② 拉回 embedding
for hit in hits:
    raw_emb = hit.get("embedding")
    if raw_emb is None:
        cosine = float(hit["_distance"])               # ← 已存在"信任距离"分支
    else:
        utt_vec = normalize(raw_emb)
        cosine  = float(np.dot(query_norm, utt_vec))   # ③ Python 侧重算 cosine
scored.sort(...); scored[:limit]                       # ④ 重排后截断
```

**这正是 `Milvus_optimization_plan.md` 方案3 要消除的"两阶段重打分"。** 关键事实:
- Milvus COSINE 对**已归一化的 float32** 向量返回的 `_distance` **就是精确 cosine**(HNSW/DiskANN 的近似性只影响"召回哪些邻居",不影响"已返回邻居的距离精度")。
- 写入向量已归一化,Schema 为 `FLOAT_VECTOR`(float32,非 float16),Milvus 距离与 Python 重算值在 float32 精度内一致(方案3 实测误差 < 1e-6)。
- 代码里 `raw_emb is None` 分支**已经在直接信任 `_distance`**,证明这条路径本就正确。
- 因此 ②③④ 都是冗余开销:每次多传 `192×4 = 768 B/条 × ann_limit` 向量,并在 Python 侧做无谓点积。

### 1.5 索引对 ASR 的依附关系(核实)

- `_run_speaker` 仅当请求 option `asr_speaker_enabled=True` 时,在 ASR stage 之后运行(`stage_executor.py` L286-287)。
- 在线构建强制要求**已发布的 ASR asset_version**(`_run_speaker` L301-302 抛错;`build_speaker_index` L225-226 抛错)。
- `build_speaker_index` 从 **Milvus ASR collection** 读取 `segment_idx/start_ms/end_ms/text`(L227-231),用于 `_adaptive_turn_units` 把说话人 turn 切到 ASR 边界。
- **含义**:ASR 优化已完成且这些字段保持不变,依附关系稳定。本计划**不改动索引构建链路**(schema 字段不变,DiskANN 迁移只改索引类型)。

### 1.6 NPZ 现状(与 OCR/ASR 对齐核实)

| 用途 | 状态 | 结论 |
|------|------|------|
| **在线读** speaker.npz | `_speaker_data_from_milvus` 是唯一在线数据源;传入的 `speaker.npz` 路径**仅用于 `.parent` 读 manifest,文件本身不被在线读取** | ✅ 已是 Milvus-only,与 OCR/ASR 删除 NPZ 在线 fallback 后的状态**一致** |
| **恢复写** speaker.npz | `save_speaker_index`(L308)写出,`SpeakerMilvusIndexer.upsert_from_npz`(L541)读回 | ✅ 与 OCR 保留 `rebuild_ocr_from_npz` + `upsert_from_npz` **一致**,应保留 |
| NPZ 中 `track_embeddings`/`track_representative_indices` | 仅被 `load_speaker_index` 消费,而 `load_speaker_index` **仅被测试引用**(`test_speaker_service.py`、`test_speaker_index.py`),恢复路径(indexer L544)只需 `utterance_*` 三数组 | ⚠️ 属"仅测试消费"的冗余;可选精简(§4.4),非必须 |

---

## 2. 借鉴什么、不借鉴什么(关键判断)

| 维度 | ASR/OCR | Speaker | 结论 |
|------|---------|---------|------|
| 查询输入 | 文本 | **音频声纹向量** | ❌ 无 BM25/hybrid:没有 text 字段,查询也不是文本 |
| 相似度 | dense(IP)+ sparse(BM25) | 纯向量 cosine | 只需单路向量检索 |
| 目标规模 | 亿级帧/分段 | **千万级 utterance(业务最终态)** | ✅ 需 DiskANN 降内存 |
| 现用索引 | DISKANN | HNSW(内存) | ✅ **迁移 DISKANN**(见 §4.2) |
| 重打分 | 已消除 | 仍在两阶段重打分 | ✅ **消除**(见 §4.1) |
| 参数配置化 | settings 下沉 | 全硬编码 | ✅ **配置化**(见 §4.3) |
| NPZ 在线 fallback | 已删 | 已是 Milvus-only | ✅ 已一致,清理残余(见 §4.4) |

**DiskANN + COSINE 在本项目已被证明可行**:`visual_embeddings` 使用 `DISKANN` + `metric_type: COSINE` + `search_list_size: 128`(`milvus_client.py` `_get_visual_index_config`),visual v2 检索用 `{"metric_type": "COSINE", "params": {"search_list": max(top_k, 100)}}`(`milvus_search_visual_v2.py`)。Speaker 迁移可**直接沿用 COSINE**,无需改度量、无需重算 embedding。

---

## 3. 已从三个模态吸取的教训 × Speaker 现状

| 教训(来源) | Speaker 现状 |
|------|------|
| 字段名/路径必须核对真实代码(`OCR_ASR_PLAN_CRITICAL_ERRORS`) | 本计划已逐一核实 |
| 每个 `search()` 必须传 `timeout`(`ASR_IMPLEMENTATION_RECORD` 决策5) | ✅ `_ann_search` L268 **已传** `timeout=milvus_query_timeout_seconds`;新增 fan-out 路径必须遵守 |
| 查询按 `asset_version` 隔离 | ✅ `_ann_search` expr L266 **已含** `asset_version` |
| 移除无效批量预取(`Visual_record` 2026-07-29) | ✅ `BULK_QUERY_FIELDS = {}`,speaker 从不批量预取 |
| `above_threshold` 只影响显示不影响聚合(`OCR_record` 问题3) | Speaker 不参与跨模态聚合,`above_threshold` 仅用于 evidence 文案,无此风险 |
| 删 collection 前检查存在性(`OCR_record` 问题4) | DiskANN 迁移涉及 drop index/collection,须复用 `utility.has_collection` 检查 |
| HNSW→DiskANN 非 in-place,需重建(`OCR_record`/`milvus_client` 校验钩子) | Speaker **仅索引类型变、schema 字段不变**,可 `drop_index → create_index` 原地重建,无需重嵌入(比 OCR/ASR 更轻) |
| 用 mock 断言防回归 | 本计划为每项改动配 mock 单测 |

---

## 4. 优化项(按优先级)

### 4.1 【P0】消除 `milvus_speaker_candidates` 两阶段重打分 + 清理死代码

**目标**:信任 Milvus COSINE 距离,不再拉回 `embedding`、不再 Python 重算;顺带删掉重打分死代码(与 ASR 删除 `lexical_score`/`_semantic_chunk_scores` 的清理一致)。

**改动**(仅 `milvus_search.py` `milvus_speaker_candidates`):

1. `output_fields` 去掉 `"embedding"`:
   ```python
   ["utterance_idx", "start_ms", "end_ms", "track_id", "asr_chunk_idx"]
   ```
2. 直接以距离为 cosine,**删除** `raw_emb`/`normalize`/`np.dot` 分支(不再保留 `if raw_emb is None` 的双分支——重打分取消后它是死代码):
   ```python
   for hit in hits:
       cosine = float(hit["_distance"])   # COSINE 度量下即精确 cosine
       scored.append((cosine, hit))
   ```
3. `ann_limit = min(limit * 2, 16_384)` 收窄为 `ann_limit = limit`(扩召回本为重排服务,重排取消后不需要)。
   - **保守落地**:第一步只删重打分、保留 `limit*2` 确保行为等价;A/B 通过后再收窄为 `limit`,避免一次改动叠加两个变量。
4. 其余(threshold、Candidate 构造、evidence、features)**保持不变**。

**行为等价性**:输出 cosine 与旧路径在 float32 精度内一致;排序/阈值/截断不变 → top-k 结果集应完全一致。

**收益**:单次检索省去 `768 B × ann_limit` 传输 + Python 点积循环;在 fan-out(路径 A)中按视频数放大,收益显著。

**风险**:极低。已归一化 float32 + COSINE,重算本就冗余;代码已有等价分支佐证。

---

### 4.2 【P0】DiskANN 迁移(面向千万级规模)

**依据**:业务最终态需千万级向量检索。HNSW 全内存,192 维千万级约 `1e7 × 192 × 4B ≈ 7.6 GB` 原始向量 + 图结构常驻内存,随规模线性增长;DiskANN 将向量与图落盘、仅 PQ 常驻内存,内存降 ~90%(与 visual/asr/ocr 一致)。

**改动**:

1. **索引配置**(`milvus_client.py`):`speaker_embeddings` 由 HNSW → DISKANN,**保留 COSINE**:
   ```python
   "speaker_embeddings": {
       "index_type": "DISKANN",
       "metric_type": "COSINE",          # visual 已证明 DiskANN 支持 COSINE
       "params": {
           "max_degree": 56,
           "search_list_size": 128,
           "pq_code_budget_gb": 0.125,
           "build_dram_budget_gb": 32.0,
       },
   }
   ```
   (参数体例对齐 asr/ocr 的 DiskANN 配置。)

2. **检索期支持 DiskANN**(`milvus_search.py`):
   - `_STATIC_INDEX_TYPES["speaker"]`:`"HNSW"` → `"DISKANN"`。
   - `_ann_search` 增加 DISKANN 分支(当前 L252-257 显式拒绝):
     ```python
     elif index_type == "DISKANN":
         # DiskANN 硬约束: search_list >= limit。limit 由调用方传入的 ann_limit
         # 决定(见下),不能取固定 setting,否则大 limit 时检索报错/截断。
         search_list = max(limit, _get_speaker_search_list())
         sp = {"metric_type": metric, "params": {"search_list": search_list}}
     ```
     `metric` 对 speaker 仍是 `"COSINE"`(`_MODALITY_METRIC` 不变)。Face 仍 IVF_FLAT,不受影响。
   - **【关键修正 · 必须动态取值】** `search_list` **不能**直接取静态 setting。`_ann_search` 收到的 `limit` 就是 `milvus_speaker_candidates` 传入的 `ann_limit`,而 `ann_limit = min(limit*2, 16_384)`,上层 `limit` 经 `VoiceSearchRequest.limit`(`le=200`,`schemas.py` L210)/ 上传路径 `min(200, limit)`(`speaker_routes.py` L108)后**最大可达 200 → ann_limit 最大 400**,远超默认 `search_list=128`。DiskANN 要求 `search_list >= 检索 limit`,固定 128 会在 `limit>64` 时直接失败或静默截断召回。**必须** `search_list = max(limit, settings.speaker_diskann_search_list)`,与 visual 的 `max(top_k, 100)` 同构(`milvus_search_visual_v2.py` 已验证)。

3. **迁移执行**(开发阶段,直接重建数据):
   - **本项目处于开发早期,不提供独立重建脚本、不做存量原地迁移。** 改配置后直接**重新建立索引数据**(重跑索引 / drop collection 后由 `_init_collections` 以新 DISKANN 配置重建)。
   - 说明为何需要显式重建:`_init_collections`(`milvus_client.py` L332-363)对**已存在**的 `speaker_embeddings` 只 `load()`,**不会**重建索引;仅改 `_STATIC_INDEX_CONFIGS` 只对全新 collection 生效。故存量数据须手动清掉重建——开发阶段直接重灌数据即可,无需迁移脚本。
   - schema 字段不变 → 重建时向量无需重新嵌入(若保留 collection 走 `drop_index → create_index(DISKANN) → load`;开发阶段更简单的是 drop collection 重灌)。

4. **索引类型校验**(与 visual `_verify_index_type` 一致):检索前一次性校验实际索引类型 == DISKANN,不匹配则 fail-fast(防止配置与线上索引漂移)。
   - ⚠️ **部署顺序含义**:一旦加了 fail-fast 校验,若 collection 仍是旧 HNSW,speaker 检索会**直接抛错**。因此**改配置 + 重建数据必须先于服务启用**。开发阶段按上一步重灌数据即可满足。

**与 4.1 的关系**:互补且应一起改。4.1 去掉 embedding 回传后,DiskANN 检索完全不必读盘上的完整向量(只用 PQ 距离 + 元数据),两者叠加收益最大。

**风险**:
- DiskANN 对**极小段**可能构建慢或有下限要求 → 开发环境视频少时先验证可建成、可检索(见 §6)。
- 冷路径(路径 B)会从盘读完整向量,略慢 —— 详见 §5.2 评定。

---

### 4.3 【P1】Speaker 检索参数配置化(消除硬编码)

**问题**:`threshold=0.50`(函数默认)、`_HNSW_EF=128`、`ann_limit` 倍数全硬编码;`settings.py` 无任何 speaker 检索项。三个模态优化均把关键参数下沉 settings。

**改动**(`backend/app/core/settings.py`,附 validator,参考 `validate_asr_positive`/区间校验):
```python
speaker_identity_threshold: float = 0.50   # CAM++ 同一说话人阈值(仅影响 above_threshold 显示)
speaker_diskann_search_list: int = 128     # DiskANN 检索期 search_list(须 >= limit)
speaker_recall_multiplier: int = 1         # ann_limit = limit * 该值(重排取消后默认 1)
```

**接线**:
- `milvus_speaker_candidates(... threshold: float | None = None ...)`:为 None 时取 `get_settings().speaker_identity_threshold`;**保持声纹检索显式传 `-1.0` 的用法与语义不变**(取全部候选、阈值在别处判定)。
- `_ann_search` 的 speaker 分支用 `max(limit, speaker_diskann_search_list)`(**不是**直接取 setting,见 §4.2 第 2 点的硬约束);**Face 分支保持不变**,避免误伤。
- `speaker_recall_multiplier` 用于 §4.1 第 3 步收窄:`ann_limit = limit * speaker_recall_multiplier`(默认 1)。

---

### 4.4 【P1】Legacy 残余清理(与 Visual/OCR/ASR 对齐)

**已一致、无需动**(核实,勿误删):
- 在线读已是 Milvus-only(`_speaker_data_from_milvus`);speaker.npz 在线不被读取。
- `save_speaker_index` + `upsert_from_npz` 恢复路径,对齐 OCR 的 `rebuild_ocr_from_npz`,**保留**。

**genuine 残余(建议清理)**:
1. **4.1 产生的重打分死代码**:`milvus_speaker_candidates` 中 `raw_emb is None` 双分支、`normalize`+`np.dot` 块、`limit*2` 扩召回注释 —— 随 4.1 一并删除。
2. **`_ann_search` 过时注释**(L253 "DiskANN and other types are not supported (face/speaker only)")—— 随 4.2 更新为准确描述。
3. **`_ann_search` 的 HNSW 分支与 `_HNSW_EF` 常量(L53、L249)**:speaker→DISKANN、face=IVF_FLAT 后,`_ann_search` 已无任何模态会命中 HNSW 分支(visual 的 HNSW 在独立的 `milvus_search_visual_v2.py`,不经 `_ann_search`)。该分支与 `_HNSW_EF` 变为**死代码**。**保守处理**:本轮可保留(无害),但需在注释中标注"当前无模态使用,保留以备将来";若清理则一并删除 `_HNSW_EF`(L53)与 L248-249 的 HNSW 分支。**建议保留 + 加注**,避免过度删改。
4. **(可选)NPZ 中 `track_embeddings`/`track_representative_indices`**:仅被 `load_speaker_index`(仅测试)消费,恢复路径不需要。可从 `save_speaker_index` 输出中移除以缩小恢复产物;但会牵动两个测试文件,**价值低,列为可选**,若做需同步更新测试。

**不做**:不删除 NPZ 写出与恢复路径(与 OCR/ASR 一致保留);不动索引构建链路。

---

### 4.5 【P1 / 按 profiling 决定】降低声纹检索跨视频 fan-out(路径 A)

**问题**:`_voice_search_vectors_milvus` 对目录中**每个视频**各发一次 ANN → RPC 数 O(视频数 × 查询声纹数)。视频规模上升后成为主导成本。详细前后对比见 §5.1。

**约束(已核实)**:`asset_version` **按视频独立**(每个视频 `milvus_meta.json` 内单调整数,不同视频取值不同)。故**不能**用单一 asset_version 跨视频检索;正确的跨视频过滤须为配对 OR:
```
(video_id == "v1" and asset_version == "3") or (video_id == "v2" and asset_version == "5") or ...
```

**方案(分批 OR 表达式 + 多查询向量单次 search)**:
1. 预取候选视频及其已发布 speaker `asset_version`(`_published_asset_version` 已逐视频提供)。
2. 视频按批(如 50–100/批)组合成 OR 表达式。
3. 每批一次 `col.search(data=[q1,q2,...], expr=<批 OR>, limit=...)`;Milvus 原生支持多查询向量,`results[i]` 对应第 i 条声纹。**每个 search 必须传 `timeout`**;`output_fields` **必须含 `video_id`**(批内混合多视频,需据此解复用)。
4. 合并各批各查询结果并叠加 overlay。

> **【关键修正 · 键必须带 video_id】** 现有 `_voice_search_vectors_milvus`(`speaker_service.py` L432-444)的 `best_by_utterance` 仅以 `candidate.unit_id`(=`utterance_idx`)为键,**这只在当前"逐视频循环内"成立**——因为循环体一次只处理一个 `video_id`。`utterance_idx` 是**每视频独立**编号(0..N),一旦批量把多视频命中合并到同一结果集,`unit_id` 会**跨视频碰撞**(v1 的 utterance 0 与 v2 的 utterance 0)。故批量路径:
> - `best_by_utterance` 键必须改为 **`(video_id, utterance_idx)`** 复合键;
> - 每条 hit 先按其 `video_id`(来自 `output_fields`)解复用,再取对应视频的 `catalog.speaker_overlays(video_id)` 叠加 `searchable`/`corrected_track_id`;
> - `milvus_speaker_candidates` 现签名固定单 `video_id` 并把它写进每个 `Candidate.video_id`,**不能原样复用于批量**——批量 search 需自行按 `(video_id, utterance_idx)` 重建候选,不能直接调它。计划早前"沿用现有 `best_by_utterance`"的表述已作废,以本条为准。

→ RPC 数从 O(N视频 × Q) 降到 O(ceil(N视频 / 批大小))。

**风险与缓解**:
- Milvus 表达式长度/解析有上限 → **先做小实验确定安全批大小**(`backend/scripts/` 探测脚本),再定 `speaker_voice_search_batch_size`(新增 settings)。
- 逻辑更复杂 → 保留旧逐视频路径作为 `expr` 超限 fallback,集成测试对比新旧结果集一致(Jaccard=1.0 期望,底层同一 ANN)。

**排序**:先做 4.1+4.2(每次 per-video RPC 已更轻),再依 profiling 决定 4.5 是否本轮落地,避免过度工程。

---

### 4.6 【P2 / 面向千万级的正确设计】说话人面板路径 B:持久化 track 聚合 + 读时叠加 overlay

**先厘清一个规模事实(纠正早前"冷路径所以无所谓"的措辞)**:路径 B 的**每次请求成本按"单视频 utterance 数"有界**(约几百行),**不随 collection 总量增长**。因此它**不在千万级的计算临界路径上**——哪怕整库到千万级,单次面板请求处理的向量数不变。所以"本轮不优化"是**安全**的,理由是"计算量按单视频有界",而非"冷路径可忽略"。

**但仍有一处真实的规模化改进空间**(与"承接千万级"初衷一致,故写入本节作为已定的正确设计,择期落地):

#### 现状的浪费
`_speaker_data_from_milvus`(`speaker_service.py` L196-207)**每次读都重算** track 质心 `track_embeddings` 与代表 `representatives`——它先全量取回该视频**所有** utterance 的 192 维向量(O(utterances)),再在 Python 侧按 `track_id` 分组算质心、选代表。**而这份聚合在构建期本就已算过**:`save_speaker_index`(`speaker.py` L161-162)已产出 `track_embeddings` / `track_representative_indices`。在线路径却弃之不用、重算一遍。

#### 为什么不能简单"推进 Milvus 内部"
- **取数**已经在 Milvus(`query()`)。留在 Python 的是**聚合**(group-by-track → 质心 → argmax 选代表 → 排序)。Milvus 是向量检索引擎,**无服务端 GROUP-BY / 质心 / argmax**,这部分计算无法像 SQL `GROUP BY` 那样下推。
- 正确方向是 **"构建期预计算 + 持久化 + 读时廉价取回"**,而不是"把聚合塞进 Milvus"。

#### 目标设计(持久化 track 级行)
1. **新增 track 级存储**:一个轻量 `speaker_tracks` collection(或在恢复产物中显式持久化),按 `(video_id, asset_version, track_id)` 存 `track_embedding`(192维,已归一化)、`representative_utterance_idx`、`utterance_count`、`duration_ms` 等面板所需聚合量。构建期由 `save_speaker_index` 已算好的数组直接写入,**无额外计算**。
2. **路径 B 改为**:`query(speaker_tracks, expr=video_id+asset_version)` 取回 **O(tracks)** 行(通常个位到十位数),而非 O(utterances) 条 192 维向量;面板预览所需的少量 utterance 再按 `representative_utterance_idx` 精确点取。读取量与计算量都**从"按 utterance"降到"按 track"**。

#### 必须正确处理的坑:overlay 不能预烘焙
用户可在读时通过 `corrected_track_id` **把 utterance 改派到别的 track**(`speaker_service.py` L249-251、L319),`_rank_speaker_utterances`(L276-288)据此在**当前(已改派)成员集**上重算质心/代表。因此:
- 持久化的 track 聚合是 **auto-track(未叠加 overlay)** 的版本;
- 读时:**无 overlay 改派的 track 直接用持久化值**;**仅对被 `corrected_track_id` 改动了成员的 track 现算**——退化时最多回到"取该 track 成员向量现算"(仍远小于全视频)。
- `display_name` / `hidden` / `representative_utterance_index` 这类 overlay 是纯展示叠加,不影响质心,读时直接盖。

#### 排序与理由
- **本轮不做**:符合开发阶段取向;且路径 B 计算按单视频有界,不阻塞千万级目标。
- **落地前置**:需要构建链路把 track 聚合写入新存储——**会触碰索引构建链路**(本轮明确不动,见 §6 末注),故独立于 P0/P1 排期。
- **落地收益**:面板读取从 O(utterances)×192维 降到 O(tracks);消除每请求的 Python 重算;与"数据后面重新建立索引"的开发节奏天然契合(重建时顺带产出 track 存储)。

> **真正决定路径 B(及路径 A)能否扛千万级的,不是上面的聚合优化,而是 §4.7 的 `video_id` 过滤可扩展性。** 聚合优化省的是"单视频内"的读/算;§4.7 省的是"从千万级里定位这一个视频的行"。两者互补,后者优先级更高但跨模态。

---

### 4.7 【跨模态 · 千万级真正依赖】`video_id` 标量过滤的可扩展性(点名,非本 speaker 计划落地)

**核实**:`video_id` 在 `_common_fields()`(`milvus_schema.py`)是普通 `DataType.VARCHAR`,**无 partition key、无标量索引**;`_init_collections` 只对 `embedding`(及 asr/ocr 的 `sparse_embedding`)建索引。

**影响**:speaker 的**两条路径都以 `video_id == X and asset_version == Y` 过滤**——路径 A 的 `_ann_search` expr(`milvus_search.py` L266)缩小 ANN 搜索域,路径 B 的 `query()`(`speaker_service.py` L76)定位单视频行。千万级下,若该过滤退化为全段扫描,**两条路径同时劣化**。这才是 speaker(乃至所有模态)承接千万级的**真正规模瓶颈**,远大于路径 B 的 Python 重算。

**方向(二选一或并用,需全模态统一评估)**:
- **`video_id` 作为 partition key**:Milvus 按 partition 物理隔离,过滤直接命中对应 partition,天然随视频数扩展;代价是 partition 数上限与建表期决策。
- **`video_id` 标量索引(INVERTED)**:保留单 collection,过滤走标量索引而非全扫。

**为何不在本 speaker 计划落地**:这是 `_common_fields()` 层面的**全模态 schema 决策**(visual/asr/ocr/face/speaker 共用),体量与影响面超出 speaker 单模态,且属**建表期不可原地变更**项(与 asr/ocr 的 `_validate_existing_*_collection` 同类,须 drop+rebuild)。**应上升到 `Milvus_optimization_plan.md` 统一规划**。本 speaker 计划仅**点名它是千万级的真正依赖**,并撤销早前把规模风险错误归因于"路径 B 向量读"的表述。

---

## 5. 两条特殊检索路径:优化前/后差异与思路评定(重点)

Speaker 的"特殊性"集中在两条与其他模态迥异的调用路径。以下逐路径给出**优化前实现 → 优化后预期 → 思路是否适宜**。

### 5.1 路径 A — 声纹检索(热路径)

**代码**:`speaker_service.py` `_voice_search_vectors_milvus` (L408-479) → `milvus_speaker_candidates` → `_ann_search`。

#### 优化前(现状)
```python
for video in catalog.list_videos():              # 目录中每个视频
    for query in queries:                         # 每条查询声纹
        milvus_speaker_candidates(client, video_id, query,
            _published_asset_version(video_id, "speaker"), limit, threshold=-1.0)
        # 内部: HNSW ANN(limit*2)→ 输出含 embedding(768B/条)→ Python 重算 cosine → 排序截断
```
- **索引**:HNSW(全内存)。
- **RPC 数**:`N_videos × Q`。
- **单 RPC 负载**:`limit*2` 条 ×(元数据 + 192维×4B embedding)。
- **CPU**:每条命中在 Python 侧 normalize + dot。

#### 优化后(4.1 + 4.2,叠加可选 4.5)
| 环节 | 优化前 | 优化后 |
|------|--------|--------|
| 索引 | HNSW 全内存 | **DiskANN**:向量/图落盘,内存 ~90%↓(千万级可行) |
| 单 RPC 输出字段 | 含 `embedding`(768B/条) | **仅元数据**,无向量回传 |
| 打分 | Python 重算 cosine | **直接用 Milvus `_distance`** |
| 扩召回 | `limit*2` | `limit`(§4.3 可配) |
| RPC 数(叠加 4.5) | `N_videos × Q` | **`ceil(N_videos / 批) `**,多查询单次 search |

#### 思路适宜性评定
- **消除重打分(4.1)——适宜,强推**:归一化 float32 + COSINE 下,`_distance` 即精确 cosine,重算是纯冗余;同索引类型下结果集 top-k 完全等价(HNSW held-fixed `allclose` 可证,见 §7)。在 fan-out 下,冗余传输/计算被视频数放大,收益被乘性放大 —— 这条路径正是收益最大处。
- **DiskANN(4.2)——面向目标规模适宜,需注意小数据延迟**:千万级下 HNSW 内存不可承受,DiskANN 是正解且与其他模态统一。**权衡**:小数据量下 DiskANN 单查询延迟略高于内存 HNSW(多一次盘 IO);但业务目标是千万级,内存约束是硬约束,延迟差可用 `search_list` 调优。结论:**适宜**,并用 §4.3 的可配 `search_list` 兜住延迟。
- **fan-out 批处理(4.5)——适宜但有前置条件**:RPC 从 O(N×Q) 降到 O(N/批),对大目录是必要优化;但受 `asset_version` 按视频独立约束,须用配对 OR 且探明表达式上限。**因此定为"先 profiling 再决定",并强制保留逐视频 fallback**,避免为不大的目录过度工程。

### 5.2 路径 B — 说话人面板(冷/管理路径)

**代码**:`speaker_service.py` `video_speakers` (L346-370) → `_load_speaker_data` → `_speaker_data_from_milvus` (L134-214)。

#### 优化前(现状)
```python
rows = _milvus_rows(video_id, "speaker", [... "embedding"])   # query_iterator 全量拉单视频所有 utterance 行(含 embedding)
# Python 侧:归一化 → 按 track_id 分组算 track 质心 track_embeddings → 选代表 representative → 返回面板
```
- 用 `query()`(非 `search()`)全量取一个视频的行。
- **每次请求都重算** track 质心与代表 utterance(不落库)。
- 单视频约 290 行 × 192 维 ≈ 55 KB,冷路径。

#### 优化后(预期)
- **4.1 不触及本路径**(4.1 只改候选检索 `milvus_speaker_candidates`,不改 `_speaker_data_from_milvus` 的 `query`)。
- **【修正 · 4.2 对本路径基本无影响】** 本路径用 `query()`(标量表达式过滤,`speaker_service.py` L95/L139),**不走 ANN 索引**——`query()` 按 `video_id+asset_version` 直接从 segment 存储读取字段(含 `embedding`),读取路径与索引类型(HNSW/DiskANN)**无关**。因此 HNSW→DiskANN **不会**改变本路径的向量读取成本。早前"DiskANN 拖慢路径 B 全量向量读"的判断有误,已更正。
- 量级小(~55 KB/视频)、冷路径,**预期可接受**。

#### 思路适宜性评定
- **本轮不优化本路径——适宜,但理由要精确**:可延后**不是**因为"冷路径可忽略",而是**计算量按单视频 utterance 数有界、不随 collection 千万级增长**,故不在千万级临界路径上。
- **DiskANN 对本路径无额外代价**:既然 `query()` 不经 ANN 索引,DiskANN 迁移在本路径**既无收益也无负向影响**。§6 仍可顺带记录面板延迟基线,但**不应**把它当作 DiskANN 的"代价"来观测。
- **仍存在与千万级初衷一致的正确优化**(见 §4.6):在线路径每次重算 track 质心/代表,而这份聚合**构建期已算好**(`save_speaker_index`)。正解是**持久化 track 级聚合 + 读时叠加 overlay**,把读/算从 O(utterances) 降到 O(tracks)——但这会触碰索引构建链路,独立于 P0/P1 排期,择期落地。
- **真正的千万级瓶颈在别处**(见 §4.7):路径 B 与路径 A 都靠 `video_id` 标量过滤,而 `video_id` 无 partition key / 标量索引。这是跨模态的真正规模依赖,优先级高于本路径的聚合优化,但超出 speaker 单模态范围。
- **结论**:路径 B 与 DiskANN 迁移正交;本轮不优化其聚合(计算按单视频有界),正确的持久化设计记于 §4.6,千万级的真正依赖记于 §4.7。

---

## 6. 实施步骤与顺序

1. **P0-a 重打分消除**(`milvus_search.py`)
   - 改 `milvus_speaker_candidates`:去 `embedding` 输出、直接用 `_distance`、删死分支;先保 `limit*2`。
   - 单测:mock `_ann_search`,断言 (a) `output_fields` **不含** `"embedding"`;(b) Candidate `score` == mock `_distance`;(c) 排序/截断/阈值不变。
2. **P0-b DiskANN 迁移**(`milvus_client.py` + `milvus_search.py`)
   - 配置 HNSW→DISKANN(保 COSINE);`_STATIC_INDEX_TYPES["speaker"]="DISKANN"`;`_ann_search` 加 DISKANN 分支(`search_list = max(limit, setting)`,见 §4.2);加索引类型校验。
   - **更新既有断言**:`test_milvus_search_metric.py` L81 现硬编码 `("speaker","COSINE","HNSW")`,须改为 `("speaker","COSINE","DISKANN")`,否则该回归用例失败。
   - 迁移(开发阶段):改配置后**直接重灌 speaker 索引数据**(drop collection → `_init_collections` 以新 DISKANN 配置重建,或重跑索引);不提供独立迁移脚本。开发环境先验证**可建成 + 可检索**。
   - 单测:speaker 走 DISKANN 分支时 `search_list` == `max(limit, setting)`;timeout 仍转发(补 `test_milvus_query_timeout.py` speaker 用例)。
3. **P1 配置化**(`settings.py` + `milvus_search.py`)
   - 新增 3 项 settings + validator;接线 threshold/search_list/multiplier;更新 `.env` 示例。
   - 单测:默认值加载;`threshold=-1.0` 语义不变;`search_list` 生效。
   - 完成后将 P0-a 的 `limit*2` 收窄为 `limit * speaker_recall_multiplier`。
4. **P1 legacy 清理**:随 P0 删死代码 + 更新过时注释;NPZ 恢复保留。
5. **P1 fan-out**(`speaker_service.py`,依 profiling)
   - 先写 expr 上限探测脚本,定批大小;实现分批 OR + 多查询(带 timeout);保留逐视频 fallback。
   - 集成测试:新旧结果集一致;overlay 正确。
6. **验证与文档**:`docs/SPEAKER_IMPLEMENTATION_RECORD.md`(对齐 ASR/OCR record 体例)。

**已定但本轮不落地(记录在案,择期推进)**:
- **路径 B 持久化 track 聚合(§4.6)**:构建期已算好的 `track_embeddings`/`representatives` 落到 track 级存储,读时叠加 overlay,把面板读/算从 O(utterances) 降到 O(tracks)。**触碰构建链路**,独立于 P0/P1;与"数据后续重建索引"的开发节奏契合。
- **`video_id` 标量过滤可扩展性(§4.7)**:partition key 或标量索引,是千万级的真正跨模态依赖,**上升到 `Milvus_optimization_plan.md` 统一规划**,不在本 speaker 计划落地。

> **构建链路(`speaker.py`/`stage_executor.py`/schema 字段/indexer 写入)本轮不改动**——已满足 timeout/asset_version 隔离,对 ASR 依附稳定,且 DiskANN 迁移不涉及 schema 字段变更。§4.6 落地时会触碰构建链路,故明确排在本轮之后。

---

## 7. 验证方案

**正确性(P0 重点,开发阶段以功能正确为主,效果测评留待后续)**:
- **重打分消除(4.1)的等价性验证——索引类型须保持不变**:同为 HNSW 下,对比"重打分开/关"两条路径,断言 `np.allclose(old_cosines, new_cosines, atol=1e-4)` 且 top-k `unit_id` 一致。4.1 是索引无关的改动(只改是否 Python 重算),这是它唯一严谨的等价性证明方式。
- **不要**把"HNSW+重算" vs "DiskANN+信任距离"直接对比并要求 top-k 恒等:HNSW 与 DiskANN 是不同近似索引,召回的邻居集本就可能不同,恒等断言会产生"非 bug 的失败"。DiskANN 的召回质量属效果测评范畴,本阶段不做。
- mock 单测防回归:`output_fields` 不含 `embedding`;timeout 仍转发;speaker 走 DISKANN 分支且 `search_list == max(limit, setting)`。

**DiskANN 迁移专项**:
- 索引类型校验通过(实际 == DISKANN)。
- 开发环境小数据可建成、可检索(验证无"段太小"构建问题)。
- (可选)面板路径 B 加载延迟基线:仅作观测记录;`query()` 不经 ANN 索引,DiskANN 迁移对本路径无影响(见 §5.2),不作为迁移代价评估。

**性能**:
- 路径 A 单次声纹检索:RPC 数、总传输、P50/P95(优化前后);内存占用(HNSW vs DiskANN)。
- fan-out 落地后:不同视频规模的 RPC 数/延迟曲线。

**回归**:
- `test_speaker_service.py`、`test_speaker_index.py`、`test_speaker_no_speech.py` 全绿(若动 NPZ 可选项,同步更新)。
- `/api/voice-search`、`/api/voice-search/upload`、`/api/videos/{id}/speakers` 端到端冒烟。

---

## 8. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 信任距离导致分数细微漂移 | 极低 | 低 | float32+归一化下等价;同索引类型下 `allclose` 验证(§7) |
| **DiskANN `search_list < limit` 致检索失败/截断** | **高(若取静态值)** | **高** | **`search_list = max(limit, setting)` 动态取值(§4.2);单测断言** |
| DiskANN 对小段构建失败/慢 | 中 | 中 | 开发环境先验证可建成;必要时先在大数据集验证 |
| fan-out 批处理 `best_by_utterance` 跨视频键碰撞 | 中 | 高 | 改 `(video_id, unit_id)` 复合键;保留逐视频 fallback(§4.5) |
| fan-out OR 表达式超 Milvus 上限 | 中 | 中 | 先探测批大小;保留逐视频 fallback |
| 误改 `threshold=-1.0` 声纹检索语义 | 低 | 高 | 显式保留传参路径;集成测试覆盖 |
| search_list 参数化误伤 face(共用 `_ann_search`) | 低 | 中 | 仅 speaker DISKANN 分支取新 setting,face IVF_FLAT 分支不变 |
| 未重灌数据即启用 fail-fast 校验致 speaker 检索全挂 | 中 | 高 | 改配置+重灌数据须先于服务启用(§4.2 部署顺序) |
| 误动构建链路破坏 ASR 依附 | 低 | 高 | 本轮不改 `speaker.py`/`stage_executor`/schema 字段 |

---

## 9. 与既有约定的一致性检查清单

- [ ] 所有新 `search()` 调用转发 `timeout`(现有已满足;fan-out 新路径必须遵守)
- [ ] 所有查询表达式含 `video_id` + `asset_version` 隔离
- [ ] 不新增 BM25/analyzer/sparse 字段(音频查询无意义)
- [ ] `speaker_embeddings` 迁移为 **DISKANN + COSINE**(对齐 visual 证明的组合)
- [ ] `_ann_search` 新增 DISKANN 分支,face(IVF_FLAT)不受影响
- [ ] **DiskANN 分支 `search_list = max(limit, setting)`(动态取值,非静态 setting)** —— 上层 limit 最大 200 → ann_limit 最大 400,静态 128 会违反 `search_list >= limit`
- [ ] **更新 `test_milvus_search_metric.py` L81 断言 `HNSW → DISKANN`**(否则既有回归用例失败)
- [ ] 消除 `milvus_speaker_candidates` 两阶段重打分并删死代码
- [ ] **随 speaker→DISKANN 清理 `_ann_search` 现已不可达的 HNSW 分支与 `_HNSW_EF` 常量**(face=IVF_FLAT、speaker=DISKANN、visual HNSW 在独立 v2 函数)
- [ ] 关键检索参数下沉 settings(threshold/search_list/multiplier)
- [ ] **若落地 fan-out(4.5):`best_by_utterance` 键改为 `(video_id, utterance_idx)`**,并从 search 结果的 `video_id` 输出字段解复用,避免跨视频 utterance_idx 碰撞
- [ ] 保留 NPZ 离线恢复(对齐 OCR/ASR);不删 `save_speaker_index`/`upsert_from_npz`
- [ ] 不改动 ASR 依附字段(`segment_idx/start_ms/end_ms/text`)与构建链路
- [ ] 每项改动配 mock 单测防回归
- [ ] (择期,非本轮)路径 B 持久化 track 级聚合 + 读时叠加 overlay,读/算从 O(utterances) 降到 O(tracks);持久化 auto-track 版本,仅对 `corrected_track_id` 改派的 track 现算(§4.6)
- [ ] (跨模态,上升到 `Milvus_optimization_plan.md`)`video_id` 加 partition key / 标量索引 —— 千万级下路径 A/B 的真正规模依赖,`_common_fields()` 层全模态决策(§4.7)

---

**结论**:Speaker 优化的核心是 **P0 双改**——(a) 消除两阶段重打分(方案3,低风险高确定性)、(b) 迁移 DiskANN + COSINE(面向千万级业务目标,对齐 Visual/OCR/ASR),两者互补应一起落地;辅以 **参数配置化(P1)**、**legacy 死代码清理(P1)** 与按需的 **声纹检索 fan-out 收敛(P1)**。**不**引入 BM25/hybrid(音频查询无意义)。

两条特殊路径中,路径 A(声纹检索)是收益最大处;路径 B(面板)走 `query()` 不经 ANN 索引,与 DiskANN 迁移正交,本轮不优化其聚合——但**面向千万级的正确设计已写入 §4.6**:构建期 track 聚合本已算好(`save_speaker_index`),在线却每次重算,正解是**持久化 track 级聚合 + 读时叠加 overlay**,把读/算从 O(utterances) 降到 O(tracks);因触碰构建链路,独立于 P0/P1 择期落地。而**真正决定两条路径能否扛千万级的是 §4.7**:`video_id` 无 partition key / 标量索引,过滤可扩展性是跨模态的真正规模依赖,优先级最高但超出 speaker 单模态,应上升到 `Milvus_optimization_plan.md`。
