# Speaker 模态优化实施记录

**分支**: `feature/Speaker_optimize`
**日期**: 2026-08-06
**前置参考**: `SPEAKER_OPTIMIZATION_PLAN.md`(v2.2)、`Milvus_optimization_plan.md`(方案3)、`Visual_record.md`、`OCR_record.md`、`ASR_IMPLEMENTATION_RECORD.md`

---

## 📋 概述

### 目标
承接前三个模态(Visual/OCR/ASR)的优化经验,对 Speaker 声纹检索做两项核心改造:

1. **消除两阶段重打分**(`Milvus_optimization_plan.md` 方案3):信任 Milvus COSINE 距离,不再拉回 `embedding` 到客户端做 Python 侧 `np.dot` 重算。
2. **HNSW → DiskANN 迁移**:面向业务最终态的千万级 utterance 规模,把向量与图落盘、PQ 常驻内存,内存占用降 ~90%,与 Visual/OCR/ASR 统一。

辅以 **检索参数配置化(P1)** 与 **legacy 死代码清理(P1)**。

### 关键判断:借鉴什么、不借鉴什么
- ✅ **借鉴**:DiskANN + COSINE(visual 已证明可行)、消除重打分、参数下沉 settings、timeout/asset_version 隔离、mock 单测防回归。
- ❌ **不套用**:BM25 / `sparse_embedding` / analyzer / `hybrid_search`。Speaker 查询是**音频声纹向量**而非文本,词面检索无意义;schema 也无 `text` 字段。

### 影响面
`backend/app/retrieval/search.py` 完全不引用 speaker,Speaker 不走通用跨模态融合,仅经 `speaker_service` 对外服务。故本次优化影响面小、风险可控。

---

## 🏗️ 实施详情

### 1. 消除两阶段重打分(P0-a,`milvus_search.py`)

**改动函数**:`milvus_speaker_candidates()`

**改动前**(两阶段):
```python
ann_limit = min(limit * 2, 16_384)                    # 2倍扩召回(为重排服务)
hits = _ann_search(..., ann_limit,
    ["utterance_idx","start_ms","end_ms","track_id","asr_chunk_idx","embedding"])  # 拉回 embedding
for hit in hits:
    raw_emb = hit.get("embedding")
    if raw_emb is None:
        cosine = float(hit["_distance"])              # 已存在的"信任距离"分支
    else:
        utt_vec = normalize(raw_emb)
        cosine  = float(np.dot(query_norm, utt_vec))  # Python 侧重算
```

**改动后**(单阶段):
```python
ann_limit = min(limit * settings.speaker_recall_multiplier, 16_384)  # 默认 multiplier=1
hits = _ann_search(..., ann_limit,
    ["utterance_idx","start_ms","end_ms","track_id","asr_chunk_idx"])  # 不含 embedding
scored = [(float(hit["_distance"]), hit) for hit in hits]  # COSINE 距离即精确 cosine
```

**等价性依据**:
- Speaker embedding 写入前已 `_normalize` 为单位向量(`speaker.py` L143/L100),读回不影响。
- 归一化 float32 + COSINE 度量下,Milvus 返回的 `_distance` **就是精确 cosine**;HNSW/DiskANN 的近似性只影响"召回哪些邻居",不影响"已返回邻居的距离精度"。
- 原代码 `raw_emb is None` 分支本就直接信任 `_distance`,证明该路径本来正确。
- 收窄 `limit*2 → limit`:重排取消后宽召回无意义。**精确检索**下 ANN 已按距离排序,`2*limit` 取 top-limit 与 `limit` 取 top-limit 结果集相同;但需注意 DiskANN 是**近似**索引,不同 `search_list`/`limit` 理论上可能召回不同邻居集,故这里并非严格的 top-k 恒等,而是"重排取消 + multiplier 默认 1"的新设计决策(召回质量属效果测评,留待后续,见 §验证)。

**收益**:单次检索省去 `192×4B = 768 B × ann_limit` 向量传输 + Python 点积循环;在跨视频 fan-out(路径 A)中按视频数放大。

---

### 2. DiskANN 迁移(P0-b,`milvus_client.py` + `milvus_search.py`)

**2.1 索引配置**(`milvus_client.py` `_STATIC_INDEX_CONFIGS["speaker_embeddings"]`):
```python
"speaker_embeddings": {
    # Migrated HNSW → DISKANN for 千万级 scale. COSINE retained: speaker
    # embeddings are normalised unit vectors, and visual proved DiskANN
    # supports COSINE in this stack.
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
参数体例对齐 asr/ocr 的 DiskANN 配置。

**2.2 检索期支持 DiskANN**(`milvus_search.py`):
- `_STATIC_INDEX_TYPES["speaker"]`:`"HNSW"` → `"DISKANN"`。
- `_ann_search` 新增 DISKANN 分支;HNSW 分支保留并加注(当前无模态命中,备将来之需):
```python
if index_type == "DISKANN":
    # DiskANN hard constraint: search_list >= limit. `limit` here is the
    # caller's ann_limit; a static setting would fail/truncate when limit
    # exceeds it. Mirror visual v2's max(top_k, 100) pattern. search_list
    # setting is modality-keyed so this generic branch cannot misapply
    # speaker's tuning to another modality.
    search_list = max(limit, _diskann_search_list_for(modality))
    sp = {"metric_type": metric, "params": {"search_list": search_list}}
elif index_type == "IVF_FLAT":   # face 不受影响
    sp = {"metric_type": metric, "params": {"nprobe": _IVF_NPROBE}}
elif index_type == "HNSW":       # 当前不可达,保留 + 加注
    sp = {"metric_type": metric, "params": {"ef": _HNSW_EF}}
```

**⚠️ 关键约束 · `search_list >= limit`**:上层 `VoiceSearchRequest.limit` `le=200`(`schemas.py` L210)、上传路径 `min(200, limit)`(`speaker_routes.py` L108),经 `ann_limit = min(limit*multiplier, 16_384)` 后传入 `_ann_search` 的 `limit`(=ann_limit)最大可达 200(multiplier=1)。DiskANN 要求 `search_list >= 检索 limit`,固定 128 会在 limit>128 时失败/截断召回。故 **必须** `search_list = max(limit, setting)`,与 visual v2 的 `max(top_k, 100)` 同构。

**🔧 复审修正 · search_list setting 按模态键控**(避免潜在耦合):`_ann_search` 的 DISKANN 分支是 face/speaker 共用的**通用**路径,早前直接硬取 `get_settings().speaker_diskann_search_list`。当前 face=IVF_FLAT 不会进此分支,是**潜在**而非现存 bug;但若将来 face 迁 DISKANN 会错用 speaker 参数。已抽出 `_diskann_search_list_for(modality)` helper 按模态返回 setting——speaker 命中 `speaker_diskann_search_list`,其它模态若未登记则显式抛 `MilvusServiceError`(强制将来新增 DISKANN 模态时登记自己的 setting,而非静默 fall through 到 speaker 值)。

**2.3 索引类型 fail-fast 校验**(`milvus_search.py` 新增 `_verify_ann_index_type_once`):
- `_init_collections` 对**已存在**的 collection 只 `load()`,不会因改配置而重建索引。若 collection 仍是旧 HNSW 而检索期期望 DISKANN,会静默失败。
- 新增**按模态键控**的一次性校验(缓存,首次检索才发 RPC):speaker→DISKANN、face→IVF_FLAT 各自比对**自己**的期望类型,不会误伤 face。
- **作用域说明(复审补记)**:凡经 `_ann_search` 的 ANN 模态都会被校验——不止 speaker,face 也随本轮 DiskANN 迁移一并纳入。face 期望 IVF_FLAT、线上也是 IVF_FLAT,因此**对正确构建的 collection 是安全网而非行为变更**(首次 face 检索多一次 `col.index()` RPC)。
- **瞬时 vs 结构性失败(复审修正 #5)**:早前对**所有** introspection 异常一律软通过 + 缓存,会让一次偶发 RPC 抖动**永久**关闭该模态漂移检测。已细分:
  - **结构性限制**(`AttributeError`/`TypeError`:轻量 mock/wrapper 根本不支持 introspection;或 index 缺失/`index_type` 非字符串)→ 软通过并**缓存**(永不恢复,避免每次检索重试刷日志)。
  - **瞬时失败**(RPC/timeout 等其它异常)→ 软通过但**不缓存**,下次检索**重试**,单次抖动不再永久致盲。

---

### 3. 检索参数配置化(P1,`settings.py`)

新增 3 项 speaker 检索 settings + validator:
```python
speaker_identity_threshold: float = 0.50  # CAM++ same-speaker cutoff
speaker_diskann_search_list: int = 128    # DiskANN search_list (dynamically raised to >= ann_limit)
speaker_recall_multiplier: int = 1        # ann_limit = limit * this (re-score removed → 1)
```
validator:`speaker_diskann_search_list`/`speaker_recall_multiplier` 须 > 0;`speaker_identity_threshold` 须 ∈ [-1.0, 1.0]。

**接线**:
- `milvus_speaker_candidates(threshold: float | None = None)`:为 None 时取 `speaker_identity_threshold`;**保持声纹检索显式传 `-1.0` 的语义不变**(取全部候选)。
- `_ann_search` DISKANN 分支经 `_diskann_search_list_for(modality)` 取 `max(limit, speaker_diskann_search_list)`(按模态键控,见 §2.2 复审修正);**face 分支不变**。
- `ann_limit = limit * speaker_recall_multiplier`。

`.env.0829` 同步新增示例:
```bash
# Speaker Voice Search Configuration (DiskANN + trusted COSINE distance)
SPEAKER_IDENTITY_THRESHOLD=0.50
SPEAKER_DISKANN_SEARCH_LIST=128
SPEAKER_RECALL_MULTIPLIER=1
```

---

### 4. Legacy 死代码清理(P1)

- **随 4.1 删除**:`milvus_speaker_candidates` 中 `raw_emb is None` 双分支、`normalize`+`np.dot` 重算块、`limit*2` 扩召回。
- **更新过时注释**:`_ann_search` 原 "DiskANN ... not supported" 注释改为准确的三分支描述。
- **HNSW 分支 + `_HNSW_EF`**:speaker→DISKANN、face=IVF_FLAT、visual HNSW 在独立 v2 函数后,`_ann_search` 已无模态命中 HNSW 分支。**保守保留 + 加注**("当前无模态使用,保留以备将来"),避免过度删改。
- **保留**(勿删):`save_speaker_index` + `SpeakerMilvusIndexer.upsert_from_npz` 离线恢复路径(对齐 OCR/ASR);不动索引构建链路。

---

## ⚖️ 关键决策与权衡

### 1. 信任 COSINE 距离(不再重打分)
归一化 float32 + COSINE 下重算是纯冗余,同索引类型下 top-k 结果集完全等价。方案3 实测误差 < 1e-6。**低风险高确定性**。

### 2. DiskANN 面向千万级
HNSW 全内存,192 维千万级约 `1e7 × 192 × 4B ≈ 7.6 GB` 原始向量 + 图结构常驻内存。DiskANN 落盘 + PQ 常驻,内存降 ~90%。**权衡**:小数据量下单查询延迟略高于内存 HNSW(多一次盘 IO),用可配 `search_list` 兜住;业务目标是千万级,内存是硬约束。

### 3. 不引入 BM25/hybrid
Speaker 查询是声纹向量,无文本、无词面语义,BM25 无意义。

### 4. `speaker_identity_threshold` 对热路径近乎装饰
声纹检索路径显式传 `threshold=-1.0`,threshold 只影响 `above_threshold`/`decision`/`evidence` 显示字段;而 `_voice_search_vectors_milvus` 并不消费这些字段(仅用 `score`/`features`/`unit_id`)。配置化仍保留以统一约定并支持将来直接调用场景。

### 5. 索引本次完成后手动重建
按用户确认,开发阶段不提供独立迁移脚本、不做存量原地迁移。改配置后 drop collection → `_init_collections` 以新 DISKANN 配置重建,或重跑索引。schema 字段不变 → 向量无需重新嵌入。**部署顺序**:改配置 + 重灌数据 **必须先于服务启用**,否则 fail-fast 校验会让 speaker 检索全部抛错。

---

## 🔀 两条特殊检索路径

### 路径 A — 声纹检索(热路径,已优化)
`speaker_service._voice_search_vectors_milvus` → `milvus_speaker_candidates` → `_ann_search`。
本次 P0 双改的收益集中处:索引 DiskANN、单 RPC 仅回元数据、直接用 `_distance`、召回收窄为 `limit`。

**未落地(按需 / P1)**:跨视频 fan-out 收敛(每视频一次 ANN → 分批 OR + 多查询单次 search)。受 `asset_version` 按视频独立约束,须用配对 OR 且探明表达式上限;批量合并时 `best_by_utterance` 键须改 `(video_id, utterance_idx)` 复合键(现按 `unit_id` 仅在 per-video 循环内安全)。**定为先 profiling 再决定,保留逐视频 fallback**,避免过度工程。

### 路径 B — 说话人面板(冷/管理路径,未改)
`speaker_service.video_speakers` → `_speaker_data_from_milvus`。走 `query()`(标量过滤)**不经 ANN 索引**,与 DiskANN 迁移**正交**——HNSW→DiskANN 不改变其向量读取成本。

**本轮不优化其聚合**:计算量按单视频 utterance 数有界(约几百行),不随 collection 千万级增长,不在临界路径上。**已定的正确设计(择期,§4.6)**:构建期 `save_speaker_index` 已算好 `track_embeddings`/`representatives`,在线却每次重算;正解是持久化 track 级聚合 + 读时叠加 overlay,把读/算从 O(utterances) 降到 O(tracks)。但是优化效果不大，而且储存增加，可能做成负优化。

### 真正的千万级瓶颈(跨模态,§4.7)
路径 A/B 都靠 `video_id == X and asset_version == Y` 过滤,而 `video_id` 在 `_common_fields()` 是普通 VARCHAR,**无 partition key / 无标量索引**。千万级下这是真正的规模依赖,远大于路径 B 的 Python 重算。属 `_common_fields()` 层全模态 schema 决策,**应上升到 `Milvus_optimization_plan.md`**,不在本 speaker 计划落地。

---

## ✅ 验证

### 代码修改清单
| 文件 | 改动 |
|------|------|
| `backend/app/vector_store/milvus/milvus_search.py` | 消除重打分;`_STATIC_INDEX_TYPES["speaker"]=DISKANN`;`_ann_search` DISKANN 分支;新增 `_verify_ann_index_type_once` 按模态 fail-fast;threshold/multiplier/search_list 接线;死代码清理 + 注释更新<br>**复审补丁**:抽出 `_diskann_search_list_for(modality)` 按模态键控 search_list(#2);`_verify_ann_index_type_once` 细分结构性(缓存)/瞬时(重试不缓存)失败(#5)+ 补 face 作用域说明(#3) |
| `backend/app/vector_store/milvus/milvus_client.py` | `speaker_embeddings` HNSW→DISKANN(保 COSINE) |
| `backend/app/core/settings.py` | 新增 3 项 speaker settings + 2 个 validator |
| `.env.0829` | 新增 speaker 配置示例 |
| `backend/tests/test_milvus_search_metric.py` | L81 断言 `HNSW → DISKANN` |
| `backend/tests/test_milvus_query_timeout.py` | 新增 speaker timeout 用例 + mock `index()` DISKANN |
| `backend/tests/test_speaker_candidates.py` | **新增**:output_fields 不含 embedding;search_list=max(ann_limit,setting);score==_distance 且排序;threshold=None/-1.0 语义 |
| `backend/tests/test_speaker_index_verify.py` | **新增**:speaker 旧 HNSW collection fail-fast;DISKANN 通过;face IVF_FLAT 不被误判 |
| `backend/tests/test_speaker_service.py` | 更新 `test_voice_search_matches_individual_utterances` 断言以匹配"信任距离"契约(收紧为 `video_id=="b"` + `score==approx(0.99)`,详见下方验证章节);**新增** `test_voice_search_skips_videos_without_published_speaker` 回归测试(见"鲁棒性修复"章节) |
| `backend/app/identity/speaker_service.py` | **鲁棒性修复**:跨视频搜索循环对无 speaker 发布版本的视频调用 `_published_asset_version` 时 try/except 捕获 `SpeakerMilvusCoverageError` 并 continue,而非令整个请求 503 |

### 单测覆盖(mock,无需运行 Milvus)
- `output_fields` 不含 `embedding`(重打分已消除)。
- DISKANN `search_list == max(ann_limit, setting)`(硬约束防回归)。
- `Candidate.score` == 信任的 Milvus `_distance`,结果按 cosine 降序。
- `threshold=None` 取配置默认;`threshold=-1.0` 保留全部候选(声纹检索语义)。
- timeout 转发到 `search()`。
- 索引漂移 fail-fast:speaker 期望 DISKANN、实际 HNSW → 抛 `MilvusServiceError`;face IVF_FLAT 用自身期望类型校验、不被误判。

### 容器内实测(0829 镜像,已通过)
在 `momentseek-0829-platform` 容器内(Python 3.11.6 + 真实 `pymilvus`/`pydantic`/`numpy` + Milvus v2.6.20)执行,**51 个测试全部通过**:
```bash
# 第一轮(优化实施后,鲁棒性修复前):50 passed
# 第二轮(鲁棒性修复 + 回归测试新增后):
docker cp backend/app/identity/speaker_service.py momentseek-0829-platform:/app/backend/app/identity/speaker_service.py
docker cp backend/tests/test_speaker_service.py momentseek-0829-platform:/app/backend/tests/test_speaker_service.py
python -m pytest tests/ -q
# → 51 passed in 1.98s
```
- 设置在容器内正常加载(threshold 0.5 / search_list 128 / mult 1),三个 validator 均正确拒绝非法值。

### 鲁棒性修复:无 speaker 数据视频不中断跨视频搜索

**现象**:重建索引后"搜索同声"报 503 `Milvus speaker version is not published for video a06335ada8a448998e5ba85231c86d3e`。

**根因**:该视频（0 个 utterance,无语音片段）从未向 Milvus 发布 speaker 资产。前端确实不会为该视频显示 speaker 面板或"搜索同声"按钮,但它仍是 catalog 成员。`_voice_search_vectors_milvus` 的跨视频循环对每一个 catalog 视频无条件调用 `_published_asset_version`,遇到无发布版本的视频便抛出 `SpeakerMilvusCoverageError`,被上层 translate 为 503——整个搜索请求因一个无关视频而失败。这是**优化前已存在的鲁棒性缺陷**,与本轮 DiskANN/重打分消除优化无关。

**修复**(`backend/app/identity/speaker_service.py` `_voice_search_vectors_milvus`,L438 附近):
```python
try:
    speaker_version = _published_asset_version(video_id, "speaker")
except SpeakerMilvusCoverageError:
    continue  # skip videos without published speaker (e.g. no-speech video)
```

**为何安全**:查询源视频始终有 speaker 数据(否则前端不展示"搜索同声"入口),因此 query 向量的提取路径(`speaker_utterance_embedding`)不受此 continue 影响。无 speaker 版本的视频本就无法提供任何 speaker 命中,跳过它们只会使结果更准确(不是少算)。

**回归测试**(`backend/tests/test_speaker_service.py::test_voice_search_skips_videos_without_published_speaker`):
- catalog 含三个视频:a(query 源)、b(有 speaker)、c(无 speech,无已发布版本);
- `_published_asset_version` 对 c 抛 `SpeakerMilvusCoverageError`;
- Milvus search mock 对 c 调用会 `AssertionError`,若修复失效则立即暴露;
- 断言:`voice_search` 不抛异常;返回结果不含 c;b 的命中在列。

#### 修正的一个既有测试(契约变更,非 bug)
`test_speaker_service.py::test_voice_search_matches_individual_utterances` 初次运行失败(`assert 0.99 > 0.99`)。根因:该用例的 mock 里 video b 的 hit 设 `distance=0.99` 但 `embedding=[0.99,0.01]` **两者不自洽**——
- 旧代码拉回 embedding 重算 → `0.99/√(0.99²+0.01²) ≈ 0.99995 > 0.99` 通过;
- 新代码信任 Milvus `_distance=0.99` → 失败。

真实 Milvus 下归一化向量的 COSINE `_distance` 本就是精确 cosine,0.99995 纯属 mock 数据不自洽的假象。断言已**收紧**(非放宽)为 `hits[0]["video_id"] == "b"` + `score == pytest.approx(0.99)` 并加注释说明契约变更。修复后全绿。

### 🔍 代码复审跟进(2026-08-06,深入审查后修正)

一轮深入代码审查后,针对发现的次要问题逐条修正(核心 P0/P1 逻辑无错漏):

1. **[已修·代码] `_ann_search` DISKANN 分支硬取 speaker setting 的潜在耦合**:该分支 face/speaker 共用,早前直接 `get_settings().speaker_diskann_search_list`。抽出 `_diskann_search_list_for(modality)` helper 按模态返回;未登记的模态显式抛 `MilvusServiceError`,强制将来新增 DISKANN 模态登记自己的 setting,杜绝静默 fall through 到 speaker 值。(详见 §2.2 复审修正)
2. **[已修·文档+注释] face 也被纳入 fail-fast 的作用域说明**:`_verify_ann_index_type_once` 现覆盖所有经 `_ann_search` 的 ANN 模态(speaker+face)。对 face 是安全网(IVF_FLAT 配置匹配 IVF_FLAT 线上索引 → 通过),非行为变更。已在 docstring 与 §2.3 明确记录。
3. **[已修·代码] 瞬时 introspection 失败永久致盲漂移检测**:早前对所有异常一律软通过 + 缓存。已细分——结构性限制(`AttributeError`/`TypeError`、index 缺失/非字符串)缓存以免刷日志;瞬时 RPC/timeout **不缓存**,下次检索重试。(详见 §2.3 复审修正 #5)
4. **[已修·文档] 等价性措辞对近似索引收紧**:`limit*2 → limit` 的"结果集完全相同"仅对 exact 检索严格成立;近似 ANN 下不同 `search_list` 可能召回不同邻居集。已改为"multiplier 默认 1 是重排取消后的新设计决策",不再以恒等为论据。(详见 §1 等价性依据)
5. **[已修·git] `speaker_service.py` 鲁棒性修复此前未 `git add`**:复审时 `git status` 显示 ` M`(工作区未暂存),已一并 stage,避免漏提。

**复审后回归**:上述代码改动(`_diskann_search_list_for` helper + `_verify_ann_index_type_once` 瞬时/结构性分流)在 `momentseek-0829-platform` 容器内重跑,speaker 相关 **47 passed**,含相关模块 **52 passed**,无回归。

### 语法校验
所有改动文件通过 `ast.parse` 语法检查。

### 等价性验证方法(落地环境执行)
- **重打分消除的等价性——索引类型须保持不变**:同为 HNSW/或同为 DISKANN 下,对比"重打分开/关"两条路径,断言 `np.allclose(old_cosines, new_cosines, atol=1e-4)` 且 top-k `unit_id` 一致。
- **不要**把 "HNSW+重算" vs "DiskANN+信任距离" 直接对比要求 top-k 恒等:两种近似索引召回的邻居集本就可能不同。DiskANN 召回质量属效果测评,本阶段只保功能正确。

---

## 🚀 后续(记录在案,本轮不落地)

1. **路径 A fan-out 收敛(P1,按 profiling)**:分批 OR + 多查询单次 search,RPC 从 O(N视频×Q) 降到 O(N视频/批)。前置:探测 Milvus 表达式上限定批大小(`backend/scripts/`);`best_by_utterance` 改 `(video_id, utterance_idx)` 复合键;保留逐视频 fallback。
2. **路径 B 持久化 track 聚合(§4.6)**:构建期 track 聚合落 track 级存储,读时叠加 overlay,面板读/算 O(utterances)→O(tracks)。**触碰构建链路**,与"数据后续重建索引"节奏契合。持久化 auto-track 版本,仅对 `corrected_track_id` 改派的 track 现算。
3. **`video_id` partition key / 标量索引(§4.7,跨模态)**:千万级下路径 A/B 的真正规模依赖,`_common_fields()` 层全模态决策,**上升到 `Milvus_optimization_plan.md`** 统一规划。

---

## 📚 参考
- `docs/SPEAKER_OPTIMIZATION_PLAN.md`(v2.2 实施方案)
- `docs/Milvus_optimization_plan.md`(方案3:消除重打分)
- `docs/ASR_IMPLEMENTATION_RECORD.md` / `docs/OCR_record.md` / `docs/Visual_record.md`(前三模态经验)
- `backend/app/vector_store/milvus/milvus_search.py` / `milvus_client.py`
- `backend/app/identity/speaker_service.py`
- `backend/app/indexing/modalities/speaker/speaker.py`(构建链路,本轮未改)
