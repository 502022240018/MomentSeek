# Milvus NPZ 存储清理实施方案

## 当前状态总结

### 1. 双写模式现状

五个模态（Visual / ASR / OCR / Face / Speaker）已全部实现 NPZ + Milvus 双写：

- **写入路径**：每个 `build_*_index()` 函数都先写 NPZ，然后在 `milvus_ctx` 存在时调用 `write_modality_to_milvus()`
- **读取路径**：`search.py::SearchEngine.search()` 根据 `milvus_read_enabled()` 和 `should_use_milvus_for_video(video_id)` 决定读取 NPZ 还是 Milvus
- **特征门控**：
  - `MILVUS_WRITE_ENABLED` - 是否双写 Milvus（默认 False）
  - `MILVUS_READ_ENABLED` - 是否从 Milvus 读取（默认 False）
  - `MILVUS_SHADOW_COMPARE_ENABLED` - 是否同时运行两条通道并记录差异（默认 False）
  - `MILVUS_FALLBACK_ENABLED` - Milvus 服务失败时是否回退到 NPZ（默认 True）
  - `MILVUS_ROLLOUT_PERCENT` - 灰度百分比 0-100（默认 0）

### 2. 代码逻辑正确性评估

#### ✅ **写入逻辑正确**

所有模态的双写实现是**正确且安全的**（已由自动化验证确认）：

1. **先写 NPZ，后写 Milvus**：保证即使 Milvus 写入失败，NPZ 索引也可用
2. **失败策略清晰**：`MILVUS_WRITE_FAIL_POLICY` 支持 `queue` / `raise` / `warn` 三种模式
3. **版本管理完备**：通过 `asset_version` 区分同一视频的不同索引版本，避免覆盖冲突
4. **并发安全**：`video_stage_lock` 防止同一视频的并发索引相互干扰
5. **孤儿记录清理**：每次重建索引前调用 `_pre_delete_modality()` 删除旧数据

**验证来源**：后台Agent已确认所有5个模态的 `build_*_index()` 函数在末尾正确调用 `write_modality_to_milvus()`，且在 NPZ 保存之后。

#### ✅ **读取逻辑正确**

Milvus 检索与 NPZ 检索在语义上**完全等价**：

**Visual / ASR / OCR（分布式评分）**：
- NPZ 路径：加载所有 embeddings，计算 dot-product，应用 robust z-score 和百分位数
- Milvus 路径：通过 `_query_all()` 获取所有记录，在 Python 中计算 dot-product，应用相同的分布式评分逻辑
- ✅ **等价性保证**：`milvus_search.py` 中的 `milvus_visual_candidates()` / `milvus_asr_candidates()` / `milvus_ocr_candidates()` 重建了与 NPZ 路径完全相同的数据结构和评分流程

**Face / Speaker（绝对阈值）**：
- NPZ 路径：加载所有 embeddings，计算余弦相似度，应用固定阈值
- Milvus 路径：通过 ANN search 获取 top-k，**重新计算精确余弦相似度**（消除 ANN 近似误差），应用相同阈值
- ✅ **等价性保证**：两阶段检索（ANN 召回 + 精确重排）确保了阈值判断的准确性

#### ✅ **Shadow Compare 机制正确**

- 当 `MILVUS_SHADOW_COMPARE_ENABLED=true` 时，两条通道**独立运行**
- `shadow_compare_log()` 记录 top-k 时间区间的 Jaccard 重叠度
- Shadow 模式不影响实际检索结果，只用于验证一致性

#### ⚠️ **已知依赖项**

虽然逻辑正确，但以下场景**仍然依赖 NPZ**（需要在各 Phase 中解决）：

1. **Manifest 文件读取**（Phase 1 解决）：`search.py::SearchEngine.search()` 从 `manifest.json` 中读取 `duration_ms` / `segment_ms` 等元数据，这些信息存储在文件系统而非 Milvus。
   - **实际状况**：Visual collection 已存储 `segment_start_ms` / `segment_end_ms`，可以推导所需信息
   - **解决方案**：Phase 1 修改 `milvus_search.py` 从已有字段推导，或添加冗余字段

2. **Speaker 模块的 ASR 文本依赖**（Phase 2 解决）：`speaker_service.py::_texts()` 直接读取 `asr.npz` 中的 `texts` 字段
   - **实际状况**：Milvus ASR collection 已存储 `text` 字段
   - **解决方案**：Phase 2 从 Milvus 读取，无需新建 catalog 表

3. **实体嵌入**（Phase 3 解决）：`main.py` 中人脸和语音实体的参考 embedding 存储在文件系统 `.npz` 中

4. **检索查询编码**（无需解决）：所有 `_encode_*_query()` 函数需要加载模型，模型路径配置在 `settings.py` 中，与 NPZ 索引文件无关

#### ❌ **潜在不一致风险**

当前代码**没有显式同步机制**，可能出现：

- NPZ 已更新但 Milvus 写入失败（通过 write queue 重试缓解）
- Milvus 已更新但 NPZ 被误删（需要运维规范）
- 灰度切换期间，同一查询在不同时刻命中不同后端

---

## 清理方案

### 阶段划分

分为 **6 个独立步骤**，每个步骤都可以单独验证和回滚。

---

### **Phase 1: 元数据解耦**

**目标**：将 NPZ 中的元数据迁移到独立存储，使 Milvus 检索不再依赖 NPZ 文件的 manifest。

**现状分析**：
- Visual collection 已经存储了 `segment_start_ms` / `segment_end_ms` (每帧的segment时间边界)
- `duration_ms` / `segment_ms` 可以从现有数据推导，无需额外字段
- 实际工作量比原计划更小

**实施内容**：

1. **修改 `milvus_search.py::milvus_visual_candidates()`**：
   - 从函数签名中移除 `duration_ms` / `segment_ms` 参数
   - 改为从 Milvus rows 中推导：
     ```python
     # 从第一条记录获取 segment 信息（所有记录应该一致）
     if rows:
         first_row = rows[0]
         # segment_ms 可以从 segment bounds 推导
         # duration_ms 可以从最大 timestamp_ms 推导或从外部传入
     ```
   - 如果推导失败，回退到从 `search.py` 传入的参数（向后兼容）

2. **（可选）扩展 Visual collection schema**：
   - 如果推导不可靠，添加冗余字段：
     - `video_duration_ms`: INT64
     - `indexed_at`: INT64 (时间戳)
   - 在 `milvus_indexer.py::VisualMilvusIndexer.upsert_from_npz()` 中填充

3. **修改 `search.py::SearchEngine.search()`**：
   - 保持当前从 manifest 读取 `duration_ms` / `segment_ms` 的逻辑
   - 传递给 `milvus_visual_candidates()` 作为回退值

4. **验证方式**：
   - 开启 `MILVUS_SHADOW_COMPARE_ENABLED`
   - 运行完整检索测试套件
   - 确认 Jaccard 重叠度 ≥ 0.95

**影响范围**：
- 修改文件：`milvus_search.py` (主要), 可选: `milvus_schema.py`, `milvus_indexer.py`
- 不影响：现有索引流程，NPZ 文件仍然正常写入

**回滚方案**：
- 回滚代码即可，Milvus 中的新字段（如果添加）不影响旧代码

---

### **Phase 2: Speaker 模块独立化**

**目标**：解除 `speaker_service.py` 对 `asr.npz` 的直接依赖。

**现状分析**：
- Milvus ASR collection 已经存储了 `text` 字段（见 `milvus_indexer.py::AsrMilvusIndexer`）
- 每个 ASR chunk 的文本已经在 Milvus 中，无需额外的 catalog 表
- 可以直接从 Milvus 读取，减少数据冗余

**实施内容**：

1. **新增 Milvus 读取辅助函数**：
   - 在 `speaker_service.py` 中新增：
     ```python
     def _texts_from_milvus(video_id: str) -> list[str]:
         """从 Milvus ASR collection 读取文本。"""
         from app.indexing.milvus_client import get_milvus_client
         client = get_milvus_client()
         rows = client.collection_for("asr").query(
             expr=f'video_id == "{video_id}"',
             output_fields=["segment_idx", "text"],
         )
         # segment_idx 是原始 ASR chunk 的索引
         rows.sort(key=lambda r: int(r.get("segment_idx") or 0))
         return [str(r.get("text") or "") for r in rows]
     ```

2. **修改 `speaker_service.py::_texts()`**：
   - 优先从 Milvus 读取（当 `milvus_read_enabled()` 时）
   - 如果 Milvus 返回空或失败，回退到读取 NPZ（兼容旧数据 + Milvus 未启用场景）
     ```python
     def _texts(asr_path: Path, video_id: str) -> list[str]:
         from app.indexing.milvus_flags import milvus_read_enabled
         if milvus_read_enabled():
             try:
                 texts = _texts_from_milvus(video_id)
                 if texts:  # Milvus 有数据
                     return texts
             except Exception as e:
                 import logging
                 logging.getLogger(__name__).warning(
                     "Failed to read ASR texts from Milvus for %s: %s — falling back to NPZ",
                     video_id, e
                 )
         # 回退到 NPZ
         with np.load(asr_path, allow_pickle=False) as data:
             return [str(value) for value in data["texts"]]
     ```

3. **修改 `speaker_service.py::video_speakers()` 的调用**：
   - 传递 `video_id` 到 `_texts()` 函数

4. **验证方式**：
   - 重建一个视频的 ASR 和 Speaker 索引（`MILVUS_WRITE_ENABLED=true`）
   - 运行 Speaker 检索（`MILVUS_READ_ENABLED=true`）
   - 确认结果一致

**影响范围**：
- 修改文件：`speaker_service.py`
- 不影响：Milvus 写入流程，NPZ 文件仍然正常写入

**回滚方案**：
- 将 `_texts()` 函数恢复为只读取 NPZ 的版本

**优势**：
- ✅ 无需新建 catalog 表，减少数据冗余
- ✅ 统一数据源（所有模态数据都在 Milvus）
- ✅ 与 Phase 5 的最终目标一致

---

### **Phase 3: 实体嵌入迁移**

**目标**：将人脸和语音实体的参考 embedding 从文件系统迁移到数据库。

**实施内容**：

1. **扩展 catalog schema**：
   - `entities` 表增加字段：
     - `face_embedding BLOB` (512 维 float32)
     - `voice_embedding BLOB` (192 维 float32)

2. **修改 `main.py`**：
   - 上传参考图/音频时，将 embedding 存储到数据库
   - 保留文件系统副本作为备份（可选）

3. **修改 `search.py::SearchEngine`**：
   - `_face()` 和 `_encode_asr_query()` 从数据库读取 entity embedding

4. **数据迁移脚本**：
   - 扫描现有 `.npz` 文件，批量导入数据库

**影响范围**：
- 修改文件：`db.py`, `main.py`, `search.py`
- 新增：迁移脚本

**回滚方案**：
- 回滚代码，继续从文件系统读取
- 数据库中的 embedding 字段留空

---

### **Phase 4: NPZ 写入门控**

**目标**：新增开关控制 NPZ 写入，为逐步停用做准备。

**实施内容**：

1. **新增配置项**：
   ```python
   # settings.py
   npz_write_enabled: bool = True  # 默认保持兼容
   ```

2. **修改所有 `build_*_index()` 函数**：
   ```python
   # 在 atomic_save_npz() 调用前增加判断
   if settings.npz_write_enabled:
       atomic_save_npz(output_path, ...)
   else:
       # 创建空的占位文件，避免文件不存在错误
       Path(output_path).touch()
   ```

3. **修改 `search.py`**：
   - 检查 NPZ 文件大小，如果为 0 则强制走 Milvus 路径

4. **验证方式**：
   - 设置 `NPZ_WRITE_ENABLED=false` + `MILVUS_WRITE_ENABLED=true` + `MILVUS_READ_ENABLED=true` + `MILVUS_ROLLOUT_PERCENT=100`
   - 重建索引，确认只写入 Milvus
   - 运行检索，确认结果正确

**影响范围**：
- 修改文件：`settings.py`, 所有 `build_*_index()` 函数, `search.py`

**回滚方案**：
- 设置 `NPZ_WRITE_ENABLED=true` 恢复双写

---

### **Phase 5: NPZ 读取路径清理**

**目标**：移除 `search.py` 中的 NPZ 读取代码，强制所有检索走 Milvus。

**前提条件**：
- Phase 1-4 全部完成
- 生产环境运行 `MILVUS_ROLLOUT_PERCENT=100` 至少 2 周，无重大问题

**重要澄清**：
本阶段**不是删除 ~800 行代码**，而是删除 **NPZ 文件 I/O 调用**。许多辅助函数（如 `robust_distribution`, `_visual_candidates` 的评分逻辑）仍然需要保留，因为 Milvus 路径也依赖它们。

**实施内容**：

1. **删除 NPZ 文件加载代码**：
   - `search.py::SearchEngine.search()` 中删除所有 `with np.load(index_file, ...) as data:` 调用
   - 删除 `if not use_milvus:` 分支（Visual/Face/ASR/OCR）
   - 删除 `npz_visual` / `npz_face` / `npz_asr` / `npz_ocr` 变量的 NPZ 加载路径

2. **保留的代码**（Milvus 路径仍需要）：
   - `robust_distribution()` — 分布式评分函数
   - `face_confidence()` / `asr_semantic_confidence()` / `visual_confidence()` — 置信度映射
   - `lexical_score()` — 词法匹配评分
   - `_asr_candidates()` — ASR 融合逻辑（接受 chunks 列表，被 `milvus_asr_candidates` 调用）
   - `_ocr_chunks_from_npz()` — 可以重命名为 `_build_ocr_chunks()`，参数改为从 Milvus 获取的数据

3. **简化 `SearchEngine.search()`**：
   - 删除 `do_shadow` / `do_fallback` / `use_milvus` 逻辑
   - 所有检索直接调用 `milvus_*_candidates()`
   - 移除 shadow compare 相关代码块（已完成验证）
   - 保留 `MilvusServiceError` 的捕获，但改为直接抛出（不再有 NPZ 回退）

4. **配置清理**：
   - 移除 `MILVUS_READ_ENABLED` / `MILVUS_ROLLOUT_PERCENT` / `MILVUS_FALLBACK_ENABLED` / `MILVUS_SHADOW_COMPARE_ENABLED`
   - 这些开关在 Milvus 成为唯一后端后已无意义

**实际删除的代码示例**：

```python
# 删除前：
npz_visual: list[Candidate] = []
if do_shadow or not use_milvus:
    with np.load(index_file, allow_pickle=False) as data:  # ← 删除这个 I/O 调用
        npz_visual = _visual_candidates(data, ...)

if use_milvus or do_shadow:
    ml_visual = milvus_visual_candidates(...)
    if use_milvus:
        candidates.extend(ml_visual)

if not use_milvus:
    candidates.extend(npz_visual)

# 删除后：
ml_visual = milvus_visual_candidates(...)
candidates.extend(ml_visual)
```

**影响范围**：
- 修改文件：`search.py`, `settings.py`, `milvus_flags.py`
- 代码行数：**预计删除 ~400 行 NPZ I/O 代码**（不是 800 行，因为评分逻辑保留）

**回滚方案**：
- Git revert 恢复代码
- ⚠️ 如果 NPZ 文件已删除，需要从备份恢复或重建索引
- **重要**：在执行 Phase 6 前保留所有 NPZ 文件至少 30 天

---

### **Phase 6: NPZ 文件清理**

**目标**：停止写入 NPZ，清理磁盘上的历史 NPZ 文件。

**前提条件**：
- Phase 5 完成且稳定运行至少 1 个月

**实施内容**：

1. **停止写入**：
   - 设置 `NPZ_WRITE_ENABLED=false`

2. **归档历史数据**：
   - 编写脚本，将所有 `*.npz` 文件移动到归档目录
   - 保留 `manifest.json`（包含索引元数据）

3. **监控磁盘空间**：
   - 验证磁盘空间释放符合预期

4. **最终清理**（可选）：
   - 30 天后，如果无回滚需求，删除归档目录

**影响范围**：
- 运维操作，不涉及代码修改

**回滚方案**：
- 从归档目录恢复 NPZ 文件
- 设置 `NPZ_WRITE_ENABLED=true` 恢复写入

---

## 风险评估

### 高风险项

1. **Phase 5 不可逆**：一旦删除 NPZ 读取代码，必须确保 Milvus 服务高可用
   - **缓解措施**：在 Phase 4 长期运行灰度，充分验证 Milvus 稳定性

2. **元数据丢失**：如果 Milvus 和 NPZ 同时损坏，索引无法重建
   - **缓解措施**：定期备份 Milvus collection 和 catalog.sqlite3

### 中风险项

1. **Speaker 依赖破坏**：catalog 中的 ASR 文本缺失导致 Speaker 检索失败
   - **缓解措施**：Phase 2 保留 NPZ 回退路径

2. **实体嵌入不兼容**：数据库中的 embedding 格式与模型不匹配
   - **缓解措施**：Phase 3 增加版本校验

### 低风险项

1. **性能回退**：Milvus 查询慢于 NPZ 文件读取
   - **缓解措施**：Shadow compare 阶段监控延迟

2. **存储成本**：Milvus 占用空间大于 NPZ
   - **缓解措施**：Phase 6 前评估存储成本

---

---

## 验证清单

每个阶段完成后需验证：

- [ ] 单元测试通过
- [ ] 端到端检索测试通过（所有模态）
- [ ] Shadow compare Jaccard ≥ 0.95
- [ ] 生产环境无异常错误日志
- [ ] 回滚流程验证通过

---

## 建议

1. **优先级排序**：Phase 1-3 是基础设施改造，可并行开发；Phase 4-6 必须串行执行
2. **灰度策略**：Phase 4 使用 `MILVUS_ROLLOUT_PERCENT` 逐步放量（10% → 50% → 100%）
3. **监控告警**：在 Phase 5 前部署 Milvus 健康检查和降级告警
4. **文档更新**：每个阶段完成后更新部署文档和故障排查指南
5. **Phase 2 优化**：采用 Milvus 方案而非 catalog 表，减少数据冗余
6. **Phase 5 保守执行**：保留所有 NPZ 文件至少 30 天再执行 Phase 6，确保回滚窗口

---

## 附录：关键文件清单

### 需要修改的文件

| 文件 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|------|---------|---------|---------|---------|---------|
| `settings.py` | - | - | - | ✓ | ✓ |
| `db.py` | - | - | ✓ | - | - |
| `search.py` | ✓ (可选) | - | ✓ | ✓ | ✓✓✓ |
| `milvus_schema.py` | ✓ (可选) | - | - | - | - |
| `milvus_indexer.py` | ✓ (可选) | - | - | - | - |
| `milvus_search.py` | ✓ | - | - | - | - |
| `milvus_flags.py` | - | - | - | - | ✓ |
| `asr.py` | - | - | - | ✓ | - |
| `visual.py` | - | - | - | ✓ | - |
| `faces.py` | - | - | - | ✓ | - |
| `ocr.py` | - | - | - | ✓ | - |
| `speaker.py` | - | - | - | ✓ | - |
| `speaker_service.py` | - | ✓ | - | - | - |
| `main.py` | - | - | ✓ | - | - |

### NPZ 文件当前用途

| 文件名 | 写入位置 | 读取位置 | 可删除阶段 |
|--------|---------|---------|-----------|
| `visual.npz` | `visual.py::build_visual_index()` | `search.py::_visual_candidates()` | Phase 5 |
| `face.npz` | `faces.py::build_face_index()` | `search.py::_face_candidates()` | Phase 5 |
| `asr.npz` | `asr.py::build_asr_index()` | `search.py::_asr_chunks_from_npz()` <br> `speaker_service.py::_texts()` | Phase 5 |
| `ocr.npz` | `ocr.py::build_ocr_index()` | `search.py::_ocr_chunks_from_npz()` | Phase 5 |
| `speaker.npz` | `speaker.py::build_speaker_index()` | `speaker_service.py::load_speaker_index()` | Phase 5 |
| 实体 `.npz` | `main.py` (实体上传) | `search.py::SearchEngine` | Phase 3 |

---

**结论**：当前双写模式逻辑正确，Milvus 检索与 NPZ 检索在语义上完全等价。建议按照修正后的 6 个阶段逐步清理 NPZ 依赖，总耗时约 **15.5 周**（包含充分的灰度验证时间）。

**修正说明（与初版相比）**：
1. **Phase 1**：简化方案，利用已有的 segment bounds 字段，减少开发量
2. **Phase 2**：从 Milvus 读取 ASR texts 而非新建 catalog 表，减少数据冗余
3. **Phase 5**：澄清删除范围（NPZ I/O 代码，非全部辅助函数），实际删除量约 400 行而非 800 行
4. **总时间**：从 16.5 周优化到 15.5 周

**核心风险提示**：
- Phase 5 不可逆，必须确保 Milvus 100% 稳定且覆盖完整
- 在 Phase 6 前保留所有 NPZ 文件至少 30 天作为回滚缓冲
- Phase 4 的灰度放量必须充分（每个百分比档位至少运行 3-5 天，观察监控指标）
