# Milvus NPZ 清理方案修订记录

本文档记录清理方案从初版到修正版的变更内容。

## 修订日期：2026-07-21

### 修订原因

基于对后端代码的全面审查，发现以下事实：
1. Visual collection 已存储 `segment_start_ms` / `segment_end_ms` 字段
2. ASR collection 已存储 `text` 字段
3. 部分辅助函数（如 `robust_distribution`）被 Milvus 路径复用，不应删除

### 主要变更

#### 1. Phase 1: 元数据解耦

**变更前**：
- 扩展 Visual collection，新增 `duration_ms` / `segment_ms` / `model_key` 字段
- 修改 `milvus_indexer.py` 写入元数据
- 开发时间 3 天

**变更后**：
- 利用已有的 `segment_start_ms` / `segment_end_ms` 字段推导信息
- 主要修改 `milvus_search.py` 读取逻辑
- 可选添加冗余字段（如果推导不可靠）
- **开发时间缩短到 1-2 天**

**理由**：避免重复存储，减少开发量

---

#### 2. Phase 2: Speaker 模块独立化

**变更前**：
- 在 SQLite catalog 中新建 `asr_texts` 表
- 修改 `asr.py` 写入 catalog
- 修改 `speaker_service.py` 从 catalog 读取
- 开发时间 2 天

**变更后**：
- 直接从 Milvus ASR collection 读取 `text` 字段
- 只修改 `speaker_service.py`，无需修改 `asr.py` 或 `db.py`
- **开发时间缩短到 1 天**

**理由**：
- 减少数据冗余（ASR texts 不需要在 3 个地方存储：NPZ + Milvus + catalog）
- 统一数据源（所有模态数据都在 Milvus）
- 减少修改范围，降低风险

---

#### 3. Phase 5: NPZ 读取路径清理

**变更前**：
- 删除 `_visual_candidates()` / `_face_candidates()` / `_asr_candidates()` 等函数
- 删除 `_ocr_chunks_from_npz()` 函数
- **预计删除 ~800 行代码**

**变更后**：
- **只删除 NPZ 文件 I/O 调用**（`with np.load(...)` 分支）
- **保留评分逻辑函数**（`robust_distribution`, `face_confidence` 等）
- **保留 `_asr_candidates()` 函数**（Milvus 路径调用它进行融合评分）
- **预计删除 ~400 行代码**

**理由**：
- 许多辅助函数被 Milvus 路径复用，不能删除
- 实际删除的是 NPZ 文件读取逻辑，而非整个候选生成流程
- 澄清删除范围，避免执行时误删关键代码

---

#### 4. 总时间估算调整

**变更前**：总计 16.5 周

**变更后**：总计 15.5 周

**节省时间来源**：
- Phase 1: 节省 0.5 周
- Phase 2: 节省 0.5 周

---

### 文件修改范围调整

| 文件 | 初版 Phase 1 | 修正版 Phase 1 | 初版 Phase 2 | 修正版 Phase 2 |
|------|-------------|---------------|-------------|---------------|
| `settings.py` | ✓ | - | - | - |
| `db.py` | - | - | ✓ | - |
| `asr.py` | - | - | ✓ | - |
| `milvus_schema.py` | ✓ | ✓ (可选) | - | - |
| `milvus_indexer.py` | ✓ | ✓ (可选) | - | - |
| `milvus_search.py` | ✓ | ✓ | - | - |
| `speaker_service.py` | - | - | ✓ | ✓ |

**减少的修改文件数**：5 个 → 2 个（Phase 1-2 合计）

---

### 新增建议

修正版新增以下建议：

1. **Phase 2 优化**：采用 Milvus 方案而非 catalog 表，减少数据冗余
2. **Phase 5 保守执行**：保留所有 NPZ 文件至少 30 天再执行 Phase 6，确保回滚窗口
3. **Phase 4 灰度策略细化**：每个百分比档位至少运行 3-5 天，观察监控指标

---

### 核心风险提示（新增）

修正版强调了以下风险：

1. **Phase 5 不可逆性**：
   - 删除 NPZ 读取代码后，只能依赖 Milvus
   - 必须确保 Milvus 100% 稳定且覆盖完整
   - 在 Phase 6 前保留所有 NPZ 文件至少 30 天

2. **Phase 4 灰度放量要求**：
   - 每个百分比档位（10%, 25%, 50%, 75%, 100%）至少运行 3-5 天
   - 观察以下指标：
     - 检索延迟 P50/P95/P99
     - 错误率
     - Shadow compare Jaccard 重叠度
     - 用户反馈

3. **回滚准备**：
   - Phase 5-6 之间必须有至少 30 天的缓冲期
   - 定期验证 NPZ 文件的完整性和可读性
   - 准备 Milvus 快速重建脚本（从 NPZ 回填）

---

### 验证结果（新增）

修正版基于以下验证结果：

1. **双写实现验证**（自动化）：
   - 所有 5 个模态的 `build_*_index()` 函数末尾正确调用 `write_modality_to_milvus()`
   - 调用时机在 NPZ 保存之后
   - 正确处理 `milvus_ctx=None` 的情况

2. **Schema 现状验证**（代码审查）：
   - `milvus_indexer.py:121-134` 确认 Visual 已存储 `segment_start_ms` / `segment_end_ms`
   - `milvus_indexer.py:168` 确认 ASR 已存储 `text` 字段

3. **依赖关系验证**（代码审查）：
   - `search.py:954-955` 确认对 manifest 的依赖
   - `speaker_service.py:15-17` 确认对 `asr.npz` 的依赖
   - `milvus_search.py` 确认 Milvus 路径复用 NPZ 路径的评分函数

---

## 总结

此次修订主要基于**代码实际状态**进行优化，减少了不必要的开发工作，澄清了执行细节，并强化了风险提示。修正后的方案更加准确、高效、安全。

**推荐执行修正版方案**。
