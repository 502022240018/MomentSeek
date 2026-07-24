# Milvus 写入性能优化方案

**文档版本**：1.0  
**创建日期**：2026-07-22  
**适用范围**：基于 MILVUS_MIGRATION_GUIDE.md 后续优化方向的深度设计

---

## 目录

1. [问题诊断与风险评估](#问题诊断与风险评估)
2. [现有优化方向评估](#现有优化方向评估)
3. [优化层次架构](#优化层次架构)
4. [P0: 可靠性修复（生产安全必做）](#p0-可靠性修复生产安全必做)
5. [P1: 自适应批量大小](#p1-自适应批量大小)
6. [P2: 流式直写（消除NPZ中间文件）](#p2-流式直写消除npz中间文件)
7. [P3: 异步写入（解耦推理与写入延迟）](#p3-异步写入解耦推理与写入延迟)
8. [实施路线图](#实施路线图)
9. [迁移注意事项](#迁移注意事项)

---

## 问题诊断与风险评估

### 代码审查发现的实际问题

通过对照实际代码实现（`stage_runner.py` / `milvus_indexer.py` / `batch_buffer.py`），确认以下问题：

#### 🔴 P0-A: Milvus 连接失败静默丢弃数据（生产数据完整性风险）

**位置**：`backend/app/stage_runner.py:32-44` (`_setup_milvus_context`)

```python
try:
    client = get_milvus_client()
    return MilvusWriteContext(...)
except Exception as exc:
    logger.warning(  # ⚠️ 仅 warning 级别
        "Milvus context init failed for video=%s: %s — continuing without Milvus write",
        video_id, exc,
    )
    return None  # ⚠️ 返回 None，索引任务继续但不写 Milvus
```

**实际影响**：
- `bump_asset_version()` 在连接尝试**之前**已经执行，版本号白白递增
- 所有 5 个模态的 `write_modality_to_milvus()` 因 `milvus_ctx=None` 被跳过
- 索引任务**返回成功**（有 NPZ 就算成功），但 Milvus 中完全没有数据
- 运维人员无法从任务状态判断数据是否真的写入了 Milvus

**风险等级**：🔴 **Critical** — 生产环境下会导致数据完整性问题，且不易被发现

---

#### 🔴 P0-B: upsert 无重试逻辑（单次网络抖动终止整个阶段）

**位置**：`backend/app/indexing/milvus_indexer.py:75-82` (`_upsert_batched`)

```python
def _upsert_batched(collection, rows: list[dict]) -> int:
    total = 0
    for i in range(0, len(rows), _BATCH):
        batch = rows[i : i + _BATCH]
        collection.upsert(batch)   # ⚠️ 无重试，直接 raise
        total += len(batch)
    return total
```

**实际影响**：
- 对于 600 帧的 Visual 索引（~18 次批次调用），任意一次网络抖动均导致整个视频索引失败
- Milvus 在高负载或 GC 停顿时偶发超时，这是生产中常见情形
- `fail_policy=raise` 下会直接中断任务，需要完整重新索引

**风险等级**：🔴 **High** — 直接影响索引成功率

---

#### 🟡 P1: `_BATCH = 200` 对所有模态一刀切（负载分配低效）

**位置**：`backend/app/indexing/milvus_indexer.py:54`

```python
_BATCH = 200  # records per upsert call  — 对全部 5 个模态统一使用
```

**实际负载分析**（每批次 payload 估算，embedding 字段 float32）：

| 模态 | 向量维度 | 200行payload估算 | 实际压力 |
|---|---|---|---|
| Visual | 1152 | ~921 KB | 可能超过 Milvus RPC 默认限制 |
| Face | 512 | ~409 KB | 中等 |
| ASR | 384 | ~307 KB | 合理 |
| OCR | 384 | ~307 KB | 合理 |
| Speaker | 192 | ~154 KB | 批次过小，浪费 RPC 次数 |

Visual 单批负载是 Speaker 的 6 倍，而两者使用完全相同的 200 行批大小。

**风险等级**：🟡 **Medium** — 影响写入吞吐量和 Milvus 服务稳定性

---

#### 🟡 P2: BatchBuffer 完全是死代码（已有设施未被使用）

**位置**：`backend/app/indexing/batch_buffer.py`（文件存在但从未被主流水线引用）

迁移指南写的是"当前仅测试使用"，但实际上连测试代码也没有调用它。`milvus_indexer.py` 自己实现了 `_upsert_batched`，两套机制并存但各自孤立。

`BatchBuffer` 设计合理（线程安全、timeout 自动 flush、context manager），直接集成比重新实现更经济。

**风险等级**：🟡 **Low** — 不影响正确性，但造成维护负担

---

#### 🟢 P3: NPZ 中间磁盘 I/O（主要是 Visual 模态）

**当前链路**：推理结果 → 写 NPZ → 读 NPZ → 写 Milvus → 删 NPZ

对于并发索引多个视频的场景，同时存在多个大 NPZ 临时文件：
- Visual NPZ：~2.7 MB（600帧 × 1152维 × float16）
- 10 路并发 = ~27 MB 临时磁盘压力（I/O 竞争）

其余模态（ASR/OCR/Face/Speaker）NPZ 合计约 0.4 MB，优先级较低。

**风险等级**：🟢 **Low** — 影响磁盘 I/O，不影响正确性

---

## 现有优化方向评估

### 迁移指南的优化建议是否准确？

**结论：方向基本正确，但层次优先级有偏差。**

| 指南提出的优化 | 评估 | 缺失点 |
|---|---|---|
| 消除 NPZ 磁盘往返（流式直写） | ✅ 方向正确 | 未区分可靠性问题与性能问题 |
| 批量缓冲（BatchBuffer） | ✅ 方向正确 | 未指出 BatchBuffer 是死代码 |
| 异步写入 | ✅ 方向正确 | 未给出具体设计 |
| 实施顺序 | ⚠️ 遗漏了 P0 | 把可靠性修复混入性能优化叙述中 |

**核心偏差**：指南把流式直写列为最高优先级，但在生产系统中，**数据不丢（P0-A）+ 可重试（P0-B）的优先级必须高于性能优化**。当前静默丢数据的风险在任何性能优化落地之前就需要先处理。

### 最佳方案是什么？

推荐的四层优化，**按优先级从高到低**：

```
P0（必做，不做则不应上生产）
  ├── P0-A: 连接失败改为尊重 fail_policy（不再静默丢数据）
  └── P0-B: upsert 加指数退避重试

P1（低成本高收益，与 P0 可并行实施）
  └── P1: 按模态计算自适应批量大小

P2（中期，消除 NPZ 磁盘往返）
  └── P2: 流式直写 + 集成现有 BatchBuffer

P3（视需要，高并发场景才有明显收益）
  └── P3: 异步写入队列，解耦推理与写入延迟
```

---

## 优化层次架构

```
┌─────────────────────────────────────────────────────────────┐
│                    索引写入链路（优化后）                      │
├─────────────────────────────────────────────────────────────┤
│  build_*_index()                                            │
│    │                                                        │
│    ├── 推理/提取（帧解码、模型推理）                          │
│    │     ↓ 每 BATCH 行（按模态自适应大小）                    │
│    │   [P3] AsyncWriteQueue.put(modality, rows)             │
│    │       └── [P2] StreamingWriter.add(rows)               │
│    │               └── [P1] 自适应批量大小                   │
│    │                       └── [P0-B] _upsert_with_retry()  │
│    │                                                        │
│    └── [P0-A] 连接失败 → 尊重 fail_policy（不再静默）        │
└─────────────────────────────────────────────────────────────┘
```

每层独立可测，可以单独上线而不依赖其他层完成。

---

## P0: 可靠性修复（生产安全必做）

### P0-A: 修复 _setup_milvus_context 静默丢弃问题

**文件**：`backend/app/stage_runner.py`

**核心问题**：
1. `bump_asset_version()` 在连接尝试前执行，连接失败时版本号白白递增
2. 连接失败只记 warning，不区分"主动禁用"和"意外连接失败"
3. 返回 `None` 后索引任务照常返回成功

**修复设计**：

```python
def _setup_milvus_context(video_id: str, video_index_dir: Path):
    """初始化 MilvusWriteContext；连接失败时根据 fail_policy 决策。

    修复要点（对比旧版本）：
    1. 先验证连接可达，再 bump asset_version（避免版本号空耗）
    2. 连接失败日志升级为 ERROR，并附带 fail_policy 说明
    3. fail_policy=raise 时直接抛出，不再静默返回 None
    """
    from app.indexing.milvus_asset_version import bump_asset_version
    from app.indexing.milvus_client import get_milvus_client
    from app.indexing.milvus_flags import milvus_write_fail_policy
    from app.indexing.milvus_indexer import MilvusWriteContext

    try:
        # 先建立连接（可能抛出），再递增版本号
        client = get_milvus_client()
    except Exception as exc:
        policy = milvus_write_fail_policy()
        logger.error(
            "Milvus 连接失败 video=%s: %s (fail_policy=%s)",
            video_id, exc, policy,
        )
        if policy == "raise":
            raise RuntimeError(
                f"Milvus 连接失败，索引中止 (video={video_id}): {exc}"
            ) from exc
        # policy == "warn" — 主动降级，明确记录数据不会写入 Milvus
        logger.warning(
            "video=%s 本次索引跳过 Milvus 写入（policy=warn）。"
            "数据将仅存于本地 NPZ，不可被检索。",
            video_id,
        )
        return None

    # 连接成功后才递增版本号
    new_version = bump_asset_version(video_index_dir)
    logger.info("Milvus asset_version video=%s → %s", video_id, new_version)
    return MilvusWriteContext(
        video_id=video_id,
        asset_version=new_version,
        client=client,
    )
```

**关键变化**：
- 连接失败时日志级别从 `warning` → `error`
- `fail_policy=raise`（生产默认）时抛出异常，不再静默
- `fail_policy=warn` 时继续，但日志明确说明"数据不可被检索"
- `bump_asset_version` 移到连接成功后，避免版本号空耗

---

### P0-B: _upsert_batched 加指数退避重试

**文件**：`backend/app/indexing/milvus_indexer.py`

**修复设计**：

```python
import time

# 可重试错误码（MilvusException.code）—— 补充实际 pymilvus 错误码
_RETRYABLE_CODES: frozenset[int] = frozenset({
    1,     # UnexpectedError（网络层）
    9999,  # RateLimit
})

def _upsert_with_retry(
    collection,
    batch: list[dict],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> None:
    """单批次 upsert，带指数退避重试。

    Args:
        max_retries: 最大重试次数（不含首次尝试），默认 3
        base_delay: 首次重试等待秒数，每次翻倍（1 → 2 → 4）
    """
    from pymilvus.exceptions import MilvusException

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            collection.upsert(batch)
            return
        except MilvusException as exc:
            last_exc = exc
            is_last = attempt == max_retries
            # 非可重试错误码或最后一次尝试 → 直接上抛
            if exc.code not in _RETRYABLE_CODES or is_last:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Milvus upsert 暂时失败 (code=%s attempt=%d/%d)，%.1fs 后重试: %s",
                exc.code, attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)
        except Exception as exc:
            # 非 MilvusException（如网络层断开）— 同样重试
            last_exc = exc
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Milvus upsert 异常 (attempt=%d/%d)，%.1fs 后重试: %s",
                attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)


def _upsert_batched(collection, rows: list[dict], batch_size: int = _BATCH) -> int:
    """分批 upsert，每批次带重试。返回总写入行数。"""
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _upsert_with_retry(collection, batch)
        total += len(batch)
    return total
```

**注意事项**：
- 重试间隔用指数退避（1s / 2s / 4s），避免惊群
- 永久性错误（schema 不匹配、PK 冲突导致的 code）不重试，立即上抛
- `_RETRYABLE_CODES` 需根据实际生产中观察到的错误码补充

---

## P1: 自适应批量大小

**文件**：`backend/app/indexing/milvus_indexer.py`

### 设计思路

目标：每次 upsert RPC 的 payload 控制在 ~256 KB（可调）。
公式：`batch_size = target_bytes ÷ row_bytes`，其中 `row_bytes ≈ dim × 4 + 元数据估算`。

```python
# 目标每批 payload 大小（字节）。可通过环境变量覆盖。
_BATCH_TARGET_BYTES: int = 256 * 1024  # 256 KB

# 每行非向量元数据的估算字节数（pk str + 数值字段 + 文本字段）
_METADATA_BYTES: dict[str, int] = {
    "visual":  256,   # pk + video_id + timestamp + segment fields
    "asr":     512,   # pk + text(最大2000字) + 数值字段
    "ocr":     512,   # pk + text + box_score + 数值字段
    "face":    128,   # pk + 时间字段
    "speaker": 128,   # pk + 时间字段
}

def _calc_batch_size(modality: str) -> int:
    """根据向量维度和元数据大小计算合理批量。"""
    dim = EMBEDDING_DIMS[modality]
    # float32 向量 + 元数据
    row_bytes = dim * 4 + _METADATA_BYTES.get(modality, 256)
    size = max(50, _BATCH_TARGET_BYTES // row_bytes)
    return min(size, 500)  # 上限 500，防止单批过大

# 预计算各模态批量（模块加载时确定，避免运行时重复计算）
_MODALITY_BATCH: dict[str, int] = {
    mod: _calc_batch_size(mod) for mod in EMBEDDING_DIMS
}
# 预期结果（256 KB 目标）：
#   visual  → ~55 行  (1152*4+256 ≈ 4864 bytes/行)
#   face    → ~120 行 (512*4+128  ≈ 2176 bytes/行)
#   asr     → ~115 行 (384*4+512  ≈ 2048 bytes/行)
#   ocr     → ~115 行 (同 asr)
#   speaker → ~290 行 (192*4+128  ≈ 896 bytes/行)
```

修改 `_upsert_batched` 签名，将 `batch_size` 改为按模态查表：

```python
def _upsert_batched(collection, rows: list[dict], modality: str) -> int:
    batch_size = _MODALITY_BATCH.get(modality, _BATCH)
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _upsert_with_retry(collection, batch)
        total += len(batch)
    return total
```

对应修改各 Indexer 的 `upsert_from_npz()` 调用：

```python
# 修改前
return _upsert_batched(col, rows)

# 修改后（以 VisualMilvusIndexer 为例）
return _upsert_batched(col, rows, modality="visual")
```

---

## P2: 流式直写（消除NPZ中间文件）

### 目标

消除"推理 → 写NPZ → 读NPZ → 写Milvus"的磁盘往返，改为推理完一批行后直接写入 Milvus。

### 集成现有 BatchBuffer

`batch_buffer.py` 已经实现了线程安全的缓冲+自动flush，直接复用：

```python
# 修改 BatchBuffer._flush_internal 使用带重试的 upsert
class BatchBuffer:
    def _flush_internal(self) -> None:
        if not self.buffer:
            return
        try:
            _upsert_with_retry(self.collection, self.buffer)  # 使用 P0-B 的重试版本
            ...
```

### 新接口：StreamingMilvusWriter

在 `milvus_indexer.py` 中增加流式写入上下文管理器：

```python
class StreamingMilvusWriter:
    """流式写入上下文：推理时每攒满一批行即写 Milvus，无需 NPZ 中间文件。

    用法：
        with StreamingMilvusWriter(ctx, "visual") as writer:
            for frame_rows in process_frames():
                writer.add_rows(frame_rows)
        # __exit__ 时自动 flush 剩余行
    """

    def __init__(self, ctx: MilvusWriteContext, modality: str):
        self.ctx = ctx
        self.modality = modality
        col = ctx.client.collection_for(modality)
        batch_size = _MODALITY_BATCH.get(modality, _BATCH)
        self._buffer = BatchBuffer(
            collection=col,
            batch_size=batch_size,
            timeout_seconds=10.0,
            pk_generator=None,  # pk 已在行构建时设置
        )
        self._count = 0

    def add_rows(self, rows: list[dict]) -> None:
        """将一批行加入缓冲（超过 batch_size 时自动 flush）。"""
        for row in rows:
            self._buffer.add(row)
        self._count += len(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._buffer.flush()
            logger.info(
                "StreamingMilvusWriter flush OK modality=%s video=%s@%s count=%d",
                self.modality, self.ctx.video_id, self.ctx.asset_version, self._count,
            )
        return False

    @property
    def count(self) -> int:
        return self._count
```

### 修改 build_visual_index（改造示例，改造收益最大）

Visual 是磁盘压力最大的模态（NPZ ~2.7 MB），优先改造。

当前 `visual.py` 末尾写入模式（简化）：

```python
# 当前（visual.py）
atomic_save_npz(output_path, **payload)
if milvus_ctx is not None:
    write_modality_to_milvus(milvus_ctx, "visual", output_path)
    Path(output_path).unlink(missing_ok=True)
```

改为流式写入模式（核心变化）：

```python
# 改造后（visual.py）
if milvus_ctx is not None:
    # 流式直写：推理时逐 segment 写 Milvus，无需临时 NPZ
    with StreamingMilvusWriter(milvus_ctx, "visual") as writer:
        for seg_idx, (seg_embeddings, seg_times) in enumerate(segments):
            rows = _build_visual_rows(ctx=milvus_ctx, seg_idx=seg_idx,
                                       embeddings=seg_embeddings, times=seg_times)
            writer.add_rows(rows)
    # 流式写入成功后 flush collection（让 downstream 可见）
    milvus_ctx.client.collection_for("visual").flush()
else:
    # 无 Milvus 时仍保留 NPZ（开发/测试用）
    atomic_save_npz(output_path, **payload)
```

> **注意**：需要重构 `build_visual_index()` 将"构建 payload 行"的逻辑提取为独立函数，使流式路径可以复用。`VisualMilvusIndexer.upsert_from_npz()` 保留，供离线重试/历史 NPZ 迁移使用。

### 改造优先级

| 模态 | NPZ 大小 | 改造难度 | 优先级 |
|---|---|---|---|
| Visual | ~2.7 MB | 高（需要重构 segment 循环） | 🔴 最高 |
| ASR | ~150 KB | 中（chunk 已顺序处理） | 🟡 中 |
| OCR | ~150 KB | 中 | 🟡 中 |
| Face | ~20 KB | 低 | 🟢 低（可保持现有流程） |
| Speaker | ~38 KB | 低 | 🟢 低 |

---

## P3: 异步写入（解耦推理与写入延迟）

### 适用场景

- 高并发场景：多视频同时索引，Milvus 写入延迟成为推理阶段的尾延迟来源
- Visual 模态：帧推理是 GPU/NPU 密集操作，不应阻塞等待 Milvus RPC

### 设计：ModalityWriteQueue

```python
import queue
import threading
from dataclasses import dataclass

@dataclass
class _WriteTask:
    collection_name: str
    rows: list[dict]
    modality: str

class ModalityWriteQueue:
    """后台写入队列：推理线程投递行，后台线程负责实际 upsert。

    特性：
    - 后台单线程串行写入（保证顺序，避免连接竞争）
    - 有界队列（max_queue_rows），背压保护（防止推理远超写入时 OOM）
    - 写入失败通过 _error 信号传回，join() 时再抛出
    - 不依赖 asyncio（与现有同步推理代码无缝集成）
    """

    def __init__(self, ctx: MilvusWriteContext, max_queue_rows: int = 2000):
        self._ctx = ctx
        self._queue: queue.Queue[_WriteTask | None] = queue.Queue(
            maxsize=max_queue_rows
        )
        self._error: Exception | None = None
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="milvus-write-worker"
        )
        self._worker.start()

    def put(self, modality: str, rows: list[dict]) -> None:
        """投递写入任务（可能阻塞，当队列满时背压推理线程）。"""
        if self._error:
            raise RuntimeError("写入队列已出错，停止接受新任务") from self._error
        col_name = self._ctx.client.collection_for(modality).name
        self._queue.put(_WriteTask(col_name, rows, modality))

    def join(self) -> None:
        """等待所有待写入任务完成；如有写入错误则抛出。"""
        self._queue.put(None)  # 哨兵值，通知 worker 退出
        self._worker.join()
        if self._error:
            raise RuntimeError("Milvus 异步写入失败") from self._error

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                col = self._ctx.client.collection_for(task.modality)
                _upsert_with_retry(col, task.rows)
            except Exception as exc:
                logger.error("Milvus 异步写入失败 modality=%s: %s", task.modality, exc)
                self._error = exc
                # 不 break：继续消费队列（防止 put() 永久阻塞），但后续 put() 会检测到 error
            finally:
                self._queue.task_done()
```

### 与 P2 的结合

P3 是 P2 的可选加速层：

```python
# P2 + P3 组合用法
write_queue = ModalityWriteQueue(ctx)

with StreamingMilvusWriter(ctx, "visual", write_queue=write_queue) as writer:
    for frame_rows in process_frames():
        writer.add_rows(frame_rows)   # 非阻塞（投递到队列）

write_queue.join()  # 等待所有写入完成后再 flush
ctx.client.collection_for("visual").flush()
```

### P3 是否必要？

对于当前单机部署，推理（GPU/NPU bound）与 Milvus 写入（I/O bound）天然串行，P3 的收益取决于：

- **有收益**：推理时间 >> Milvus 写入时间（GPU 推理 + 等待写入完成才推进下一帧）
- **无明显收益**：Milvus 写入已经够快，或推理与写入时间相当

建议先实施 P0+P1，观察 Milvus 写入延迟占比后再决定是否引入 P3。

---

## 实施路线图

### 阶段一：可靠性修复（1-2天，必做）

**目标**：消除生产数据丢失风险，不改变整体架构。

| 任务 | 文件 | 工作量 |
|---|---|---|
| 修复 `_setup_milvus_context` 失败策略 | `stage_runner.py` | 小（~30行） |
| 实现 `_upsert_with_retry` | `milvus_indexer.py` | 小（~50行） |
| 替换 `_upsert_batched` 中的裸 upsert 调用 | `milvus_indexer.py` | 小（~5行） |
| 补充相关单元测试 | `tests/test_milvus_retry.py` | 中 |

**验收标准**：
- Milvus 连接失败时，`fail_policy=raise` 下任务明确报错（不再静默成功）
- 模拟 Milvus 单次超时，任务自动重试并最终成功

---

### 阶段二：批量大小优化（0.5天，与阶段一并行）

**目标**：按模态匹配最优 RPC payload 大小。

| 任务 | 文件 | 工作量 |
|---|---|---|
| 实现 `_calc_batch_size` 和 `_MODALITY_BATCH` | `milvus_indexer.py` | 小（~20行） |
| 修改 5 个 `upsert_from_npz()` 传入 modality | `milvus_indexer.py` | 小（~10行） |
| 基准测试：不同批量下 Milvus 写入吞吐量 | `tests/bench_batch_size.py` | 中 |

**验收标准**：
- Visual 单批 payload < 300 KB
- Speaker 单批 payload ≈ 256 KB（减少 RPC 次数）

---

### 阶段三：流式直写（1-2周，改动较大）

**目标**：消除 Visual 模态的 NPZ 中间文件（~2.7 MB/视频）。

| 任务 | 文件 | 工作量 |
|---|---|---|
| 修复 BatchBuffer 使用 `_upsert_with_retry` | `batch_buffer.py` | 小 |
| 实现 `StreamingMilvusWriter` | `milvus_indexer.py` | 中 |
| 重构 `build_visual_index` 提取行构建逻辑 | `visual.py` | 大 |
| 集成 `StreamingMilvusWriter` 到 `build_visual_index` | `visual.py` | 中 |
| ASR/OCR 流式改造（可选，NPZ 较小） | `asr.py` / `ocr.py` | 中 |
| 集成测试：流式路径与 NPZ 路径结果一致性验证 | `tests/` | 大 |

**验收标准**：
- `MILVUS_WRITE_ENABLED=true` 时，Visual 阶段不再产生临时 `.npz` 文件
- 并发 10 路索引时磁盘临时文件总大小 < 5 MB
- 检索结果与原 NPZ 路径结果完全一致

**风险提示**：
- `build_visual_index` 重构范围较大，需要全面回归测试
- 流式路径需保留 `upsert_from_npz()` 作为离线重试/批量补写的入口，不可删除

---

### 阶段四：异步写入（按需，高并发场景）

**前置条件**：阶段一+二已稳定运行，监控数据显示 Milvus 写入延迟影响推理吞吐量。

| 任务 | 文件 | 工作量 |
|---|---|---|
| 实现 `ModalityWriteQueue` | `milvus_indexer.py` | 中 |
| 集成到 `StreamingMilvusWriter`（可选模式） | `milvus_indexer.py` | 中 |
| 压测：验证背压机制 + OOM 边界 | `tests/bench_async_write.py` | 大 |

---

## 迁移注意事项

### 向后兼容

- `upsert_from_npz()` 接口**不得删除**：`reindex_from_file()` 和历史 NPZ 批量补写脚本（`backfill_milvus.py`）依赖它
- `_BATCH` 全局常量在阶段二改造后可以保留作为 fallback 默认值，但不再是各模态的实际批量

### 测试策略

```
unit tests（单元，无 Milvus）
  ├── test_upsert_retry: 模拟网络超时，验证重试次数和退避间隔
  ├── test_batch_size: 验证各模态 batch_size 计算结果在合理范围
  └── test_streaming_writer: mock collection，验证 flush 触发条件

integration tests（集成，需要 Milvus）
  ├── test_connection_fail_policy: 模拟连接失败，验证 raise/warn 两种策略
  ├── test_streaming_vs_npz: 流式写入与 NPZ 路径写入结果一致性
  └── test_concurrent_index: 并发 5 路索引，验证无数据丢失、无竞态
```

### 监控指标

阶段一上线后建议新增以下指标：

| 指标 | 含义 | 告警阈值 |
|---|---|---|
| `milvus_upsert_retries_total` | 重试次数（按模态） | > 10/min |
| `milvus_context_init_failures_total` | 连接初始化失败次数 | > 0 |
| `milvus_upsert_latency_p99` | upsert P99 延迟 | > 2s |
| `milvus_write_skipped_total` | 因 warn 策略跳过的写入次数 | > 0 |

---

## 总结

| 优化层 | 解决的核心问题 | 实施代价 | 建议 |
|---|---|---|---|
| **P0-A** | 连接失败静默丢数据 | 极低 | ✅ 立即实施 |
| **P0-B** | 单次网络抖动终止索引 | 低 | ✅ 立即实施 |
| **P1** | 一刀切批量大小低效 | 极低 | ✅ 与 P0 并行 |
| **P2** | NPZ 磁盘 I/O 往返 | 高（重构） | ⏳ 阶段三 |
| **P3** | 推理阻塞等待写入 | 中 | ⏳ 按需引入 |

**最重要的行动项**：P0-A 和 P0-B 是生产安全问题，不依赖任何性能优化，应当在当前迭代内修复，优先级高于所有性能优化工作。

---

*文档创建：2026-07-22*  
*关联文档：[[MILVUS_MIGRATION_GUIDE]] | [[MILVUS_DEPLOYMENT_GUIDE]]*

