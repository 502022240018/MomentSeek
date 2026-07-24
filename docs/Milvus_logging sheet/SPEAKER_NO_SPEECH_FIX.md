# Speaker 索引在无人声场景下的异常修复报告

## 问题描述

用户报告：使用仅含背景音乐无人声的视频进行测试时，发现 speaker.npz 文件仍然被写入。

## 根本原因分析

### 问题定位

**位置**：`backend/app/indexing/speaker.py:257-267`（修复前）

```python
eligible = np.asarray([...])  # 过滤出有意义的 ASR chunks
if not len(eligible):
    result = save_speaker_index(  # ❌ 问题：仍然写入空的 NPZ 文件
        output_path, 
        utterance_times_ms=np.empty((0, 2), np.int32),
        ...
    )
    return {**result, "elapsed_seconds": ...}
```

### 问题影响

1. **资源浪费**：无人声视频仍然创建无意义的 speaker.npz 文件
2. **逻辑不一致**：
   - ASR 阶段：无音频时不写入 NPZ（`no_audio` 分支）
   - Speaker 阶段：无有效人声时仍写入空 NPZ（不一致）
3. **Milvus 模式下的混淆**：Phase 4 会删除 NPZ，但中间步骤仍然执行了无效的写入操作

### 连带问题：NPZ 读取错误

**位置**：`speaker.py:254`

```python
with np.load(asr_path, allow_pickle=False) as asr:  # ❌ 无法读取文本数组
    texts = [str(value) for value in asr["texts"]]
```

ASR 的 `texts` 字段是 `dtype=object` 的 NumPy 数组，需要 `allow_pickle=True` 才能读取。

## 修复方案

### 修复 1：提前返回，不写入文件

**文件**：`backend/app/indexing/speaker.py:261-267`

```python
if not len(eligible):
    # ASR 中无有效人声片段（纯背景音乐/无音频），跳过 speaker 索引
    # 不写入任何文件（包括 NPZ），直接返回空结果
    return {
        "utterances": 0,
        "tracks": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
```

**效果**：
- 无人声场景直接返回，不创建 speaker.npz
- 与 ASR 的 `no_audio` 行为一致
- Milvus 模式下不会触发无效的写入操作

### 修复 2：允许读取文本数组

**文件**：`backend/app/indexing/speaker.py:254`

```python
with np.load(asr_path, allow_pickle=True) as asr:  # ✓ 允许读取 object 数组
    times = asr["chunk_times_ms"].astype(np.int32)
    texts = [str(value) for value in asr["texts"]]
```

## 验证测试

### 测试 1：NPZ 模式下无人声场景

```python
# 输入：ASR 返回 0 个 chunk
asr.npz: chunk_times_ms=(0,2), texts=[]

# 输出：
result = {"utterances": 0, "tracks": 0, "elapsed_seconds": 0.001}
speaker.npz exists: False  ✓
```

### 测试 2：Milvus 模式下无人声场景

```python
# 输入：ASR 返回 0 个 chunk + milvus_ctx 存在
milvus_ctx = MilvusWriteContext(...)

# 输出：
speaker.npz exists: False  ✓
Milvus rows: 0  ✓
```

### 测试 3：正常功能不受影响

有效的 ASR 数据（`eligible > 0`）时，speaker 索引正常执行到 VAD 和聚类阶段。

## 影响范围

### 修改的文件

- `backend/app/indexing/speaker.py`（2 处修改）

### 依赖检查

- ✓ `main.py` 中的 `speaker_indexed` 判断完全依赖 Milvus，不依赖 NPZ 文件
- ✓ `speaker_service.py` 读取 NPZ 时会检查文件是否存在
- ✓ `stage_runner.py` 不强制要求 speaker.npz 存在

### 向后兼容性

- ✓ 已索引的视频不受影响
- ✓ 正常的 speaker 索引流程不受影响
- ✓ Phase 4（纯 Milvus）模式行为更加正确

## 结论

修复成功解决了无人声视频异常写入 speaker.npz 的问题，同时修复了 NPZ 读取的兼容性问题。所有测试通过，正常功能不受影响。
