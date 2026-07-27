# Visual模态优化实施方案

**版本**: 1.0  
**日期**: 2026-07-27  
**目标**: 将Visual模态从全量query转为DiskANN + 混合采样，降低延迟60-80%

---

## 📋 实施步骤总览

### 阶段1: 配置和基础设施 (本阶段)
- [x] 创建实施文档
- [ ] 添加配置开关到settings.py
- [ ] 创建新的索引配置（支持DiskANN）
- [ ] 实现索引管理脚本

### 阶段2: 实现新检索逻辑
- [ ] 实现 `milvus_visual_candidates_ann()` - ANN版本
- [ ] 实现分层随机采样逻辑
- [ ] 实现分布估算和z-score计算
- [ ] 保留旧版本作为fallback

### 阶段3: Shadow测试
- [ ] 实现A/B对比逻辑
- [ ] 收集对比数据（Jaccard相似度）
- [ ] 分析准确性影响

### 阶段4: 灰度上线
- [ ] 5% 流量测试
- [ ] 25% → 50% → 100% 逐步放量
- [ ] 监控性能指标

---

## 🔧 技术实现

### 1. 配置增强

#### 1.1 新增配置项 (`backend/app/settings.py`)

```python
# Visual模态优化配置
visual_use_diskann: bool = False  # 是否使用DiskANN索引
visual_use_ann_search: bool = False  # 是否使用ANN检索（vs 全量query）
visual_ann_top_k: int = 500  # ANN召回数量
visual_sample_size: int = 500  # 分布估算采样数量
visual_sample_strategy: Literal["random", "stratified", "systematic"] = "stratified"
```

**说明**:
- `visual_use_diskann`: 控制索引类型，需要重建索引
- `visual_use_ann_search`: 控制检索策略，可以在运行时切换
- `visual_ann_top_k`: ANN召回候选数，trade-off召回率vs延迟
- `visual_sample_size`: 用于分布估算的采样帧数
- `visual_sample_strategy`: 
  - `random`: 完全随机采样
  - `stratified`: 按segment_id分层采样（推荐）
  - `systematic`: 系统采样（每N帧取1帧）

#### 1.2 索引配置更新 (`backend/app/indexing/milvus_client.py`)

```python
def get_visual_index_config(use_diskann: bool = False):
    """根据配置返回Visual索引参数"""
    if use_diskann:
        return {
            "index_type": "DISKANN",
            "metric_type": "COSINE",
            "params": {
                "search_list_size": 200,  # DiskANN参数
            },
        }
    else:
        # 保留HNSW作为默认
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

# 更新_COLLECTION_CONFIGS
_COLLECTION_CONFIGS["visual_embeddings"]["index"] = get_visual_index_config()
```

---

### 2. 新检索实现

#### 2.1 ANN召回 + 分层采样

新文件: `backend/app/indexing/milvus_search_visual_v2.py`

```python
"""Visual模态优化检索实现 - ANN + 混合采样策略"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

from app.search import Candidate, robust_distribution
from app.settings import get_settings

logger = logging.getLogger(__name__)


def milvus_visual_candidates_ann(
    client: "MilvusClient",
    video_id: str,
    query_texts: List[np.ndarray],  # 多个子查询
    limit: int = 20,
    profiler: "RetrievalProfiler" | None = None,
) -> List[Candidate]:
    """Visual检索 - ANN + 混合采样版本
    
    策略:
    1. ANN召回top-K候选帧
    2. 分层随机采样N帧用于分布估算
    3. 计算采样帧分数，估算全局分布参数（median, MAD）
    4. 用估算的分布对ANN候选进行z-score评分
    5. 按segment聚合，生成候选
    
    Args:
        client: Milvus客户端
        video_id: 视频ID
        query_texts: 查询向量列表（已编码的子查询）
        limit: 返回候选数量
        profiler: 性能分析器
    
    Returns:
        候选列表
    """
    settings = get_settings()
    ann_top_k = settings.visual_ann_top_k
    sample_size = settings.visual_sample_size
    
    # 归一化查询向量
    query_values = np.stack([_normalize(q) for q in query_texts])  # [N_queries, 1152]
    
    # Step 1: ANN召回（多查询并行）
    ann_results = _ann_recall_multi_query(
        client, video_id, query_values, ann_top_k, profiler
    )
    
    # Step 2: 分层随机采样
    sample_results = _stratified_sample(
        client, video_id, sample_size, profiler
    )
    
    # Step 3: 计算采样帧分数，估算分布
    distribution = _estimate_distribution(sample_results, query_values)
    
    # Step 4: 对ANN候选打分
    scored_frames = _score_ann_candidates(ann_results, query_values, distribution)
    
    # Step 5: 按segment聚合
    candidates = _aggregate_by_segment(
        scored_frames, video_id, limit, distribution
    )
    
    return candidates


def _ann_recall_multi_query(
    client, video_id: str, query_values: np.ndarray, top_k: int, profiler
) -> List[Dict]:
    """ANN召回 - 支持多查询"""
    collection = client.get_collection("visual")
    
    results = []
    for i, query_vec in enumerate(query_values):
        hits = collection.search(
            data=[query_vec.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            expr=f'video_id == "{video_id}"',
            output_fields=[
                "frame_idx", "timestamp_ms", "segment_id",
                "segment_start_ms", "segment_end_ms"
            ],
            # 关键: 不返回embedding，减少传输
        )
        
        for hit in hits[0]:
            results.append({
                "query_idx": i,
                "frame_idx": hit.entity.get("frame_idx"),
                "timestamp_ms": hit.entity.get("timestamp_ms"),
                "segment_id": hit.entity.get("segment_id"),
                "segment_start_ms": hit.entity.get("segment_start_ms"),
                "segment_end_ms": hit.entity.get("segment_end_ms"),
                "cosine": hit.distance,  # COSINE metric返回的就是余弦值
            })
    
    return results


def _stratified_sample(
    client, video_id: str, sample_size: int, profiler
) -> List[Dict]:
    """分层随机采样 - 按segment_id分层"""
    collection = client.get_collection("visual")
    
    # 先获取该视频的segment列表
    segments_query = collection.query(
        expr=f'video_id == "{video_id}"',
        output_fields=["segment_id"],
        limit=16384,  # 假设最多1万个segment
    )
    
    unique_segments = sorted(set(row["segment_id"] for row in segments_query))
    n_segments = len(unique_segments)
    
    if n_segments == 0:
        return []
    
    # 每个segment采样多少帧
    per_segment = max(1, sample_size // n_segments)
    
    sample_rows = []
    for seg_id in unique_segments:
        # 系统采样: 每N帧取1帧
        # 计算采样率
        seg_query = collection.query(
            expr=f'video_id == "{video_id}" AND segment_id == {seg_id}',
            output_fields=["frame_idx"],
            limit=10000,  # 每个segment最多1万帧
        )
        
        total_frames_in_seg = len(seg_query)
        if total_frames_in_seg == 0:
            continue
        
        sample_rate = max(1, total_frames_in_seg // per_segment)
        
        # 采样
        seg_samples = collection.query(
            expr=f'video_id == "{video_id}" AND segment_id == {seg_id} AND frame_idx % {sample_rate} == 0',
            output_fields=["embedding", "frame_idx"],
            limit=per_segment,
        )
        
        sample_rows.extend(seg_samples)
        
        if len(sample_rows) >= sample_size:
            break
    
    return sample_rows[:sample_size]


def _estimate_distribution(
    sample_rows: List[Dict], query_values: np.ndarray
) -> Dict[str, Any]:
    """估算全局分布参数"""
    if not sample_rows:
        # Fallback: 无法采样时返回默认分布
        return {"median": 0.5, "mad": 0.1, "percentiles": None}
    
    # 提取采样帧的embedding
    sample_embeddings = np.array([row["embedding"] for row in sample_rows])  # [N_samples, 1152]
    
    # 计算采样帧在每个子查询下的分数
    sample_scores = sample_embeddings @ query_values.T  # [N_samples, N_queries]
    
    # 对每个子查询，计算跨帧的分布
    # 这里简化为:取所有子查询×所有采样帧的分数pool
    all_scores = sample_scores.flatten()
    
    # 计算robust统计量
    median = float(np.median(all_scores))
    mad = float(np.median(np.abs(all_scores - median)))
    
    # 用于百分位数估算
    sorted_scores = np.sort(all_scores)
    
    return {
        "median": median,
        "mad": mad if mad > 1e-6 else 0.1,  # 避免除零
        "sorted_scores": sorted_scores,
    }


def _score_ann_candidates(
    ann_results: List[Dict], 
    query_values: np.ndarray, 
    distribution: Dict
) -> List[Dict]:
    """对ANN候选计算z-score和percentile"""
    median = distribution["median"]
    mad = distribution["mad"]
    sorted_scores = distribution.get("sorted_scores", np.array([]))
    
    MAD_TO_SIGMA = 0.67448975  # MAD → 标准差转换系数
    
    scored = []
    for result in ann_results:
        raw_score = result["cosine"]
        z_score = MAD_TO_SIGMA * (raw_score - median) / mad
        
        # 估算百分位数
        if len(sorted_scores) > 0:
            percentile = float(np.searchsorted(sorted_scores, raw_score) / len(sorted_scores))
        else:
            percentile = 0.5
        
        scored.append({
            **result,
            "raw_score": raw_score,
            "z_score": z_score,
            "percentile": percentile,
        })
    
    return scored


def _aggregate_by_segment(
    scored_frames: List[Dict], 
    video_id: str, 
    limit: int,
    distribution: Dict
) -> List[Candidate]:
    """按segment聚合，生成Candidate"""
    from collections import defaultdict
    
    # 按segment分组
    seg_frames = defaultdict(list)
    for frame in scored_frames:
        seg_id = frame["segment_id"]
        seg_frames[seg_id].append(frame)
    
    # 对每个segment，聚合多子查询的分数
    segment_candidates = []
    for seg_id, frames in seg_frames.items():
        # 取该segment内所有帧的分数
        scores = [f["raw_score"] for f in frames]
        z_scores = [f["z_score"] for f in frames]
        percentiles = [f["percentile"] for f in frames]
        
        # 聚合策略: 取top-3的mean
        top3_scores = sorted(scores, reverse=True)[:3]
        aggregate_score = float(np.mean(top3_scores))
        
        top3_z = sorted(z_scores, reverse=True)[:3]
        aggregate_z = float(np.mean(top3_z))
        
        max_percentile = max(percentiles)
        
        # 决策逻辑（保持与旧版本一致）
        if aggregate_z >= 2.0 or max_percentile >= 0.975:
            decision = "strong"
        elif max_percentile >= 0.80:
            decision = "fuzzy" if aggregate_z >= 1.0 else "weak"
        else:
            decision = "weak"
        
        # 获取segment边界
        seg_start_ms = frames[0]["segment_start_ms"]
        seg_end_ms = frames[0]["segment_end_ms"]
        
        segment_candidates.append({
            "segment_id": seg_id,
            "score": aggregate_score,
            "z_score": aggregate_z,
            "percentile": max_percentile,
            "decision": decision,
            "start_ms": seg_start_ms,
            "end_ms": seg_end_ms,
            "frame_count": len(frames),
        })
    
    # 排序并返回top-limit
    segment_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    candidates = []
    for seg in segment_candidates[:limit]:
        candidates.append(Candidate(
            video_id=video_id,
            start_time=seg["start_ms"] / 1000.0,
            end_time=seg["end_ms"] / 1000.0,
            score=seg["score"],
            modality="visual",
            evidence=f"[milvus_ann] z={seg['z_score']:.2f} p={seg['percentile']:.3f} ({seg['frame_count']} frames)",
            raw_score=seg["score"],
            decision=seg["decision"],
            above_threshold=seg["decision"] in ("strong", "fuzzy"),
            best_time=seg["start_ms"] / 1000.0,
            unit_type="segment",
            unit_id=seg["segment_id"],
            best_ms=seg["start_ms"],
            features={
                "z_score": seg["z_score"],
                "percentile": seg["percentile"],
                "segment_id": seg["segment_id"],
                "frame_count": seg["frame_count"],
                "source": "milvus_ann",
            },
        ))
    
    return candidates


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2归一化"""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm
```

---

### 3. 集成到现有代码

#### 3.1 修改 `backend/app/indexing/milvus_search.py`

在文件顶部添加导入:
```python
from .milvus_search_visual_v2 import milvus_visual_candidates_ann
```

在 `milvus_visual_candidates()` 函数开头添加分支:
```python
def milvus_visual_candidates(...):
    settings = get_settings()
    
    # 新版本: ANN + 混合采样
    if settings.visual_use_ann_search:
        return milvus_visual_candidates_ann(
            client, video_id, query_texts, limit, profiler
        )
    
    # 旧版本: 全量query（保留作为fallback）
    rows = _query_all(client, "visual", video_id, [...])
    # ... 原有逻辑
```

---

### 4. Shadow测试实现

#### 4.1 新增配置
```python
# settings.py
visual_shadow_test_enabled: bool = False  # Shadow测试开关
visual_shadow_test_sample_rate: float = 0.1  # 采样率（10%查询）
```

#### 4.2 实现对比逻辑

新文件: `backend/app/indexing/milvus_shadow_visual.py`

```python
"""Visual模态Shadow测试 - 对比新旧版本"""
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def shadow_compare_visual(
    client,
    video_id: str,
    query_texts: list,
    limit: int,
    profiler,
):
    """对比新旧版本的Visual检索结果"""
    import time
    from .milvus_search import milvus_visual_candidates  # 旧版本
    from .milvus_search_visual_v2 import milvus_visual_candidates_ann  # 新版本
    
    # 旧版本
    start_old = time.time()
    old_results = milvus_visual_candidates(
        client, video_id, query_texts, limit, profiler
    )
    latency_old = time.time() - start_old
    
    # 新版本
    start_new = time.time()
    new_results = milvus_visual_candidates_ann(
        client, video_id, query_texts, limit, profiler
    )
    latency_new = time.time() - start_new
    
    # 计算Jaccard相似度（top-10的segment_id集合）
    old_segments = set(c.features.get("segment_id") for c in old_results[:10])
    new_segments = set(c.features.get("segment_id") for c in new_results[:10])
    
    intersection = len(old_segments & new_segments)
    union = len(old_segments | new_segments)
    jaccard = intersection / union if union > 0 else 0.0
    
    # 记录日志
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_id": video_id,
        "limit": limit,
        "latency_old_ms": latency_old * 1000,
        "latency_new_ms": latency_new * 1000,
        "speedup": latency_old / latency_new if latency_new > 0 else 0,
        "jaccard_top10": jaccard,
        "old_top10_segments": list(old_segments),
        "new_top10_segments": list(new_segments),
    }
    
    # 写入日志文件
    log_dir = Path("/app/runtime/shadow_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"visual_shadow_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    logger.info(
        f"Visual shadow compare: video={video_id}, "
        f"jaccard={jaccard:.3f}, "
        f"speedup={log_entry['speedup']:.2f}x"
    )
    
    # 返回新版本结果（shadow模式不影响实际返回）
    return new_results
```

---

## 📊 验证计划

### 1. 单元测试
- [ ] 测试分层采样逻辑（不同segment数量）
- [ ] 测试分布估算（边界情况：0帧、1帧、大量帧）
- [ ] 测试z-score计算准确性

### 2. 集成测试
- [ ] 在测试Milvus实例上创建小数据集（10个视频）
- [ ] 对比新旧版本的top-10结果
- [ ] 验证Jaccard > 0.85

### 3. 性能测试
- [ ] 测量延迟（p50, p95, p99）
- [ ] 测量网络传输量
- [ ] 测量内存峰值

### 4. Shadow测试（生产环境）
- [ ] 启用10%采样
- [ ] 收集7天数据
- [ ] 分析Jaccard分布、延迟改进

---

## 📈 成功标准

### 必须满足（Go/No-Go）
- ✅ Jaccard相似度 >= 0.85（top-10）
- ✅ 延迟降低 >= 50%
- ✅ 无新增错误或崩溃

### 期望达到
- 🎯 Jaccard >= 0.90
- 🎯 延迟降低 60-80%
- 🎯 网络传输降低 > 90%

---

## 🚨 回滚计划

### 触发条件
- Jaccard < 0.80
- 新版本延迟 > 旧版本
- 错误率上升

### 回滚步骤
1. 设置 `visual_use_ann_search = False`
2. 重启服务
3. 验证旧版本恢复正常

---

## 📝 下一步

当前阶段完成后，进入阶段2实施代码实现。

**预计工作量**: 2-3个工作日

**负责人**: [待指定]
