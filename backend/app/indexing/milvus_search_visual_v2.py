"""Visual模态优化检索实现 - ANN + 混合采样策略

替代全量query的新检索方案:
1. ANN召回top-K候选帧 (减少网络传输)
2. 分层随机采样N帧估算分布 (保留z-score语义)
3. 用估算分布对ANN候选评分
4. 按segment聚合生成候选

性能目标: 延迟降低60-80%, 准确性Jaccard > 0.85
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from app.search import Candidate

if TYPE_CHECKING:
    from app.profiling import RetrievalProfiler
    from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)

# MAD (Median Absolute Deviation) 到标准差的转换系数
# 假设正态分布: sigma ≈ 1.4826 * MAD, 反过来 MAD ≈ 0.6745 * sigma
MAD_TO_SIGMA = 0.67448975


def milvus_visual_candidates_ann(
    client: MilvusClient,
    video_id: str,
    query_texts: list[np.ndarray],
    limit: int = 20,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Visual检索 - ANN + 混合采样版本

    Args:
        client: Milvus客户端
        video_id: 视频ID
        query_texts: 查询向量列表（已编码的子查询）
        limit: 返回候选数量
        profiler: 性能分析器

    Returns:
        候选列表，按分数降序排列
    """
    from app.settings import get_settings

    settings = get_settings()
    ann_top_k = settings.visual_ann_top_k
    sample_size = settings.visual_sample_size

    if profiler:
        profiler.mark("visual_ann_start")

    # 归一化查询向量
    query_values = np.stack([_normalize(q) for q in query_texts])  # [N_queries, 1152]

    # Step 1: ANN召回候选帧
    ann_results = _ann_recall_multi_query(
        client, video_id, query_values, ann_top_k, profiler
    )

    if not ann_results:
        logger.warning(f"Visual ANN: no results for video {video_id}")
        return []

    if profiler:
        profiler.mark("visual_ann_recall_done")

    # Step 2: 系统采样
    sample_results = _systematic_sample(
        client, video_id, sample_size, profiler
    )

    if profiler:
        profiler.mark("visual_sample_done")

    # Step 3: 估算全局分布
    distribution = _estimate_distribution(sample_results, query_values)

    if profiler:
        profiler.mark("visual_distribution_done")

    # Step 4: 对ANN候选打分
    scored_frames = _score_ann_candidates(ann_results, query_values, distribution)

    # Step 5: 按segment聚合
    candidates = _aggregate_by_segment(
        scored_frames, video_id, limit, distribution
    )

    if profiler:
        profiler.mark("visual_ann_complete")

    logger.debug(
        f"Visual ANN: video={video_id}, "
        f"ann_recalled={len(ann_results)}, "
        f"sampled={len(sample_results)}, "
        f"candidates={len(candidates)}"
    )

    return candidates


def _ann_recall_multi_query(
    client: MilvusClient,
    video_id: str,
    query_values: np.ndarray,
    top_k: int,
    profiler: RetrievalProfiler | None,
) -> list[dict[str, Any]]:
    """ANN召回 - 批量查询优化

    使用批量search减少RPC往返次数
    """
    collection = client.collection_for("visual")

    try:
        # 批量查询：一次调用处理所有子查询
        hits = collection.search(
            data=query_values.tolist(),  # 批量查询向量
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            expr=f'video_id == "{video_id}"',
            output_fields=[
                "frame_idx",
                "timestamp_ms",
                "segment_id",
                "segment_start_ms",
                "segment_end_ms",
            ],
            # 关键: 不返回embedding字段，减少网络传输
        )

        results = []
        for query_idx, query_hits in enumerate(hits):
            for hit in query_hits:
                entity = hit.entity
                results.append({
                    "query_idx": query_idx,
                    "frame_idx": int(entity.get("frame_idx", 0)),
                    "timestamp_ms": int(entity.get("timestamp_ms", 0)),
                    "segment_id": int(entity.get("segment_id", 0)),
                    "segment_start_ms": int(entity.get("segment_start_ms", 0)),
                    "segment_end_ms": int(entity.get("segment_end_ms", 0)),
                    "cosine": float(hit.distance),  # COSINE metric返回余弦值
                })

        return results

    except Exception as e:
        logger.error(f"Visual ANN batch search failed: {e}")
        return []


def _systematic_sample(
    client: MilvusClient,
    video_id: str,
    sample_size: int,
    profiler: RetrievalProfiler | None,
) -> list[dict[str, Any]]:
    """系统采样 - 每N帧取1帧，带随机偏移避免周期性偏差

    注意：这是系统采样（systematic sampling），不是分层采样（stratified sampling）。
    采用随机偏移确保在周期性数据中也能获得无偏样本。
    """
    import random

    collection = client.collection_for("visual")

    try:
        # 查询少量帧估算总数
        probe = collection.query(
            expr=f'video_id == "{video_id}"',
            output_fields=["frame_idx"],
            limit=100,
        )

        if not probe:
            return []

        # 估算总帧数
        max_frame_idx = max(row["frame_idx"] for row in probe)
        estimated_total = max_frame_idx + 1

        # 计算采样率
        if estimated_total <= sample_size:
            # 小数据集：全量拉取
            sample_rate = 1
            offset = 0
        else:
            sample_rate = max(1, estimated_total // sample_size)
            # 随机偏移：避免周期性数据的采样偏差
            offset = random.randint(0, sample_rate - 1)

        # 系统采样: (frame_idx + offset) % sample_rate == 0
        if offset == 0:
            expr = f'video_id == "{video_id}" AND frame_idx % {sample_rate} == 0'
        else:
            expr = f'video_id == "{video_id}" AND (frame_idx + {offset}) % {sample_rate} == 0'

        sample_rows = collection.query(
            expr=expr,
            output_fields=["embedding", "frame_idx"],
            limit=sample_size,
        )

        logger.debug(
            f"Visual systematic sampling: total≈{estimated_total}, "
            f"sample_rate={sample_rate}, offset={offset}, sampled={len(sample_rows)}"
        )

        return sample_rows

    except Exception as e:
        logger.error(f"Visual sampling failed for video {video_id}: {e}")
        return []


def _estimate_distribution(
    sample_rows: list[dict[str, Any]],
    query_values: np.ndarray,
) -> dict[str, Any]:
    """估算全局分布参数

    计算采样帧在所有子查询下的分数，提取robust统计量
    """
    if not sample_rows:
        # Fallback: 无法采样时返回默认分布
        logger.warning("Visual distribution estimation: no samples, using defaults")
        return {
            "median": 0.5,
            "mad": 0.1,
            "sorted_scores": np.array([0.5]),
        }

    # 提取采样帧的embedding
    sample_embeddings = np.array([row["embedding"] for row in sample_rows])  # [N_samples, 1152]

    # 计算采样帧在每个子查询下的分数
    sample_scores = sample_embeddings @ query_values.T  # [N_samples, N_queries]

    # 合并所有分数用于分布估算
    all_scores = sample_scores.flatten()

    # 计算robust统计量
    median = float(np.median(all_scores))
    mad = float(np.median(np.abs(all_scores - median)))

    # 避免除零
    if mad < 1e-6:
        mad = 0.1

    # 排序用于百分位数估算
    sorted_scores = np.sort(all_scores)

    logger.debug(
        f"Visual distribution: n_samples={len(sample_rows)}, "
        f"median={median:.3f}, mad={mad:.3f}"
    )

    return {
        "median": median,
        "mad": mad,
        "sorted_scores": sorted_scores,
    }


def _score_ann_candidates(
    ann_results: list[dict[str, Any]],
    query_values: np.ndarray,
    distribution: dict[str, Any],
) -> list[dict[str, Any]]:
    """对ANN候选计算z-score和percentile"""
    median = distribution["median"]
    mad = distribution["mad"]
    sorted_scores = distribution["sorted_scores"]

    scored = []
    for result in ann_results:
        raw_score = result["cosine"]

        # 计算z-score (MAD归一化)
        z_score = MAD_TO_SIGMA * (raw_score - median) / mad

        # 估算百分位数
        percentile = float(np.searchsorted(sorted_scores, raw_score) / len(sorted_scores))

        scored.append({
            **result,
            "raw_score": raw_score,
            "z_score": z_score,
            "percentile": percentile,
        })

    return scored


def _aggregate_by_segment(
    scored_frames: list[dict[str, Any]],
    video_id: str,
    limit: int,
    distribution: dict[str, Any],
) -> list[Candidate]:
    """按segment聚合，生成Candidate

    聚合策略:
    - 每个segment内取top-3帧的平均分作为segment分数
    - 决策逻辑保持与旧版本一致
    """
    # 按segment分组
    seg_frames: dict[int, list[dict]] = defaultdict(list)
    for frame in scored_frames:
        seg_id = frame["segment_id"]
        seg_frames[seg_id].append(frame)

    # 对每个segment聚合
    segment_candidates = []
    for seg_id, frames in seg_frames.items():
        # 取该segment内所有帧的分数
        scores = [f["raw_score"] for f in frames]
        z_scores = [f["z_score"] for f in frames]
        percentiles = [f["percentile"] for f in frames]

        # 聚合策略: top-3 mean + min的加权组合
        # 与旧版本保持一致: 0.65 * mean(top3) + 0.35 * min(top3)
        top3_scores = sorted(scores, reverse=True)[:3]
        if len(top3_scores) >= 3:
            aggregate_score = 0.65 * np.mean(top3_scores) + 0.35 * min(top3_scores)
        else:
            aggregate_score = float(np.mean(top3_scores))

        # Z-score聚合: 取top-3均值
        top3_z = sorted(z_scores, reverse=True)[:3]
        aggregate_z = float(np.mean(top3_z))

        # Percentile: 取最大值
        max_percentile = max(percentiles)

        # 决策逻辑（与旧版本milvus_search.py保持一致）
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
        candidates.append(
            Candidate(
                video_id=video_id,
                start_time=seg["start_ms"] / 1000.0,
                end_time=seg["end_ms"] / 1000.0,
                score=seg["score"],
                modality="visual",
                evidence=(
                    f"[milvus_ann] z={seg['z_score']:.2f} "
                    f"p={seg['percentile']:.3f} "
                    f"({seg['frame_count']} frames)"
                ),
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
            )
        )

    return candidates


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2归一化"""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm
