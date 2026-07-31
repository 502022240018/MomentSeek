"""Visual modality ANN-based retrieval (simplified).

Replaces full-query approach with pure ANN recall:
1. ANN recall of top-K candidate frames per subquery
2. Aggregate multi-query results with legacy semantics (0.65*mean + 0.35*min)
3. Aggregate by segment and generate candidates

Performance target: 60-80% latency reduction, maintains semantic correctness.

NOTE: No distribution sampling/z-score normalization - results go to VLM reranking.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from app.retrieval.search import Candidate, _seconds, visual_confidence

if TYPE_CHECKING:
    from app.retrieval.retrieval_metrics import RetrievalProfiler

    from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)

# Index verification cache: stores the last `expect_diskann` value that was verified.
# None means "not yet verified". Avoids one extra Milvus RPC per search call.
_verified_for_diskann: bool | None = None
_verify_lock = threading.Lock()


class MilvusVisualSearchError(RuntimeError):
    """Raised on Milvus query failures; NOT on empty result sets."""


def _reset_index_verification() -> None:
    """Reset the cached index-type verification result.

    Call this in tests or after a live configuration change (e.g. switching
    visual_use_diskann) so the next search re-verifies against the real index.
    """
    global _verified_for_diskann
    with _verify_lock:
        _verified_for_diskann = None


def milvus_visual_candidates_ann(
    client: MilvusClient,
    video_id: str,
    query_texts: list[np.ndarray],
    limit: int = 20,
    profile: str = "balanced",
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Visual retrieval using ANN recall with multi-query aggregation.

    Args:
        client: Milvus client
        video_id: Video ID
        query_texts: List of query vectors (encoded subqueries)
        limit: Number of candidates to return
        profile: Search profile ("precision", "balanced", "recall")
        profiler: Performance profiler

    Returns:
        List of candidates sorted by score descending

    Raises:
        MilvusVisualSearchError: On Milvus query failures
    """
    from app.core.settings import get_settings

    settings = get_settings()
    ann_top_k = settings.visual_ann_top_k
    segment_top_n = settings.visual_ann_segment_top_n

    if profiler:
        profiler.mark("visual_ann_start")

    # Verify index type matches configuration (cached to avoid extra RPC per search)
    _verify_index_type_once(client, settings.visual_use_diskann)

    # Normalize query vectors
    query_values = np.stack([_normalize(q) for q in query_texts])

    # ANN recall of candidate frames (multi-query batch)
    ann_results = _ann_recall_multi_query(
        client, video_id, query_values, ann_top_k,
        settings.visual_use_diskann, profiler
    )

    if not ann_results:
        logger.info(f"Visual ANN: no results for video {video_id}")
        return []

    if profiler:
        profiler.mark("visual_ann_recall_done")

    # Aggregate by segment with multi-query semantics
    candidates = _aggregate_by_segment(
        ann_results, video_id, limit, profile, len(query_texts), segment_top_n
    )

    if profiler:
        profiler.mark("visual_ann_complete")

    logger.info(
        f"Visual ANN: video={video_id}, profile={profile}, "
        f"queries={len(query_texts)}, recalled={len(ann_results)}, "
        f"candidates={len(candidates)}"
    )

    return candidates


def _verify_index_type_once(client: MilvusClient, expect_diskann: bool) -> None:
    """Cached wrapper for _verify_index_type.

    Skips the Milvus RPC if the index has already been verified for the current
    configuration. Acquires a lock so concurrent first-calls are safe.
    """
    global _verified_for_diskann
    if _verified_for_diskann == expect_diskann:
        return  # Fast path: already verified, no RPC needed
    with _verify_lock:
        if _verified_for_diskann == expect_diskann:
            return  # Another thread already verified while we waited
        _verify_index_type(client, expect_diskann)
        _verified_for_diskann = expect_diskann


def _verify_index_type(client: MilvusClient, expect_diskann: bool) -> None:
    """Verify visual collection index type matches configuration.

    Args:
        client: Milvus client
        expect_diskann: Expected index type from configuration

    Raises:
        MilvusVisualSearchError: Index type mismatch requiring rebuild
    """
    try:
        col = client.collection_for("visual")
        index_info = col.index()

        if not index_info:
            logger.warning("Visual collection has no index; first indexing will create it")
            return

        actual_type = index_info.params.get("index_type", "UNKNOWN")

        if expect_diskann and actual_type != "DISKANN":
            raise MilvusVisualSearchError(
                f"Index type mismatch: config expects DISKANN but collection has {actual_type}. "
                "Run backend/scripts/rebuild_visual_index.py to rebuild."
            )
        elif not expect_diskann and actual_type == "DISKANN":
            raise MilvusVisualSearchError(
                "Index type mismatch: config expects HNSW but collection has DISKANN. "
                "Run backend/scripts/rebuild_visual_index.py to rebuild."
            )

        logger.debug(f"Visual index type verified: {actual_type}")

    except MilvusVisualSearchError:
        raise
    except Exception as e:
        logger.warning(f"Failed to verify index type: {e}")


def _ann_recall_multi_query(
    client: MilvusClient,
    video_id: str,
    query_values: np.ndarray,
    top_k: int,
    use_diskann: bool,
    profiler: RetrievalProfiler | None,
) -> list[dict[str, Any]]:
    """ANN recall with batch multi-query support.

    Uses correct parameters for HNSW vs DiskANN:
    - HNSW: ef parameter (must be >= top_k)
    - DiskANN: search_list parameter (must be >= top_k)

    Returns:
        List of frame hits with fields: query_idx, frame_idx, timestamp_ms,
        segment_id, segment_start_ms, segment_end_ms, cosine
    """
    collection = client.collection_for("visual")

    try:
        # Use correct parameters based on index type
        if use_diskann:
            # DiskANN parameter: search_list >= top_k
            search_params = {
                "metric_type": "COSINE",
                "params": {"search_list": max(top_k, 100)},
            }
        else:
            # HNSW parameter: ef >= top_k
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": max(top_k, 128)},
            }

        # Batch search: process all subqueries in one call
        hits = collection.search(
            data=query_values.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'video_id == "{video_id}"',
            output_fields=[
                "frame_idx",
                "timestamp_ms",
                "segment_id",
                "segment_start_ms",
                "segment_end_ms",
            ],
            # Do NOT return embedding field to reduce network transfer
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
                    "cosine": float(hit.distance),  # COSINE metric returns cosine value
                })

        return results

    except Exception as e:
        logger.error(f"Visual ANN batch search failed: {e}")
        raise MilvusVisualSearchError(f"ANN search failed for video {video_id}") from e


def _aggregate_by_segment(
    ann_results: list[dict[str, Any]],
    video_id: str,
    limit: int,
    profile: str,
    n_queries: int,
    segment_top_n: int = 3,
) -> list[Candidate]:
    """Aggregate ANN frames by segment with multi-query support.

    Multi-query aggregation (matches legacy semantics):
    - If single query: use max frame score directly
    - If multi queries: 0.65 * mean(per_query_max) + 0.35 * min(per_query_max)
      This ensures "simultaneously satisfying multiple constraints"

    Segment aggregation:
    - Per segment: mean of top-N frames' aggregate scores (N configurable via segment_top_n)
    - Profile affects selection cap (recall=500, others=limit)

    Args:
        ann_results: ANN search results
        video_id: Video ID
        limit: Number of candidates to return
        profile: Search profile
        n_queries: Number of query vectors
        segment_top_n: Number of top frames per segment for score aggregation (default: 3)
    """
    # Group frames by (segment_id, frame_idx, query_idx)
    frame_scores: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    frame_meta: dict[tuple[int, int], dict] = {}

    for result in ann_results:
        seg_id = result["segment_id"]
        frame_idx = result["frame_idx"]
        query_idx = result["query_idx"]
        cosine = result["cosine"]

        key = (seg_id, frame_idx)
        frame_scores[key][query_idx] = cosine

        if key not in frame_meta:
            frame_meta[key] = {
                "timestamp_ms": result["timestamp_ms"],
                "segment_start_ms": result["segment_start_ms"],
                "segment_end_ms": result["segment_end_ms"],
            }

    # Aggregate per frame across queries
    frame_aggregates: dict[tuple[int, int], float] = {}
    for key, query_scores in frame_scores.items():
        if n_queries == 1:
            # Single query: use score directly
            aggregate = list(query_scores.values())[0]
        else:
            # Multi-query: 0.65 * mean + 0.35 * min (legacy semantics)
            scores = [query_scores.get(q_idx, 0.0) for q_idx in range(n_queries)]
            aggregate = 0.65 * np.mean(scores) + 0.35 * np.min(scores)

        frame_aggregates[key] = float(aggregate)

    if not frame_aggregates:
        return []

    # Group by segment
    seg_frames: dict[int, list[tuple[int, float, dict]]] = defaultdict(list)
    for (seg_id, frame_idx), score in frame_aggregates.items():
        meta = frame_meta[(seg_id, frame_idx)]
        seg_frames[seg_id].append((frame_idx, score, meta))

    # Aggregate per segment
    segment_scores = []
    for seg_id, frames in seg_frames.items():
        scores = [score for _, score, _ in frames]

        # Segment score: mean of top-N frames (N configurable via segment_top_n)
        topn_scores = sorted(scores, reverse=True)[:segment_top_n]
        segment_score = float(np.mean(topn_scores))

        # Best frame for timestamp
        best_idx = scores.index(max(scores))
        best_meta = frames[best_idx][2]

        segment_scores.append({
            "segment_id": seg_id,
            "score": segment_score,
            "start_ms": best_meta["segment_start_ms"],
            "end_ms": best_meta["segment_end_ms"],
            "best_ms": best_meta["timestamp_ms"],
            "frame_count": len(frames),
            "max_frame_score": max(scores),
        })

    # Sort by score descending
    segment_scores.sort(key=lambda x: x["score"], reverse=True)

    # Apply profile cap
    cap = 500 if profile == "recall" else limit

    # Generate candidates
    candidates: list[Candidate] = []
    for seg in segment_scores[:cap]:
        raw = seg["score"]
        rank_score = visual_confidence(raw)

        evidence = (
            f"[milvus_ann] score={raw:.3f} · rank={rank_score:.3f} · "
            f"{seg['frame_count']} frames · {n_queries} queries"
        )

        candidates.append(
            Candidate(
                video_id=video_id,
                start_time=_seconds(seg["start_ms"]),
                end_time=_seconds(seg["end_ms"]),
                score=rank_score,
                modality="visual",
                evidence=evidence,
                raw_score=raw,
                best_time=_seconds(seg["best_ms"]),
                unit_type="segment",
                unit_id=seg["segment_id"],
                best_ms=seg["best_ms"],
                features={
                    "visual_rank_score": rank_score,
                    "segment_id": seg["segment_id"],
                    "frame_count": seg["frame_count"],
                    "source": "milvus_ann",
                },
            )
        )

        if len(candidates) >= limit and profile != "recall":
            break

    return candidates


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalization."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm
