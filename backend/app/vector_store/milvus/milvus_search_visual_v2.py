"""Visual modality publication-scoped ANN retrieval.

1. Search all selected, currently-published frame vectors once per model cohort
2. ANN recall of global top-K candidate frames per subquery
3. Aggregate multi-query results with legacy semantics (0.65*mean + 0.35*min)
4. Aggregate by publication-safe segment keys and generate candidates

No distribution sampling/z-score normalization is applied; candidates continue
through the shared fusion and optional VLM reranking path.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
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
    asset_version: str,
    query_texts: list[np.ndarray],
    limit: int = 20,
    profile: str = "balanced",
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Compatibility wrapper for one published video version."""
    return milvus_visual_candidates_global_ann(
        client,
        {video_id: asset_version},
        query_texts,
        limit,
        profile,
        profiler,
    )


def milvus_visual_candidates_global_ann(
    client: MilvusClient,
    publication_versions: Mapping[str, str],
    query_texts: list[np.ndarray],
    limit: int = 20,
    profile: str = "balanced",
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Search one compatible publication cohort in a single ANN RPC.

    Args:
        client: Milvus client
        publication_versions: Exact ``video_id -> asset_version`` allowlist
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

    versions = _normalize_publication_versions(publication_versions)
    if not versions or not query_texts or limit <= 0:
        return []

    settings = get_settings()
    ann_top_k = settings.visual_ann_top_k
    segment_top_n = settings.visual_ann_segment_top_n

    # Verify index type matches configuration (cached to avoid extra RPC per search)
    _verify_index_type_once(client, settings.visual_use_diskann)

    # Normalize query vectors
    query_values = np.stack([_normalize(q) for q in query_texts])
    if query_values.ndim != 2 or not np.all(np.isfinite(query_values)):
        raise ValueError("Visual ANN query vectors must form one finite 2D batch")
    if np.any(np.linalg.norm(query_values, axis=1) < 1e-8):
        raise ValueError("Visual ANN query vectors must have non-zero norm")

    # ANN recall of candidate frames (multi-query batch)
    ann_results = _ann_recall_global_multi_query(
        client, versions, query_values, ann_top_k,
        settings.visual_use_diskann, profiler
    )

    if not ann_results:
        logger.info(
            "Visual ANN: no results for publication cohort (%d videos)",
            len(versions),
        )
        return []

    # Aggregate by segment with multi-query semantics
    aggregate_span = (
        profiler.span("local_processing", "visual_candidate_build")
        if profiler
        else nullcontext()
    )
    with aggregate_span:
        candidates = _aggregate_global_by_segment(
            ann_results,
            versions,
            limit,
            profile,
            len(query_texts),
            segment_top_n,
        )

    logger.info(
        "Visual ANN: cohort_videos=%d, profile=%s, queries=%d, "
        "recalled=%d, candidates=%d",
        len(versions),
        profile,
        len(query_texts),
        len(ann_results),
        len(candidates),
    )

    return candidates


def _normalize_publication_versions(
    publication_versions: Mapping[str, str],
) -> dict[str, str]:
    """Normalize a deterministic non-empty publication pair allowlist."""
    versions: dict[str, str] = {}
    for raw_video_id, raw_asset_version in publication_versions.items():
        if raw_video_id is None or raw_asset_version is None:
            raise ValueError("Visual publication scope cannot contain null values")
        video_id = str(raw_video_id)
        asset_version = str(raw_asset_version)
        if not video_id.strip() or not asset_version.strip():
            raise ValueError("Visual publication scope cannot contain blank values")
        previous = versions.get(video_id)
        if previous is not None and previous != asset_version:
            raise ValueError(
                f"Conflicting Visual asset versions for video_id={video_id!r}"
            )
        versions[video_id] = asset_version
    return dict(sorted(versions.items()))


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
        if _verify_index_type(client, expect_diskann):
            _verified_for_diskann = expect_diskann


def _verify_index_type(client: MilvusClient, expect_diskann: bool) -> bool:
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
            return True

        index_params = index_info.params or {}
        actual_type = index_params.get("index_type", "UNKNOWN")
        actual_metric = index_params.get("metric_type", "UNKNOWN")
        expected_type = "DISKANN" if expect_diskann else "HNSW"

        if not isinstance(actual_type, str) or not isinstance(actual_metric, str):
            logger.warning(
                "Visual index metadata did not expose string index/metric types; "
                "skipping drift check"
            )
            return True

        if actual_type != expected_type:
            raise MilvusVisualSearchError(
                f"Index type mismatch: config expects {expected_type} but collection "
                f"has {actual_type}. "
                "Run backend/scripts/rebuild_visual_index.py to rebuild."
            )
        if actual_metric != "COSINE":
            raise MilvusVisualSearchError(
                "Metric type mismatch: visual config expects COSINE but collection "
                f"has {actual_metric}. Rebuild the visual vector index before serving."
            )

        logger.debug(
            "Visual ANN index verified: index_type=%s metric_type=%s",
            actual_type,
            actual_metric,
        )
        return True

    except MilvusVisualSearchError:
        raise
    except (AttributeError, TypeError) as exc:
        # Structural limitation of a lightweight wrapper/test double. Retrying
        # cannot make index metadata introspectable, so this state is cacheable.
        logger.warning(
            "Visual index metadata is not introspectable (%s); skipping drift check",
            exc,
        )
        return True
    except Exception as exc:
        # RPC and timeout failures may recover. Continue serving this request, but
        # deliberately do not cache success so the next request retries the check.
        logger.warning(
            "Transient failure verifying visual index metadata: %s; will retry",
            exc,
        )
        return False


def _ann_recall_multi_query(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
    query_values: np.ndarray,
    top_k: int,
    use_diskann: bool,
    profiler: RetrievalProfiler | None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for one publication-scoped ANN RPC."""
    return _ann_recall_global_multi_query(
        client,
        {video_id: asset_version},
        query_values,
        top_k,
        use_diskann,
        profiler,
    )


def _ann_recall_global_multi_query(
    client: MilvusClient,
    publication_versions: Mapping[str, str],
    query_values: np.ndarray,
    top_k: int,
    use_diskann: bool,
    profiler: RetrievalProfiler | None,
) -> list[dict[str, Any]]:
    """Recall global top-K frames for each subquery in one Milvus RPC."""
    from app.core.settings import get_settings

    versions = _normalize_publication_versions(publication_versions)
    if not versions or top_k <= 0:
        return []
    try:
        collection = client.collection_for("visual")
        expression_started = time.perf_counter()
        scope_expression = _published_scope_expression(versions)
        expression_seconds = time.perf_counter() - expression_started
        if profiler:
            profiler.increment(
                "visual_scope", "publication_pair_count", len(versions)
            )
            profiler.increment(
                "visual_scope",
                "expr_utf8_bytes",
                len(scope_expression.encode("utf-8")),
            )
            profiler.add_seconds(
                "visual_scope", "expr_build", expression_seconds
            )

        if use_diskann:
            search_params = {
                "metric_type": "COSINE",
                # Keep ANN breadth equal to the configured global recall K.
                "params": {"search_list": top_k},
            }
        else:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": max(top_k, 128)},
            }

        rpc_span = (
            profiler.span("milvus_rpc", "visual")
            if profiler
            else nullcontext()
        )
        # ``visual_requests`` counts attempted Milvus RPCs, including calls that
        # raise. Increment immediately before collection.search so diagnostics do
        # not under-report failing backends.
        if profiler:
            profiler.increment("milvus", "visual_requests")
            profiler.increment("milvus", "visual_rpc_attempted")
        try:
            with rpc_span:
                hits = collection.search(
                    data=query_values.tolist(),
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=scope_expression,
                    output_fields=[
                        "video_id",
                        "asset_version",
                        "frame_idx",
                        "timestamp_ms",
                        "segment_id",
                        "segment_start_ms",
                        "segment_end_ms",
                    ],
                    timeout=get_settings().milvus_query_timeout_seconds,
                )
        except Exception:
            if profiler:
                profiler.increment("milvus", "visual_rpc_failed")
            raise
        else:
            if profiler:
                profiler.increment("milvus", "visual_rpc_succeeded")

        # A partial batch is ambiguous: query indexes would no longer identify
        # the input subqueries reliably, so do not return degraded candidates.
        if len(hits) != len(query_values):
            raise ValueError(
                "Visual ANN returned an unexpected result-set count: "
                f"expected={len(query_values)} actual={len(hits)}"
            )

        raw_hit_count = sum(len(query_hits) for query_hits in hits)
        results: list[dict[str, Any]] = []
        malformed_hits = 0
        for query_idx, query_hits in enumerate(hits):
            for hit in query_hits:
                entity = hit.entity
                try:
                    # Required fields intentionally have no compatibility defaults:
                    # missing scope/time metadata must not become a plausible hit.
                    results.append({
                        "query_idx": query_idx,
                        "video_id": str(_required_entity_field(entity, "video_id")),
                        "asset_version": str(
                            _required_entity_field(entity, "asset_version")
                        ),
                        "frame_idx": int(_required_entity_field(entity, "frame_idx")),
                        "timestamp_ms": int(
                            _required_entity_field(entity, "timestamp_ms")
                        ),
                        "segment_id": int(_required_entity_field(entity, "segment_id")),
                        "segment_start_ms": int(
                            _required_entity_field(entity, "segment_start_ms")
                        ),
                        "segment_end_ms": int(
                            _required_entity_field(entity, "segment_end_ms")
                        ),
                        "cosine": float(hit.distance),
                    })
                except (KeyError, TypeError, ValueError, OverflowError):
                    malformed_hits += 1

        valid_results = _valid_global_visual_results(results, versions)
        dropped_hits = malformed_hits + len(results) - len(valid_results)
        if dropped_hits:
            logger.warning(
                "Visual ANN dropped %d invalid/out-of-scope hit(s) for "
                "publication cohort (%d videos)",
                dropped_hits,
                len(versions),
            )

        if profiler:
            profiler.increment("milvus", "visual_rows", len(valid_results))
            if dropped_hits:
                profiler.increment("milvus", "visual_invalid_rows", dropped_hits)
            profiler.increment("milvus_rows", "visual_raw", raw_hit_count)
            profiler.increment("milvus_rows", "visual_valid", len(valid_results))
            profiler.increment("milvus_rows", "visual_invalid", dropped_hits)
        return valid_results

    except MilvusVisualSearchError:
        raise
    except Exception as exc:
        logger.error("Visual ANN global batch search failed: %s", exc)
        raise MilvusVisualSearchError(
            f"ANN search failed for publication cohort ({len(versions)} videos)"
        ) from exc


def _published_scope_expression(publication_versions: Mapping[str, str]) -> str:
    """Build an escaped exact-pair allowlist for the active publications."""
    versions = _normalize_publication_versions(publication_versions)
    if not versions:
        raise ValueError("Visual publication scope cannot be empty")
    return " or ".join(
        "(video_id == "
        f"{json.dumps(video_id, ensure_ascii=False)} and asset_version == "
        f"{json.dumps(asset_version, ensure_ascii=False)})"
        for video_id, asset_version in versions.items()
    )


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
    ann_results = _valid_visual_results(ann_results, video_id=video_id)

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


def _aggregate_global_by_segment(
    ann_results: list[dict[str, Any]],
    publication_versions: Mapping[str, str],
    limit: int,
    profile: str,
    n_queries: int,
    segment_top_n: int = 3,
) -> list[Candidate]:
    """Aggregate a global recall pool without crossing publication boundaries.

    The score calculation intentionally matches ``_aggregate_by_segment``.
    Scope fields are only added to every grouping key and to the resulting
    candidates so videos with identical segment/frame ids remain independent.
    """
    versions = _normalize_publication_versions(publication_versions)
    ann_results = _valid_global_visual_results(ann_results, versions)

    frame_scores: dict[
        tuple[str, str, int, int], dict[int, float]
    ] = defaultdict(dict)
    frame_meta: dict[tuple[str, str, int, int], dict[str, int]] = {}

    for result in ann_results:
        key = (
            result["video_id"],
            result["asset_version"],
            result["segment_id"],
            result["frame_idx"],
        )
        # Assignment (rather than max) preserves the singleton legacy behavior
        # should Milvus ever return the same frame/query more than once.
        frame_scores[key][result["query_idx"]] = result["cosine"]
        if key not in frame_meta:
            frame_meta[key] = {
                "timestamp_ms": result["timestamp_ms"],
                "segment_start_ms": result["segment_start_ms"],
                "segment_end_ms": result["segment_end_ms"],
            }

    frame_aggregates: dict[tuple[str, str, int, int], float] = {}
    for key, query_scores in frame_scores.items():
        if n_queries == 1:
            aggregate = list(query_scores.values())[0]
        else:
            scores = [query_scores.get(query_idx, 0.0) for query_idx in range(n_queries)]
            aggregate = 0.65 * np.mean(scores) + 0.35 * np.min(scores)
        frame_aggregates[key] = float(aggregate)

    if not frame_aggregates:
        return []

    segment_frames: dict[
        tuple[str, str, int], list[tuple[int, float, dict[str, int]]]
    ] = defaultdict(list)
    for (video_id, asset_version, segment_id, frame_idx), score in (
        frame_aggregates.items()
    ):
        frame_key = (video_id, asset_version, segment_id, frame_idx)
        segment_frames[(video_id, asset_version, segment_id)].append(
            (frame_idx, score, frame_meta[frame_key])
        )

    segment_scores: list[dict[str, Any]] = []
    for (video_id, asset_version, segment_id), frames in segment_frames.items():
        scores = [score for _, score, _ in frames]
        topn_scores = sorted(scores, reverse=True)[:segment_top_n]
        segment_score = float(np.mean(topn_scores))
        best_index = scores.index(max(scores))
        best_meta = frames[best_index][2]
        segment_scores.append({
            "video_id": video_id,
            "asset_version": asset_version,
            "segment_id": segment_id,
            "score": segment_score,
            "start_ms": best_meta["segment_start_ms"],
            "end_ms": best_meta["segment_end_ms"],
            "best_ms": best_meta["timestamp_ms"],
            "frame_count": len(frames),
        })

    # Python's stable sort retains Milvus/aggregation insertion order for exact
    # score ties, matching the singleton path's historic behavior.
    segment_scores.sort(key=lambda item: item["score"], reverse=True)
    cap = 500 if profile == "recall" else limit

    candidates: list[Candidate] = []
    for segment in segment_scores[:cap]:
        raw_score = segment["score"]
        rank_score = visual_confidence(raw_score)
        evidence = (
            f"[milvus_ann] score={raw_score:.3f} · rank={rank_score:.3f} · "
            f"{segment['frame_count']} frames · {n_queries} queries"
        )
        candidates.append(
            Candidate(
                video_id=segment["video_id"],
                start_time=_seconds(segment["start_ms"]),
                end_time=_seconds(segment["end_ms"]),
                score=rank_score,
                modality="visual",
                evidence=evidence,
                raw_score=raw_score,
                best_time=_seconds(segment["best_ms"]),
                unit_type="segment",
                unit_id=segment["segment_id"],
                best_ms=segment["best_ms"],
                features={
                    "visual_rank_score": rank_score,
                    "segment_id": segment["segment_id"],
                    "frame_count": segment["frame_count"],
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


def _required_entity_field(entity: Any, field: str) -> Any:
    """Read a required Milvus field without supplying a compatibility default."""
    value = entity.get(field)
    if value is None:
        raise KeyError(field)
    return value


def _valid_visual_results(
    results: list[dict[str, Any]],
    *,
    video_id: str,
) -> list[dict[str, Any]]:
    """Fail closed on invalid or internally inconsistent visual time metadata."""
    structurally_valid: list[dict[str, Any]] = []
    bounds_by_segment: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for result in results:
        try:
            query_idx = int(result["query_idx"])
            frame_idx = int(result["frame_idx"])
            timestamp_ms = int(result["timestamp_ms"])
            segment_id = int(result["segment_id"])
            start_ms = int(result["segment_start_ms"])
            end_ms = int(result["segment_end_ms"])
            cosine = float(result["cosine"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if (
            query_idx < 0
            or frame_idx < 0
            or segment_id < 0
            or timestamp_ms < 0
            or start_ms < 0
            or end_ms <= start_ms
            or timestamp_ms < start_ms
            or timestamp_ms > end_ms
            or not np.isfinite(cosine)
        ):
            continue

        normalized = dict(result)
        normalized.update({
            "query_idx": query_idx,
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "segment_id": segment_id,
            "segment_start_ms": start_ms,
            "segment_end_ms": end_ms,
            "cosine": cosine,
        })
        structurally_valid.append(normalized)
        bounds_by_segment[segment_id].add((start_ms, end_ms))

    inconsistent_segments = {
        segment_id
        for segment_id, bounds in bounds_by_segment.items()
        if len(bounds) != 1
    }
    if inconsistent_segments:
        logger.warning(
            "Visual ANN ignored segments with inconsistent time bounds "
            "video=%s segment_ids=%s",
            video_id,
            sorted(inconsistent_segments),
        )
    return [
        result
        for result in structurally_valid
        if result["segment_id"] not in inconsistent_segments
    ]


def _valid_global_visual_results(
    results: list[dict[str, Any]],
    publication_versions: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Validate time metadata and re-check the exact publication allowlist."""
    versions = _normalize_publication_versions(publication_versions)
    structurally_valid: list[dict[str, Any]] = []
    bounds_by_segment: dict[
        tuple[str, str, int], set[tuple[int, int]]
    ] = defaultdict(set)

    for result in results:
        try:
            video_id = str(result["video_id"])
            asset_version = str(result["asset_version"])
            query_idx = int(result["query_idx"])
            frame_idx = int(result["frame_idx"])
            timestamp_ms = int(result["timestamp_ms"])
            segment_id = int(result["segment_id"])
            start_ms = int(result["segment_start_ms"])
            end_ms = int(result["segment_end_ms"])
            cosine = float(result["cosine"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if versions.get(video_id) != asset_version:
            continue
        if (
            query_idx < 0
            or frame_idx < 0
            or segment_id < 0
            or timestamp_ms < 0
            or start_ms < 0
            or end_ms <= start_ms
            or timestamp_ms < start_ms
            or timestamp_ms > end_ms
            or not np.isfinite(cosine)
        ):
            continue

        normalized = dict(result)
        normalized.update({
            "video_id": video_id,
            "asset_version": asset_version,
            "query_idx": query_idx,
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "segment_id": segment_id,
            "segment_start_ms": start_ms,
            "segment_end_ms": end_ms,
            "cosine": cosine,
        })
        structurally_valid.append(normalized)
        bounds_by_segment[(video_id, asset_version, segment_id)].add(
            (start_ms, end_ms)
        )

    inconsistent_segments = {
        segment_key
        for segment_key, bounds in bounds_by_segment.items()
        if len(bounds) != 1
    }
    if inconsistent_segments:
        logger.warning(
            "Visual ANN ignored publication segments with inconsistent time "
            "bounds: %s",
            sorted(inconsistent_segments),
        )
    return [
        result
        for result in structurally_valid
        if (
            result["video_id"],
            result["asset_version"],
            result["segment_id"],
        )
        not in inconsistent_segments
    ]
