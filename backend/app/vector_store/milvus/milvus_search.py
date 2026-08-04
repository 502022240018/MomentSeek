"""Milvus-side candidate generation for all five modalities.

Design principle
----------------
Visual / ASR / OCR rely on *distribution-aware* scoring: robust z-scores and
empirical percentiles are computed over ALL embeddings in the video, not just the
top-k ANN hits.  A top-k ANN search would give the wrong distribution sample, so
these three modalities use collection.query() to fetch every row for the video,
then compute dot-products in Python — exactly as the NPZ path does.

Face and Speaker use absolute-threshold scoring (no distribution normalization
needed), so ANN search is appropriate and efficient for them.

All functions return identical list[Candidate] types so the existing fusion,
grouping, and ranking code in search.py needs no changes.

Fallback contract
-----------------
MilvusServiceError is raised *only* on connection / timeout failures.
"Milvus returned 0 results" is NOT a failure — it is a valid empty answer.
The caller (SearchEngine.search) decides whether to fall back to NPZ on
MilvusServiceError, controlled by MILVUS_FALLBACK_ENABLED.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np

from app.retrieval.retrieval_metrics import RetrievalProfiler
from app.retrieval.search import (
    Candidate,
    _seconds,
    face_confidence,
    normalize,
)
from app.core.settings import get_settings

if TYPE_CHECKING:
    from app.vector_store.milvus.milvus_client import MilvusClient

# Visual modality optimization: ANN + sampling implementation (v2)
from .milvus_search_visual_v2 import milvus_visual_candidates_ann

logger = logging.getLogger(__name__)


# Milvus ANN search params (used only for face / speaker).
_HNSW_EF    = 128
_IVF_NPROBE = 64

# Per-modality metric config — static mapping, must stay in sync with
# _COLLECTION_CONFIGS in milvus_client.py.
_MODALITY_METRIC: dict[str, str] = {
    "visual":  "COSINE",
    "asr":     "IP",
    "ocr":     "IP",
    "face":    "L2",
    "speaker": "COSINE",
}

# Static index types for non-visual modalities
_STATIC_INDEX_TYPES: dict[str, str] = {
    "asr":     "DISKANN",
    "ocr":     "DISKANN",
    "face":    "IVF_FLAT",
    "speaker": "HNSW",
}


def get_modality_index_type(modality: str) -> str:
    """Get index type for a modality (dynamic for visual, static for others).

    This function is called at runtime to support dynamic configuration changes,
    particularly for visual modality which can switch between DISKANN and HNSW.

    Args:
        modality: Modality name ("visual", "asr", "ocr", "face", "speaker")

    Returns:
        Index type string ("DISKANN", "HNSW", "IVF_FLAT")
    """
    if modality == "visual":
        settings = get_settings()
        return "DISKANN" if settings.visual_use_diskann else "HNSW"
    return _STATIC_INDEX_TYPES[modality]


# Deprecated: Use get_modality_index_type() for runtime access
# This dict exists only for backward compatibility with test assertions
_MODALITY_INDEX_TYPE: dict[str, str] = _STATIC_INDEX_TYPES.copy()

# Batch size for QueryIterator (and fallback offset-pagination).
# Milvus recommends iterator for entity traversal; 1 000–4 000 is a practical
# sweet-spot that keeps per-page latency low while amortising round-trip cost.
_QUERY_BATCH = 2_000

# visual / ocr / asr are all intentionally absent: each uses ANN/hybrid search
# and issues its own collection.search() / hybrid_search() call without consuming
# pre-fetched rows. Including them would trigger a full query_iterator traversal
# that reads every embedding before the search runs, wasting significant I/O for
# no benefit. This dict is therefore empty — no modality is bulk-prefetched.
BULK_QUERY_FIELDS: dict[str, list[str]] = {}


class MilvusServiceError(RuntimeError):
    """Raised on connection / timeout failures; NOT on empty result sets."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _schema_available_fields(col, requested: list[str]) -> list[str]:
    """Return the subset of *requested* fields that exist in *col*'s schema.

    Provides backward compatibility when a collection was created with an older
    schema that lacks recently-added fields (e.g. ``has_embedding``).  Missing
    fields are logged at WARNING level so operators know a schema migration is
    needed.
    """
    try:
        schema_fields = {field.name for field in col.schema.fields}
    except TypeError:
        # Lightweight unit-test clients and older wrappers may not expose
        # schema metadata. Let Milvus validate the requested fields directly.
        return requested
    available = [f for f in requested if f in schema_fields]
    missing = set(requested) - schema_fields
    if missing:
        logger.warning(
            "Collection '%s' is missing schema fields %s — "
            "run migrate_milvus_schema.py to upgrade; "
            "omitting missing fields (backward-compat mode)",
            col.name, sorted(missing),
        )
    return available


def query_rows_for_videos(
    client: MilvusClient,
    modality: str,
    video_ids: list[str],
    output_fields: list[str],
    profiler: RetrievalProfiler | None = None,
) -> dict[str, list[dict]]:
    """Traverse one collection once and group rows for a batch of videos."""
    unique_ids = list(dict.fromkeys(str(value) for value in video_ids if value))
    grouped = {video_id: [] for video_id in unique_ids}
    if not unique_ids:
        return grouped

    col = client.collection_for(modality)
    requested_fields = list(dict.fromkeys(["video_id", *output_fields]))
    available_fields = _schema_available_fields(col, requested_fields)
    if "video_id" not in available_fields:
        raise MilvusServiceError(
            f"Milvus collection for modality={modality} has no video_id field"
        )
    expr = f"video_id in {json.dumps(unique_ids, ensure_ascii=False)}"
    timeout = get_settings().milvus_query_timeout_seconds
    row_count = 0
    try:
        span = profiler.span("milvus_rpc", modality) if profiler else nullcontext()
        with span:
            if hasattr(col, "query_iterator"):
                iterator = col.query_iterator(
                    batch_size=_QUERY_BATCH,
                    expr=expr,
                    output_fields=available_fields,
                    timeout=timeout,
                )
                try:
                    while True:
                        page = iterator.next()
                        if not page:
                            break
                        for row in page:
                            video_id = str(row.get("video_id") or "")
                            if video_id in grouped:
                                grouped[video_id].append(row)
                                row_count += 1
                        if profiler:
                            profiler.increment("milvus", f"{modality}_pages")
                finally:
                    iterator.close()
            else:
                offset = 0
                while True:
                    page = col.query(
                        expr=expr,
                        output_fields=available_fields,
                        limit=_QUERY_BATCH,
                        offset=offset,
                        timeout=timeout,
                    )
                    for row in page:
                        video_id = str(row.get("video_id") or "")
                        if video_id in grouped:
                            grouped[video_id].append(row)
                            row_count += 1
                    if profiler:
                        profiler.increment("milvus", f"{modality}_pages")
                    if len(page) < _QUERY_BATCH:
                        break
                    offset += _QUERY_BATCH
    except MilvusServiceError:
        raise
    except Exception as exc:
        raise MilvusServiceError(
            f"Milvus batch query failed for modality={modality}: {exc}"
        ) from exc

    if profiler:
        profiler.increment("milvus", f"{modality}_rows", row_count)
        profiler.increment("milvus", f"{modality}_requests")
        profiler.increment("milvus", f"{modality}_video_batches")
    return grouped


def _ann_search(
    client: MilvusClient,
    modality: str,
    video_id: str,
    query: list[float],
    limit: int,
    output_fields: list[str],
    profiler: RetrievalProfiler | None = None,
) -> list[dict]:
    """Execute a per-video ANN search; used only by face and speaker."""
    col = client.collection_for(modality)
    metric     = _MODALITY_METRIC[modality]
    index_type = get_modality_index_type(modality)
    if index_type == "HNSW":
        sp = {"metric_type": metric, "params": {"ef": _HNSW_EF}}
    elif index_type == "IVF_FLAT":
        sp = {"metric_type": metric, "params": {"nprobe": _IVF_NPROBE}}
    else:
        # DiskANN and other types are not supported in _ann_search (face/speaker only)
        raise MilvusServiceError(
            f"_ann_search does not support index_type={index_type!r} "
            f"for modality={modality!r}; only HNSW and IVF_FLAT are supported."
        )
    try:
        span = profiler.span("milvus_rpc", modality) if profiler else nullcontext()
        with span:
            results = col.search(
                data=[query],
                anns_field="embedding",
                param=sp,
                limit=limit,
                expr=f'video_id == "{video_id}"',
                output_fields=output_fields,
                timeout=get_settings().milvus_query_timeout_seconds,
            )
    except Exception as exc:
        raise MilvusServiceError(
            f"Milvus ANN search failed for modality={modality}: {exc}"
        ) from exc
    hits: list[dict] = []
    for hit in results[0]:
        row = {"_distance": float(hit.distance)}
        for f in output_fields:
            row[f] = hit.entity.get(f)
        hits.append(row)
    if profiler:
        profiler.increment("milvus", f"{modality}_rows", len(hits))
        profiler.increment("milvus", f"{modality}_requests")
    return hits


# ---------------------------------------------------------------------------
# Visual — query-all + segment-aware distribution scoring
# ---------------------------------------------------------------------------

def milvus_visual_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    duration_ms: int | None = None,
    segment_ms: int | None = None,
    profile: str = "balanced",
    limit: int = 72,
    profiler: RetrievalProfiler | None = None,
    rows: list[dict] | None = None,
) -> list[Candidate]:
    """Visual recall via Milvus ANN search.

    Uses ANN (HNSW or DiskANN) to recall top-K candidate frames, then aggregates
    by segment with multi-query support. Results go directly to VLM reranking.

    Args:
        query: Query embedding(s), shape [D] or [N_queries, D]
        profile: "precision", "balanced", or "recall"
        limit: Number of candidates to return
        duration_ms: Deprecated – no longer used by the ANN implementation.
        segment_ms: Deprecated – no longer used by the ANN implementation.
        rows: Deprecated – no longer used by the ANN implementation.
    """
    if rows is not None or duration_ms is not None or segment_ms is not None:
        logger.warning(
            "milvus_visual_candidates: parameters 'rows', 'duration_ms', and "
            "'segment_ms' are not used by the ANN-based v2 implementation. "
            "These arguments are silently ignored; remove them from the call site."
        )

    # Convert query format to list (support multi-query)
    query_texts = [query] if query.ndim == 1 else list(query)

    return milvus_visual_candidates_ann(
        client, video_id, query_texts, limit, profile, profiler
    )


# ---------------------------------------------------------------------------
# ASR — DiskANN + BM25 hybrid search
# ---------------------------------------------------------------------------

def milvus_asr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray | None,
    limit: int,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """ASR hybrid search: DiskANN (semantic) + BM25 (lexical).

    Uses Milvus Function Field for server-side BM25 computation and WeightedRanker
    for result fusion. Semantic-first strategy (dense_weight > sparse_weight),
    reflecting that ASR transcripts are longer and semantically richer than the
    single-frame OCR text that favours the lexical channel.

    Three-way fallback:
      * hybrid       — query_embedding present AND query_text non-empty
      * dense-only   — query_embedding present, query_text empty
      * bm25-only    — query_embedding is None, query_text non-empty

    The fused ``hybrid_score`` (dense IP ∈ [-1,1] plus unbounded BM25) does NOT
    fall in [0,1]; ``above_threshold`` is left True here and finalised by the
    global dynamic threshold in search.py after all videos are collected.

    Args:
        client: Milvus client instance.
        video_id: Target video ID to filter candidates.
        query_text: Query text for BM25 lexical search.
        query_embedding: Query embedding for DiskANN semantic search; None
            triggers BM25-only fallback.
        limit: Maximum number of candidates to return.
        profiler: Optional profiler for per-span latency tracking.

    Returns:
        List of Candidate objects ordered by hybrid score descending.
        ``above_threshold`` is always True on return; search.py applies the
        global dynamic threshold after collecting all videos.
    """
    from pymilvus import AnnSearchRequest, WeightedRanker

    settings = get_settings()
    col = client.collection_for("asr")

    recall_size = settings.asr_hybrid_recall_size
    semantic_weight = settings.asr_semantic_weight
    lexical_weight = 1.0 - semantic_weight
    search_list = settings.asr_diskann_search_list

    output_fields = ["segment_idx", "start_ms", "end_ms", "text", "has_embedding"]

    # Handle None query_embedding — BM25-only fallback.
    if query_embedding is None:
        logger.info(
            "No semantic embedding for ASR, using BM25-only search: video_id=%s",
            video_id,
        )
        if not query_text or not query_text.strip():
            logger.warning(
                "Empty query for ASR with no semantic embedding, returning empty "
                "results: video_id=%s",
                video_id,
            )
            return []

        with (profiler.span("milvus_rpc", "asr_bm25_only") if profiler else nullcontext()):
            results = col.search(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=f'video_id == "{video_id}"',
                output_fields=output_fields,
            )
    else:
        query_norm = normalize(np.asarray(query_embedding, dtype=np.float32))

        # Empty query text: dense-only fallback.
        if not query_text or not query_text.strip():
            logger.warning(
                "Empty query_text for ASR, falling back to dense-only: video_id=%s",
                video_id,
            )
            with (profiler.span("milvus_rpc", "asr_dense_only") if profiler else nullcontext()):
                results = col.search(
                    data=[query_norm.tolist()],
                    anns_field="embedding",
                    param={
                        "metric_type": "IP",
                        "params": {"search_list": search_list},
                    },
                    limit=limit,
                    expr=f'video_id == "{video_id}" AND has_embedding == True',
                    output_fields=output_fields,
                )
        else:
            # Hybrid search: Dense + Sparse.
            dense_req = AnnSearchRequest(
                data=[query_norm.tolist()],
                anns_field="embedding",
                param={
                    "metric_type": "IP",
                    "params": {"search_list": search_list},
                },
                limit=recall_size,
                expr=f'video_id == "{video_id}" AND has_embedding == True',
            )

            sparse_req = AnnSearchRequest(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=recall_size,
                expr=f'video_id == "{video_id}"',
            )

            with (profiler.span("milvus_rpc", "asr_hybrid") if profiler else nullcontext()):
                results = col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=WeightedRanker(semantic_weight, lexical_weight),
                    limit=limit,
                    output_fields=output_fields,
                )

    # Convert to Candidate objects (threshold applied globally later in search.py).
    candidates: list[Candidate] = []
    for hit in results[0]:
        hybrid_score = float(hit.score)
        text = str(hit.entity.get("text") or "")
        start_ms = int(hit.entity.get("start_ms") or 0)
        end_ms = int(hit.entity.get("end_ms") or 0)
        segment_idx = int(hit.entity.get("segment_idx") or 0)

        # above_threshold stays True here; the global dynamic threshold in
        # search.py updates it and appends the "· 低于阈值" suffix if needed.
        evidence_text = f"[asr_hybrid] {text[:100]} · hybrid={hybrid_score:.3f}"

        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=hybrid_score,
            modality="asr",
            evidence=evidence_text,
            raw_score=hybrid_score,
            above_threshold=True,
            best_time=_seconds(start_ms),
            unit_type="chunk",
            unit_id=segment_idx,
            best_ms=start_ms,
            text=text,
            features={
                "hybrid_score": hybrid_score,
                "source": "milvus_hybrid",
                "has_embedding": bool(hit.entity.get("has_embedding", True)),
            },
        ))

    return candidates


# ---------------------------------------------------------------------------
# OCR — DiskANN + BM25 hybrid search
# ---------------------------------------------------------------------------

def milvus_ocr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray | None,
    limit: int,
    profiler: RetrievalProfiler | None = None,
    rows: list[dict] | None = None,
) -> list[Candidate]:
    """OCR hybrid search: DiskANN (semantic) + BM25 (lexical).

    Uses Milvus Function Field for server-side BM25 computation and WeightedRanker
    for result fusion. Lexical-first strategy (sparse_weight > dense_weight).

    When query_embedding is None, falls back to BM25-only search.

    Args:
        client: Milvus client
        video_id: Video ID
        query_text: Query text (for BM25)
        query_embedding: Query embedding (for DiskANN), can be None
        limit: Number of results to return
        profiler: Performance profiler
        rows: DEPRECATED - ignored for compatibility

    Returns:
        List of Candidate objects with hybrid scores
    """
    from pymilvus import AnnSearchRequest, WeightedRanker

    if rows is not None:
        logger.debug(
            "milvus_ocr_candidates_hybrid: 'rows' parameter is deprecated "
            "and ignored in the hybrid search implementation."
        )

    settings = get_settings()
    col = client.collection_for("ocr")

    # Get configuration from settings
    recall_size = settings.ocr_hybrid_recall_size
    lexical_weight = settings.ocr_lexical_weight
    semantic_weight = 1.0 - lexical_weight
    search_list = settings.ocr_diskann_search_list

    # Handle None query_embedding - use BM25-only fallback
    if query_embedding is None:
        logger.info(
            "No semantic embedding for OCR, using BM25-only search: video_id=%s",
            video_id
        )
        if not query_text or not query_text.strip():
            logger.warning("Empty query for OCR with no semantic embedding, returning empty results: video_id=%s", video_id)
            return []

        with (profiler.span("milvus_rpc", "ocr_bm25_only") if profiler else nullcontext()):
            results = col.search(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=f'video_id == "{video_id}"',
                output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                              "text", "avg_box_score", "has_embedding"],
            )
    else:
        # Normalize query embedding for semantic search
        query_norm = normalize(np.asarray(query_embedding, dtype=np.float32))

        # Empty query: dense-only fallback
        if not query_text or not query_text.strip():
            logger.warning(
                "Empty query_text for OCR, falling back to dense-only: video_id=%s",
                video_id
            )
            with (profiler.span("milvus_rpc", "ocr_dense_only") if profiler else nullcontext()):
                results = col.search(
                    data=[query_norm.tolist()],
                    anns_field="embedding",
                    param={
                        "metric_type": "IP",
                        "params": {"search_list": search_list},
                    },
                    limit=limit,
                    expr=f'video_id == "{video_id}" AND has_embedding == True',
                    output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                                  "text", "avg_box_score", "has_embedding"],
                )
        else:
            # Hybrid search: Dense + Sparse
            dense_req = AnnSearchRequest(
                data=[query_norm.tolist()],
                anns_field="embedding",
                param={
                    "metric_type": "IP",
                    "params": {"search_list": search_list},
                },
                limit=recall_size,
                expr=f'video_id == "{video_id}" AND has_embedding == True',
            )

            sparse_req = AnnSearchRequest(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=recall_size,
                expr=f'video_id == "{video_id}"',
            )

            with (profiler.span("milvus_rpc", "ocr_hybrid") if profiler else nullcontext()):
                results = col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=WeightedRanker(semantic_weight, lexical_weight),
                    limit=limit,
                    output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                                  "text", "avg_box_score", "has_embedding"],
                )

    # Convert to Candidate objects (threshold will be applied globally later)
    candidates: list[Candidate] = []
    for hit in results[0]:
        hybrid_score = float(hit.score)
        # Note: above_threshold will be set to True initially and updated globally later
        # in search.py after collecting all candidates from all videos
        above_threshold = True
        frame_ms = int(hit.entity.get("frame_ms") or 0)
        start_ms = int(hit.entity.get("start_ms") or -1)
        end_ms = int(hit.entity.get("end_ms") or -1)
        text = str(hit.entity.get("text") or "")

        # Handle legacy data without frame windows
        if start_ms < 0:
            start_ms = max(0, frame_ms - 500)
            end_ms = frame_ms + 500

        evidence_text = f"[ocr_hybrid] {text[:100]} · hybrid={hybrid_score:.3f}"
        # Note: "低于阈值" suffix will be added later after global threshold calculation

        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=hybrid_score,
            modality="ocr",
            evidence=evidence_text,
            above_threshold=above_threshold,
            best_time=_seconds(frame_ms),
            unit_type="frame",
            unit_id=int(hit.entity.get("frame_idx") or 0),
            best_ms=frame_ms,
            text=text,
            features={
                "ocr_confidence": float(hit.entity.get("avg_box_score") or 0.0),
                "has_embedding": bool(hit.entity.get("has_embedding", True)),
            },
        ))

    return candidates


# ---------------------------------------------------------------------------
# Face — ANN search with absolute threshold (no distribution normalization)
# ---------------------------------------------------------------------------

def milvus_face_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    limit: int,
    threshold: float = 0.35,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Face track recall: ANN candidate expansion → exact cosine re-score → threshold.

    Two-phase approach:
    1. ANN search with expanded limit (limit * 2) to compensate for recall loss
       from approximate indexing.
    2. Retrieve embedding vectors alongside metadata; recompute exact cosine as
       dot(query_norm, track_norm) rather than trusting the ANN distance value.
       This eliminates floating-point approximation errors introduced by
       IVF_FLAT quantisation and L2↔cosine conversion.
    3. Apply the identity threshold on the exact cosine; sort and truncate.

    Face uses L2 metric on unit vectors.  The exact cosine is simply the dot
    product of two unit vectors — no conversion formula needed.
    """
    query_norm = normalize(np.asarray(query, dtype=np.float32))
    # Expand recall to guard against ANN miss-rate at the threshold boundary.
    ann_limit = min(limit * 2, 16_384)
    hits = _ann_search(
        client, "face", video_id, query_norm.tolist(),
        ann_limit,
        ["track_idx", "start_ms", "end_ms", "best_ms", "embedding"],
        profiler,
    )
    scoring_started = time.perf_counter()
    scored: list[tuple[float, dict]] = []
    for hit in hits:
        raw_emb = hit.get("embedding")
        if raw_emb is None:
            # Milvus reports squared L2 distance.  For unit vectors:
            # squared_l2 = 2 - 2*cosine.
            squared_l2 = float(hit["_distance"])
            cosine = max(-1.0, min(1.0, 1.0 - squared_l2 / 2.0))
        else:
            track_vec = normalize(np.asarray(raw_emb, dtype=np.float32))
            cosine = float(np.dot(query_norm, track_vec))
        scored.append((cosine, hit))

    # Sort by exact cosine descending, then truncate to requested limit.
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates: list[Candidate] = []
    for cosine, hit in scored[:limit]:
        above    = cosine >= threshold
        conf     = face_confidence(cosine)
        start_ms = int(hit.get("start_ms") or 0)
        end_ms   = int(hit.get("end_ms")   or 0)
        best_ms  = int(hit.get("best_ms")  or start_ms)
        detail   = f"[milvus] face cosine={cosine:.3f} · confidence={conf * 100:.1f}%"
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=conf,
            modality="face",
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=cosine,
            decision="absolute_hit" if above else "weak",
            above_threshold=above,
            best_time=_seconds(best_ms),
            unit_type="track",
            unit_id=int(hit.get("track_idx") or 0),
            best_ms=best_ms,
            features={"face_cosine": cosine, "source": "milvus"},
        ))
    if profiler:
        profiler.add_seconds(
            "local_processing",
            "face_scoring",
            time.perf_counter() - scoring_started,
        )
    return candidates


# ---------------------------------------------------------------------------
# Speaker — ANN candidate expansion + exact cosine re-score
# ---------------------------------------------------------------------------

def milvus_speaker_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    limit: int,
    threshold: float = 0.50,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Speaker utterance recall: ANN expansion → exact cosine re-score → threshold.

    Same two-phase strategy as face:
    1. ANN with expanded limit (HNSW COSINE metric).
    2. Recompute exact cosine from retrieved utterance embeddings to eliminate
       any HNSW approximation error near the identity threshold.
    3. Apply threshold, sort, truncate.

    threshold=0.50 is calibrated for CAM++ (3D-Speaker); same-speaker utterances
    typically land 0.6–0.9, different speakers below 0.4.
    """
    query_norm = normalize(np.asarray(query, dtype=np.float32))
    ann_limit  = min(limit * 2, 16_384)
    hits = _ann_search(
        client, "speaker", video_id, query_norm.tolist(),
        ann_limit,
        ["utterance_idx", "start_ms", "end_ms", "track_id", "asr_chunk_idx", "embedding"],
        profiler,
    )
    scored: list[tuple[float, dict]] = []
    for hit in hits:
        raw_emb = hit.get("embedding")
        if raw_emb is None:
            # COSINE metric: distance value IS the cosine similarity.
            cosine = float(hit["_distance"])
        else:
            utt_vec = normalize(np.asarray(raw_emb, dtype=np.float32))
            cosine  = float(np.dot(query_norm, utt_vec))
        scored.append((cosine, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates: list[Candidate] = []
    for cosine, hit in scored[:limit]:
        above    = cosine >= threshold
        start_ms = int(hit.get("start_ms") or 0)
        end_ms   = int(hit.get("end_ms")   or 0)
        track_id = int(hit.get("track_id") or -1)
        detail   = f"[milvus] speaker cosine={cosine:.3f} track_id={track_id}"
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=cosine,
            modality="speaker",
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=cosine,
            decision="absolute_hit" if above else "weak",
            above_threshold=above,
            best_time=_seconds(start_ms),
            unit_type="utterance",
            unit_id=int(hit.get("utterance_idx") or 0),
            best_ms=start_ms,
            features={
                "speaker_cosine": cosine,
                "track_id":       track_id,
                "asr_chunk_idx":  int(hit.get("asr_chunk_idx") or -1),
                "source":         "milvus",
            },
        ))
    return candidates


# ---------------------------------------------------------------------------
# Shadow compare helper
# ---------------------------------------------------------------------------

def shadow_compare_log(
    video_id: str,
    modality: str,
    npz_candidates: list[Candidate],
    milvus_candidates: list[Candidate],
    top_k: int = 5,
) -> None:
    """Log top-k divergence between NPZ and Milvus results for the same video+modality.

    Only above-threshold candidates are compared.  The Jaccard overlap on the
    top-k time intervals is reported as a single INFO log line.
    """
    def _top_intervals(cands: list[Candidate]) -> set[tuple[float, float]]:
        above = sorted(
            [c for c in cands if c.above_threshold],
            key=lambda c: c.score, reverse=True,
        )[:top_k]
        return {(round(c.start_time, 1), round(c.end_time, 1)) for c in above}

    npz_top    = _top_intervals(npz_candidates)
    milvus_top = _top_intervals(milvus_candidates)
    union      = npz_top | milvus_top
    inter      = npz_top & milvus_top
    jaccard    = len(inter) / max(1, len(union)) if union else 1.0
    logger.info(
        "shadow_compare video=%s modality=%s npz_total=%d milvus_total=%d "
        "npz_above=%d milvus_above=%d top_k=%d jaccard=%.2f "
        "only_npz=%s only_milvus=%s",
        video_id, modality,
        len(npz_candidates), len(milvus_candidates),
        len(npz_top), len(milvus_top),
        top_k, jaccard,
        sorted(npz_top - milvus_top),
        sorted(milvus_top - npz_top),
    )



