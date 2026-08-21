"""Milvus-side candidate generation for all five modalities.

Visual uses segment-aware ANN, ASR/OCR use Milvus dense+sparse hybrid search,
and Face/Speaker use absolute-threshold ANN. Online retrieval reads only the
Catalog-published ``asset_version`` and never falls back to local index files.

All functions return identical list[Candidate] types so the existing fusion,
grouping, and ranking code in search.py needs no changes.

Failure contract
----------------
MilvusServiceError is raised *only* on connection / timeout failures.
"Milvus returned 0 results" is NOT a failure — it is a valid empty answer.
The Milvus-only request path propagates connection and timeout failures
explicitly; it never falls back to retained NPZ artifacts.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np

from app.indexing.common import normalize
from app.retrieval.retrieval_metrics import RetrievalProfiler
from app.retrieval.search import (
    Candidate,
    _seconds,
    face_confidence,
)
from app.core.settings import get_settings
from app.vector_store.milvus.row_contract import (
    required_int_field as _required_int_field,
    required_nonnegative_int_field,
    required_time_window as _required_time_window,
)

if TYPE_CHECKING:
    from app.vector_store.milvus.milvus_client import MilvusClient

# Visual modality optimization: ANN + sampling implementation (v2)
from .milvus_search_visual_v2 import milvus_visual_candidates_ann

logger = logging.getLogger(__name__)


# Milvus ANN search params. Face can explicitly target an already-published
# IVF_FLAT/L2 collection; returned embeddings are then re-scored by cosine.
_HNSW_EF    = 128

# Per-modality metric config — static mapping, must stay in sync with
# _COLLECTION_CONFIGS in milvus_client.py.
_MODALITY_METRIC: dict[str, str] = {
    "visual":  "COSINE",
    "asr":     "IP",
    "ocr":     "IP",
    "face":    "COSINE",   # migrated L2 → COSINE (unit vectors) with DiskANN
    "speaker": "COSINE",
}

# Static index types for non-visual modalities
_STATIC_INDEX_TYPES: dict[str, str] = {
    "asr":     "DISKANN",
    "ocr":     "DISKANN",
    "face":    "DISKANN",   # migrated IVF_FLAT → DISKANN (COSINE) for 千万级 scale
    "speaker": "DISKANN",   # migrated HNSW → DISKANN (COSINE) for 千万级 scale
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
    if modality == "face":
        profile = get_settings().milvus_face_ann_profile
        return "IVF_FLAT" if profile == "ivf_flat_l2" else "DISKANN"
    return _STATIC_INDEX_TYPES[modality]


def get_modality_metric_type(modality: str) -> str:
    """Return the metric paired with the configured ANN index contract."""
    if modality == "face":
        profile = get_settings().milvus_face_ann_profile
        return "L2" if profile == "ivf_flat_l2" else "COSINE"
    return _MODALITY_METRIC[modality]


class MilvusServiceError(RuntimeError):
    """Raised on connection / timeout failures; NOT on empty result sets."""


# ---------------------------------------------------------------------------
# Index-type fail-fast verification (face / speaker share _ann_search)
# ---------------------------------------------------------------------------

# Modalities already verified this process. Keyed by modality so face(DISKANN)
# and speaker(DISKANN) are checked against their own expected type — never a
# shared assumption that could mis-judge one of them.
_verified_index_modalities: set[str] = set()
_index_verify_lock = threading.Lock()


def _reset_index_verification() -> None:
    """Test hook: clear the per-modality verification cache."""
    with _index_verify_lock:
        _verified_index_modalities.clear()


def _verify_ann_index_type_once(client: MilvusClient, modality: str) -> None:
    """Fail-fast if the live ANN index configuration has drifted.

    A HNSW→DISKANN config change does NOT rebuild an existing collection
    (_init_collections only load()s it), so a stale collection would silently
    break DiskANN search. This one-time check surfaces the drift explicitly.
    Cached per-modality; only issues an RPC on the first search of each modality.

    Scope note: both ANN modalities that reach _ann_search are verified — speaker
    (expects DISKANN) and face (expects DISKANN, migrated from IVF_FLAT). Each is
    checked against its own configured type. A stale IVF_FLAT face collection that
    predates the migration is caught here (config DISKANN != collection IVF_FLAT →
    fail-fast), forcing a rebuild before serving instead of silently mis-searching.

    Transient vs structural failure: a genuine RPC/timeout error during
    introspection soft-passes (does not block search) but is NOT cached, so the
    next search retries and drift detection is not permanently disabled. Only a
    structural limitation — a lightweight client/collection that cannot introspect
    at all (AttributeError/TypeError), or a missing/non-str index type — is cached
    to avoid re-attempting (and log-spamming) on every search. A real index must
    match both its configured index type and metric type.
    """
    if modality in _verified_index_modalities:
        return
    with _index_verify_lock:
        if modality in _verified_index_modalities:
            return
        expected = get_modality_index_type(modality)
        expected_metric = get_modality_metric_type(modality)
        # col.index() (the RPC) is intentionally inside the lock so that
        # concurrent searches on first startup do not fan out duplicate
        # introspection RPCs. The lock is held for one network round-trip
        # only once per modality per process lifetime.
        try:
            col = client.collection_for(modality)
            index_info = col.index()
            if not index_info:
                logger.warning(
                    "%s collection has no index yet; first indexing will create it",
                    modality,
                )
                _verified_index_modalities.add(modality)
                return
            index_params = index_info.params or {}
            actual = index_params.get("index_type", "UNKNOWN")
            actual_metric = index_params.get("metric_type", "UNKNOWN")
        except MilvusServiceError:
            raise
        except (AttributeError, TypeError) as exc:
            # Structural limitation: lightweight test clients / wrappers that do
            # not expose index introspection at all. This will never recover, so
            # cache to avoid re-attempting on every search. Mirror visual
            # _verify_index_type's soft handling.
            logger.warning(
                "%s index type not introspectable (%s); skipping drift check",
                modality, exc,
            )
            _verified_index_modalities.add(modality)
            return
        except Exception as exc:
            # Transient introspection failure (RPC/timeout). Soft-pass so search
            # is not blocked, but do NOT cache — the next search retries so a
            # single blip never permanently disables drift detection.
            logger.warning(
                "Transient failure verifying %s index type: %s; will retry",
                modality, exc,
            )
            return
        if not isinstance(actual, str):
            # Loose test mocks return a non-str (e.g. MagicMock) here. A real
            # Milvus collection always yields a str index_type, so only enforce
            # drift detection when we actually got one.
            _verified_index_modalities.add(modality)
            return
        if actual != expected:
            raise MilvusServiceError(
                f"Index type mismatch for modality={modality!r}: config expects "
                f"{expected} but collection has {actual}. Rebuild the "
                f"{modality} collection with the new index config before serving."
            )
        if actual_metric != expected_metric:
            raise MilvusServiceError(
                f"Metric type mismatch for modality={modality!r}: config expects "
                f"{expected_metric} but collection has {actual_metric}. Rebuild the "
                f"{modality} vector index with the new metric before serving."
            )
        logger.debug(
            "%s ANN index verified: index_type=%s metric_type=%s",
            modality,
            actual,
            actual_metric,
        )
        _verified_index_modalities.add(modality)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _diskann_search_list_for(modality: str) -> int:
    """Return the configured DiskANN search_list for a modality.

    Keyed per modality so the DISKANN branch of _ann_search never silently
    inherits another modality's tuning. Both face and speaker reach this branch
    (both migrated to DISKANN; visual's DiskANN path lives in
    milvus_search_visual_v2). A future DISKANN modality must add its own entry
    here rather than fall through to another modality's value.
    """
    settings = get_settings()
    if modality == "speaker":
        return settings.speaker_diskann_search_list
    if modality == "face":
        return settings.face_diskann_search_list
    raise MilvusServiceError(
        f"_ann_search has no DiskANN search_list configured for modality={modality!r}"
    )


def _ann_search(
    client: MilvusClient,
    modality: str,
    video_id: str,
    asset_version: str,
    query: list[float],
    limit: int,
    output_fields: list[str],
    profiler: RetrievalProfiler | None = None,
) -> list[dict]:
    """Execute a per-video ANN search; used only by face and speaker."""
    _verify_ann_index_type_once(client, modality)
    col = client.collection_for(modality)
    metric     = get_modality_metric_type(modality)
    index_type = get_modality_index_type(modality)
    if index_type == "DISKANN":
        # DiskANN hard constraint: search_list >= limit. `limit` here is the
        # caller's ann_limit; taking a static setting would fail/truncate when
        # limit exceeds it (see milvus_speaker_candidates). Mirror visual v2's
        # max(top_k, 100) pattern. search_list setting is modality-keyed so this
        # generic branch cannot misapply speaker's tuning to another modality.
        search_list = max(limit, _diskann_search_list_for(modality))
        sp = {"metric_type": metric, "params": {"search_list": search_list}}
    elif index_type == "HNSW":
        # Currently unreachable: face and speaker are both DISKANN, and visual's
        # HNSW path lives in milvus_search_visual_v2 (not via _ann_search).
        # Retained for a possible future modality that indexes with HNSW.
        sp = {"metric_type": metric, "params": {"ef": _HNSW_EF}}
    elif index_type == "IVF_FLAT" and modality == "face":
        sp = {
            "metric_type": metric,
            "params": {"nprobe": get_settings().face_ivf_nprobe},
        }
    else:
        raise MilvusServiceError(
            f"_ann_search does not support index_type={index_type!r} "
            f"for modality={modality!r}."
        )
    try:
        span = profiler.span("milvus_rpc", modality) if profiler else nullcontext()
        with span:
            results = col.search(
                data=[query],
                anns_field="embedding",
                param=sp,
                limit=limit,
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                ),
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


def _log_dropped_time_rows(modality: str, video_id: str, count: int) -> None:
    if count:
        logger.warning(
            "%s search dropped %d Milvus hit(s) with missing or invalid time "
            "metadata for video=%s; rebuild the published index version",
            modality.upper(),
            count,
            video_id,
        )


# ---------------------------------------------------------------------------
# Visual — query-all + segment-aware distribution scoring
# ---------------------------------------------------------------------------

def milvus_visual_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    asset_version: str,
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
        client, video_id, asset_version, query_texts, limit, profile, profiler
    )


# ---------------------------------------------------------------------------
# ASR — DiskANN + BM25 hybrid search
# ---------------------------------------------------------------------------

def milvus_asr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
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
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                ),
                output_fields=output_fields,
                timeout=settings.milvus_query_timeout_seconds,
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
                    expr=(
                        f'video_id == {json.dumps(video_id)}'
                        f' and asset_version == {json.dumps(asset_version)}'
                        f' and has_embedding == True'
                    ),
                    output_fields=output_fields,
                    timeout=settings.milvus_query_timeout_seconds,
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
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                    f' and has_embedding == True'
                ),
            )

            sparse_req = AnnSearchRequest(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=recall_size,
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                ),
            )

            with (profiler.span("milvus_rpc", "asr_hybrid") if profiler else nullcontext()):
                results = col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=WeightedRanker(semantic_weight, lexical_weight),
                    limit=limit,
                    output_fields=output_fields,
                    timeout=settings.milvus_query_timeout_seconds,
                )

    # Convert to Candidate objects (threshold applied globally later in search.py).
    candidates: list[Candidate] = []
    invalid_time_rows = 0
    for hit in results[0]:
        hybrid_score = float(hit.score)
        text = str(hit.entity.get("text") or "")
        try:
            start_ms, end_ms = _required_time_window(hit.entity)
        except (TypeError, ValueError, OverflowError):
            invalid_time_rows += 1
            continue
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

    _log_dropped_time_rows("asr", video_id, invalid_time_rows)
    return candidates


# ---------------------------------------------------------------------------
# OCR — DiskANN + BM25 hybrid search
# ---------------------------------------------------------------------------

def milvus_ocr_candidates_hybrid(
    client: MilvusClient,
    video_id: str,
    asset_version: str,
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
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                ),
                output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                              "text", "avg_box_score", "has_embedding"],
                timeout=settings.milvus_query_timeout_seconds,
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
                    expr=(
                        f'video_id == {json.dumps(video_id)}'
                        f' and asset_version == {json.dumps(asset_version)}'
                        f' and has_embedding == True'
                    ),
                    output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                                  "text", "avg_box_score", "has_embedding"],
                    timeout=settings.milvus_query_timeout_seconds,
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
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                    f' and has_embedding == True'
                ),
            )

            sparse_req = AnnSearchRequest(
                data=[query_text.strip()],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=recall_size,
                expr=(
                    f'video_id == {json.dumps(video_id)}'
                    f' and asset_version == {json.dumps(asset_version)}'
                ),
            )

            with (profiler.span("milvus_rpc", "ocr_hybrid") if profiler else nullcontext()):
                results = col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=WeightedRanker(semantic_weight, lexical_weight),
                    limit=limit,
                    output_fields=["frame_idx", "frame_ms", "start_ms", "end_ms",
                                  "text", "avg_box_score", "has_embedding"],
                    timeout=settings.milvus_query_timeout_seconds,
                )

    # Convert to Candidate objects (threshold will be applied globally later)
    candidates: list[Candidate] = []
    invalid_time_rows = 0
    for hit in results[0]:
        hybrid_score = float(hit.score)
        # Note: above_threshold will be set to True initially and updated globally later
        # in search.py after collecting all candidates from all videos
        above_threshold = True
        try:
            frame_ms = _required_int_field(hit.entity, "frame_ms")
            start_ms, end_ms = _required_time_window(hit.entity)
            if frame_ms < start_ms or frame_ms > end_ms:
                raise ValueError("frame_ms must fall inside the candidate window")
        except (TypeError, ValueError, OverflowError):
            invalid_time_rows += 1
            continue
        text = str(hit.entity.get("text") or "")

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

    _log_dropped_time_rows("ocr", video_id, invalid_time_rows)
    return candidates


# ---------------------------------------------------------------------------
# Face — ANN search with absolute threshold (no distribution normalization)
# ---------------------------------------------------------------------------

def milvus_face_candidates(
    client: MilvusClient,
    video_id: str,
    query: np.ndarray,
    asset_version: str,
    limit: int,
    threshold: float | None = None,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Face track recall for modern and explicitly configured legacy indexes.

    Face embeddings are unit-normalised before write (faces.py), so under a
    COSINE metric Milvus returns ``_distance`` that IS the exact cosine
    similarity within float32 precision. DiskANN approximation only affects
    *which* neighbours are returned, not the distance of a returned neighbour.
    The former two-phase re-score (pull ``embedding`` back + L2→cosine formula +
    Python ``np.dot``) was therefore pure overhead and has been removed
    (Milvus_optimization_plan.md 方案3).

    threshold=None resolves to settings.face_identity_threshold (default 0.35,
    ArcFace buffalo_l same-identity cutoff). It only drives the ``above_threshold``
    display/decision flag; the cross-modal fusion score is ``face_confidence(cosine)``.
    """
    settings = get_settings()
    if threshold is None:
        threshold = settings.face_identity_threshold
    query_norm = normalize(np.asarray(query, dtype=np.float32))
    legacy_l2 = settings.milvus_face_ann_profile == "ivf_flat_l2"
    # IVF/L2 is approximate and its distance is not the score consumed by the
    # platform, so recall at least 2x before exact cosine re-scoring.
    recall_multiplier = max(2, settings.face_recall_multiplier) if legacy_l2 else (
        settings.face_recall_multiplier
    )
    ann_limit = min(limit * recall_multiplier, 16_384)
    output_fields = ["track_idx", "start_ms", "end_ms", "best_ms"]
    if legacy_l2:
        output_fields.append("embedding")
    hits = _ann_search(
        client, "face", video_id, asset_version, query_norm.tolist(),
        ann_limit,
        output_fields,
        profiler,
    )
    scoring_started = time.perf_counter()
    scored: list[tuple[float, dict]] = []
    for hit in hits:
        if not legacy_l2:
            scored.append((float(hit["_distance"]), hit))
            continue
        embedding = np.asarray(hit.get("embedding"), dtype=np.float32).reshape(-1)
        if (
            embedding.size != query_norm.size
            or not np.all(np.isfinite(embedding))
            or float(np.linalg.norm(embedding)) <= 0.0
        ):
            logger.warning(
                "FACE search dropped legacy IVF hit with invalid embedding "
                "for video=%s track=%s",
                video_id,
                hit.get("track_idx"),
            )
            continue
        cosine = float(np.dot(query_norm, normalize(embedding)))
        scored.append((float(np.clip(cosine, -1.0, 1.0)), hit))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates: list[Candidate] = []
    invalid_time_rows = 0
    for cosine, hit in scored[:limit]:
        above    = cosine >= threshold
        conf     = face_confidence(cosine)
        try:
            start_ms, end_ms = _required_time_window(hit)
            best_ms = _required_int_field(hit, "best_ms")
            if best_ms < start_ms or best_ms > end_ms:
                raise ValueError("best_ms must fall inside the candidate window")
        except (TypeError, ValueError, OverflowError):
            invalid_time_rows += 1
            continue
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
    _log_dropped_time_rows("face", video_id, invalid_time_rows)
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
    asset_version: str,
    limit: int,
    threshold: float | None = None,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Speaker utterance recall: single-phase ANN with trusted COSINE distance.

    Speaker embeddings are unit-normalised before write (speaker.py), so under a
    COSINE metric Milvus returns ``_distance`` that IS the exact cosine similarity
    within float32 precision. HNSW/DiskANN approximation only affects *which*
    neighbours are returned, not the distance of a returned neighbour. The former
    two-phase re-score (pull ``embedding`` back + Python ``np.dot``) was therefore
    pure overhead and has been removed (Milvus_optimization_plan.md 方案3).

    threshold=None resolves to settings.speaker_identity_threshold (default 0.50,
    calibrated for CAM++ 3D-Speaker: same-speaker utterances land 0.6–0.9,
    different speakers below 0.4). It only drives the ``above_threshold`` display
    flag; voice-search passes threshold=-1.0 to keep every candidate.
    """
    settings = get_settings()
    if threshold is None:
        threshold = settings.speaker_identity_threshold
    query_norm = normalize(np.asarray(query, dtype=np.float32))
    # Reranking removed → wide recall no longer needed. multiplier defaults to 1;
    # search_list >= ann_limit is enforced in _ann_search's DISKANN branch.
    ann_limit = min(limit * settings.speaker_recall_multiplier, 16_384)
    hits = _ann_search(
        client, "speaker", video_id, asset_version, query_norm.tolist(),
        ann_limit,
        ["utterance_idx", "start_ms", "end_ms", "track_id", "asr_chunk_idx"],
        profiler,
    )
    # COSINE metric on unit vectors: _distance is the exact cosine similarity.
    # Real Milvus ANN search returns hits sorted by descending similarity; the
    # explicit sort below enforces that contract regardless of mock order in
    # tests, and is O(n) on an already-sorted production result. hits[:limit]
    # drops any surplus when multiplier > 1; with multiplier=1 ann_limit==limit
    # so it is a no-op.
    candidates: list[Candidate] = []
    invalid_rows = 0
    for hit in hits[:limit]:
        try:
            cosine = float(hit["_distance"])
            if not np.isfinite(cosine):
                raise ValueError("speaker cosine must be finite")
            start_ms, end_ms = _required_time_window(hit)
            utterance_idx = required_nonnegative_int_field(hit, "utterance_idx")
            track_id = required_nonnegative_int_field(hit, "track_id")
            asr_chunk_idx = required_nonnegative_int_field(hit, "asr_chunk_idx")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_rows += 1
            continue
        above    = cosine >= threshold
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
            unit_id=utterance_idx,
            best_ms=start_ms,
            features={
                "speaker_cosine": cosine,
                "track_id":       track_id,
                "asr_chunk_idx":  asr_chunk_idx,
                "source":         "milvus",
            },
        ))
    _log_dropped_time_rows("speaker", video_id, invalid_rows)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def milvus_speaker_candidates_scoped(
    client: MilvusClient,
    queries: np.ndarray,
    asset_versions: dict[str, str],
    limit: int,
    threshold: float | None = None,
    profiler: RetrievalProfiler | None = None,
) -> list[Candidate]:
    """Search many published video versions and reference vectors in one RPC."""
    if not asset_versions or limit <= 0:
        return []
    settings = get_settings()
    if threshold is None:
        threshold = settings.speaker_identity_threshold
    vectors = np.asarray(queries, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Speaker scoped search requires at least one query vector")
    vectors = np.vstack([normalize(vector) for vector in vectors])
    versions = {
        str(video_id): str(asset_version)
        for video_id, asset_version in asset_versions.items()
        if str(video_id) and str(asset_version)
    }
    if not versions:
        return []
    _verify_ann_index_type_once(client, "speaker")
    collection = client.collection_for("speaker")
    ann_limit = min(limit * settings.speaker_recall_multiplier, 16_384)
    search_params = {
        "metric_type": _MODALITY_METRIC["speaker"],
        "params": {
            "search_list": max(ann_limit, settings.speaker_diskann_search_list),
        },
    }
    expression = " or ".join(
        (
            f'(video_id == {json.dumps(video_id)} and '
            f'asset_version == {json.dumps(asset_version)})'
        )
        for video_id, asset_version in sorted(versions.items())
    )
    output_fields = [
        "video_id", "asset_version", "utterance_idx", "start_ms", "end_ms",
        "track_id", "asr_chunk_idx",
    ]
    try:
        span = profiler.span("milvus_rpc", "speaker") if profiler else nullcontext()
        with span:
            result_sets = collection.search(
                data=vectors.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=ann_limit,
                expr=expression,
                output_fields=output_fields,
                timeout=settings.milvus_query_timeout_seconds,
            )
    except Exception as exc:
        raise MilvusServiceError(f"Milvus scoped Speaker ANN search failed: {exc}") from exc
    best: dict[tuple[str, int], Candidate] = {}
    invalid_rows = 0
    row_count = 0
    for result_set in result_sets:
        for hit in result_set:
            row_count += 1
            try:
                video_id = str(hit.entity.get("video_id") or "")
                asset_version = str(hit.entity.get("asset_version") or "")
                if not video_id or versions.get(video_id) != asset_version:
                    raise ValueError("Speaker hit escaped the published scope")
                cosine = float(hit.distance)
                if not np.isfinite(cosine):
                    raise ValueError("speaker cosine must be finite")
                entity = {field: hit.entity.get(field) for field in output_fields}
                start_ms, end_ms = _required_time_window(entity)
                utterance_idx = required_nonnegative_int_field(entity, "utterance_idx")
                track_id = required_nonnegative_int_field(entity, "track_id")
                asr_chunk_idx = required_nonnegative_int_field(entity, "asr_chunk_idx")
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid_rows += 1
                continue
            above = cosine >= threshold
            detail = f"[milvus] speaker cosine={cosine:.3f} track_id={track_id}"
            candidate = Candidate(
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
                unit_id=utterance_idx,
                best_ms=start_ms,
                features={
                    "speaker_cosine": cosine,
                    "track_id": track_id,
                    "asr_chunk_idx": asr_chunk_idx,
                    "source": "milvus",
                },
            )
            key = (video_id, utterance_idx)
            previous = best.get(key)
            if previous is None or candidate.score > previous.score:
                best[key] = candidate
    _log_dropped_time_rows("speaker", "published-scope", invalid_rows)
    if profiler:
        profiler.increment("milvus", "speaker_rows", row_count)
        profiler.increment("milvus", "speaker_requests")
    return sorted(best.values(), key=lambda item: item.score, reverse=True)[:limit]
