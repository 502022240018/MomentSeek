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

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from app.search import (
    Candidate,
    _asr_candidates,
    _seconds,
    face_confidence,
    normalize,
    robust_distribution,
    visual_confidence,
)

if TYPE_CHECKING:
    from app.indexing.milvus_client import MilvusClient

logger = logging.getLogger(__name__)

# Milvus ANN search params (used only for face / speaker).
_HNSW_EF    = 128
_IVF_NPROBE = 64

# Per-modality index config — must stay in sync with _COLLECTION_CONFIGS in
# milvus_client.py.  Stored here because pymilvus doesn't surface index params
# reliably via schema inspection.
_MODALITY_METRIC: dict[str, str] = {
    "visual":  "COSINE",
    "asr":     "IP",
    "ocr":     "IP",
    "face":    "L2",
    "speaker": "COSINE",
}
_MODALITY_INDEX_TYPE: dict[str, str] = {
    "visual":  "HNSW",
    "asr":     "HNSW",
    "ocr":     "HNSW",
    "face":    "IVF_FLAT",
    "speaker": "HNSW",
}

# Batch size for QueryIterator (and fallback offset-pagination).
# Milvus recommends iterator for entity traversal; 1 000–4 000 is a practical
# sweet-spot that keeps per-page latency low while amortising round-trip cost.
_QUERY_BATCH = 2_000


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


def _query_all(
    client: "MilvusClient",
    modality: str,
    video_id: str,
    output_fields: list[str],
) -> list[dict]:
    """Return ALL rows for *video_id* from the given modality collection.

    Uses QueryIterator (pymilvus ≥ 2.3) for cursor-based traversal, which
    avoids the deep-offset random-access penalty of the limit/offset pattern
    and is the approach recommended by Milvus for bulk entity retrieval.

    Falls back to limit/offset pagination on older pymilvus versions that do
    not expose query_iterator().  Both paths are semantically identical; the
    iterator path is preferred in production.

    output_fields are automatically filtered to fields present in the
    collection schema so that queries against collections built from older
    schema versions do not raise "field X not exist" errors.
    """
    col  = client.collection_for(modality)
    # Guard against schema drift — omit fields absent from the live collection.
    output_fields = _schema_available_fields(col, output_fields)
    expr = f'video_id == "{video_id}"'
    rows: list[dict] = []

    try:
        # --- QueryIterator path (pymilvus ≥ 2.3) ----------------------------
        if hasattr(col, "query_iterator"):
            iterator = col.query_iterator(
                batch_size=_QUERY_BATCH,
                expr=expr,
                output_fields=output_fields,
            )
            try:
                while True:
                    page = iterator.next()
                    if not page:
                        break
                    rows.extend(page)
            finally:
                iterator.close()
        else:
            # --- Offset-pagination fallback (pymilvus < 2.3) -----------------
            # Each page hits a different QueryNode shard; this is less
            # efficient than the iterator but functionally correct.
            offset = 0
            while True:
                page = col.query(
                    expr=expr,
                    output_fields=output_fields,
                    limit=_QUERY_BATCH,
                    offset=offset,
                )
                rows.extend(page)
                if len(page) < _QUERY_BATCH:
                    break
                offset += _QUERY_BATCH

    except MilvusServiceError:
        raise
    except Exception as exc:
        raise MilvusServiceError(
            f"Milvus query failed for modality={modality} video={video_id}: {exc}"
        ) from exc

    return rows


def _ann_search(
    client: "MilvusClient",
    modality: str,
    video_id: str,
    query: list[float],
    limit: int,
    output_fields: list[str],
) -> list[dict]:
    """Execute a per-video ANN search; used only by face and speaker."""
    col = client.collection_for(modality)
    metric     = _MODALITY_METRIC[modality]
    index_type = _MODALITY_INDEX_TYPE[modality]
    sp = (
        {"metric_type": metric, "params": {"ef": _HNSW_EF}}
        if index_type == "HNSW"
        else {"metric_type": metric, "params": {"nprobe": _IVF_NPROBE}}
    )
    try:
        results = col.search(
            data=[query],
            anns_field="embedding",
            param=sp,
            limit=limit,
            expr=f'video_id == "{video_id}"',
            output_fields=output_fields,
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
    return hits


# ---------------------------------------------------------------------------
# Visual — query-all + segment-aware distribution scoring
# ---------------------------------------------------------------------------

def milvus_visual_candidates(
    client: "MilvusClient",
    video_id: str,
    query: np.ndarray,
    duration_ms: int | None = None,
    segment_ms: int | None = None,
    profile: str = "balanced",
    limit: int = 72,
) -> list[Candidate]:
    """Full-video visual recall via Milvus; functionally equivalent to _visual_candidates().

    Fetches every frame embedding for the video, computes dot-products in Python,
    then applies the identical segment-aggregation and robust-distribution logic
    used by the NPZ path.  This is the only way to get a correct per-video score
    distribution — a top-k ANN search would distort the z-score and percentile
    normalization.

    Parameters duration_ms and segment_ms are now optional — when not provided,
    they are inferred from the Milvus data itself. They are kept as parameters
    for backward compatibility and as fallback values when Milvus data is incomplete.
    """
    rows = _query_all(
        client, "visual", video_id,
        ["frame_idx", "timestamp_ms", "segment_id", "segment_start_ms", "segment_end_ms", "embedding"],
    )
    if not rows:
        return []

    # Sort by frame_idx for deterministic ordering (query() order is undefined).
    rows.sort(key=lambda r: int(r.get("frame_idx") or 0))

    query_values = np.asarray(query, dtype=np.float32)
    if query_values.ndim == 1:
        query_values = query_values.reshape(1, -1)
    frame_embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    if query_values.ndim != 2 or query_values.shape[1] != frame_embeddings.shape[1]:
        raise ValueError("visual query embedding shape does not match the Milvus index")
    query_values = np.stack([normalize(value) for value in query_values])
    frame_scores = frame_embeddings @ query_values.T  # shape [frames, subqueries]
    frame_times      = [int(r.get("timestamp_ms") or 0) for r in rows]

    # Infer segment_ms from Milvus data if not provided
    if segment_ms is None:
        # Try to infer from explicit segment boundaries
        inferred_segment_ms = None
        for row in rows:
            start_value = row.get("segment_start_ms")
            end_value = row.get("segment_end_ms")
            ss = int(start_value) if start_value is not None else -1
            se = int(end_value) if end_value is not None else -1
            if ss >= 0 and se > ss:
                inferred_segment_ms = se - ss
                break
        segment_ms = inferred_segment_ms if inferred_segment_ms else 5000  # fallback: 5s default

    # Infer duration_ms from Milvus data if not provided
    if duration_ms is None:
        # Use the maximum timestamp_ms as duration estimate
        duration_ms = max(frame_times) if frame_times else 0

    # Group frames into segments.
    # segment_id == -1 means the field wasn't present in older index entries;
    # fall back to fixed-window bucketing via timestamp_ms // segment_ms.
    seg_to_frames: dict[int, list[tuple[np.ndarray, int, int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        seg_id_raw = row.get("segment_id")
        if seg_id_raw is None or int(seg_id_raw) < 0:
            seg_id = int(frame_times[idx]) // max(1, segment_ms)
        else:
            seg_id = int(seg_id_raw)
        seg_to_frames[seg_id].append((
            frame_scores[idx],
            frame_times[idx],
            (
                int(row["segment_start_ms"])
                if row.get("segment_start_ms") is not None
                else -1
            ),
            (
                int(row["segment_end_ms"])
                if row.get("segment_end_ms") is not None
                else -1
            ),
        ))

    # Aggregate per-segment statistics.
    segment_ids_list: list[int]   = []
    raw_scores:       list[float] = []
    top3_scores:      list[float] = []
    mean_scores:      list[float] = []
    subquery_scores:  list[list[float]] = []
    best_times_ms:    list[int]   = []
    seg_time_map:     dict[int, tuple[int, int]] = {}

    for seg_id in sorted(seg_to_frames):
        frames = seg_to_frames[seg_id]
        bucket = np.stack([s for s, _, _, _ in frames]).astype(np.float32)
        per_query_top = np.max(bucket, axis=0)
        if bucket.shape[1] == 1:
            aggregate_score = float(per_query_top[0])
            frame_aggregate = bucket[:, 0]
        else:
            aggregate_score = float(
                0.65 * np.mean(per_query_top) + 0.35 * np.min(per_query_top)
            )
            frame_aggregate = (
                0.65 * np.mean(bucket, axis=1) + 0.35 * np.min(bucket, axis=1)
            )
        order = np.argsort(frame_aggregate)[::-1]

        best_score_idx = int(order[0])
        best_ts = frames[best_score_idx][1]

        segment_ids_list.append(seg_id)
        raw_scores.append(aggregate_score)
        top3_scores.append(float(np.mean(frame_aggregate[order[:min(3, len(order))]])))
        mean_scores.append(float(np.mean(frame_aggregate)))
        subquery_scores.append([float(value) for value in per_query_top])
        best_times_ms.append(best_ts)

        # Segment time bounds: prefer explicit shot boundaries; fall back to fixed.
        ss = frames[0][2]
        se = frames[0][3]
        if ss >= 0 and se >= 0:
            seg_time_map[seg_id] = (ss, se)
        else:
            seg_time_map[seg_id] = (
                seg_id * segment_ms,
                min((seg_id + 1) * segment_ms, duration_ms or (seg_id + 1) * segment_ms),
            )

    if not raw_scores:
        return []

    raw_values  = np.asarray(raw_scores, dtype=np.float32)
    dist        = robust_distribution(raw_values)
    z_scores    = dist["z_scores"]
    percentiles = dist["percentiles"]
    reliable    = dist["reliable"]

    raw_order  = np.argsort(raw_values)[::-1]
    fb_counts  = {"recall": 3, "balanced": 2, "precision": 1}
    fb_indices = {int(i) for i in raw_order[:min(len(raw_order), fb_counts.get(profile, 2))]}

    candidates: list[Candidate] = []
    cap = 500 if profile == "recall" else limit
    for local_idx in raw_order[:cap]:
        local_idx   = int(local_idx)
        seg_id      = segment_ids_list[local_idx]
        raw         = float(raw_values[local_idx])
        z           = float(z_scores[local_idx])
        pct         = float(percentiles[local_idx])
        rank_score  = visual_confidence(raw)
        top3        = float(top3_scores[local_idx])
        mean        = float(mean_scores[local_idx])
        best_ms_val = int(best_times_ms[local_idx])

        if reliable:
            if z >= 2.0 or pct >= 0.975:
                decision, above = "strong", True
            elif pct >= 0.80:
                qualifies = not (
                    (profile == "balanced" and not (z >= 1.0 or pct >= 0.90))
                    or profile == "precision"
                )
                decision, above = ("fuzzy", True) if qualifies else ("weak", False)
            else:
                decision, above = "weak", False
            detail = (
                f"[milvus] visual score={raw:.3f} · rank_score={rank_score:.3f}"
                f" · percentile={pct * 100:.1f}% · robust_z={z:.2f}"
            )
        else:
            decision, above = ("fallback", True) if local_idx in fb_indices else ("weak", False)
            detail = (
                f"[milvus] visual score={raw:.3f} · rank_score={rank_score:.3f}"
                f" · distribution fallback (n={len(raw_values)})"
            )

        start_ms, end_ms = seg_time_map[seg_id]
        detail += f" · best_frame={best_ms_val / 1000:.2f}s · top1={raw:.3f} · top3={top3:.3f} · mean={mean:.3f}"
        if query_values.shape[0] > 1:
            detail += " · subqueries=" + ",".join(
                f"{value:.3f}" for value in subquery_scores[local_idx]
            )

        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=rank_score,
            modality="visual",
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=raw,
            robust_z=z,
            percentile=pct,
            decision=decision,
            above_threshold=above,
            distribution_reliable=reliable,
            distribution_median=dist["median"],
            distribution_mad=dist["mad"],
            best_time=_seconds(best_ms_val),
            visual_top1=raw,
            visual_top3=top3,
            visual_mean=mean,
            unit_type="segment",
            unit_id=seg_id,
            best_ms=best_ms_val,
            features={
                "visual_top1":           raw,
                "visual_top3":           top3,
                "visual_mean":           mean,
                "visual_rank_score":     rank_score,
                "visual_subquery_scores": subquery_scores[local_idx],
                "visual_subquery_count": int(query_values.shape[0]),
                "percentile":            pct,
                "robust_z":              z,
                "source":                "milvus",
                # Provide defaults so downstream code that reads these keys
                # (e.g. logs, analytics) works the same as the NPZ path.
                "segment_strategy":      "fixed",
                "segment_time_source":   "fixed",
            },
        ))
        if len(candidates) >= limit and profile != "recall":
            break
    return candidates


# ---------------------------------------------------------------------------
# ASR — query-all + full-video distribution scoring
# ---------------------------------------------------------------------------

def milvus_asr_candidates(
    client: "MilvusClient",
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
) -> list[Candidate]:
    """Full-video ASR recall via Milvus; functionally equivalent to the NPZ path.

    ALL ASR chunks are stored in Milvus (has_embedding=False for lexical-only
    chunks, has_embedding=True for chunks with real semantic vectors).  This
    mirrors the NPZ layout exactly:

      NPZ path:  _asr_chunks_from_npz()   → all N chunks (lexical base)
                 _semantic_arrays()        → sparse M embeddings + indices
      Milvus path: ALL rows from query()  → same N chunks
                   rows with has_embedding → same M embeddings + local indices

    The downstream _asr_candidates() call is therefore identical in both paths.
    """
    rows = _query_all(
        client, "asr", video_id,
        ["segment_idx", "start_ms", "end_ms", "text", "has_embedding", "embedding"],
    )
    if not rows:
        return []

    # Sort by segment_idx for deterministic ordering.
    rows.sort(key=lambda r: int(r.get("segment_idx") or 0))

    # Rebuild the complete chunks list — used for lexical scoring over ALL chunks,
    # exactly as _asr_chunks_from_npz() provides for the NPZ path.
    chunks: list[dict] = []
    for idx, row in enumerate(rows):
        chunks.append({
            "chunk_id": int(row.get("segment_idx") or idx),
            "start_ms": int(row.get("start_ms") or 0),
            "end_ms":   int(row.get("end_ms")   or 0),
            "text":     str(row.get("text") or ""),
        })

    # Rebuild sparse semantic arrays — only rows that carry a real embedding.
    # has_embedding defaults to True for rows written before the field was added
    # (old schema rows all had real embeddings).
    semantic_local_indices = [
        i for i, r in enumerate(rows)
        if r.get("has_embedding", True)
    ]
    if semantic_local_indices and query_embedding is not None:
        semantic_embeddings     = np.array(
            [rows[i]["embedding"] for i in semantic_local_indices], dtype=np.float32
        )
        # embedding_chunk_indices are LOCAL positions into the chunks list above
        # (not segment_idx values), matching the contract of _asr_candidates().
        embedding_chunk_indices = np.array(semantic_local_indices, dtype=np.int32)
    else:
        semantic_embeddings     = None
        embedding_chunk_indices = None

    return _asr_candidates(
        chunks, query_text, video_id, limit,
        modality="asr",
        semantic_embeddings=semantic_embeddings,
        embedding_chunk_indices=embedding_chunk_indices,
        semantic_query=query_embedding,
    )


# ---------------------------------------------------------------------------
# OCR — query-all + full-video distribution scoring
# ---------------------------------------------------------------------------

def milvus_ocr_candidates(
    client: "MilvusClient",
    video_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    limit: int,
) -> list[Candidate]:
    """Full-video OCR recall via Milvus; functionally equivalent to the NPZ path.

    ALL OCR frames are stored in Milvus (has_embedding=False for lexical-only
    frames, has_embedding=True for frames with real semantic vectors).  This
    mirrors the NPZ layout exactly:

      NPZ path:  _ocr_chunks_from_npz()   → all N frames (lexical base)
                 data["embeddings"]        → sparse M embeddings
                 data["embedding_frame_indices"] → sparse indices
      Milvus path: ALL rows from query()  → same N frames
                   rows with has_embedding → same M embeddings + local indices

    The downstream _asr_candidates(modality="ocr") call is identical in both paths.
    """
    rows = _query_all(
        client, "ocr", video_id,
        ["frame_idx", "region_idx", "frame_ms", "start_ms", "end_ms",
         "text", "avg_box_score", "has_embedding", "embedding"],
    )
    if not rows:
        return []

    # Sort by frame_ms for correct temporal ordering.
    rows.sort(key=lambda r: int(r.get("frame_ms") or 0))

    # Rebuild the complete chunks list — all frames for lexical scoring,
    # exactly as _ocr_chunks_from_npz() provides for the NPZ path.
    chunks: list[dict] = []
    for idx, row in enumerate(rows):
        frame_ms = int(row.get("frame_ms") or 0)
        start_ms = int(row.get("start_ms") or -1)
        end_ms   = int(row.get("end_ms")   or -1)
        # start_ms == -1 means an older index entry without frame windows.
        if start_ms < 0:
            start_ms = max(0, frame_ms - 500)
            end_ms   = frame_ms + 500
        chunks.append({
            "chunk_id":  idx,
            "start_ms":  start_ms,
            "end_ms":    end_ms,
            "frame_ms":  frame_ms,
            "text":      str(row.get("text") or ""),
            "score":     float(row.get("avg_box_score") or 0.0),
        })

    # Rebuild sparse semantic arrays — only rows with real embeddings.
    # has_embedding defaults to True for rows from old schema (all had real vectors).
    semantic_local_indices = [
        i for i, r in enumerate(rows)
        if r.get("has_embedding", True)
    ]
    if semantic_local_indices and query_embedding is not None:
        semantic_embeddings     = np.array(
            [rows[i]["embedding"] for i in semantic_local_indices], dtype=np.float32
        )
        # Local positions into the chunks list — matches the NPZ path contract.
        embedding_chunk_indices = np.array(semantic_local_indices, dtype=np.int32)
    else:
        semantic_embeddings     = None
        embedding_chunk_indices = None

    return _asr_candidates(
        chunks, query_text, video_id, limit,
        modality="ocr",
        semantic_embeddings=semantic_embeddings,
        embedding_chunk_indices=embedding_chunk_indices,
        semantic_query=query_embedding,
    )


# ---------------------------------------------------------------------------
# Face — ANN search with absolute threshold (no distribution normalization)
# ---------------------------------------------------------------------------

def milvus_face_candidates(
    client: "MilvusClient",
    video_id: str,
    query: np.ndarray,
    limit: int,
    threshold: float = 0.35,
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
    )
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
    return candidates


# ---------------------------------------------------------------------------
# Speaker — ANN candidate expansion + exact cosine re-score
# ---------------------------------------------------------------------------

def milvus_speaker_candidates(
    client: "MilvusClient",
    video_id: str,
    query: np.ndarray,
    limit: int,
    threshold: float = 0.50,
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
