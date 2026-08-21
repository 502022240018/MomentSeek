"""Milvus-only index writers for all five online modalities.

Every indexer accepts in-memory arrays and writes them directly to Milvus.  The
write layer deliberately has no file import, NPZ recovery, or fallback API: a
failed write aborts the indexing attempt and its unpublished ``asset_version``
remains invisible to readers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .milvus_schema import (
    EMBEDDING_DIMS,
    MODEL_VERSIONS,
    asr_pk,
    face_pk,
    face_group_pk,
    ocr_pk,
    speaker_pk,
    truncate_text_for_milvus,
    visual_pk,
)

if TYPE_CHECKING:
    from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# P1: Adaptive batch sizing — target ~256 KB payload per upsert RPC call.
# ---------------------------------------------------------------------------

_BATCH_TARGET_BYTES: int = 256 * 1024

# Estimated non-vector bytes per row for each modality (pk + scalar fields).
_METADATA_BYTES: dict[str, int] = {
    "visual":  256,   # pk + video_id + timestamp_ms + segment fields
    "asr":     512,   # pk + text (up to 2000 chars) + scalar fields
    "ocr":     512,   # pk + text + avg_box_score + scalar fields
    "face":    128,   # pk + start_ms / end_ms / best_ms
    "speaker": 128,   # pk + start_ms / end_ms + asr_chunk_idx + track_id
}


def _calc_batch_size(modality: str) -> int:
    """Derive a reasonable batch size so each upsert RPC stays near _BATCH_TARGET_BYTES."""
    dim = EMBEDDING_DIMS[modality]
    row_bytes = dim * 4 + _METADATA_BYTES.get(modality, 256)  # float32 = 4 bytes/element
    return max(50, min(500, _BATCH_TARGET_BYTES // row_bytes))


# Pre-computed per-modality batch sizes (computed once at import time).
# Expected values at 256 KB target:
#   visual  → ~55 rows  (1152*4+256 ≈ 4864 B/row)
#   asr     → ~115 rows ( 384*4+512 ≈ 2048 B/row)
#   ocr     → ~115 rows ( 384*4+512 ≈ 2048 B/row)
#   face    → ~120 rows ( 512*4+128 ≈ 2176 B/row)
#   speaker → ~290 rows ( 192*4+128 ≈  896 B/row)
_MODALITY_BATCH: dict[str, int] = {mod: _calc_batch_size(mod) for mod in EMBEDDING_DIMS}

# Fallback for callers that do not pass modality (backwards-compat only).
_BATCH: int = 200

# ---------------------------------------------------------------------------
# P0-B: Resilient upsert with exponential-backoff retry.
# ---------------------------------------------------------------------------

# MilvusException error codes that indicate a transient failure worth retrying.
# Extend this set as additional transient codes are observed in production.
_RETRYABLE_CODES: frozenset[int] = frozenset({
    1,     # UnexpectedError — often a transient RPC issue
    9999,  # RateLimit
})

_RETRY_MAX: int = 3            # retries after the initial attempt
_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles each attempt (1 → 2 → 4)


def _upsert_with_retry(
    collection,
    batch: list[dict],
    *,
    max_retries: int = _RETRY_MAX,
    base_delay: float = _RETRY_BASE_DELAY,
) -> None:
    """Upsert one batch with exponential-backoff retry on transient errors.

    Permanent errors (schema mismatch, invalid data) are re-raised immediately.
    Transient MilvusException codes listed in _RETRYABLE_CODES are retried up to
    max_retries times.  Non-MilvusException transport errors are also retried.
    """
    try:
        from pymilvus.exceptions import MilvusException as _MilvusExc
    except ImportError:
        # pymilvus not installed (stub / unit-test environment) — no retry wrapper.
        collection.upsert(batch)
        return

    for attempt in range(max_retries + 1):
        try:
            collection.upsert(batch)
            return
        except _MilvusExc as exc:
            is_last = attempt == max_retries
            if exc.code not in _RETRYABLE_CODES or is_last:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Milvus upsert transient error code=%s attempt=%d/%d, "
                "retrying in %.1fs: %s",
                exc.code, attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)
        except Exception:
            # Non-MilvusException (socket reset, OS timeout, etc.) — also retry.
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Milvus upsert exception attempt=%d/%d, retrying in %.1fs",
                attempt + 1, max_retries, delay,
                exc_info=True,
            )
            time.sleep(delay)


@dataclass
class MilvusWriteContext:
    """Carries per-job metadata; injected into build_* functions."""

    video_id: str
    asset_version: str
    client: "MilvusClient"
    # Optional overrides — defaults from milvus_schema.MODEL_VERSIONS
    model_versions: dict[str, str] = field(default_factory=dict)

    def model_ver(self, modality: str) -> str:
        return self.model_versions.get(modality, MODEL_VERSIONS[modality])


# ---------------------------------------------------------------------------
# Internal batch-upsert helper
# ---------------------------------------------------------------------------

def _upsert_batched(collection, rows: list[dict], modality: str = "") -> int:
    """Split rows into modality-appropriate batches and upsert each with retry.

    Args:
        collection: Milvus Collection instance.
        rows:       Rows to upsert.
        modality:   Modality name used to look up the adaptive batch size.
                    Falls back to _BATCH (200) when omitted or unknown.

    Returns:
        Total number of rows upserted.
    """
    batch_size = _MODALITY_BATCH.get(modality, _BATCH)
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _upsert_with_retry(collection, batch)
        total += len(batch)
    return total


# ---------------------------------------------------------------------------
# Modality indexers
# ---------------------------------------------------------------------------

class VisualMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        embeddings: np.ndarray,
        frame_times_ms: np.ndarray,
        segment_frame_offsets: np.ndarray,
        segment_times_ms: np.ndarray,
        duration_ms: int,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        times_ms = np.asarray(frame_times_ms, dtype=np.int64)
        offsets = np.asarray(segment_frame_offsets, dtype=np.int64)
        segment_times = np.asarray(segment_times_ms, dtype=np.int64)
        duration_ms = int(duration_ms)

        if embeddings.ndim != 2:
            raise ValueError("visual embeddings must be a 2-D array")
        if embeddings.shape[1] != EMBEDDING_DIMS["visual"]:
            raise ValueError(
                "visual embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMS['visual']}, got {embeddings.shape[1]}"
            )
        frame_count = int(embeddings.shape[0])
        if times_ms.ndim != 1 or len(times_ms) != frame_count:
            raise ValueError("visual frame_times_ms must be a 1-D array matching embeddings")
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError("visual segment_frame_offsets must contain at least start and end")
        segment_count = len(offsets) - 1
        if segment_times.shape != (segment_count, 2):
            raise ValueError(
                "visual segment_times_ms must have shape "
                f"({segment_count}, 2), got {segment_times.shape}"
            )
        if duration_ms <= 0:
            raise ValueError("visual duration_ms must be positive")
        if int(offsets[0]) != 0 or int(offsets[-1]) != frame_count:
            raise ValueError(
                "visual segment_frame_offsets must cover every frame exactly "
                f"(expected 0..{frame_count})"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError("visual segment_frame_offsets must be non-decreasing")
        starts = segment_times[:, 0]
        ends = segment_times[:, 1]
        if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends > duration_ms):
            raise ValueError(
                "visual segment bounds must satisfy 0 <= start < end <= duration_ms"
            )
        if len(segment_times) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError("visual segment bounds must be ordered and non-overlapping")
        if np.any(times_ms < 0) or np.any(times_ms > duration_ms):
            raise ValueError("visual frame timestamps must be within video duration")

        frame_segment_ids = np.full(frame_count, -1, dtype=np.int32)
        frame_seg_start_ms = np.full(frame_count, -1, dtype=np.int64)
        frame_seg_end_ms = np.full(frame_count, -1, dtype=np.int64)
        for seg_idx in range(segment_count):
            start_f = int(offsets[seg_idx])
            end_f = int(offsets[seg_idx + 1])
            segment_start_ms = int(segment_times[seg_idx, 0])
            segment_end_ms = int(segment_times[seg_idx, 1])
            segment_frame_times = times_ms[start_f:end_f]
            if np.any(segment_frame_times < segment_start_ms):
                raise ValueError(
                    f"visual segment {seg_idx} contains a frame before its start boundary"
                )
            if segment_end_ms < duration_ms:
                outside_end = segment_frame_times >= segment_end_ms
            else:
                outside_end = segment_frame_times > segment_end_ms
            if np.any(outside_end):
                raise ValueError(
                    f"visual segment {seg_idx} contains a frame outside its end boundary"
                )
            frame_segment_ids[start_f:end_f] = seg_idx
            frame_seg_start_ms[start_f:end_f] = segment_start_ms
            frame_seg_end_ms[start_f:end_f] = segment_end_ms

        if frame_count and np.any(frame_segment_ids < 0):
            raise ValueError("visual segment offsets left one or more frames unassigned")
        if frame_count == 0:
            return 0

        model_ver = ctx.model_ver("visual")
        col = ctx.client.collection_for("visual")
        rows = [
            {
                "pk":               visual_pk(ctx.video_id, ctx.asset_version, idx, model_ver),
                "video_id":         ctx.video_id,
                "asset_version":    ctx.asset_version,
                "model_version":    model_ver,
                "frame_idx":        idx,
                "timestamp_ms":     int(times_ms[idx]),
                "segment_id":       int(frame_segment_ids[idx]),
                "segment_start_ms": int(frame_seg_start_ms[idx]),
                "segment_end_ms":   int(frame_seg_end_ms[idx]),
                "embedding":        embeddings[idx].tolist(),
            }
            for idx in range(len(embeddings))
        ]
        return _upsert_batched(col, rows, "visual")


class AsrMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        chunk_times_ms: np.ndarray,
        texts: "list[str]",
        embeddings: "np.ndarray | None" = None,
        embedding_chunk_indices: "np.ndarray | None" = None,
    ) -> int:
        """P2 direct path: build rows from in-memory data and upsert."""
        times    = np.asarray(chunk_times_ms, dtype=np.int32)
        n_chunks = len(times)
        if n_chunks == 0:
            return 0

        # Build chunk_idx → embedding mapping.
        has_semantic = embeddings is not None and embedding_chunk_indices is not None
        chunk_to_embedding: dict[int, np.ndarray] = {}
        if has_semantic:
            emb_arr     = np.asarray(embeddings, dtype=np.float32)
            embed_idx_a = np.asarray(embedding_chunk_indices, dtype=np.int32)
            dim = emb_arr.shape[1] if emb_arr.ndim == 2 else EMBEDDING_DIMS["asr"]
            for e_idx, c_idx in enumerate(embed_idx_a):
                c_idx = int(c_idx)
                if 0 <= c_idx < n_chunks:
                    chunk_to_embedding[c_idx] = emb_arr[e_idx]
        else:
            dim = EMBEDDING_DIMS["asr"]

        zero_vec  = [0.0] * dim
        model_ver = ctx.model_ver("asr")
        col       = ctx.client.collection_for("asr")
        schema_fields      = {f.name for f in col.schema.fields}
        write_has_embedding = "has_embedding" in schema_fields
        if not write_has_embedding:
            logger.warning(
                "ASR collection is missing 'has_embedding' field — "
                "run migrate_milvus_schema.py to upgrade the schema"
            )
        rows = []
        for chunk_idx in range(n_chunks):
            emb     = chunk_to_embedding.get(chunk_idx)
            has_emb = emb is not None
            row = {
                "pk":            asr_pk(ctx.video_id, ctx.asset_version, chunk_idx, model_ver),
                "video_id":      ctx.video_id,
                "asset_version": ctx.asset_version,
                "model_version": model_ver,
                "segment_idx":   chunk_idx,
                "start_ms":      int(times[chunk_idx, 0]),
                "end_ms":        int(times[chunk_idx, 1]),
                "text":          truncate_text_for_milvus(texts[chunk_idx]) if chunk_idx < len(texts) else "",
                "embedding":     emb.tolist() if has_emb else zero_vec,
            }
            if write_has_embedding:
                row["has_embedding"] = has_emb
            rows.append(row)
        return _upsert_batched(col, rows, "asr")


class OcrMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        frame_times_ms: np.ndarray,
        frame_windows_ms: np.ndarray,
        embeddings: "np.ndarray | None" = None,
        embedding_frame_indices: "np.ndarray | None" = None,
        box_frame_indices: "np.ndarray | None" = None,
        box_texts: "list[str] | None" = None,
        box_scores: "np.ndarray | None" = None,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
        frame_times   = np.asarray(frame_times_ms, dtype=np.int32)
        frame_windows = np.asarray(frame_windows_ms, dtype=np.int32)
        n_frames = len(frame_times)
        if n_frames == 0:
            return 0

        # Pre-compute per-frame text aggregation and mean confidence.
        frame_text_map:  dict[int, str]   = {}
        frame_score_map: dict[int, float] = {}
        if box_frame_indices is not None and box_texts is not None:
            bfi = np.asarray(box_frame_indices, dtype=np.int32)
            for fi in range(n_frames):
                mask = np.flatnonzero(bfi == fi)
                texts_here  = [box_texts[int(i)].strip() for i in mask if box_texts[int(i)].strip()]
                scores_here = (
                    [float(box_scores[int(i)]) for i in mask if box_texts[int(i)].strip()]
                    if box_scores is not None else []
                )
                frame_text_map[fi]  = truncate_text_for_milvus(" ".join(texts_here))
                frame_score_map[fi] = float(np.mean(scores_here)) if scores_here else 0.0

        # Build frame_idx → embedding mapping.
        has_semantic = embeddings is not None and embedding_frame_indices is not None
        frame_to_embedding: dict[int, np.ndarray] = {}
        if has_semantic:
            emb_arr  = np.asarray(embeddings, dtype=np.float32)
            fidx_arr = np.asarray(embedding_frame_indices, dtype=np.int32)
            dim = emb_arr.shape[1] if emb_arr.ndim == 2 else EMBEDDING_DIMS["ocr"]
            for e_idx, fi in enumerate(fidx_arr):
                fi = int(fi)
                if 0 <= fi < n_frames:
                    frame_to_embedding[fi] = emb_arr[e_idx]
        else:
            dim = EMBEDDING_DIMS["ocr"]

        zero_vec  = [0.0] * dim
        model_ver = ctx.model_ver("ocr")
        col       = ctx.client.collection_for("ocr")
        schema_fields       = {f.name for f in col.schema.fields}
        write_has_embedding = "has_embedding" in schema_fields
        if not write_has_embedding:
            logger.warning(
                "OCR collection is missing 'has_embedding' field — "
                "run migrate_milvus_schema.py to upgrade the schema"
            )
        rows = []
        for fi in range(n_frames):
            emb     = frame_to_embedding.get(fi)
            has_emb = emb is not None
            row = {
                "pk":            ocr_pk(ctx.video_id, ctx.asset_version, fi, 0, model_ver),
                "video_id":      ctx.video_id,
                "asset_version": ctx.asset_version,
                "model_version": model_ver,
                "frame_idx":     fi,
                "region_idx":    0,
                "frame_ms":      int(frame_times[fi]),
                "start_ms":      int(frame_windows[fi, 0]),
                "end_ms":        int(frame_windows[fi, 1]),
                "text":          frame_text_map.get(fi, ""),
                "avg_box_score": frame_score_map.get(fi, 0.0),
                "embedding":     emb.tolist() if has_emb else zero_vec,
            }
            if write_has_embedding:
                row["has_embedding"] = has_emb
            rows.append(row)
        return _upsert_batched(col, rows, "ocr")


class FaceMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        embeddings: np.ndarray,
        track_times_ms: np.ndarray,
        group_model_version: str,
        group_embeddings: np.ndarray | None = None,
        group_track_indices: np.ndarray | None = None,
        group_times_ms: np.ndarray | None = None,
        group_bboxes: np.ndarray | None = None,
        group_qualities: np.ndarray | None = None,
        group_durations_ms: np.ndarray | None = None,
        group_occurrence_counts: np.ndarray | None = None,
        group_importance_scores: np.ndarray | None = None,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
        group_model_version = str(group_model_version).strip()
        if not group_model_version:
            raise ValueError("face group_model_version is required")
        emb_arr   = np.asarray(embeddings, dtype=np.float32)
        times_arr = np.asarray(track_times_ms, dtype=np.int32)
        if not len(emb_arr):
            return 0

        model_ver = ctx.model_ver("face")
        col = ctx.client.collection_for("face")
        rows = [
            {
                "pk":            face_pk(ctx.video_id, ctx.asset_version, idx, model_ver),
                "video_id":      ctx.video_id,
                "asset_version": ctx.asset_version,
                "model_version": model_ver,
                "track_idx":     idx,
                "start_ms":      int(times_arr[idx, 0]),
                "end_ms":        int(times_arr[idx, 1]),
                "best_ms":       int(times_arr[idx, 2]),
                "embedding":     emb_arr[idx].tolist(),
            }
            for idx in range(len(emb_arr))
        ]
        track_count = _upsert_batched(col, rows, "face")

        upsert_face_group_rows(
            ctx,
            group_model_version=group_model_version,
            group_embeddings=group_embeddings,
            group_track_indices=group_track_indices,
            group_times_ms=group_times_ms,
            group_bboxes=group_bboxes,
            group_qualities=group_qualities,
            group_durations_ms=group_durations_ms,
            group_occurrence_counts=group_occurrence_counts,
            group_importance_scores=group_importance_scores,
        )
        return track_count


def upsert_face_group_rows(
    ctx: MilvusWriteContext,
    *,
    group_model_version: str,
    group_embeddings: np.ndarray | None,
    group_track_indices: np.ndarray | None,
    group_times_ms: np.ndarray | None,
    group_bboxes: np.ndarray | None,
    group_qualities: np.ndarray | None,
    group_durations_ms: np.ndarray | None,
    group_occurrence_counts: np.ndarray | None,
    group_importance_scores: np.ndarray | None,
) -> int:
    """Write one immutable derived Face group generation without touching tracks."""
    model_version = str(group_model_version).strip()
    if not model_version:
        raise ValueError("face group_model_version is required")
    vectors = (
        np.asarray(group_embeddings, dtype=np.float32)
        if group_embeddings is not None
        else np.empty((0, 512), dtype=np.float32)
    )
    count = len(vectors)
    if vectors.ndim != 2 or (count and vectors.shape[1] != 512):
        raise ValueError("face group embeddings must have shape (N, 512)")
    if not count:
        return 0
    arrays = {
        "track_indices": np.asarray(group_track_indices, dtype=np.int64),
        "times": np.asarray(group_times_ms, dtype=np.int64),
        "bboxes": np.asarray(group_bboxes, dtype=np.float32),
        "qualities": np.asarray(group_qualities, dtype=np.float32),
        "durations": np.asarray(group_durations_ms, dtype=np.int64),
        "occurrences": np.asarray(group_occurrence_counts, dtype=np.int64),
        "importance": np.asarray(group_importance_scores, dtype=np.float32),
    }
    expected_shapes = {
        "track_indices": (count,),
        "times": (count, 3),
        "bboxes": (count, 4),
        "qualities": (count,),
        "durations": (count,),
        "occurrences": (count,),
        "importance": (count,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"face group {name} must have shape {expected}, "
                f"got {arrays[name].shape}"
            )
    if (
        not np.isfinite(vectors).all()
        or not np.isfinite(arrays["bboxes"]).all()
        or not np.isfinite(arrays["qualities"]).all()
        or not np.isfinite(arrays["importance"]).all()
    ):
        raise ValueError("face group arrays must be finite")
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError("face group embeddings must be non-zero")
    if np.any(arrays["track_indices"] < 0):
        raise ValueError("face group representative track indices must be non-negative")
    if np.any(arrays["durations"] <= 0):
        raise ValueError("face group durations must be positive")
    if np.any(arrays["occurrences"] <= 0):
        raise ValueError("face group occurrence counts must be positive")
    if np.any((arrays["qualities"] < 0) | (arrays["qualities"] > 1)):
        raise ValueError("face group qualities must be between 0 and 1")

    rows = []
    for idx in range(count):
        start_ms, end_ms, best_ms = (int(value) for value in arrays["times"][idx])
        if start_ms < 0 or end_ms <= start_ms or not start_ms <= best_ms <= end_ms:
            raise ValueError(f"face group {idx} has invalid time bounds")
        bbox = arrays["bboxes"][idx]
        missing_bbox = bool(np.all(bbox == -1.0))
        valid_bbox = bool(
            np.all((bbox >= 0.0) & (bbox <= 1.0))
            and bbox[2] > bbox[0]
            and bbox[3] > bbox[1]
        )
        if not missing_bbox and not valid_bbox:
            raise ValueError(f"face group {idx} has invalid representative bbox")
        rows.append({
            "pk": face_group_pk(
                ctx.video_id,
                ctx.asset_version,
                idx,
                model_version,
            ),
            "video_id": ctx.video_id,
            "asset_version": ctx.asset_version,
            "model_version": model_version,
            "group_idx": idx,
            "representative_track_idx": int(arrays["track_indices"][idx]),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "best_ms": best_ms,
            "bbox_x1": float(bbox[0]),
            "bbox_y1": float(bbox[1]),
            "bbox_x2": float(bbox[2]),
            "bbox_y2": float(bbox[3]),
            "representative_quality": float(arrays["qualities"][idx]),
            "duration_ms": int(arrays["durations"][idx]),
            "occurrence_count": int(arrays["occurrences"][idx]),
            "importance_score": float(arrays["importance"][idx]),
            "embedding": vectors[idx].tolist(),
        })
    collection = ctx.client.collection("face_groups")
    written = _upsert_batched(collection, rows, "face")
    collection.flush()
    return written


class SpeakerMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        utterance_embeddings: np.ndarray,
        utterance_times_ms: np.ndarray,
        utterance_refs: np.ndarray,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
        emb_arr   = np.asarray(utterance_embeddings, dtype=np.float32)
        times_arr = np.asarray(utterance_times_ms, dtype=np.int32)
        refs_arr  = np.asarray(utterance_refs, dtype=np.int32)
        if not len(emb_arr):
            return 0

        model_ver = ctx.model_ver("speaker")
        col = ctx.client.collection_for("speaker")
        rows = [
            {
                "pk":             speaker_pk(ctx.video_id, ctx.asset_version, idx, model_ver),
                "video_id":       ctx.video_id,
                "asset_version":  ctx.asset_version,
                "model_version":  model_ver,
                "utterance_idx":  idx,
                "start_ms":       int(times_arr[idx, 0]),
                "end_ms":         int(times_arr[idx, 1]),
                "asr_chunk_idx":  int(refs_arr[idx, 0]),
                "track_id":       int(refs_arr[idx, 1]),
                "embedding":      emb_arr[idx].tolist(),
            }
            for idx in range(len(emb_arr))
        ]
        return _upsert_batched(col, rows, "speaker")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_INDEXERS: dict[str, Any] = {
    "visual":  VisualMilvusIndexer(),
    "asr":     AsrMilvusIndexer(),
    "ocr":     OcrMilvusIndexer(),
    "face":    FaceMilvusIndexer(),
    "speaker": SpeakerMilvusIndexer(),
}


# ---------------------------------------------------------------------------
# Public write hooks — called from each build_* function
# ---------------------------------------------------------------------------

def write_modality_from_memory(
    ctx: MilvusWriteContext,
    modality: str,
    arrays: dict[str, Any],
) -> int:
    """Write one in-memory modality payload and fail closed on any error.

    Args:
        ctx:      MilvusWriteContext with video_id, asset_version, and client.
        modality: ``visual`` / ``asr`` / ``ocr`` / ``face`` / ``speaker``.
        arrays:   Keyword arguments for the modality indexer's in-memory writer.

    The caller publishes ``asset_version`` only after this function returns and
    the persisted row count is verified.  No recovery artifact is written.
    """
    indexer = _INDEXERS[modality]
    try:
        count = indexer.upsert_from_memory(ctx, **arrays)
        # A failed flush is a failed write attempt: readers must never publish a
        # generation that has not been made queryable and verified.
        ctx.client.collection_for(modality).flush()
        logger.info(
            "Milvus direct-write OK modality=%s video=%s@%s count=%d",
            modality, ctx.video_id, ctx.asset_version, count,
        )
        return int(count)
    except Exception as exc:
        _handle_write_failure(ctx, modality, exc)


def _handle_write_failure(
    ctx: MilvusWriteContext,
    modality: str,
    exc: Exception,
) -> None:
    logger.error(
        "Milvus write failed modality=%s video=%s@%s: %s",
        modality, ctx.video_id, ctx.asset_version, exc,
    )

    raise RuntimeError(
        f"Milvus write failed (fail-closed) modality={modality} "
        f"video={ctx.video_id}: {exc}"
    ) from exc
