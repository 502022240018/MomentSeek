"""Milvus indexers for all five modalities.

Each indexer exposes two write methods:

  upsert_from_memory(**arrays) — P2 direct path: accepts in-memory numpy arrays
      and upserts directly to Milvus without writing an intermediate NPZ file.
      This is the hot path used by build_* functions when milvus_ctx is available.

  upsert_from_npz(npz_path) — legacy / recovery path: loads a previously-written
      NPZ from disk and delegates to upsert_from_memory.  Used by reindex_from_file()
      for manual recovery and backfill scripts.

Public write hooks:
  write_modality_from_memory() — P2 hook; build_* functions call this instead of
      saving an NPZ first.  On failure it invokes recovery_save_fn (if supplied)
      so the NPZ is only written when actually needed.
  write_modality_to_milvus()   — legacy hook; reads from an NPZ path.  Kept for
      reindex_from_file() and any caller that already has a NPZ on disk.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .milvus_schema import (
    EMBEDDING_DIMS,
    MODEL_VERSIONS,
    asr_pk,
    face_pk,
    ocr_pk,
    speaker_pk,
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
        segment_times_ms: "np.ndarray | None" = None,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        times_ms   = np.asarray(frame_times_ms, dtype=np.int32)
        offsets    = np.asarray(segment_frame_offsets, dtype=np.int32)
        if not len(embeddings):
            return 0

        frame_segment_ids  = np.full(len(embeddings), -1, dtype=np.int32)
        frame_seg_start_ms = np.full(len(embeddings), -1, dtype=np.int64)
        frame_seg_end_ms   = np.full(len(embeddings), -1, dtype=np.int64)
        for seg_idx in range(len(offsets) - 1):
            start_f = int(offsets[seg_idx])
            end_f   = int(offsets[seg_idx + 1])
            frame_segment_ids[start_f:end_f] = seg_idx
            if segment_times_ms is not None and seg_idx < len(segment_times_ms):
                frame_seg_start_ms[start_f:end_f] = int(segment_times_ms[seg_idx, 0])
                frame_seg_end_ms[start_f:end_f]   = int(segment_times_ms[seg_idx, 1])

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

    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        """Legacy / recovery path: load NPZ and delegate to upsert_from_memory."""
        with np.load(npz_path, allow_pickle=False) as data:
            required = {"frame_embeddings", "frame_times_ms", "segment_frame_offsets"}
            if not required.issubset(set(data.files)):
                raise ValueError(
                    "visual.npz missing frame_embeddings, frame_times_ms, or segment_frame_offsets"
                )
            return self.upsert_from_memory(
                ctx,
                embeddings=np.asarray(data["frame_embeddings"], dtype=np.float32),
                frame_times_ms=np.asarray(data["frame_times_ms"], dtype=np.int32),
                segment_frame_offsets=np.asarray(data["segment_frame_offsets"], dtype=np.int32),
                segment_times_ms=(
                    np.asarray(data["segment_times_ms"], dtype=np.int32)
                    if "segment_times_ms" in data.files else None
                ),
            )


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
                "text":          texts[chunk_idx][:2000] if chunk_idx < len(texts) else "",
                "embedding":     emb.tolist() if has_emb else zero_vec,
            }
            if write_has_embedding:
                row["has_embedding"] = has_emb
            rows.append(row)
        return _upsert_batched(col, rows, "asr")

    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        """Legacy / recovery path: load NPZ and delegate to upsert_from_memory."""
        with np.load(npz_path, allow_pickle=False) as data:
            if "chunk_times_ms" not in data.files or "texts" not in data.files:
                return 0
            texts = [str(t) for t in data["texts"].tolist()]
            has_semantic = (
                "embeddings" in data.files
                and "embedding_chunk_indices" in data.files
            )
            return self.upsert_from_memory(
                ctx,
                chunk_times_ms=np.asarray(data["chunk_times_ms"], dtype=np.int32),
                texts=texts,
                embeddings=(
                    np.asarray(data["embeddings"], dtype=np.float32)
                    if has_semantic else None
                ),
                embedding_chunk_indices=(
                    np.asarray(data["embedding_chunk_indices"], dtype=np.int32)
                    if has_semantic else None
                ),
            )


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
                frame_text_map[fi]  = " ".join(texts_here)[:2000]
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

    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        """Legacy / recovery path: load NPZ and delegate to upsert_from_memory."""
        with np.load(npz_path, allow_pickle=False) as data:
            required = {"frame_times_ms", "frame_windows_ms"}
            if not required.issubset(set(data.files)):
                return 0
            has_semantic = (
                "embeddings" in data.files
                and "embedding_frame_indices" in data.files
            )
            has_boxes = (
                "box_frame_indices" in data.files
                and "box_texts" in data.files
            )
            return self.upsert_from_memory(
                ctx,
                frame_times_ms=np.asarray(data["frame_times_ms"], dtype=np.int32),
                frame_windows_ms=np.asarray(data["frame_windows_ms"], dtype=np.int32),
                embeddings=(
                    np.asarray(data["embeddings"], dtype=np.float32)
                    if has_semantic else None
                ),
                embedding_frame_indices=(
                    np.asarray(data["embedding_frame_indices"], dtype=np.int32)
                    if has_semantic else None
                ),
                box_frame_indices=(
                    np.asarray(data["box_frame_indices"], dtype=np.int32)
                    if has_boxes else None
                ),
                box_texts=(
                    [str(t) for t in data["box_texts"].tolist()]
                    if has_boxes else None
                ),
                box_scores=(
                    np.asarray(data["box_scores"], dtype=np.float32)
                    if has_boxes and "box_scores" in data.files else None
                ),
            )


class FaceMilvusIndexer:
    def upsert_from_memory(
        self,
        ctx: MilvusWriteContext,
        *,
        embeddings: np.ndarray,
        track_times_ms: np.ndarray,
    ) -> int:
        """P2 direct path: build rows from in-memory arrays and upsert."""
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
        return _upsert_batched(col, rows, "face")

    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        """Legacy / recovery path: load NPZ and delegate to upsert_from_memory."""
        with np.load(npz_path, allow_pickle=False) as data:
            if "embeddings" not in data.files or "track_times_ms" not in data.files:
                raise ValueError("face.npz missing embeddings or track_times_ms")
            return self.upsert_from_memory(
                ctx,
                embeddings=np.asarray(data["embeddings"], dtype=np.float32),
                track_times_ms=np.asarray(data["track_times_ms"], dtype=np.int32),
            )


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

    def upsert_from_npz(self, ctx: MilvusWriteContext, npz_path: str | Path) -> int:
        """Legacy / recovery path: load NPZ and delegate to upsert_from_memory."""
        with np.load(npz_path, allow_pickle=False) as data:
            required = {"utterance_embeddings", "utterance_times_ms", "utterance_refs"}
            if not required.issubset(set(data.files)):
                raise ValueError(f"speaker.npz missing: {required - set(data.files)}")
            return self.upsert_from_memory(
                ctx,
                utterance_embeddings=np.asarray(data["utterance_embeddings"], dtype=np.float32),
                utterance_times_ms=np.asarray(data["utterance_times_ms"], dtype=np.int32),
                utterance_refs=np.asarray(data["utterance_refs"], dtype=np.int32),
            )


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
    *,
    recovery_save_fn: "Callable[[], None] | None" = None,
) -> None:
    """P2 direct-write hook: upsert in-memory arrays to Milvus; save NPZ on failure.

    Args:
        ctx:              MilvusWriteContext with video_id, asset_version, client.
        modality:         "visual" / "asr" / "ocr" / "face" / "speaker".
        arrays:           kwargs dict to pass to the indexer's upsert_from_memory().
        recovery_save_fn: Optional callable that saves the NPZ to disk.  Invoked
                          ONLY when Milvus write fails and fail_policy="warn",
                          ensuring the data survives for manual recovery.

    On success: data is flushed and immediately queryable.
    On failure: invokes recovery_save_fn (if supplied) before handling the failure.
    """
    indexer = _INDEXERS[modality]
    try:
        count = indexer.upsert_from_memory(ctx, **arrays)
        # Flush to ensure buffered records are queryable by downstream stages.
        try:
            ctx.client.collection_for(modality).flush()
        except Exception as flush_exc:
            logger.warning(
                "Milvus flush failed modality=%s video=%s@%s: %s — "
                "downstream readers may not see these records immediately",
                modality, ctx.video_id, ctx.asset_version, flush_exc,
            )
        logger.info(
            "Milvus direct-write OK modality=%s video=%s@%s count=%d",
            modality, ctx.video_id, ctx.asset_version, count,
        )
    except Exception as exc:
        # Save NPZ before handling failure so it's available for recovery.
        if recovery_save_fn is not None:
            try:
                recovery_save_fn()
                logger.info(
                    "Milvus write failed but NPZ saved for recovery modality=%s video=%s@%s",
                    modality, ctx.video_id, ctx.asset_version,
                )
            except Exception as save_exc:
                logger.error(
                    "Recovery NPZ save also failed modality=%s video=%s@%s: %s",
                    modality, ctx.video_id, ctx.asset_version, save_exc,
                    exc_info=True,
                )
        _handle_write_failure(ctx, modality, exc)


def write_modality_to_milvus(
    ctx: MilvusWriteContext,
    modality: str,
    npz_path: str | Path,
) -> None:
    """Legacy write hook: upsert from an already-written NPZ file.

    Used by:
      - reindex_from_file() for manual recovery
      - any caller that already has a NPZ on disk (offline backfill, etc.)

    After a successful upsert the collection is flushed so the written data is
    immediately visible to subsequent queries in the same indexing job.
    """
    indexer = _INDEXERS[modality]
    try:
        count = indexer.upsert_from_npz(ctx, npz_path)
        # Flush ensures buffered records are sealed and queryable before any
        # downstream stage reads from the same collection.
        try:
            ctx.client.collection_for(modality).flush()
        except Exception as flush_exc:
            logger.warning(
                "Milvus flush failed modality=%s video=%s@%s: %s — "
                "downstream readers may not see these records immediately",
                modality, ctx.video_id, ctx.asset_version, flush_exc,
            )
        logger.info(
            "Milvus upsert OK modality=%s video=%s@%s count=%d",
            modality, ctx.video_id, ctx.asset_version, count,
        )
    except Exception as exc:
        _handle_write_failure(ctx, modality, exc)


def _handle_write_failure(
    ctx: MilvusWriteContext,
    modality: str,
    exc: Exception,
) -> None:
    from .milvus_flags import milvus_write_fail_policy

    policy = milvus_write_fail_policy()
    logger.error(
        "Milvus write failed modality=%s video=%s@%s: %s",
        modality, ctx.video_id, ctx.asset_version, exc,
    )

    if policy == "raise":
        raise RuntimeError(
            f"Milvus write failed modality={modality} video={ctx.video_id}: {exc}"
        ) from exc

    # policy == "warn" — logged above, indexing continues without this modality in Milvus.


# ---------------------------------------------------------------------------
# Re-index entry point kept for manual recovery; not used by the write queue
# (which has been removed).  Call directly when a modality's NPZ is available.
# ---------------------------------------------------------------------------

def reindex_from_file(
    *,
    client: "MilvusClient",
    modality: str,
    video_id: str,
    asset_version: str,
    model_version: str,
    npz_path: str,
) -> None:
    """Manual recovery helper: re-upsert one modality from a temporary NPZ."""
    ctx = MilvusWriteContext(
        video_id=video_id,
        asset_version=asset_version,
        client=client,
        model_versions={modality: model_version},
    )
    indexer = _INDEXERS[modality]
    indexer.upsert_from_npz(ctx, npz_path)
    ctx.client.collection_for(modality).flush()
