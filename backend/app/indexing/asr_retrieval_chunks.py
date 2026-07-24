from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk
from app.indexing.asr_text import normalize_asr_text, normalize_search_text, semantic_text_quality


@dataclass(frozen=True)
class RetrievalChunkConfig:
    normal_gap_ms: int = 500
    short_gap_ms: int = 1000
    same_bucket_gap_ms: int = 1000
    target_max_duration_ms: int = 8000
    soft_max_duration_ms: int = 12000
    merge_max_duration_ms: int = 12000
    short_text_chars: int = 8
    max_text_chars: int = 180
    bucket_ms: int = 5000
    same_unit_only: bool = True


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？"))


def _ends_soft_punctuation(text: str) -> bool:
    return text.rstrip().endswith((",", ";", "，", "；", "、", ":", "："))


def _compact_length(text: str) -> int:
    return len(normalize_search_text(text))


def _merge_labels(*values: str) -> str:
    labels: list[str] = []
    for value in values:
        for part in str(value or "").split("|"):
            label = part.strip().casefold()
            if label and label not in labels:
                labels.append(label)
    return "|".join(labels)


def _is_short_text(text: str, config: RetrievalChunkConfig) -> bool:
    return _compact_length(text) <= config.short_text_chars


def _same_bucket(left: RetrievalChunk, right: RawTranscriptItem, config: RetrievalChunkConfig) -> bool:
    return left.start_ms // config.bucket_ms == right.start_ms // config.bucket_ms


def _join_text(left_text: str, right_text: str) -> str:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left:
        return right
    if not right:
        return left
    if _is_cjk(left[-1]) and _is_cjk(right[0]):
        return f"{left}{right}"
    if _ends_soft_punctuation(left):
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_decision(
    current: RetrievalChunk,
    item: RawTranscriptItem,
    *,
    config: RetrievalChunkConfig,
    current_unit_id: int | None,
) -> tuple[bool, list[str]]:
    gap_ms = max(0, item.start_ms - current.end_ms)
    candidate_duration_ms = max(current.end_ms, item.end_ms) - current.start_ms
    flags: list[str] = []

    if (
        config.same_unit_only
        and current_unit_id is not None
        and item.unit_id is not None
        and current_unit_id != item.unit_id
    ):
        return False, flags
    candidate_text = _join_text(current.text, item.text)
    if _compact_length(candidate_text) > config.max_text_chars:
        return False, flags
    if candidate_duration_ms > config.merge_max_duration_ms:
        return False, flags
    if _ends_sentence(current.text):
        return False, flags
    if _is_short_text(current.text, config) or _is_short_text(item.text, config):
        return gap_ms <= config.short_gap_ms, flags
    if gap_ms <= config.normal_gap_ms:
        return True, flags
    if (
        _same_bucket(current, item, config)
        and candidate_duration_ms <= config.target_max_duration_ms
        and gap_ms <= config.same_bucket_gap_ms
    ):
        return True, flags
    return False, flags


def _normalize_raw_items(source_items: list[RawTranscriptItem]) -> tuple[list[RawTranscriptItem], int]:
    normalized = []
    dropped_empty = 0
    for item in source_items:
        text = normalize_asr_text(item.text)
        if not text:
            dropped_empty += 1
            continue
        normalized.append(RawTranscriptItem(
            item_id=item.item_id,
            start_ms=item.start_ms,
            end_ms=max(item.start_ms, item.end_ms),
            text=text,
            source=item.source,
            unit_id=item.unit_id,
            emotion=item.emotion,
            audio_event=item.audio_event,
            diagnostics=item.diagnostics,
        ))
    return normalized, dropped_empty


def _chunk_from_item(item: RawTranscriptItem, chunk_id: int) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        start_ms=item.start_ms,
        end_ms=item.end_ms,
        text=item.text,
        source_item_ids=[item.item_id],
        emotion=item.emotion,
        audio_event=item.audio_event,
    )


def _merge_normalized_items(
    normalized: list[RawTranscriptItem],
    config: RetrievalChunkConfig,
) -> tuple[list[RetrievalChunk], int, int]:
    chunks: list[RetrievalChunk] = []
    chunk_unit_ids: list[int | None] = []
    merged_items = 0
    cross_unit_merge_blocks = 0
    for item in normalized:
        if not chunks:
            chunks.append(_chunk_from_item(item, 0))
            chunk_unit_ids.append(item.unit_id)
            continue
        current = chunks[-1]
        current_unit_id = chunk_unit_ids[-1]
        cross_unit = (
            config.same_unit_only
            and current_unit_id is not None
            and item.unit_id is not None
            and current_unit_id != item.unit_id
        )
        allowed, flags = _merge_decision(current, item, config=config, current_unit_id=current_unit_id)
        if not allowed:
            cross_unit_merge_blocks += int(cross_unit)
            chunks.append(_chunk_from_item(item, len(chunks)))
            chunk_unit_ids.append(item.unit_id)
            continue
        chunks[-1] = RetrievalChunk(
            chunk_id=current.chunk_id,
            start_ms=current.start_ms,
            end_ms=max(current.end_ms, item.end_ms),
            text=_join_text(current.text, item.text),
            source_item_ids=[*current.source_item_ids, item.item_id],
            quality_flags=list(dict.fromkeys([*current.quality_flags, *flags])),
            emotion=_merge_labels(current.emotion, item.emotion),
            audio_event=_merge_labels(current.audio_event, item.audio_event),
        )
        merged_items += 1
    return chunks, merged_items, cross_unit_merge_blocks


def _finalize_chunks(
    chunks: list[RetrievalChunk],
    config: RetrievalChunkConfig,
) -> tuple[list[RetrievalChunk], dict[str, int]]:
    final_chunks = []
    counters = {"semantic_ineligible_chunks": 0, "long_chunks": 0, "low_boundary_chunks": 0}
    for chunk in chunks:
        duration_ms = chunk.end_ms - chunk.start_ms
        quality = semantic_text_quality(chunk.text, duration_ms=duration_ms)
        quality_flags = list(chunk.quality_flags)
        if not _ends_sentence(chunk.text):
            quality_flags.append("non_terminal_boundary")
            counters["low_boundary_chunks"] += 1
        if duration_ms > config.soft_max_duration_ms:
            quality_flags.append("long_chunk")
            counters["long_chunks"] += 1
        if not quality.eligible:
            quality_flags.append(f"embedding_ineligible:{quality.reason}")
            counters["semantic_ineligible_chunks"] += 1
        final_chunks.append(RetrievalChunk(
            chunk_id=len(final_chunks),
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            text=chunk.text,
            source_item_ids=chunk.source_item_ids,
            semantic_eligible=bool(quality.eligible),
            semantic_reason=quality.reason,
            quality_flags=list(dict.fromkeys(quality_flags)),
            emotion=chunk.emotion,
            audio_event=chunk.audio_event,
        ))
    return final_chunks, counters


def build_retrieval_chunks(
    raw_items: Iterable[RawTranscriptItem],
    *,
    config: RetrievalChunkConfig | None = None,
) -> tuple[list[RetrievalChunk], dict[str, int]]:
    config = config or RetrievalChunkConfig()
    source_items = list(raw_items)
    normalized, dropped_empty = _normalize_raw_items(source_items)
    chunks, merged_items, cross_unit_merge_blocks = _merge_normalized_items(normalized, config)
    final_chunks, quality_counters = _finalize_chunks(chunks, config)

    return final_chunks, {
        "raw_items": len(source_items),
        "normalized_items": len(normalized),
        "retrieval_chunks": len(final_chunks),
        "dropped_empty_items": dropped_empty,
        "merged_items": merged_items,
        "cross_unit_merge_blocks": cross_unit_merge_blocks,
        **quality_counters,
    }
