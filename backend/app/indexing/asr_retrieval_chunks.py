from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk
from app.indexing.asr_text import normalize_asr_text, normalize_search_text, semantic_text_quality


@dataclass(frozen=True)
class RetrievalChunkConfig:
    normal_gap_ms: int = 700
    short_gap_ms: int = 1800
    same_bucket_gap_ms: int = 1800
    false_gap_repair_ms: int = 8000
    target_max_duration_ms: int = 18000
    soft_max_duration_ms: int = 25000
    hard_max_duration_ms: int = 35000
    short_text_chars: int = 8
    max_text_chars: int = 180
    bucket_ms: int = 5000


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？"))


def _ends_soft_punctuation(text: str) -> bool:
    return text.rstrip().endswith((",", ";", "，", "；", "、", ":", "："))


def _compact_length(text: str) -> int:
    return len(normalize_search_text(text))


def _last_run_after_punctuation(text: str) -> str:
    stripped = text.rstrip()
    boundary = max(stripped.rfind(mark) for mark in ".!?。！？,;，；、:：")
    return stripped[boundary + 1 :].strip()


def _first_run_before_punctuation(text: str) -> str:
    stripped = text.lstrip()
    positions = [index for index, char in enumerate(stripped) if char in ".!?。！？,;，；、:："]
    if not positions:
        return stripped
    return stripped[: positions[0]].strip()


def _is_short_text(text: str, config: RetrievalChunkConfig) -> bool:
    return _compact_length(text) <= config.short_text_chars


def _same_bucket(left: RetrievalChunk, right: RawTranscriptItem, config: RetrievalChunkConfig) -> bool:
    return left.start_ms // config.bucket_ms == right.start_ms // config.bucket_ms


def _needs_cjk_boundary_repair(left_text: str, right_text: str) -> bool:
    if _ends_sentence(left_text):
        return False
    left_tail = _last_run_after_punctuation(left_text)
    right_head = _first_run_before_punctuation(right_text)
    if not left_tail or not right_head:
        return False
    if not _is_cjk(left_tail[-1]) or not _is_cjk(right_head[0]):
        return False
    if len(left_tail) <= 4:
        return True
    if _ends_soft_punctuation(left_text):
        return True
    return False


def _needs_latin_boundary_repair(left_text: str, right_text: str) -> bool:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left or not right:
        return False
    if _ends_sentence(left_text):
        return False
    return left[-1].isalpha() and right[0].isalpha()


def _join_text(left_text: str, right_text: str, *, boundary_repair: bool) -> str:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left:
        return right
    if not right:
        return left
    if boundary_repair and _is_cjk(left[-1]) and _is_cjk(right[0]):
        return f"{left}{right}"
    if boundary_repair and left[-1].isalpha() and right[0].isalpha():
        return f"{left}{right}"
    if _ends_soft_punctuation(left):
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_decision(
    current: RetrievalChunk,
    item: RawTranscriptItem,
    *,
    config: RetrievalChunkConfig,
) -> tuple[bool, bool, list[str]]:
    gap_ms = max(0, item.start_ms - current.end_ms)
    candidate_duration_ms = max(current.end_ms, item.end_ms) - current.start_ms
    cjk_repair = _needs_cjk_boundary_repair(current.text, item.text)
    latin_repair = _needs_latin_boundary_repair(current.text, item.text)
    boundary_repair = cjk_repair or latin_repair
    flags: list[str] = []
    if cjk_repair:
        flags.append("cjk_boundary_repair")
    if latin_repair:
        flags.append("latin_boundary_repair")
    if gap_ms > config.short_gap_ms and boundary_repair:
        flags.append("fake_gap_repair")

    candidate_text = _join_text(current.text, item.text, boundary_repair=boundary_repair)
    if _compact_length(candidate_text) > config.max_text_chars:
        return False, boundary_repair, flags
    if candidate_duration_ms > config.hard_max_duration_ms:
        return False, boundary_repair, flags
    if boundary_repair and gap_ms <= config.false_gap_repair_ms:
        return True, boundary_repair, flags
    if _ends_sentence(current.text):
        return False, boundary_repair, flags
    if _is_short_text(current.text, config) or _is_short_text(item.text, config):
        return gap_ms <= config.short_gap_ms, boundary_repair, flags
    if gap_ms <= config.normal_gap_ms:
        return True, boundary_repair, flags
    if (
        _same_bucket(current, item, config)
        and candidate_duration_ms <= config.target_max_duration_ms
        and gap_ms <= config.same_bucket_gap_ms
    ):
        return True, boundary_repair, flags
    return False, boundary_repair, flags


def build_retrieval_chunks(
    raw_items: Iterable[RawTranscriptItem],
    *,
    config: RetrievalChunkConfig | None = None,
) -> tuple[list[RetrievalChunk], dict[str, int]]:
    config = config or RetrievalChunkConfig()
    source_items = list(raw_items)
    normalized: list[RawTranscriptItem] = []
    dropped_empty = 0
    for item in source_items:
        text = normalize_asr_text(item.text)
        if not text:
            dropped_empty += 1
            continue
        normalized.append(
            RawTranscriptItem(
                item_id=item.item_id,
                start_ms=item.start_ms,
                end_ms=max(item.start_ms, item.end_ms),
                text=text,
                source=item.source,
                unit_id=item.unit_id,
                diagnostics=item.diagnostics,
            )
        )

    chunks: list[RetrievalChunk] = []
    word_boundary_repairs = 0
    fake_gap_repairs = 0
    merged_items = 0

    for item in normalized:
        if not chunks:
            chunks.append(
                RetrievalChunk(
                    chunk_id=0,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                    source_item_ids=[item.item_id],
                )
            )
            continue
        current = chunks[-1]
        allowed, boundary_repair, flags = _merge_decision(current, item, config=config)
        if not allowed:
            chunks.append(
                RetrievalChunk(
                    chunk_id=len(chunks),
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                    source_item_ids=[item.item_id],
                )
            )
            continue
        merged_text = _join_text(current.text, item.text, boundary_repair=boundary_repair)
        merged_flags = list(dict.fromkeys([*current.quality_flags, *flags]))
        chunks[-1] = RetrievalChunk(
            chunk_id=current.chunk_id,
            start_ms=current.start_ms,
            end_ms=max(current.end_ms, item.end_ms),
            text=merged_text,
            source_item_ids=[*current.source_item_ids, item.item_id],
            quality_flags=merged_flags,
        )
        merged_items += 1
        if boundary_repair:
            word_boundary_repairs += 1
        if "fake_gap_repair" in flags:
            fake_gap_repairs += 1

    final_chunks: list[RetrievalChunk] = []
    semantic_ineligible = 0
    long_chunks = 0
    for chunk in chunks:
        quality = semantic_text_quality(chunk.text)
        duration_ms = chunk.end_ms - chunk.start_ms
        if duration_ms > config.soft_max_duration_ms:
            long_chunks += 1
        if not quality.eligible:
            semantic_ineligible += 1
        final_chunks.append(
            RetrievalChunk(
                chunk_id=len(final_chunks),
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                text=chunk.text,
                source_item_ids=chunk.source_item_ids,
                semantic_eligible=bool(quality.eligible),
                semantic_reason=quality.reason,
                quality_flags=chunk.quality_flags,
            )
        )

    return final_chunks, {
        "raw_items": len(source_items),
        "normalized_items": len(normalized),
        "retrieval_chunks": len(final_chunks),
        "dropped_empty_items": dropped_empty,
        "merged_items": merged_items,
        "word_boundary_repairs": word_boundary_repairs,
        "fake_gap_repairs": fake_gap_repairs,
        "long_chunks": long_chunks,
        "semantic_ineligible_chunks": semantic_ineligible,
    }
