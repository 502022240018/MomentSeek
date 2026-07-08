from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.indexing.asr_text import normalize_asr_text, normalize_search_text, semantic_text_quality


@dataclass(frozen=True)
class AsrPostprocessConfig:
    normal_gap_ms: int = 700
    short_gap_ms: int = 1500
    same_segment_normal_gap_ms: int = 1200
    same_segment_short_gap_ms: int = 1800
    cross_segment_short_gap_ms: int = 900
    hard_max_duration_ms: int = 8000
    short_text_chars: int = 8
    max_text_chars: int = 160


def strategy_config(name: str) -> AsrPostprocessConfig:
    base = AsrPostprocessConfig()
    if name == "gap_only":
        return replace(
            base,
            same_segment_normal_gap_ms=base.normal_gap_ms,
            same_segment_short_gap_ms=base.short_gap_ms,
            cross_segment_short_gap_ms=base.short_gap_ms,
        )
    if name == "bucket_bonus":
        return replace(base, same_segment_normal_gap_ms=1200, same_segment_short_gap_ms=1800)
    if name == "shot_bonus":
        return replace(base, same_segment_normal_gap_ms=1500, same_segment_short_gap_ms=2600)
    if name == "conservative":
        return replace(base, normal_gap_ms=450, short_gap_ms=900, cross_segment_short_gap_ms=450)
    if name == "aggressive_short":
        return replace(base, normal_gap_ms=900, short_gap_ms=2200, same_segment_short_gap_ms=3200)
    raise ValueError(f"unknown ASR postprocess strategy: {name}")


def default_strategy_names() -> list[str]:
    return ["gap_only", "bucket_bonus", "shot_bonus", "conservative", "aggressive_short"]


def _to_ms(value: Any) -> int:
    return int(round(float(value) * 1000.0))


def _chunk_times_ms(chunk: dict[str, Any]) -> tuple[int, int]:
    if "start_ms" in chunk and "end_ms" in chunk:
        return int(chunk["start_ms"]), int(chunk["end_ms"])
    return _to_ms(chunk.get("start_time", chunk.get("start", 0))), _to_ms(chunk.get("end_time", chunk.get("end", 0)))


def _compact_length(text: str) -> int:
    return len(normalize_search_text(text))


def _is_short_text(text: str, config: AsrPostprocessConfig) -> bool:
    return _compact_length(text) <= config.short_text_chars


def _merge_allowed(
    current: dict[str, Any],
    item: dict[str, Any],
    *,
    config: AsrPostprocessConfig,
) -> tuple[bool, bool]:
    if not current.get("merge_eligible", True) or not item.get("merge_eligible", True):
        return False, False
    gap_ms = max(0, int(item["start_ms"]) - int(current["end_ms"]))
    current_segment = current.get("segment_id")
    item_segment = item.get("segment_id")
    has_segments = current_segment is not None and item_segment is not None
    same_segment = has_segments and current_segment == item_segment
    cross_segment = has_segments and current_segment != item_segment
    short_side = _is_short_text(str(current["text"]), config) or _is_short_text(str(item["text"]), config)
    if same_segment and short_side:
        return gap_ms <= config.same_segment_short_gap_ms, False
    if same_segment:
        return gap_ms <= config.same_segment_normal_gap_ms, False
    if short_side:
        return gap_ms <= config.cross_segment_short_gap_ms or gap_ms <= config.short_gap_ms, cross_segment
    return gap_ms <= config.normal_gap_ms, cross_segment


def postprocess_asr_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    segment_ids: Iterable[int | None] | None = None,
    config: AsrPostprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config = config or AsrPostprocessConfig()
    raw_chunks = list(chunks)
    segment_values = list(segment_ids) if segment_ids is not None else []
    normalized: list[dict[str, Any]] = []
    dropped_empty = 0

    for index, chunk in enumerate(raw_chunks):
        start_ms, end_ms = _chunk_times_ms(chunk)
        text = normalize_asr_text(str(chunk.get("text") or ""))
        if not text:
            dropped_empty += 1
            continue
        quality = semantic_text_quality(text)
        segment_id = segment_values[index] if index < len(segment_values) else None
        normalized.append({
            "source_chunk_ids": [index],
            "start_ms": start_ms,
            "end_ms": max(start_ms, end_ms),
            "start_time": start_ms / 1000.0,
            "end_time": max(start_ms, end_ms) / 1000.0,
            "text": text,
            "segment_id": segment_id,
            "merge_eligible": quality.eligible,
        })

    processed: list[dict[str, Any]] = []
    cross_segment_merges = 0
    for item in normalized:
        if not processed:
            processed.append(item)
            continue
        current = processed[-1]
        allowed, cross_segment = _merge_allowed(current, item, config=config)
        candidate_duration_ms = int(item["end_ms"]) - int(current["start_ms"])
        candidate_text = f'{current["text"]} {item["text"]}'.strip()
        if (
            allowed
            and candidate_duration_ms <= config.hard_max_duration_ms
            and _compact_length(candidate_text) <= config.max_text_chars
        ):
            current["end_ms"] = item["end_ms"]
            current["end_time"] = item["end_time"]
            current["text"] = candidate_text
            current["source_chunk_ids"].extend(item["source_chunk_ids"])
            current["merge_eligible"] = current.get("merge_eligible", True) and item.get("merge_eligible", True)
            if cross_segment:
                cross_segment_merges += 1
            continue
        processed.append(item)

    semantic_ineligible = 0
    long_low_info = 0
    for item in processed:
        quality = semantic_text_quality(str(item["text"]))
        duration_ms = int(item["end_ms"]) - int(item["start_ms"])
        if duration_ms > config.hard_max_duration_ms and not quality.eligible:
            long_low_info += 1
        item["semantic_eligible"] = bool(quality.eligible)
        item["semantic_reason"] = quality.reason
        if not quality.eligible:
            semantic_ineligible += 1
        item.pop("merge_eligible", None)

    stats = {
        "raw_chunks": len(raw_chunks),
        "normalized_chunks": len(normalized),
        "processed_chunks": len(processed),
        "dropped_empty_chunks": dropped_empty,
        "merged_chunks": max(0, len(normalized) - len(processed)),
        "cross_segment_merges": cross_segment_merges,
        "semantic_ineligible_chunks": semantic_ineligible,
        "long_low_info_chunks": long_low_info,
    }
    return processed, stats
