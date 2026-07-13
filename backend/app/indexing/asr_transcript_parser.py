from __future__ import annotations

import re
from typing import Any, Iterable

from app.indexing.asr_pipeline_types import RawTranscriptItem


_SENSEVOICE_TAG_RE = re.compile(r"<\|([^|<>]+)\|>")
_SENSEVOICE_EMOTION_TAGS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
}
_SENSEVOICE_AUDIO_EVENT_TAGS = {
    "speech": "speech",
    "bgm": "bgm",
    "music": "music",
    "laughter": "laughter",
    "applause": "applause",
    "cry": "cry",
    "sneeze": "sneeze",
    "breath": "breath",
    "cough": "cough",
    "sing": "sing",
    "noise": "noise",
    "silence": "silence",
}
_STRONG_PUNCT = set(".!?") | {"\u3002", "\uff01", "\uff1f"}
_SOFT_PUNCT = set(",;:") | {"\uff0c", "\uff1b", "\u3001", "\uff1a"}


def _dedupe_pipe_labels(labels: Iterable[str]) -> str:
    seen: list[str] = []
    for label in labels:
        for part in str(label or "").split("|"):
            value = part.strip().casefold()
            if value and value not in seen:
                seen.append(value)
    return "|".join(seen)


def _extract_sensevoice_tags(text: str) -> tuple[str, str, str]:
    raw_text = str(text or "")
    emotions: list[str] = []
    audio_events: list[str] = []
    for tag in _SENSEVOICE_TAG_RE.findall(raw_text):
        key = re.sub(r"[^a-z0-9]+", "", tag.casefold())
        if key in _SENSEVOICE_EMOTION_TAGS:
            emotions.append(_SENSEVOICE_EMOTION_TAGS[key])
        elif key in _SENSEVOICE_AUDIO_EVENT_TAGS:
            audio_events.append(_SENSEVOICE_AUDIO_EVENT_TAGS[key])
    return (
        _SENSEVOICE_TAG_RE.sub("", raw_text).strip(),
        _dedupe_pipe_labels(emotions),
        _dedupe_pipe_labels(audio_events),
    )


def _clean_funasr_text(text: str, is_sensevoice: bool) -> str:
    text = str(text or "").strip()
    if not is_sensevoice:
        return text
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return str(rich_transcription_postprocess(text)).strip()
    except Exception:
        return text


def _timed_char(value: str) -> bool:
    return bool(re.match(r"[\w\u3400-\u9fff]", value, flags=re.UNICODE))


def _valid_timestamp_pairs(timestamps: object) -> list[tuple[int, int]]:
    if not isinstance(timestamps, list):
        return []
    pairs: list[tuple[int, int]] = []
    for pair in timestamps:
        if isinstance(pair, list) and len(pair) >= 2 and pair[0] is not None and pair[1] is not None:
            pairs.append((int(pair[0]), int(pair[1])))
    return pairs


def _align_token_spans(text: str, tokens: list[object]) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for raw_token in tokens:
        token = str(raw_token or "").strip()
        if not token:
            return None
        index = text.find(token, cursor)
        if index < 0:
            return None
        spans.append((index, index + len(token)))
        cursor = index + len(token)
    return spans


def _raw_item_from_parts(
    *,
    item_id: int,
    start_ms: int,
    end_ms: int,
    text: str,
    source: str,
    is_sensevoice: bool,
    emotion: str = "",
    audio_event: str = "",
    diagnostics: dict[str, Any] | None = None,
) -> RawTranscriptItem | None:
    if is_sensevoice and not (emotion or audio_event):
        text, emotion, audio_event = _extract_sensevoice_tags(text)
    cleaned = _clean_funasr_text(text, is_sensevoice)
    if not cleaned or end_ms < start_ms:
        return None
    return RawTranscriptItem(
        item_id=int(item_id),
        start_ms=int(start_ms),
        end_ms=int(end_ms),
        text=cleaned,
        source=source,
        emotion=emotion,
        audio_event=audio_event,
        diagnostics=diagnostics or {},
    )


def _reindex_raw_items(items: Iterable[RawTranscriptItem]) -> list[RawTranscriptItem]:
    return [
        RawTranscriptItem(
            item_id=index,
            start_ms=int(item.start_ms),
            end_ms=int(item.end_ms),
            text=str(item.text),
            source=str(item.source),
            unit_id=item.unit_id,
            emotion=str(item.emotion or ""),
            audio_event=str(item.audio_event or ""),
            diagnostics=dict(item.diagnostics or {}),
        )
        for index, item in enumerate(items)
    ]


def _item_from_seconds(index: int, chunk: dict[str, Any], source: str) -> RawTranscriptItem | None:
    text = str(chunk.get("text") or "").strip()
    if not text:
        return None
    emotion = str(chunk.get("emotion") or "").strip()
    audio_event = str(chunk.get("audio_event") or chunk.get("audio_events") or "").strip()
    if "start_ms" in chunk:
        start_ms = int(chunk.get("start_ms") or 0)
        end_ms = int(chunk.get("end_ms", start_ms) or start_ms)
    else:
        start_ms = int(round(float(chunk.get("start_time", chunk.get("start", 0))) * 1000.0))
        end_ms = int(round(float(chunk.get("end_time", chunk.get("end", start_ms / 1000.0))) * 1000.0))
    return RawTranscriptItem(
        item_id=index,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        text=text,
        source=source,
        unit_id=None if chunk.get("unit_id") is None else int(chunk["unit_id"]),
        emotion=emotion,
        audio_event=audio_event,
    )


def raw_items_from_chunks(chunks: Iterable[dict[str, Any]], *, source: str) -> list[RawTranscriptItem]:
    items: list[RawTranscriptItem] = []
    for chunk in chunks:
        item = _item_from_seconds(len(items), chunk, source)
        if item is not None:
            items.append(item)
    return items


def split_sensevoice_timestamp_text(
    text: str,
    pairs: list[tuple[int, int]],
    *,
    source: str,
) -> list[RawTranscriptItem]:
    tagless, emotion, audio_event = _extract_sensevoice_tags(text)
    timed_count = sum(1 for char in tagless if _timed_char(char))
    if not pairs or (timed_count >= 8 and len(pairs) < int(timed_count * 0.4)):
        return []

    items: list[RawTranscriptItem] = []
    current: list[str] = []
    current_start: int | None = None
    current_end: int | None = None
    pair_index = 0

    def flush(reason: str) -> None:
        nonlocal current, current_start, current_end
        chunk_text = "".join(current).strip()
        if chunk_text and current_start is not None and current_end is not None:
            diagnostics: dict[str, Any] = {"timestamp_split": reason}
            if current_end - current_start > 12000:
                diagnostics["long_without_safe_boundary"] = True
            item = _raw_item_from_parts(
                item_id=len(items),
                start_ms=current_start,
                end_ms=current_end,
                text=chunk_text,
                source=source,
                is_sensevoice=True,
                emotion=emotion,
                audio_event=audio_event,
                diagnostics=diagnostics,
            )
            if item is not None:
                items.append(item)
        current = []
        current_start = None
        current_end = None

    for char in tagless:
        if not char.strip():
            if current and current[-1] != " ":
                current.append(" ")
            continue
        if _timed_char(char):
            if pair_index >= len(pairs):
                if current:
                    current.append(char)
                continue
            start_ms, end_ms = pairs[pair_index]
            pair_index += 1
            if current_start is None:
                current_start = start_ms
            current_end = max(start_ms, end_ms)
            current.append(char)
        elif current:
            current.append(char)
        if current_start is None or current_end is None:
            continue
        duration_ms = current_end - current_start
        if char in _STRONG_PUNCT and duration_ms >= 1500:
            flush("strong_punctuation")
        elif char in _SOFT_PUNCT and duration_ms >= 8000:
            flush("soft_punctuation_after_8s")

    flush("end")
    return items


def split_sensevoice_word_timestamp_text(
    text: str,
    words: list[object],
    pairs: list[tuple[int, int]],
    *,
    source: str,
) -> list[RawTranscriptItem]:
    if not words or len(words) != len(pairs):
        return []
    tagless, emotion, audio_event = _extract_sensevoice_tags(text)
    spans = _align_token_spans(tagless, words)
    if spans is None:
        return []

    items: list[RawTranscriptItem] = []
    start_index = 0

    def flush(end_index: int, reason: str) -> None:
        nonlocal start_index
        left = spans[start_index][0]
        right = spans[end_index + 1][0] if end_index + 1 < len(spans) else len(tagless)
        chunk_text = tagless[left:right].strip()
        if chunk_text:
            item = _raw_item_from_parts(
                item_id=len(items),
                start_ms=int(pairs[start_index][0]),
                end_ms=int(pairs[end_index][1]),
                text=chunk_text,
                source=source,
                is_sensevoice=True,
                emotion=emotion,
                audio_event=audio_event,
                diagnostics={"timestamp_split": reason, "text_source": "raw_text_slice"},
            )
            if item is not None:
                items.append(item)
        start_index = end_index + 1

    for index, (raw_word, (start_ms, end_ms)) in enumerate(zip(words, pairs)):
        word = str(raw_word or "").strip()
        if not word:
            continue
        if index > start_index and int(start_ms) - int(pairs[index - 1][1]) > 1500:
            flush(index - 1, "word_gap")
        duration_ms = int(end_ms) - int(pairs[start_index][0])
        boundary_right = spans[index + 1][0] if index + 1 < len(spans) else len(tagless)
        source_token = tagless[spans[index][0]:boundary_right].rstrip()
        if source_token.endswith(tuple(_STRONG_PUNCT)) and duration_ms >= 1500:
            flush(index, "strong_punctuation")
        elif source_token.endswith(tuple(_SOFT_PUNCT)) and duration_ms >= 8000:
            flush(index, "soft_punctuation_after_8s")

    if start_index < len(words):
        flush(len(words) - 1, "end")
    return items


def split_raw_item_on_safe_punctuation(item: RawTranscriptItem, *, max_ms: int = 12000) -> list[RawTranscriptItem]:
    duration_ms = int(item.end_ms) - int(item.start_ms)
    text = str(item.text)
    if duration_ms <= max_ms:
        return [item]

    breakpoints: list[int] = []
    last_break = 0
    for index, char in enumerate(text):
        since_last = duration_ms * (index + 1 - last_break) / max(1, len(text))
        if char in _STRONG_PUNCT and since_last >= 1500:
            breakpoints.append(index + 1)
            last_break = index + 1
        elif char in _SOFT_PUNCT and since_last >= 8000:
            breakpoints.append(index + 1)
            last_break = index + 1

    if not breakpoints:
        return [
            RawTranscriptItem(
                item_id=int(item.item_id),
                start_ms=int(item.start_ms),
                end_ms=int(item.end_ms),
                text=text,
                source=str(item.source),
                unit_id=item.unit_id,
                emotion=str(item.emotion or ""),
                audio_event=str(item.audio_event or ""),
                diagnostics={
                    **dict(item.diagnostics or {}),
                    "long_without_safe_boundary": True,
                    "safe_split": "kept_whole",
                },
            )
        ]

    pieces: list[tuple[int, int]] = []
    previous = 0
    for breakpoint in breakpoints:
        if breakpoint > previous:
            pieces.append((previous, breakpoint))
        previous = breakpoint
    if previous < len(text):
        pieces.append((previous, len(text)))

    output: list[RawTranscriptItem] = []
    for piece_index, (left, right) in enumerate(pieces):
        piece_text = text[left:right].strip()
        if not piece_text:
            continue
        start_ms = int(item.start_ms + round(duration_ms * left / max(1, len(text))))
        end_ms = int(item.start_ms + round(duration_ms * right / max(1, len(text))))
        output.append(
            RawTranscriptItem(
                item_id=piece_index,
                start_ms=start_ms,
                end_ms=max(start_ms, end_ms),
                text=piece_text,
                source=f"{item.source}_safe_split",
                unit_id=item.unit_id,
                emotion=str(item.emotion or ""),
                audio_event=str(item.audio_event or ""),
                diagnostics={**dict(item.diagnostics or {}), "safe_split": "punctuation_linear_time"},
            )
        )
    return output or [item]


def apply_safe_raw_split(raw_items: Iterable[RawTranscriptItem], *, max_ms: int = 12000) -> list[RawTranscriptItem]:
    split_items: list[RawTranscriptItem] = []
    for item in raw_items:
        split_items.extend(split_raw_item_on_safe_punctuation(item, max_ms=max_ms))
    return _reindex_raw_items(split_items)


def parse_funasr_raw_transcript(
    result: object,
    *,
    is_sensevoice: bool,
    split_timestamp_text: bool = False,
    fallback_start_ms: int | None = None,
    fallback_end_ms: int | None = None,
) -> tuple[list[RawTranscriptItem], dict[str, int]]:
    items: list[RawTranscriptItem] = []
    timestamp_mismatch_items = 0
    timestamp_jump_warnings = 0
    timestamp_split_items = 0
    timestamp_fallback_items = 0
    long_without_safe_boundary = 0

    def add_item(
        start_ms: int,
        end_ms: int,
        text: str,
        source: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        item = _raw_item_from_parts(
            item_id=len(items),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            source=source,
            is_sensevoice=is_sensevoice,
            diagnostics=diagnostics,
        )
        if item is not None:
            items.append(
                item
            )

    source_items = result if isinstance(result, list) else [result]
    for raw in source_items:
        if not isinstance(raw, dict):
            continue
        sentence_info = raw.get("sentence_info") or []
        if isinstance(sentence_info, list) and sentence_info:
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                start_ms = int(sentence.get("start", sentence.get("start_ms", 0)) or 0)
                end_ms = int(sentence.get("end", sentence.get("end_ms", start_ms)) or start_ms)
                add_item(start_ms, end_ms, str(sentence.get("text", sentence.get("sentence", ""))), "funasr_sentence")
            continue

        text = str(raw.get("text") or "").strip()
        if not text:
            continue

        timestamps = _valid_timestamp_pairs(raw.get("timestamp"))
        words = raw.get("words")
        word_aligned_timestamps = isinstance(words, list) and bool(timestamps) and len(words) == len(timestamps)
        if is_sensevoice and split_timestamp_text and word_aligned_timestamps:
            split_items = split_sensevoice_word_timestamp_text(
                text,
                words,
                timestamps,
                source="funasr_word_timestamp_split",
            )
            if split_items:
                for item in split_items:
                    items.append(
                        RawTranscriptItem(
                            item_id=len(items),
                            start_ms=item.start_ms,
                            end_ms=item.end_ms,
                            text=item.text,
                            source=item.source,
                            emotion=item.emotion,
                            audio_event=item.audio_event,
                            diagnostics=item.diagnostics,
                        )
                    )
                timestamp_split_items += len(split_items)
                continue

        timestamp_text = "".join(str(word) for word in words) if word_aligned_timestamps else text
        timed_count = sum(1 for char in timestamp_text if _timed_char(char))
        diagnostics: dict[str, Any] = {}
        if timestamps and not word_aligned_timestamps and timed_count >= 8 and len(timestamps) < int(timed_count * 0.4):
            timestamp_mismatch_items += 1
            diagnostics["timestamp_mismatch"] = True
            timestamps = []
        if timestamps:
            starts = [pair[0] for pair in timestamps]
            jumps = sum(1 for left, right in zip(starts, starts[1:]) if right - left > 5000)
            timestamp_jump_warnings += jumps
            if jumps:
                diagnostics["timestamp_jumps"] = jumps
            if is_sensevoice and split_timestamp_text:
                split_items = split_sensevoice_timestamp_text(text, timestamps, source="funasr_timestamp_split")
                if split_items:
                    for item in split_items:
                        if item.diagnostics.get("long_without_safe_boundary"):
                            long_without_safe_boundary += 1
                        items.append(
                            RawTranscriptItem(
                                item_id=len(items),
                                start_ms=item.start_ms,
                                end_ms=item.end_ms,
                                text=item.text,
                                source=item.source,
                                emotion=item.emotion,
                                audio_event=item.audio_event,
                                diagnostics=item.diagnostics,
                            )
                        )
                    timestamp_split_items += len(split_items)
                    continue
                timestamp_fallback_items += 1
            add_item(timestamps[0][0], timestamps[-1][1], text, "funasr_timestamp", diagnostics)
            continue

        has_raw_start = "start" in raw or "start_ms" in raw
        has_raw_end = "end" in raw or "end_ms" in raw
        start_ms = int(raw.get("start", raw.get("start_ms", fallback_start_ms or 0)) or 0)
        end_default = fallback_end_ms if fallback_end_ms is not None else start_ms
        end_ms = int(raw.get("end", raw.get("end_ms", end_default)) or end_default)
        if not has_raw_start or not has_raw_end:
            timestamp_fallback_items += 1
            diagnostics["timestamp_fallback_bounds"] = True
        add_item(start_ms, end_ms, text, "funasr_text", diagnostics)

    return items, {
        "raw_items": len(items),
        "timestamp_mismatch_items": timestamp_mismatch_items,
        "timestamp_jump_warnings": timestamp_jump_warnings,
        "timestamp_split_items": timestamp_split_items,
        "timestamp_fallback_items": timestamp_fallback_items,
        "long_without_safe_boundary": long_without_safe_boundary,
    }
