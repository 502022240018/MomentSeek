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


def parse_funasr_raw_transcript(result: object, *, is_sensevoice: bool) -> tuple[list[RawTranscriptItem], dict[str, int]]:
    items: list[RawTranscriptItem] = []
    timestamp_mismatch_items = 0
    timestamp_jump_warnings = 0

    def add_item(
        start_ms: int,
        end_ms: int,
        text: str,
        source: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        emotion = ""
        audio_event = ""
        if is_sensevoice:
            text, emotion, audio_event = _extract_sensevoice_tags(text)
        cleaned = _clean_funasr_text(text, is_sensevoice)
        if cleaned and end_ms >= start_ms:
            items.append(
                RawTranscriptItem(
                    item_id=len(items),
                    start_ms=int(start_ms),
                    end_ms=int(end_ms),
                    text=cleaned,
                    source=source,
                    emotion=emotion,
                    audio_event=audio_event,
                    diagnostics=diagnostics or {},
                )
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
        timestamp_text = "".join(str(word) for word in words) if isinstance(words, list) and len(words) == len(timestamps) else text
        timed_count = sum(1 for char in timestamp_text if _timed_char(char))
        diagnostics: dict[str, Any] = {}
        if timestamps and timed_count >= 8 and len(timestamps) < int(timed_count * 0.4):
            timestamp_mismatch_items += 1
            diagnostics["timestamp_mismatch"] = True
            timestamps = []
        if timestamps:
            starts = [pair[0] for pair in timestamps]
            jumps = sum(1 for left, right in zip(starts, starts[1:]) if right - left > 5000)
            timestamp_jump_warnings += jumps
            if jumps:
                diagnostics["timestamp_jumps"] = jumps
            add_item(timestamps[0][0], timestamps[-1][1], text, "funasr_timestamp", diagnostics)
            continue

        start_ms = int(raw.get("start", raw.get("start_ms", 0)) or 0)
        end_ms = int(raw.get("end", raw.get("end_ms", start_ms)) or start_ms)
        add_item(start_ms, end_ms, text, "funasr_text", diagnostics)

    return items, {
        "raw_items": len(items),
        "timestamp_mismatch_items": timestamp_mismatch_items,
        "timestamp_jump_warnings": timestamp_jump_warnings,
    }
