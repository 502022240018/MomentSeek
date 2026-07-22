from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import types
import wave
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np

from app.indexing.asr_debug import write_asr_debug_artifacts
from app.indexing.asr_retrieval_chunks import RetrievalChunkConfig, build_retrieval_chunks
from app.indexing.asr_text import asr_text_profile, normalize_search_text
from app.indexing.asr_pipeline_types import RawTranscriptItem
from app.indexing.asr_transcript_parser import (
    apply_safe_raw_split,
    parse_funasr_raw_transcript,
    raw_items_from_chunks,
)
from app.indexing.common import atomic_save_npz
from app.indexing.text_semantic import build_text_semantic_arrays, resolve_text_embedding_device
from app.media import extract_audio, parse_timecode
from app.model_sources import (
    offline_env,
    resolve_faster_whisper_model_source,
    resolve_modelscope_model_source,
)


SAMPLE_RATE = 16000
FASTER_WHISPER_WINDOW_SECONDS = 24.0
FASTER_WHISPER_WINDOW_JOIN_GAP_SECONDS = 5.0
FASTER_WHISPER_FALLBACK_SPAN_SECONDS = 12.0
_CHINESE_ASR_LANGUAGES = {"zh", "zh-cn", "zh-tw", "cmn", "yue", "chinese"}
_STRONG_PUNCTUATION = (".", "!", "?", "\u3002", "\uff01", "\uff1f")
_SOFT_PUNCTUATION = (",", ";", ":", "\uff0c", "\uff1b", "\uff1a", "\u3001")


def resolve_asr_device(device: str, cuda_enabled: bool = False, npu_enabled: bool = False, npu_device_id: int = 0) -> str:
    """Pick the torch device for Whisper/FunASR.

    'auto' = CUDA when enabled and present, else Ascend NPU when enabled and
    torch_npu is importable, else CPU. An explicit device string is honored as-is.
    """
    if device and device != "auto":
        return device
    if cuda_enabled:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if npu_enabled:
        try:
            import torch_npu  # noqa: F401

            return f"npu:{npu_device_id}"
        except Exception:
            pass
    return "cpu"


def load_sidecar(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("segments", payload) if isinstance(payload, dict) else payload
        return [
            {
                "start_time": float(item.get("start_time", item.get("start", 0))),
                "end_time": float(item.get("end_time", item.get("end", 0))),
                "text": str(item.get("text", "")).strip(),
            }
            for item in items
            if str(item.get("text", "")).strip()
        ]
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    chunks = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start, end = lines[timing_index].split("-->", 1)
        text = " ".join(lines[timing_index + 1:]).strip()
        if text:
            chunks.append({"start_time": parse_timecode(start), "end_time": parse_timecode(end), "text": text})
    return chunks


def _is_sensevoice_model(model_name: str, model_source: str = "") -> bool:
    return "sensevoice" in f"{model_name} {model_source}".casefold()


def _is_auto_language(language: str) -> bool:
    return not language or language.casefold() == "auto"


def _is_chinese_language(language: str) -> bool:
    return str(language or "").strip().casefold() in _CHINESE_ASR_LANGUAGES


def _parse_funasr_chunks(
    result: object,
    *,
    is_sensevoice: bool,
    split_timestamp_text: bool = False,
) -> list[dict]:
    raw_items, _diagnostics = parse_funasr_raw_transcript(
        result,
        is_sensevoice=is_sensevoice,
        split_timestamp_text=split_timestamp_text,
    )
    return [item.to_dict() for item in raw_items]


def _write_wav_mono(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(int16.tobytes())


def _build_silero_groups(
    audio: np.ndarray,
    vad_model: object,
    *,
    max_group_seconds: float = 12.0,
    max_group_gap_seconds: float = 2.0,
) -> tuple[list[tuple[int, float, float]], dict[str, object]]:
    from silero_vad import get_speech_timestamps

    try:
        import torch

        vad_audio = torch.from_numpy(audio)
    except Exception:
        vad_audio = audio
    speech = get_speech_timestamps(
        vad_audio,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=200,
    )
    intervals = [
        (max(0, int(item["start"])) / SAMPLE_RATE, min(len(audio), int(item["end"])) / SAMPLE_RATE)
        for item in speech
    ]
    groups: list[tuple[int, float, float]] = []
    current_start: float | None = None
    current_end: float | None = None
    for start, end in intervals:
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        gap = max(0.0, start - current_end)
        if end - current_start <= max_group_seconds and gap <= max_group_gap_seconds:
            current_end = end
        else:
            groups.append((len(groups), current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        groups.append((len(groups), current_start, current_end))

    return groups, {
        "speech_intervals": len(intervals),
        "vad_groups": len(groups),
        "max_group_seconds": max_group_seconds,
        "speech_seconds": round(sum(max(0.0, end - start) for start, end in intervals), 3),
        "group_audio_seconds": round(sum(max(0.0, end - start) for _, start, end in groups), 3),
        "groups_gt_cap": sum(1 for _, start, end in groups if end - start > max_group_seconds),
    }


def _load_silero_onnx_vad() -> object:
    """Load Silero's bundled ONNX model without importing torchaudio.

    MomentSeek decodes and resamples audio itself, so only Silero's timestamp
    functions are needed.  A minimal module placeholder prevents the upstream
    utility module's optional audio I/O import from pulling a torch-specific
    torchaudio build into CUDA/Ascend runtimes.
    """
    if "torchaudio" not in sys.modules:
        try:
            __import__("torchaudio")
        except (ImportError, OSError):
            sys.modules["torchaudio"] = types.ModuleType("torchaudio")

    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def _offset_raw_items(
    items: list[RawTranscriptItem],
    offset_ms: int,
    *,
    unit_id: int | None = None,
) -> list[RawTranscriptItem]:
    return [
        RawTranscriptItem(
            item_id=index,
            start_ms=int(item.start_ms) + offset_ms,
            end_ms=int(item.end_ms) + offset_ms,
            text=str(item.text),
            source=str(item.source),
            unit_id=item.unit_id if unit_id is None else unit_id,
            emotion=str(item.emotion or ""),
            audio_event=str(item.audio_event or ""),
            diagnostics=dict(item.diagnostics or {}),
        )
        for index, item in enumerate(items)
    ]


def _reindex_raw_items(items: list[RawTranscriptItem]) -> list[RawTranscriptItem]:
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


def _sensevoice_silero(
    audio_path: str,
    model_kwargs: dict,
    *,
    language: str,
    local_files_only: bool,
    temp_dir: str | Path | None,
) -> list[dict]:
    from funasr import AutoModel

    with offline_env(local_files_only):
        model = AutoModel(**model_kwargs)
    vad_model = _load_silero_onnx_vad()
    audio = load_wav_mono(audio_path)
    groups, _vad_stats = _build_silero_groups(audio, vad_model, max_group_seconds=12.0)
    if not groups:
        return []

    clip_dir = Path(temp_dir) if temp_dir is not None else Path(audio_path).parent / "silero_vad_clips"
    raw_items: list[RawTranscriptItem] = []
    for group_index, start_s, end_s in groups:
        start_sample = int(round(start_s * SAMPLE_RATE))
        end_sample = int(round(end_s * SAMPLE_RATE))
        group_audio = audio[start_sample:end_sample]
        group_path = clip_dir / f"silero_{group_index:04d}_{int(start_s * 1000)}_{int(end_s * 1000)}.wav"
        _write_wav_mono(group_path, group_audio)
        generate_kwargs = {
            "input": str(group_path),
            "cache": {},
            "batch_size_s": 60,
            "use_itn": True,
            "merge_vad": False,
            "merge_length_s": 8,
            "output_timestamp": True,
            "return_time_stamps": True,
        }
        if language and language != "auto":
            generate_kwargs["language"] = language
        result = model.generate(**generate_kwargs)
        parsed_items, _stats = parse_funasr_raw_transcript(
            result,
            is_sensevoice=True,
            split_timestamp_text=True,
            fallback_start_ms=0,
            fallback_end_ms=int(round((end_s - start_s) * 1000)),
        )
        parsed_items = apply_safe_raw_split(parsed_items)
        raw_items.extend(
            _offset_raw_items(
                parsed_items,
                int(round(start_s * 1000)),
                unit_id=group_index,
            )
        )
        try:
            group_path.unlink(missing_ok=True)
        except OSError:
            pass

    return [item.to_dict() for item in _reindex_raw_items(raw_items)]


def _funasr(
    audio_path: str,
    model_name: str,
    device: str,
    model_root: str | Path | None = None,
    local_files_only: bool = True,
    language: str = "auto",
    vad_strategy: str = "funasr_fsmn",
    temp_dir: str | Path | None = None,
) -> list[dict]:
    model_source = resolve_modelscope_model_source(model_root, model_name, local_files_only=local_files_only)
    is_sensevoice = _is_sensevoice_model(model_name, model_source)
    normalized_vad_strategy = str(vad_strategy or "funasr_fsmn").replace("-", "_").casefold()
    use_silero = is_sensevoice and normalized_vad_strategy in {"silero", "silero_12s", "silero_external"}
    vad_source = None if use_silero else resolve_modelscope_model_source(
        model_root,
        "fsmn-vad",
        local_files_only=local_files_only,
    )
    punc_source = None if is_sensevoice else resolve_modelscope_model_source(
        model_root, "ct-punc", local_files_only=local_files_only
    )
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before model load
    from funasr import AutoModel

    model_kwargs = {
        "model": model_source,
        "device": device,
        "disable_update": True,
    }
    if vad_source is not None:
        model_kwargs["vad_model"] = vad_source
    if is_sensevoice:
        if use_silero:
            return _sensevoice_silero(
                audio_path,
                model_kwargs,
                language=language,
                local_files_only=local_files_only,
                temp_dir=temp_dir,
            )
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    else:
        model_kwargs["punc_model"] = punc_source
    with offline_env(local_files_only):
        model = AutoModel(**model_kwargs)
    generate_kwargs = {"input": audio_path, "cache": {}, "batch_size_s": 60 if is_sensevoice else 300}
    if language and language != "auto":
        generate_kwargs["language"] = language
    if is_sensevoice:
        generate_kwargs.update({
            "use_itn": True,
            "merge_vad": True,
            "merge_length_s": 15,
            "output_timestamp": True,
            "return_time_stamps": True,
        })
    else:
        generate_kwargs["sentence_timestamp"] = True
    result = model.generate(**generate_kwargs)
    return _parse_funasr_chunks(result, is_sensevoice=is_sensevoice)


def _resolve_whisper_model_source(model_name: str, model_dir: str | Path, local_files_only: bool) -> str:
    explicit_path = Path(model_name)
    if explicit_path.exists():
        return str(explicit_path)
    cached = Path(model_dir) / f"{model_name}.pt"
    if cached.is_file() and cached.stat().st_size > 0:
        return str(cached)
    if local_files_only:
        raise FileNotFoundError(f"本地 Whisper 模型缺失: {model_name}; model_dir={model_dir}")
    return model_name


def _whisper(
    audio_path: str,
    model_name: str,
    device: str,
    model_dir: str,
    language: str = "auto",
    local_files_only: bool = True,
) -> tuple[list[dict], dict]:
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before .to(device)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_source = _resolve_whisper_model_source(model_name, model_dir, local_files_only)
    import whisper

    model = whisper.load_model(model_source, device=device, download_root=model_dir)
    options = {"fp16": device != "cpu", "task": "transcribe"}
    if language and language != "auto":
        options["language"] = language
    result = model.transcribe(load_wav_mono(audio_path), **options)
    chunks = [
        {"start_time": float(item["start"]), "end_time": float(item["end"]), "text": item["text"].strip()}
        for item in result.get("segments", [])
        if item.get("text", "").strip()
    ]
    return chunks, {
        "task": "transcribe",
        "requested_language": language or "auto",
        "detected_language": str(result.get("language") or ""),
    }


def _word_text(value: object) -> str:
    return str(getattr(value, "word", value) or "").strip()


def _word_start(value: object, default: float) -> float:
    raw = getattr(value, "start", default)
    return default if raw is None else float(raw)


def _word_end(value: object, default: float) -> float:
    raw = getattr(value, "end", default)
    return default if raw is None else float(raw)


def _join_word_texts(parts: list[str]) -> str:
    cleaned = [part.strip() for part in parts if part.strip()]
    if not cleaned:
        return ""
    text = " ".join(cleaned)
    text = re.sub(r"\s+([.,!?;:\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001])", r"\1", text)
    return text.strip()


@dataclass
class _WhisperWordChunk:
    words: list[str] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    def add(self, text: str, start_time: float, end_time: float) -> None:
        if self.start_time is None:
            self.start_time = start_time
        self.end_time = max(start_time, end_time)
        self.words.append(text)

    def flush(self) -> dict | None:
        text = _join_word_texts(self.words)
        result = None
        if text and self.start_time is not None and self.end_time is not None:
            result = {
                "start_time": float(self.start_time),
                "end_time": max(float(self.start_time), float(self.end_time)),
                "text": text,
            }
        self.words.clear()
        self.start_time = None
        self.end_time = None
        return result


def _word_chunk_boundary(
    text: str,
    duration: float,
    max_chunk_seconds: float,
    soft_punctuation_seconds: float,
) -> bool:
    stripped = text.rstrip()
    return (
        (stripped.endswith(_STRONG_PUNCTUATION) and duration >= 1.5)
        or (stripped.endswith(_SOFT_PUNCTUATION) and duration >= soft_punctuation_seconds)
        or duration >= max_chunk_seconds
    )


def _split_faster_whisper_segment(
    segment: object,
    *,
    max_chunk_seconds: float,
    soft_punctuation_seconds: float,
    speech_gap_seconds: float,
) -> list[dict]:
    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
    segment_end = float(getattr(segment, "end", segment_start) or segment_start)
    words = list(getattr(segment, "words", None) or [])
    if not words:
        text = str(getattr(segment, "text", "") or "").strip()
        return [{"start_time": segment_start, "end_time": max(segment_start, segment_end), "text": text}] if text else []
    chunks = []
    current = _WhisperWordChunk()
    for word in words:
        text = _word_text(word)
        if not text:
            continue
        start_time = _word_start(word, segment_start)
        end_time = _word_end(word, start_time)
        if current.end_time is not None and start_time - current.end_time > speech_gap_seconds:
            flushed = current.flush()
            if flushed is not None:
                chunks.append(flushed)
        current.add(text, start_time, end_time)
        if _word_chunk_boundary(
            text,
            float(current.end_time - current.start_time),
            max_chunk_seconds,
            soft_punctuation_seconds,
        ):
            flushed = current.flush()
            if flushed is not None:
                chunks.append(flushed)
    flushed = current.flush()
    if flushed is not None:
        chunks.append(flushed)
    return chunks


def _split_faster_whisper_segments(
    segments: Iterable[object],
    *,
    max_chunk_seconds: float = 12.0,
    soft_punctuation_seconds: float = 8.0,
    speech_gap_seconds: float = 1.5,
) -> list[dict]:
    chunks: list[dict] = []
    for segment in segments:
        chunks.extend(_split_faster_whisper_segment(
            segment,
            max_chunk_seconds=max_chunk_seconds,
            soft_punctuation_seconds=soft_punctuation_seconds,
            speech_gap_seconds=speech_gap_seconds,
        ))
    return chunks


def _faster_whisper_vad_options(**kwargs):
    try:
        from faster_whisper.vad import VadOptions
    except Exception:
        return kwargs
    return VadOptions(**kwargs)


def _build_faster_whisper_windows(
    audio: np.ndarray,
    *,
    cap_seconds: float = FASTER_WHISPER_WINDOW_SECONDS,
) -> tuple[list[tuple[int, int]], dict[str, float | int]]:
    from faster_whisper.vad import get_speech_timestamps

    options = _faster_whisper_vad_options(
        threshold=0.5,
        min_speech_duration_ms=0,
        max_speech_duration_s=max(8.0, cap_seconds - 1.0),
        min_silence_duration_ms=500,
        speech_pad_ms=400,
    )
    speech = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    windows: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for interval in speech:
        start = max(0, int(interval["start"]))
        end = min(len(audio), int(interval["end"]))
        if end <= start:
            continue
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        gap_seconds = (start - current_end) / SAMPLE_RATE
        span_seconds = (end - current_start) / SAMPLE_RATE
        if gap_seconds > FASTER_WHISPER_WINDOW_JOIN_GAP_SECONDS or span_seconds > cap_seconds:
            windows.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = end
    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))
    return windows, {
        "speech_intervals": len(speech),
        "windows": len(windows),
        "source_seconds": round(len(audio) / SAMPLE_RATE, 3),
        "window_audio_seconds": round(sum(end - start for start, end in windows) / SAMPLE_RATE, 3),
        "max_window_seconds": round(
            max(((end - start) / SAMPLE_RATE for start, end in windows), default=0.0),
            3,
        ),
    }


def _faster_whisper_segment_row(
    segment: object,
    *,
    offset_seconds: float,
    unit_id: int,
) -> dict:
    local_start = float(getattr(segment, "start", 0.0) or 0.0)
    local_end = float(getattr(segment, "end", local_start) or local_start)
    return {
        "start_time": offset_seconds + local_start,
        "end_time": offset_seconds + max(local_start, local_end),
        "text": str(getattr(segment, "text", "") or "").strip(),
        "unit_id": int(unit_id),
    }


def _longest_unpunctuated_span(rows: Iterable[dict]) -> float:
    longest = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for row in sorted(rows, key=lambda value: (float(value["start_time"]), float(value["end_time"]))):
        text = str(row.get("text") or "").strip()
        if current_start is None:
            current_start = float(row["start_time"])
        current_end = float(row["end_time"])
        longest = max(longest, current_end - current_start)
        if text.endswith(_STRONG_PUNCTUATION):
            current_start = None
            current_end = None
    return longest


def _adjacent_duplicate_count(rows: Iterable[dict]) -> int:
    texts = [str(row.get("text") or "").strip().casefold() for row in rows]
    return sum(texts[index] == texts[index - 1] for index in range(1, len(texts)) if texts[index])


def _faster_whisper_window_needs_fallback(rows: list[dict]) -> bool:
    if _longest_unpunctuated_span(rows) > FASTER_WHISPER_FALLBACK_SPAN_SECONDS:
        return True
    return any(
        float(row["end_time"]) - float(row["start_time"]) > 15.0
        and sum(str(row.get("text") or "").count(mark) for mark in _STRONG_PUNCTUATION) < 2
        for row in rows
    )


def _accept_faster_whisper_fallback(
    primary: list[dict],
    fallback: list[dict],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
) -> tuple[bool, str]:
    if not fallback:
        return False, "empty_fallback"
    if any(
        float(row["start_time"]) < window_start_seconds - 0.1
        or float(row["end_time"]) > window_end_seconds + 0.1
        or float(row["end_time"]) < float(row["start_time"])
        for row in fallback
    ):
        return False, "timestamp_out_of_window"
    primary_chars = len("".join(str(row.get("text") or "") for row in primary).replace(" ", ""))
    fallback_chars = len("".join(str(row.get("text") or "") for row in fallback).replace(" ", ""))
    if primary_chars and fallback_chars < int(primary_chars * 0.5):
        return False, "text_coverage_drop"
    primary_text = normalize_search_text(" ".join(str(row.get("text") or "") for row in primary))
    fallback_text = normalize_search_text(" ".join(str(row.get("text") or "") for row in fallback))
    if primary_text and fallback_text and SequenceMatcher(None, primary_text, fallback_text).ratio() < 0.35:
        return False, "text_content_mismatch"
    primary_span = _longest_unpunctuated_span(primary)
    fallback_span = _longest_unpunctuated_span(fallback)
    if fallback_span >= primary_span - 0.25:
        return False, "no_boundary_improvement"
    if _adjacent_duplicate_count(fallback) > _adjacent_duplicate_count(primary) + 1:
        return False, "repetition_regression"
    return True, "improved_unpunctuated_span"


def _detect_language_with_faster_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    model_dir: str,
    *,
    local_files_only: bool = True,
    probe_seconds: float = 20.0,
) -> tuple[str, float]:
    from faster_whisper import WhisperModel

    model_source = resolve_faster_whisper_model_source(model_dir, model_name, local_files_only=local_files_only)
    fw_device = "cuda" if str(device).startswith("cuda") else "cpu"
    compute_type = "float16" if fw_device == "cuda" else "int8"
    with offline_env(local_files_only):
        model = WhisperModel(model_source, device=fw_device, compute_type=compute_type)
    audio = load_wav_mono(audio_path)
    if audio.size == 0:
        return "", 0.0
    max_samples = max(1, int(round(probe_seconds * SAMPLE_RATE)))
    if len(audio) <= max_samples * 2:
        probes = [audio[:max_samples]]
    else:
        max_start = len(audio) - max_samples
        starts = [
            0,
            int(round(max_start * 0.45)),
            int(round(max_start * 0.80)),
        ]
        probes = [audio[start:start + max_samples] for start in starts]

    votes: dict[str, dict[str, float | int]] = {}
    for probe_audio in probes:
        try:
            language, probability, _all_probabilities = model.detect_language(
                audio=probe_audio,
                vad_filter=True,
                vad_parameters=_faster_whisper_vad_options(min_silence_duration_ms=500),
                language_detection_segments=1,
                language_detection_threshold=0.5,
            )
        except ValueError:
            continue
        normalized = str(language or "").strip().casefold()
        if not normalized:
            continue
        vote = votes.setdefault(normalized, {"count": 0, "probability_sum": 0.0})
        vote["count"] = int(vote["count"]) + 1
        vote["probability_sum"] = float(vote["probability_sum"]) + float(probability or 0.0)
    if not votes:
        return "", 0.0
    winner, winning_vote = max(
        votes.items(),
        key=lambda item: (int(item[1]["count"]), float(item[1]["probability_sum"])),
    )
    average_probability = float(winning_vote["probability_sum"]) / max(1, int(winning_vote["count"]))
    return winner, average_probability


def _faster_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    model_dir: str,
    language: str = "auto",
    local_files_only: bool = True,
) -> tuple[list[dict], dict]:
    from faster_whisper import WhisperModel

    model_source = resolve_faster_whisper_model_source(model_dir, model_name, local_files_only=local_files_only)
    fw_device = "cuda" if str(device).startswith("cuda") else "cpu"
    compute_type = "float16" if fw_device == "cuda" else "int8"
    with offline_env(local_files_only):
        model = WhisperModel(model_source, device=fw_device, compute_type=compute_type)
    audio = load_wav_mono(audio_path)
    windows, window_stats = _build_faster_whisper_windows(audio)
    active_language = None if not language or language == "auto" else language
    detected_language = active_language or ""
    chunks: list[dict] = []
    fallback_windows: list[int] = []
    fallback_rejected: list[dict[str, str | int]] = []
    decode_seconds = 0.0
    fallback_decode_seconds = 0.0
    timestamp_clamps = 0

    for unit_id, (start_sample, end_sample) in enumerate(windows):
        offset_seconds = start_sample / SAMPLE_RATE
        window_end_seconds = end_sample / SAMPLE_RATE
        window_audio = audio[start_sample:end_sample]
        started = time.perf_counter()
        segments_iter, info = model.transcribe(
            window_audio,
            language=active_language,
            task="transcribe",
            vad_filter=False,
            condition_on_previous_text=True,
            beam_size=5,
            word_timestamps=True,
        )
        primary = [
            _faster_whisper_segment_row(segment, offset_seconds=offset_seconds, unit_id=unit_id)
            for segment in segments_iter
        ]
        decode_seconds += time.perf_counter() - started
        primary = [row for row in primary if str(row["text"]).strip()]
        if not detected_language:
            detected_language = str(getattr(info, "language", "") or "")
            if detected_language:
                active_language = detected_language

        if _faster_whisper_window_needs_fallback(primary):
            fallback_started = time.perf_counter()
            fallback_iter, _fallback_info = model.transcribe(
                window_audio,
                language=active_language,
                task="transcribe",
                vad_filter=True,
                vad_parameters=_faster_whisper_vad_options(min_silence_duration_ms=500),
                condition_on_previous_text=True,
                beam_size=5,
                word_timestamps=True,
            )
            fallback = [
                _faster_whisper_segment_row(segment, offset_seconds=offset_seconds, unit_id=unit_id)
                for segment in fallback_iter
            ]
            fallback_decode_seconds += time.perf_counter() - fallback_started
            fallback = [row for row in fallback if str(row["text"]).strip()]
            accepted, reason = _accept_faster_whisper_fallback(
                primary,
                fallback,
                window_start_seconds=offset_seconds,
                window_end_seconds=window_end_seconds,
            )
            if accepted:
                primary = fallback
                fallback_windows.append(unit_id)
            else:
                fallback_rejected.append({"unit_id": unit_id, "reason": reason})

        for row in primary:
            start_time = max(offset_seconds, float(row["start_time"]))
            end_time = min(window_end_seconds, max(start_time, float(row["end_time"])))
            if start_time != float(row["start_time"]) or end_time != float(row["end_time"]):
                timestamp_clamps += 1
            chunks.append({**row, "start_time": start_time, "end_time": end_time})

    chunks.sort(key=lambda row: (float(row["start_time"]), float(row["end_time"])))
    return chunks, {
        "task": "transcribe",
        "requested_language": language or "auto",
        "detected_language": detected_language,
        "compute_type": compute_type,
        "word_timestamps": True,
        "vad_strategy": "contiguous_24s_local_fallback",
        "window_stats": window_stats,
        "decode_seconds": round(decode_seconds, 3),
        "fallback_decode_seconds": round(fallback_decode_seconds, 3),
        "fallback_windows": fallback_windows,
        "fallback_rejected": fallback_rejected,
        "timestamp_clamps": timestamp_clamps,
    }


def load_wav_mono(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported, got sample_width={sample_width}")
    data = np.frombuffer(frames, np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32) / 32768.0


def _empty_chunk_builder_stats() -> dict[str, int]:
    return {
        "raw_items": 0,
        "normalized_items": 0,
        "retrieval_chunks": 0,
        "dropped_empty_items": 0,
        "merged_items": 0,
        "cross_unit_merge_blocks": 0,
        "long_chunks": 0,
        "low_boundary_chunks": 0,
        "semantic_ineligible_chunks": 0,
    }


@dataclass(frozen=True)
class _AsrDecodeRequest:
    audio_path: Path
    engine: str
    model_name: str
    device: str
    model_dir: str
    requested_language: str
    funasr_model: str
    funasr_model_dir: str | None
    faster_whisper_model_dir: str | None
    model_local_files_only: bool
    working_dir: str
    vad_strategy: str


@dataclass
class _AsrDecodeResult:
    raw_items: list[RawTranscriptItem]
    used_engine: str
    effective_model: str
    requested_language: str
    detected_language: str = ""
    task: str = "transcribe"
    tag_source: str = ""
    language_route: str = ""
    route_reason: str = ""
    vad_strategy: str = "funasr_fsmn"
    raw_parser_stats: dict[str, int] = field(default_factory=dict)


def _resolve_decode_route(request: _AsrDecodeRequest) -> tuple[str, str, str, str, str]:
    normalized_engine = request.engine.replace("_", "-").casefold()
    route_engine = normalized_engine
    detected_language = ""
    language_route = ""
    route_reason = ""
    if normalized_engine == "auto":
        if _is_auto_language(request.requested_language):
            detected_language, probability = _detect_language_with_faster_whisper(
                str(request.audio_path),
                request.model_name,
                request.device,
                request.faster_whisper_model_dir or str(Path(request.model_dir).parent / "faster-whisper"),
                local_files_only=request.model_local_files_only,
            )
            route_engine = "funasr" if _is_chinese_language(detected_language) else "faster-whisper"
            language_route = f"auto:probe={detected_language or 'unknown'}->{route_engine}"
            route_reason = f"auto language probe probability={probability:.3f}"
        elif _is_chinese_language(request.requested_language):
            route_engine = "funasr"
            language_route = "auto:explicit-zh->funasr"
            route_reason = "explicit Chinese ASR language"
        else:
            route_engine = "faster-whisper"
            language_route = f"auto:explicit-{request.requested_language}->faster-whisper"
            route_reason = "explicit non-Chinese ASR language"
    decode_language = request.requested_language
    if normalized_engine == "auto" and _is_auto_language(request.requested_language) and detected_language:
        decode_language = detected_language
    return route_engine, detected_language, decode_language, language_route, route_reason


def _decode_faster_whisper_route(
    request: _AsrDecodeRequest,
    *,
    decode_language: str,
    detected_language: str,
    language_route: str,
    route_reason: str,
    fallback: bool = False,
) -> _AsrDecodeResult:
    model_chunks, metadata = _faster_whisper(
        audio_path=str(request.audio_path),
        model_name=request.model_name,
        device=request.device,
        model_dir=request.faster_whisper_model_dir or str(Path(request.model_dir).parent / "faster-whisper"),
        language=decode_language,
        local_files_only=request.model_local_files_only,
    )
    detected_language = str(metadata.get("detected_language") or "")
    if fallback:
        language_route = language_route or "auto:funasr-fallback->faster-whisper"
        route_reason = route_reason or "FunASR route failed; fell back to faster-whisper"
    else:
        language_route = language_route or f"faster-whisper:{detected_language or request.requested_language}"
        route_reason = route_reason or (
            "explicit language" if request.requested_language != "auto" else "explicit faster-whisper engine"
        )
    return _AsrDecodeResult(
        raw_items=raw_items_from_chunks(model_chunks, source="faster_whisper"),
        used_engine="faster-whisper",
        effective_model=request.model_name,
        requested_language=request.requested_language,
        detected_language=detected_language,
        task=str(metadata.get("task") or "transcribe"),
        language_route=language_route,
        route_reason=route_reason,
        vad_strategy=str(metadata.get("vad_strategy") or request.vad_strategy),
    )


def _decode_funasr_route(
    request: _AsrDecodeRequest,
    *,
    normalized_engine: str,
    decode_language: str,
    detected_language: str,
    language_route: str,
    route_reason: str,
) -> _AsrDecodeResult:
    try:
        model_chunks = _funasr(
            str(request.audio_path),
            request.funasr_model,
            request.device,
            model_root=request.funasr_model_dir or str(Path(request.model_dir).parent / "funasr"),
            local_files_only=request.model_local_files_only,
            language=request.requested_language,
            vad_strategy=request.vad_strategy,
            temp_dir=Path(request.working_dir) / "asr_silero_clips",
        )
    except Exception:
        if normalized_engine == "funasr":
            raise
        return _decode_faster_whisper_route(
            request,
            decode_language=decode_language,
            detected_language=detected_language,
            language_route=language_route,
            route_reason=route_reason,
            fallback=True,
        )
    if (
        detected_language
        or request.requested_language == "zh"
        or "zh" in request.funasr_model.casefold()
        or "paraformer" in request.funasr_model.casefold()
    ):
        detected_language = detected_language or "zh"
    return _AsrDecodeResult(
        raw_items=raw_items_from_chunks(model_chunks, source="funasr"),
        used_engine="funasr",
        effective_model=request.funasr_model,
        requested_language=request.requested_language,
        detected_language=detected_language,
        tag_source="sensevoice" if _is_sensevoice_model(request.funasr_model) else "",
        language_route=language_route,
        route_reason=route_reason,
        vad_strategy=request.vad_strategy,
    )


def _decode_whisper_route(
    request: _AsrDecodeRequest,
    *,
    language_route: str,
    route_reason: str,
) -> _AsrDecodeResult:
    model_chunks, metadata = _whisper(
        str(request.audio_path),
        request.model_name,
        request.device,
        request.model_dir,
        request.requested_language,
        local_files_only=request.model_local_files_only,
    )
    detected_language = str(metadata.get("detected_language") or "")
    return _AsrDecodeResult(
        raw_items=raw_items_from_chunks(model_chunks, source="whisper"),
        used_engine="whisper",
        effective_model=request.model_name,
        requested_language=request.requested_language,
        detected_language=detected_language,
        task=str(metadata.get("task") or "transcribe"),
        language_route=language_route or f"whisper:{detected_language or request.requested_language}",
        route_reason=route_reason or (
            "explicit language" if request.requested_language != "auto" else "explicit whisper engine"
        ),
        vad_strategy=request.vad_strategy,
    )


def _decode_audio(request: _AsrDecodeRequest) -> _AsrDecodeResult:
    route_engine, detected, decode_language, language_route, route_reason = _resolve_decode_route(request)
    normalized_engine = request.engine.replace("_", "-").casefold()
    if route_engine == "funasr":
        return _decode_funasr_route(
            request,
            normalized_engine=normalized_engine,
            decode_language=decode_language,
            detected_language=detected,
            language_route=language_route,
            route_reason=route_reason,
        )
    if route_engine == "faster-whisper":
        return _decode_faster_whisper_route(
            request,
            decode_language=decode_language,
            detected_language=detected,
            language_route=language_route,
            route_reason=route_reason,
        )
    return _decode_whisper_route(request, language_route=language_route, route_reason=route_reason)


def _no_audio_result(
    output_path: str,
    semantic_target: Path | None,
    *,
    model_name: str,
    requested_language: str,
    vad_strategy: str,
) -> dict:
    _save_asr_npz(output_path, [], np.empty((0, 0), dtype=np.float16), np.empty((0,), dtype=np.int32))
    if semantic_target is not None:
        semantic_target.unlink(missing_ok=True)
    return {
        "chunks": 0,
        "raw_chunks": 0,
        "raw_items": 0,
        "retrieval_chunks": 0,
        "engine": "no_audio",
        "model": model_name,
        "language": requested_language,
        "task": "transcribe",
        "requested_language": requested_language,
        "detected_language": "",
        "language_route": "no_audio",
        "route_reason": "no audio stream found",
        "vad_strategy": vad_strategy,
        "raw_parser_stats": {},
        "chunk_builder_stats": _empty_chunk_builder_stats(),
        "text_profile": asr_text_profile([]),
        "schema_version": 3,
        "decode_status": "empty",
        "semantic_status": "empty",
        "semantic_chunks": 0,
        "warning": "no audio stream found",
    }


def _build_asr_semantic_result(
    chunks: list[dict],
    *,
    enabled: bool,
    target: Path | None,
    model_name: str,
    model_dir: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> dict:
    if not enabled:
        if target is not None:
            target.unlink(missing_ok=True)
        return {
            "embeddings": np.empty((0, 0), dtype=np.float16),
            "embedding_chunk_indices": np.empty((0,), dtype=np.int32),
            "semantic_chunks": 0,
            "semantic_status": "disabled",
        }
    try:
        return build_text_semantic_arrays(
            chunks=chunks,
            model_name=model_name,
            model_dir=model_dir,
            device=resolve_text_embedding_device(device, cuda_enabled=False),
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if target is not None:
            target.unlink(missing_ok=True)
        return {
            "embeddings": np.empty((0, 0), dtype=np.float16),
            "embedding_chunk_indices": np.empty((0,), dtype=np.int32),
            "semantic_chunks": 0,
            "semantic_model": model_name,
            "semantic_status": "unavailable",
            "semantic_error": str(exc),
        }


def _asr_result_payload(
    decode: _AsrDecodeResult,
    chunks: list[dict],
    chunk_builder_stats: dict[str, int],
    semantic_result: dict,
) -> dict:
    result = {
        "chunks": len(chunks),
        "raw_chunks": len(decode.raw_items),
        "raw_items": len(decode.raw_items),
        "retrieval_chunks": len(chunks),
        "engine": decode.used_engine,
        "model": decode.effective_model,
        "language": decode.detected_language or decode.requested_language,
        "task": decode.task,
        "requested_language": decode.requested_language,
        "detected_language": decode.detected_language,
        "language_route": decode.language_route or f"{decode.used_engine}:{decode.detected_language or decode.requested_language}",
        "route_reason": decode.route_reason or (
            "explicit language" if decode.requested_language != "auto" else "default auto language"
        ),
        "vad_strategy": decode.vad_strategy,
        "raw_parser_stats": decode.raw_parser_stats,
        "chunk_builder_stats": chunk_builder_stats,
        "text_profile": asr_text_profile(chunk.get("text", "") for chunk in chunks),
        "tag_source": decode.tag_source,
        "schema_version": 3,
        "decode_status": "complete" if chunks else "empty",
    }
    result.update({
        key: value
        for key, value in semantic_result.items()
        if key not in {"embeddings", "embedding_chunk_indices"}
    })
    return result


def build_asr_index(
    video_path: str,
    output_path: str,
    working_dir: str,
    engine: str,
    model_name: str,
    device: str,
    model_dir: str,
    language: str = "auto",
    sidecar_path: str | None = None,
    funasr_model: str = "paraformer-zh",
    funasr_model_dir: str | None = None,
    faster_whisper_model_dir: str | None = None,
    model_local_files_only: bool = True,
    semantic_enabled: bool = True,
    semantic_output_path: str | None = None,
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    semantic_device: str = "cpu",
    semantic_model_dir: str | None = None,
    semantic_batch_size: int = 32,
    semantic_local_files_only: bool = True,
    debug_artifacts_enabled: bool = False,
    save_raw_transcript: bool = False,
    debug_output_dir: str | None = None,
    vad_strategy: str = "funasr_fsmn",
) -> dict:
    semantic_target = Path(semantic_output_path) if semantic_output_path else None
    requested_language = language or "auto"
    if sidecar_path:
        decode = _AsrDecodeResult(
            raw_items=raw_items_from_chunks(load_sidecar(sidecar_path), source="sidecar"),
            used_engine="sidecar",
            effective_model=model_name,
            requested_language=requested_language,
            vad_strategy=vad_strategy,
        )
    else:
        try:
            audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        except subprocess.CalledProcessError:
            return _no_audio_result(
                output_path,
                semantic_target,
                model_name=model_name,
                requested_language=requested_language,
                vad_strategy=vad_strategy,
            )
        decode = _decode_audio(_AsrDecodeRequest(
            audio_path=Path(audio_path),
            engine=engine,
            model_name=model_name,
            device=device,
            model_dir=model_dir,
            requested_language=requested_language,
            funasr_model=funasr_model,
            funasr_model_dir=funasr_model_dir,
            faster_whisper_model_dir=faster_whisper_model_dir,
            model_local_files_only=model_local_files_only,
            working_dir=working_dir,
            vad_strategy=vad_strategy,
        ))

    retrieval_chunks, chunk_builder_stats = build_retrieval_chunks(
        decode.raw_items,
        config=RetrievalChunkConfig(),
    )
    chunks = [chunk.to_search_dict() for chunk in retrieval_chunks]
    semantic_result = _build_asr_semantic_result(
        chunks,
        enabled=semantic_enabled,
        target=semantic_target,
        model_name=semantic_model,
        model_dir=semantic_model_dir or str(Path(model_dir).parent / "text-embeddings"),
        device=semantic_device,
        batch_size=semantic_batch_size,
        local_files_only=semantic_local_files_only,
    )

    debug_dir = Path(debug_output_dir) if debug_output_dir else Path(output_path).parent / "debug"
    write_asr_debug_artifacts(
        debug_dir=debug_dir,
        enabled=debug_artifacts_enabled,
        save_raw_transcript=save_raw_transcript,
        raw_items=decode.raw_items,
        retrieval_chunks=retrieval_chunks,
        repair_stats={**chunk_builder_stats, **decode.raw_parser_stats},
    )

    _save_asr_npz(
        output_path,
        chunks,
        np.asarray(semantic_result["embeddings"], dtype=np.float16),
        np.asarray(semantic_result["embedding_chunk_indices"], dtype=np.int32),
    )
    return _asr_result_payload(decode, chunks, chunk_builder_stats, semantic_result)


def _save_asr_npz(
    output_path: str | Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_chunk_indices: np.ndarray,
) -> None:
    def utf8_bytes_array(values: list[str]) -> np.ndarray:
        width = max((len(value.encode("utf-8")) for value in values), default=1)
        return np.asarray(values, dtype=f"S{width}")

    def chunk_time_ms(chunk: dict, ms_key: str, seconds_key: str, legacy_key: str) -> int:
        if ms_key in chunk:
            return int(chunk[ms_key])
        return int(round(float(chunk.get(seconds_key, chunk.get(legacy_key, 0))) * 1000))

    chunk_times_ms = np.asarray([
        [
            chunk_time_ms(chunk, "start_ms", "start_time", "start"),
            chunk_time_ms(chunk, "end_ms", "end_time", "end"),
        ]
        for chunk in chunks
    ], dtype=np.int32).reshape((-1, 2))
    texts = np.asarray([str(chunk.get("text", "")).strip() for chunk in chunks], dtype="U")
    chunk_emotions = utf8_bytes_array([str(chunk.get("emotion", "")).strip() for chunk in chunks])
    chunk_audio_events = utf8_bytes_array([str(chunk.get("audio_event", "")).strip() for chunk in chunks])
    atomic_save_npz(
        output_path,
        chunk_times_ms=chunk_times_ms,
        texts=texts,
        chunk_emotions=chunk_emotions,
        chunk_audio_events=chunk_audio_events,
        embeddings=np.asarray(embeddings, dtype=np.float16),
        embedding_chunk_indices=np.asarray(embedding_chunk_indices, dtype=np.int32),
    )
