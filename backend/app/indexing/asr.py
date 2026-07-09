from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.indexing.asr_debug import write_asr_debug_artifacts
from app.indexing.asr_retrieval_chunks import RetrievalChunkConfig, build_retrieval_chunks
from app.indexing.asr_text import asr_text_profile
from app.indexing.asr_transcript_parser import parse_funasr_raw_transcript, raw_items_from_chunks
from app.indexing.common import atomic_save_npz
from app.indexing.text_semantic import build_text_semantic_arrays, resolve_text_embedding_device
from app.media import extract_audio, parse_timecode
from app.model_sources import (
    offline_env,
    resolve_faster_whisper_model_source,
    resolve_modelscope_model_source,
)


DEFAULT_ASR_POSTPROCESS_STRATEGY = "bucket_bonus"


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


def _parse_funasr_chunks(result: object, *, is_sensevoice: bool) -> list[dict]:
    raw_items, _diagnostics = parse_funasr_raw_transcript(result, is_sensevoice=is_sensevoice)
    return [item.to_dict() for item in raw_items]


def _funasr(
    audio_path: str,
    model_name: str,
    device: str,
    model_root: str | Path | None = None,
    local_files_only: bool = True,
    language: str = "auto",
) -> list[dict]:
    model_source = resolve_modelscope_model_source(model_root, model_name, local_files_only=local_files_only)
    vad_source = resolve_modelscope_model_source(model_root, "fsmn-vad", local_files_only=local_files_only)
    is_sensevoice = _is_sensevoice_model(model_name, model_source)
    punc_source = None if is_sensevoice else resolve_modelscope_model_source(
        model_root, "ct-punc", local_files_only=local_files_only
    )
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before model load
    from funasr import AutoModel

    model_kwargs = {
        "model": model_source,
        "vad_model": vad_source,
        "device": device,
        "disable_update": True,
    }
    if is_sensevoice:
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
    transcribe_language = None if not language or language == "auto" else language
    segments_iter, info = model.transcribe(
        audio_path,
        language=transcribe_language,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
        beam_size=5,
    )
    chunks = [
        {"start_time": float(item.start), "end_time": float(item.end), "text": str(item.text).strip()}
        for item in segments_iter
        if str(item.text).strip()
    ]
    return chunks, {
        "task": "transcribe",
        "requested_language": language or "auto",
        "detected_language": str(getattr(info, "language", "") or ""),
        "compute_type": compute_type,
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


def _fixed_bucket_segment_ids(chunks: list[dict], bucket_ms: int = 5000) -> list[int]:
    ids: list[int] = []
    for chunk in chunks:
        if "start_ms" in chunk:
            start_ms = int(chunk["start_ms"])
        else:
            start_ms = int(round(float(chunk.get("start_time", chunk.get("start", 0))) * 1000))
        ids.append(max(0, start_ms // bucket_ms))
    return ids


def _empty_postprocess_stats() -> dict[str, int]:
    return {
        "raw_chunks": 0,
        "normalized_chunks": 0,
        "processed_chunks": 0,
        "dropped_empty_chunks": 0,
        "merged_chunks": 0,
        "cross_segment_merges": 0,
        "semantic_ineligible_chunks": 0,
        "long_low_info_chunks": 0,
    }


def _empty_chunk_builder_stats() -> dict[str, int]:
    return {
        "raw_items": 0,
        "normalized_items": 0,
        "retrieval_chunks": 0,
        "dropped_empty_items": 0,
        "merged_items": 0,
        "word_boundary_repairs": 0,
        "fake_gap_repairs": 0,
        "long_chunks": 0,
        "semantic_ineligible_chunks": 0,
    }


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
    effective_model = model_name
    semantic_result: dict | None = None
    semantic_target = Path(semantic_output_path) if semantic_output_path else None
    requested_language = language or "auto"
    detected_language = ""
    task = "transcribe"
    raw_parser_stats: dict[str, int] = {}

    if sidecar_path:
        raw_items = raw_items_from_chunks(load_sidecar(sidecar_path), source="sidecar")
        used_engine = "sidecar"
    else:
        try:
            audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        except subprocess.CalledProcessError:
            chunks: list[dict] = []
            used_engine = "no_audio"
            empty_chunk_builder_stats = _empty_chunk_builder_stats()
            _save_asr_npz(output_path, chunks, np.empty((0, 0), dtype=np.float16), np.empty((0,), dtype=np.int32))
            if semantic_target is not None:
                semantic_target.unlink(missing_ok=True)
            return {
                "chunks": 0,
                "raw_chunks": 0,
                "raw_items": 0,
                "retrieval_chunks": 0,
                "engine": used_engine,
                "model": effective_model,
                "language": requested_language,
                "task": task,
                "requested_language": requested_language,
                "detected_language": detected_language,
                "language_route": "no_audio",
                "route_reason": "no audio stream found",
                "vad_strategy": vad_strategy,
                "raw_parser_stats": {},
                "chunk_builder_stats": empty_chunk_builder_stats,
                "postprocess_strategy": DEFAULT_ASR_POSTPROCESS_STRATEGY,
                "postprocess_stats": _empty_postprocess_stats(),
                "text_profile": asr_text_profile([]),
                "schema_version": 3,
                "decode_status": "empty",
                "semantic_status": "empty",
                "semantic_chunks": 0,
                "warning": "no audio stream found",
            }

        # Chinese transcription quality: prefer FunASR/Paraformer only when the
        # request explicitly asks for Chinese, or when FunASR is explicitly chosen.
        # True language auto needs Whisper's language detection.
        normalized_engine = engine.replace("_", "-").casefold()
        prefer_funasr = normalized_engine == "funasr" or (normalized_engine == "auto" and requested_language == "zh")
        if prefer_funasr:
            try:
                model_chunks = _funasr(
                    str(audio_path),
                    funasr_model,
                    device,
                    model_root=funasr_model_dir or str(Path(model_dir).parent / "funasr"),
                    local_files_only=model_local_files_only,
                    language=requested_language,
                )
                raw_items = raw_items_from_chunks(model_chunks, source="funasr")
                used_engine = "funasr"
                effective_model = funasr_model
                if (
                    requested_language == "zh"
                    or "zh" in funasr_model.casefold()
                    or "paraformer" in funasr_model.casefold()
                ):
                    detected_language = "zh"
            except Exception:
                if engine == "funasr":
                    raise
                model_chunks, whisper_metadata = _whisper(
                    str(audio_path),
                    model_name,
                    device,
                    model_dir,
                    requested_language,
                    local_files_only=model_local_files_only,
                )
                raw_items = raw_items_from_chunks(model_chunks, source="whisper")
                used_engine = "whisper"
                task = str(whisper_metadata.get("task") or task)
                detected_language = str(whisper_metadata.get("detected_language") or "")
        elif normalized_engine == "faster-whisper":
            model_chunks, whisper_metadata = _faster_whisper(
                str(audio_path),
                model_name,
                device,
                faster_whisper_model_dir or str(Path(model_dir).parent / "faster-whisper"),
                requested_language,
                local_files_only=model_local_files_only,
            )
            raw_items = raw_items_from_chunks(model_chunks, source="faster_whisper")
            used_engine = "faster-whisper"
            task = str(whisper_metadata.get("task") or task)
            detected_language = str(whisper_metadata.get("detected_language") or "")
        else:
            model_chunks, whisper_metadata = _whisper(
                str(audio_path),
                model_name,
                device,
                model_dir,
                requested_language,
                local_files_only=model_local_files_only,
            )
            raw_items = raw_items_from_chunks(model_chunks, source="whisper")
            used_engine = "whisper"
            task = str(whisper_metadata.get("task") or task)
            detected_language = str(whisper_metadata.get("detected_language") or "")

    retrieval_chunks, chunk_builder_stats = build_retrieval_chunks(
        raw_items,
        config=RetrievalChunkConfig(),
    )
    chunks = [chunk.to_search_dict() for chunk in retrieval_chunks]
    postprocess_stats = {
        "raw_chunks": len(raw_items),
        "normalized_chunks": chunk_builder_stats["normalized_items"],
        "processed_chunks": len(chunks),
        "dropped_empty_chunks": chunk_builder_stats["dropped_empty_items"],
        "merged_chunks": chunk_builder_stats["merged_items"],
        "cross_segment_merges": 0,
        "semantic_ineligible_chunks": chunk_builder_stats["semantic_ineligible_chunks"],
        "long_low_info_chunks": chunk_builder_stats["long_chunks"],
    }

    if not semantic_enabled:
        if semantic_target is not None:
            semantic_target.unlink(missing_ok=True)
        semantic_result = {
            "embeddings": np.empty((0, 0), dtype=np.float16),
            "embedding_chunk_indices": np.empty((0,), dtype=np.int32),
            "semantic_chunks": 0,
            "semantic_status": "disabled",
        }
    else:
        try:
            resolved_device = resolve_text_embedding_device(semantic_device, cuda_enabled=False)
            semantic_result = build_text_semantic_arrays(
                chunks=chunks,
                model_name=semantic_model,
                model_dir=semantic_model_dir or str(Path(model_dir).parent / "text-embeddings"),
                device=resolved_device,
                batch_size=semantic_batch_size,
                local_files_only=semantic_local_files_only,
            )
        except Exception as exc:
            # Keep ASR itself usable even if the optional semantic model is not
            # installed or unavailable. Search falls back to lexical matching.
            if semantic_target is not None:
                semantic_target.unlink(missing_ok=True)
            semantic_result = {
                "embeddings": np.empty((0, 0), dtype=np.float16),
                "embedding_chunk_indices": np.empty((0,), dtype=np.int32),
                "semantic_chunks": 0,
                "semantic_model": semantic_model,
                "semantic_status": "unavailable",
                "semantic_error": str(exc),
            }

    debug_dir = Path(debug_output_dir) if debug_output_dir else Path(output_path).parent / "debug"
    write_asr_debug_artifacts(
        debug_dir=debug_dir,
        enabled=debug_artifacts_enabled,
        save_raw_transcript=save_raw_transcript,
        raw_items=raw_items,
        retrieval_chunks=retrieval_chunks,
        repair_stats={**chunk_builder_stats, **raw_parser_stats},
    )

    _save_asr_npz(
        output_path,
        chunks,
        np.asarray(semantic_result["embeddings"], dtype=np.float16)
        if semantic_result is not None
        else np.empty((0, 0), dtype=np.float16),
        np.asarray(semantic_result["embedding_chunk_indices"], dtype=np.int32)
        if semantic_result is not None
        else np.empty((0,), dtype=np.int32),
    )
    result = {
        "chunks": len(chunks),
        "raw_chunks": len(raw_items),
        "raw_items": len(raw_items),
        "retrieval_chunks": len(chunks),
        "engine": used_engine,
        "model": effective_model,
        "language": detected_language or requested_language,
        "task": task,
        "requested_language": requested_language,
        "detected_language": detected_language,
        "language_route": f"{used_engine}:{detected_language or requested_language}",
        "route_reason": "explicit language" if requested_language != "auto" else "default auto language",
        "vad_strategy": vad_strategy,
        "raw_parser_stats": raw_parser_stats,
        "chunk_builder_stats": chunk_builder_stats,
        "postprocess_strategy": DEFAULT_ASR_POSTPROCESS_STRATEGY,
        "postprocess_stats": postprocess_stats,
        "text_profile": asr_text_profile(chunk.get("text", "") for chunk in chunks),
        "schema_version": 3,
        "decode_status": "complete" if chunks else "empty",
    }
    if semantic_result is not None:
        result.update({
            key: value
            for key, value in semantic_result.items()
            if key not in {"embeddings", "embedding_chunk_indices"}
        })
    return result


def _save_asr_npz(
    output_path: str | Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_chunk_indices: np.ndarray,
) -> None:
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
    atomic_save_npz(
        output_path,
        chunk_times_ms=chunk_times_ms,
        texts=texts,
        embeddings=np.asarray(embeddings, dtype=np.float16),
        embedding_chunk_indices=np.asarray(embedding_chunk_indices, dtype=np.int32),
    )
