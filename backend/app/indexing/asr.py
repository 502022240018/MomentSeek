from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.indexing.asr_postprocess import postprocess_asr_chunks, strategy_config
from app.indexing.asr_text import asr_text_profile
from app.indexing.common import atomic_save_npz
from app.indexing.text_semantic import build_text_semantic_arrays, resolve_text_embedding_device
from app.media import extract_audio, parse_timecode


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


def _funasr(audio_path: str, model_name: str, device: str) -> list[dict]:
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before model load
    from funasr import AutoModel

    model = AutoModel(
        model=model_name,
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=device,
        disable_update=True,
    )
    result = model.generate(input=audio_path, batch_size_s=300, sentence_timestamp=True)
    if not result:
        return []
    item = result[0]
    sentence_info = item.get("sentence_info") or []
    chunks = []
    for sentence in sentence_info:
        chunks.append({
            "start_time": float(sentence.get("start", 0)) / 1000,
            "end_time": float(sentence.get("end", 0)) / 1000,
            "text": str(sentence.get("text", "")).strip(),
        })
    if chunks:
        return [chunk for chunk in chunks if chunk["text"]]
    text = str(item.get("text", "")).strip()
    timestamps = item.get("timestamp") or []
    if text and timestamps:
        return [{"start_time": timestamps[0][0] / 1000, "end_time": timestamps[-1][1] / 1000, "text": text}]
    return [{"start_time": 0, "end_time": 0, "text": text}] if text else []


def _whisper(
    audio_path: str,
    model_name: str,
    device: str,
    model_dir: str,
    language: str = "auto",
) -> tuple[list[dict], dict]:
    import whisper

    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before .to(device)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, device=device, download_root=model_dir)
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
    semantic_enabled: bool = True,
    semantic_output_path: str | None = None,
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    semantic_device: str = "cpu",
    semantic_model_dir: str | None = None,
    semantic_batch_size: int = 32,
    semantic_local_files_only: bool = True,
) -> dict:
    effective_model = model_name
    semantic_result: dict | None = None
    semantic_target = Path(semantic_output_path) if semantic_output_path else None
    requested_language = language or "auto"
    detected_language = ""
    task = "transcribe"

    if sidecar_path:
        raw_chunks = load_sidecar(sidecar_path)
        used_engine = "sidecar"
    else:
        try:
            audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        except subprocess.CalledProcessError:
            chunks: list[dict] = []
            used_engine = "no_audio"
            _save_asr_npz(output_path, chunks, np.empty((0, 0), dtype=np.float16), np.empty((0,), dtype=np.int32))
            if semantic_target is not None:
                semantic_target.unlink(missing_ok=True)
            return {
                "chunks": 0,
                "raw_chunks": 0,
                "engine": used_engine,
                "model": effective_model,
                "language": requested_language,
                "task": task,
                "requested_language": requested_language,
                "detected_language": detected_language,
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
        prefer_funasr = engine == "funasr" or (engine == "auto" and requested_language == "zh")
        if prefer_funasr:
            try:
                raw_chunks = _funasr(str(audio_path), funasr_model, device)
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
                raw_chunks, whisper_metadata = _whisper(
                    str(audio_path),
                    model_name,
                    device,
                    model_dir,
                    requested_language,
                )
                used_engine = "whisper"
                task = str(whisper_metadata.get("task") or task)
                detected_language = str(whisper_metadata.get("detected_language") or "")
        else:
            raw_chunks, whisper_metadata = _whisper(str(audio_path), model_name, device, model_dir, requested_language)
            used_engine = "whisper"
            task = str(whisper_metadata.get("task") or task)
            detected_language = str(whisper_metadata.get("detected_language") or "")

    segment_ids = _fixed_bucket_segment_ids(raw_chunks, bucket_ms=5000)
    chunks, postprocess_stats = postprocess_asr_chunks(
        raw_chunks,
        segment_ids=segment_ids,
        config=strategy_config(DEFAULT_ASR_POSTPROCESS_STRATEGY),
    )

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
        "raw_chunks": len(raw_chunks),
        "engine": used_engine,
        "model": effective_model,
        "language": detected_language or requested_language,
        "task": task,
        "requested_language": requested_language,
        "detected_language": detected_language,
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
