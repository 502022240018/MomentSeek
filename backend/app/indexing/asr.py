from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.indexing.common import atomic_save_json
from app.indexing.text_semantic import build_text_semantic_index, resolve_text_embedding_device
from app.media import extract_audio, parse_timecode


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
            for item in items if str(item.get("text", "")).strip()
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


def _whisper(audio_path: str, model_name: str, device: str, model_dir: str, language: str = "auto") -> list[dict]:
    import whisper

    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401  # register Ascend backend before .to(device)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, device=device, download_root=model_dir)
    options = {"fp16": device != "cpu"}
    if language and language != "auto":
        options["language"] = language
    if language == "zh":
        options["initial_prompt"] = "以下是普通话简体中文转写。"
    result = model.transcribe(load_wav_mono(audio_path), **options)
    return [
        {"start_time": float(item["start"]), "end_time": float(item["end"]), "text": item["text"].strip()}
        for item in result.get("segments", []) if item.get("text", "").strip()
    ]


def load_wav_mono(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())
    if sample_width != 2:
        raise ValueError(f"仅支持 16-bit PCM wav，当前 sample_width={sample_width}")
    data = np.frombuffer(frames, np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32) / 32768.0


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
    semantic_target = Path(semantic_output_path) if semantic_output_path else Path(output_path).with_name("asr_semantic.npz")
    if sidecar_path:
        chunks = load_sidecar(sidecar_path)
        used_engine = "sidecar"
    else:
        try:
            audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        except subprocess.CalledProcessError:
            chunks = []
            used_engine = "no_audio"
            atomic_save_json(output_path, {"engine": used_engine, "model": model_name, "language": language, "chunks": chunks})
            semantic_target.unlink(missing_ok=True)
            return {"chunks": 0, "engine": used_engine, "warning": "no audio stream found"}
        # Chinese transcription quality: prefer FunASR/Paraformer when available
        # (far better on Mandarin than small Whisper); otherwise fall back to the
        # requested Whisper model — never silently downgrade to tiny.
        if engine in {"auto", "funasr"}:
            try:
                chunks = _funasr(str(audio_path), funasr_model, device)
                used_engine = "funasr"
                effective_model = funasr_model
            except Exception:
                if engine == "funasr":
                    raise
                chunks = _whisper(str(audio_path), model_name, device, model_dir, language)
                used_engine = "whisper"
        else:
            chunks = _whisper(str(audio_path), model_name, device, model_dir, language)
            used_engine = "whisper"
    atomic_save_json(output_path, {"engine": used_engine, "model": effective_model, "language": language, "chunks": chunks})

    if not semantic_enabled:
        semantic_target.unlink(missing_ok=True)
    else:
        try:
            resolved_device = resolve_text_embedding_device(semantic_device, cuda_enabled=False)
            semantic_result = build_text_semantic_index(
                chunks=chunks,
                output_path=semantic_target,
                model_name=semantic_model,
                model_dir=semantic_model_dir or str(Path(model_dir).parent / "text-embeddings"),
                device=resolved_device,
                batch_size=semantic_batch_size,
                local_files_only=semantic_local_files_only,
            )
        except Exception as exc:
            # Keep ASR itself usable even if the optional semantic model is not
            # installed or unavailable on the current server. Search will fall
            # back to lexical matching when asr_semantic.npz is absent.
            semantic_target.unlink(missing_ok=True)
            semantic_result = {
                "semantic_chunks": 0,
                "semantic_model": semantic_model,
                "semantic_status": "unavailable",
                "semantic_error": str(exc),
            }

    result = {"chunks": len(chunks), "engine": used_engine, "model": effective_model, "language": language}
    if semantic_result is not None:
        result.update(semantic_result)
    return result
