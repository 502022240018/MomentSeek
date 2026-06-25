from __future__ import annotations

import json
import re
from pathlib import Path

from app.indexing.common import atomic_save_json
from app.media import extract_audio, parse_timecode


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


def _whisper(audio_path: str, model_name: str, device: str, model_dir: str) -> list[dict]:
    import whisper

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, device=device, download_root=model_dir)
    result = model.transcribe(audio_path, fp16=device != "cpu")
    return [
        {"start_time": float(item["start"]), "end_time": float(item["end"]), "text": item["text"].strip()}
        for item in result.get("segments", []) if item.get("text", "").strip()
    ]


def build_asr_index(
    video_path: str,
    output_path: str,
    working_dir: str,
    engine: str,
    model_name: str,
    device: str,
    model_dir: str,
    sidecar_path: str | None = None,
) -> dict:
    if sidecar_path:
        chunks = load_sidecar(sidecar_path)
        used_engine = "sidecar"
    else:
        audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        if engine in {"auto", "funasr"}:
            try:
                chunks = _funasr(str(audio_path), model_name, device)
                used_engine = "funasr"
            except Exception:
                if engine == "funasr":
                    raise
                chunks = _whisper(str(audio_path), "small", "cpu", model_dir)
                used_engine = "whisper"
        else:
            chunks = _whisper(str(audio_path), model_name, device, model_dir)
            used_engine = "whisper"
    atomic_save_json(output_path, {"engine": used_engine, "model": model_name, "chunks": chunks})
    return {"chunks": len(chunks), "engine": used_engine}
