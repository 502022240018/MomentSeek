from __future__ import annotations

from pathlib import Path
from typing import Any

from app.indexing.manifest import update_channel_manifest
from app.settings import Settings


def visual_embedding_space(model_key: str) -> str:
    if model_key.startswith("siglip2"):
        return "siglip2-image-text"
    if model_key.startswith("chinese-clip"):
        return "chinese-clip-image-text"
    if model_key.startswith("openclip"):
        return "openclip-image-text"
    return "visual-image-text"


def text_embedding_space(model_name: str) -> str:
    if "MiniLM" in model_name or "minilm" in model_name.casefold():
        return "minilm-text-semantic"
    return "text-semantic"


def channel_manifest(
    stage: str,
    *,
    result: dict[str, Any],
    options: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    if stage == "visual":
        model_key = str(result.get("visual_model") or options.get("visual_model") or settings.visual_model)
        return {
            "file": "visual.npz",
            "model_key": model_key,
            "embedding_space": visual_embedding_space(model_key),
            "sample_fps": float(options.get("visual_sample_fps", settings.visual_sample_fps)),
            "decode_status": str(result.get("decode_status") or "unknown"),
        }
    if stage == "face":
        return {
            "file": "face.npz",
            "model_key": settings.face_model,
            "embedding_space": "arcface-identity",
            "sample_fps": float(options.get("face_sample_fps", settings.face_sample_fps)),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "provider": str(result.get("provider") or settings.face_provider),
        }
    if stage == "asr":
        semantic_model = settings.asr_semantic_model
        return {
            "file": "asr.npz",
            "engine": str(result.get("engine") or settings.asr_engine),
            "model_key": str(result.get("model") or options.get("asr_model") or settings.asr_model),
            "language": str(result.get("language") or options.get("asr_language") or settings.asr_language),
            "semantic_model_key": semantic_model,
            "embedding_space": text_embedding_space(semantic_model),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "semantic_status": str(result.get("semantic_status") or ("complete" if settings.asr_semantic_enabled else "disabled")),
        }
    if stage == "ocr":
        semantic_model = settings.asr_semantic_model
        return {
            "file": "ocr.npz",
            "engine": str(result.get("engine") or settings.ocr_engine),
            "model_key": str(result.get("ocr_version") or settings.ocr_version),
            "semantic_model_key": semantic_model,
            "embedding_space": text_embedding_space(semantic_model),
            "sample_fps": float(options.get("ocr_sample_fps", settings.ocr_sample_fps)),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "semantic_status": str(result.get("semantic_status") or ("complete" if settings.ocr_semantic_enabled else "disabled")),
        }
    raise ValueError(f"未知索引阶段: {stage}")


def write_stage_manifest(
    stage: str,
    *,
    index_dir: str | Path,
    video: dict[str, Any],
    options: dict[str, Any],
    settings: Settings,
    result: dict[str, Any],
) -> dict[str, Any]:
    segment_seconds = float(options.get("visual_segment_seconds", settings.visual_segment_seconds))
    return update_channel_manifest(
        index_dir,
        video_id=str(video["id"]),
        duration_seconds=float(video.get("duration") or 0),
        segment_seconds=segment_seconds,
        channel=stage,
        channel_manifest=channel_manifest(stage, result=result, options=options, settings=settings),
    )
