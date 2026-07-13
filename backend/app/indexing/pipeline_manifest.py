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
        payload = {
            "file": "visual.npz",
            "model_key": model_key,
            "embedding_space": visual_embedding_space(model_key),
            "sample_fps": float(options.get("visual_sample_fps", settings.visual_sample_fps)),
            "decode_status": str(result.get("decode_status") or "unknown"),
        }
        strategy = str(
            result.get("segment_strategy")
            or options.get("visual_segment_strategy")
            or getattr(settings, "visual_segment_strategy", "fixed")
            or "fixed"
        )
        if strategy != "fixed":
            min_segment_seconds = float(
                options.get("visual_min_segment_seconds", getattr(settings, "visual_min_segment_seconds", 0.8))
            )
            max_segment_seconds = float(
                options.get("visual_max_segment_seconds", getattr(settings, "visual_max_segment_seconds", 8.0))
            )
            shot_threshold = float(
                options.get("visual_shot_threshold", getattr(settings, "visual_shot_threshold", 0.20))
            )
            shot_detector = str(
                result.get("shot_detector")
                or options.get("visual_shot_detector")
                or getattr(settings, "visual_shot_detector", "simple")
                or "simple"
            )
            payload.update({
                "segment_strategy": strategy,
                "segment_times": str(result.get("segment_times") or "explicit"),
                "min_segment_ms": max(1, int(round(min_segment_seconds * 1000))),
                "max_segment_ms": max(1, int(round(max_segment_seconds * 1000))),
                "shot_detector": shot_detector,
                "shot_threshold": shot_threshold,
            })
        return payload
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
        requested_language = str(
            result.get("requested_language")
            or options.get("asr_language")
            or settings.asr_language
        )
        detected_language = str(result.get("detected_language") or "")
        payload = {
            "file": "asr.npz",
            "engine": str(result.get("engine") or settings.asr_engine),
            "model_key": str(result.get("model") or options.get("asr_model") or settings.asr_model),
            "language": str(result.get("language") or detected_language or requested_language),
            "task": str(result.get("task") or "transcribe"),
            "requested_language": requested_language,
            "detected_language": detected_language,
            "semantic_model_key": semantic_model,
            "embedding_space": text_embedding_space(semantic_model),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "semantic_status": str(result.get("semantic_status") or ("complete" if settings.asr_semantic_enabled else "disabled")),
            "language_route": result.get("language_route"),
            "route_reason": result.get("route_reason"),
            "vad_strategy": result.get("vad_strategy"),
            "raw_items": result.get("raw_items"),
            "retrieval_chunks": result.get("retrieval_chunks"),
            "chunk_builder_stats": result.get("chunk_builder_stats") or {},
            "text_profile": result.get("text_profile") or {},
        }
        if result.get("tag_source"):
            payload["tag_source"] = str(result["tag_source"])
        return payload
    if stage == "ocr":
        semantic_model = settings.asr_semantic_model
        return {
            "file": "ocr.npz",
            "schema_version": int(result.get("schema_version") or 3),
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
