from __future__ import annotations

from typing import Any

from app.core.settings import Settings


def visual_embedding_space(model_key: str) -> str:
    if model_key.startswith("siglip2"):
        return "siglip2-image-text"
    if model_key.startswith("chinese-clip"):
        return "chinese-clip-image-text"
    if model_key.startswith("openclip"):
        return "openclip-image-text"
    return "visual-image-text"


def text_embedding_space(model_name: str) -> str:
    if "minilm" in model_name.casefold():
        return "minilm-text-semantic"
    return "text-semantic"


def channel_metadata(
    stage: str,
    *,
    result: dict[str, Any],
    options: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Build compact Catalog metadata for one verified Milvus publication."""
    if stage == "visual":
        model_key = str(
            result.get("visual_model")
            or options.get("visual_model")
            or settings.visual_model
        )
        strategy = str(
            result.get("segment_strategy")
            or options.get("visual_segment_strategy")
            or settings.visual_segment_strategy
            or "fixed"
        )
        payload: dict[str, Any] = {
            "model_key": model_key,
            "embedding_space": visual_embedding_space(model_key),
            "sample_fps": float(
                options.get("visual_sample_fps", settings.visual_sample_fps)
            ),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "segment_strategy": strategy,
            "segment_times": "explicit",
        }
        if strategy != "fixed":
            payload.update({
                "min_segment_ms": max(1, int(round(float(options.get(
                    "visual_min_segment_seconds", settings.visual_min_segment_seconds,
                )) * 1000))),
                "max_segment_ms": max(1, int(round(float(options.get(
                    "visual_max_segment_seconds", settings.visual_max_segment_seconds,
                )) * 1000))),
                "shot_detector": str(
                    result.get("shot_detector")
                    or options.get("visual_shot_detector")
                    or settings.visual_shot_detector
                    or "simple"
                ),
                "shot_threshold": float(options.get(
                    "visual_shot_threshold", settings.visual_shot_threshold,
                )),
            })
        return payload
    if stage == "face":
        return {
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
            "engine": str(result.get("engine") or settings.asr_engine),
            "model_key": str(
                result.get("model") or options.get("asr_model") or settings.asr_model
            ),
            "language": str(
                result.get("language") or detected_language or requested_language
            ),
            "task": str(result.get("task") or "transcribe"),
            "requested_language": requested_language,
            "detected_language": detected_language,
            "semantic_model_key": semantic_model,
            "embedding_space": text_embedding_space(semantic_model),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "semantic_status": str(
                result.get("semantic_status")
                or ("complete" if settings.asr_semantic_enabled else "disabled")
            ),
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
    if stage == "speaker":
        model_key = str(
            result.get("embedding_model")
            or "iic/speech_campplus_sv_zh_en_16k-common_advanced"
        )
        source_asr_asset_version = str(
            result.get("source_asr_asset_version") or ""
        ).strip()
        if not source_asr_asset_version:
            raise ValueError(
                "speaker publication requires source_asr_asset_version"
            )
        return {
            "schema_version": 1,
            "model_key": model_key,
            "diarization_model": str(
                result.get("diarization_model") or "modelscope/3D-Speaker"
            ),
            "voice_embedding_model": model_key,
            "embedding_space": str(
                result.get("embedding_space") or "3dspeaker-campplus-zh-en-192-v1"
            ),
            "embedding_normalized": True,
            "utterances": int(result.get("utterances") or 0),
            "tracks": int(result.get("tracks") or 0),
            "decode_status": str(result.get("decode_status") or "complete"),
            "source_asr_asset_version": source_asr_asset_version,
        }
    if stage == "ocr":
        semantic_model = settings.asr_semantic_model
        return {
            "schema_version": int(result.get("schema_version") or 3),
            "engine": str(result.get("engine") or settings.ocr_engine),
            "model_key": str(result.get("ocr_version") or settings.ocr_version),
            "semantic_model_key": semantic_model,
            "embedding_space": text_embedding_space(semantic_model),
            "sample_fps": float(options.get("ocr_sample_fps", settings.ocr_sample_fps)),
            "decode_status": str(result.get("decode_status") or "unknown"),
            "semantic_status": str(
                result.get("semantic_status")
                or ("complete" if settings.ocr_semantic_enabled else "disabled")
            ),
        }
    raise ValueError(f"未知索引阶段: {stage}")
