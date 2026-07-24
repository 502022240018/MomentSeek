from __future__ import annotations

from contextlib import nullcontext
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.indexing.pipeline_manifest import write_stage_manifest
from app.model_pool import ModelPool
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageContext:
    video: dict
    options: dict
    settings: Settings
    pool: ModelPool | None
    video_path: str
    index_dir: Path
    working_dir: Path
    milvus_ctx: Any | None = None


def execute_stage(
    stage: str,
    video: dict,
    options: dict,
    settings: Settings,
    pool: ModelPool | None = None,
) -> dict:
    """Execute one indexing stage with identical behavior in every worker mode."""
    index_dir = settings.index_dir / video["id"]
    working_dir = index_dir / "work"
    index_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    runners: dict[str, Callable[[StageContext], dict]] = {
        "visual": _run_visual,
        "face": _run_face,
        "asr": _run_asr,
        "speaker": _run_speaker,
        "ocr": _run_ocr,
    }
    try:
        runner = runners[stage]
    except KeyError as exc:
        raise ValueError(f"未知索引阶段: {stage}") from exc
    lock = nullcontext()
    if settings.milvus_enabled and settings.milvus_write_enabled:
        from app.indexing.milvus_stage_lock import video_stage_lock

        lock = video_stage_lock(index_dir, video_id=video["id"], stage=stage)
    with lock:
        milvus_ctx = _setup_milvus_context(video["id"], index_dir, settings)
        if milvus_ctx is not None:
            _pre_delete_modality(milvus_ctx, video["id"], stage)
        context = StageContext(
            video=video,
            options=options,
            settings=settings,
            pool=pool,
            video_path=str(settings.resolve_path(video["file_path"])),
            index_dir=index_dir,
            working_dir=working_dir,
            milvus_ctx=milvus_ctx,
        )
        return runner(context)


def _setup_milvus_context(
    video_id: str,
    index_dir: Path,
    settings: Settings | None = None,
):
    if settings is not None and not (
        settings.milvus_enabled and settings.milvus_write_enabled
    ):
        return None
    try:
        from app.indexing.milvus_asset_version import bump_asset_version
        from app.indexing.milvus_client import get_milvus_client
        from app.indexing.milvus_indexer import MilvusWriteContext

        client = get_milvus_client()
        return MilvusWriteContext(
            video_id=video_id,
            asset_version=bump_asset_version(index_dir),
            client=client,
        )
    except Exception as exc:
        from app.indexing.milvus_flags import milvus_write_fail_policy

        if milvus_write_fail_policy() == "raise":
            raise RuntimeError(
                f"Milvus connection failed，索引已中止: video={video_id}: {exc}"
            ) from exc
        logger.warning(
            "Milvus unavailable for video=%s; retaining NPZ-only recovery copy: %s",
            video_id,
            exc,
        )
        return None


def _pre_delete_modality(milvus_ctx, video_id: str, modality: str) -> None:
    deleted = milvus_ctx.client.delete_video_modality(video_id, modality)
    if deleted >= 0:
        return
    from app.indexing.milvus_flags import milvus_write_fail_policy

    message = (
        f"Pre-index Milvus cleanup failed for video={video_id} "
        f"modality={modality}"
    )
    if milvus_write_fail_policy() == "raise":
        raise RuntimeError(message)
    logger.warning(message)


def _write_manifest(stage: str, context: StageContext, result: dict) -> None:
    write_stage_manifest(
        stage,
        index_dir=context.index_dir,
        video=context.video,
        options=context.options,
        settings=context.settings,
        result=result,
    )


def _run_visual(context: StageContext) -> dict:
    from app.indexing.visual import ClipEncoder, build_visual_index, resolve_device

    settings = context.settings
    options = context.options
    device = resolve_device(settings.npu_enabled, settings.npu_device_id, settings.cuda_enabled)
    visual_model = str(options.get("visual_model", settings.visual_model))
    model_cache_dir = str(settings.resolve_path(settings.visual_hf_cache_dir))
    encoder = None
    if context.pool is not None:
        key = f"clip:{visual_model}:{device}"
        encoder = context.pool.get(
            key,
            lambda: ClipEncoder(
                settings.clip_model,
                settings.clip_pretrained,
                device,
                visual_model=visual_model,
                model_cache_dir=model_cache_dir,
            ),
        )
    result = build_visual_index(
        video_path=context.video_path,
        output_path=str(context.index_dir / "visual.npz"),
        model_name=settings.clip_model,
        pretrained=settings.clip_pretrained,
        sample_fps=float(options.get("visual_sample_fps", settings.visual_sample_fps)),
        segment_seconds=float(options.get("visual_segment_seconds", settings.visual_segment_seconds)),
        batch_size=int(options.get("visual_batch_size", settings.visual_batch_size)),
        npu_enabled=settings.npu_enabled,
        npu_device_id=settings.npu_device_id,
        cuda_enabled=settings.cuda_enabled,
        encoder=encoder,
        visual_model=visual_model,
        model_cache_dir=model_cache_dir,
        decode_height=settings.visual_decode_height,
        prefer_ffmpeg=settings.frame_reader == "ffmpeg",
        duration_seconds=float(context.video.get("duration") or 0),
        segment_strategy=str(options.get("visual_segment_strategy", settings.visual_segment_strategy)),
        min_segment_seconds=float(options.get("visual_min_segment_seconds", settings.visual_min_segment_seconds)),
        max_segment_seconds=float(options.get("visual_max_segment_seconds", settings.visual_max_segment_seconds)),
        shot_detector=str(options.get("visual_shot_detector", settings.visual_shot_detector)),
        shot_detector_threshold=float(options.get("visual_shot_threshold", settings.visual_shot_threshold)),
        milvus_ctx=context.milvus_ctx,
    )
    _write_manifest("visual", context, result)
    return result


def _run_face(context: StageContext) -> dict:
    from app.indexing.faces import FaceEncoder, build_face_index

    settings = context.settings
    options = context.options
    model_root = str(settings.app_model_dir / "insightface")
    encoder = None
    if context.pool is not None:
        key = f"face:{settings.face_model}:{settings.face_provider}:{settings.npu_device_id}"
        encoder = context.pool.get(
            key,
            lambda: FaceEncoder(
                settings.face_model,
                settings.face_provider,
                settings.npu_device_id,
                model_root,
                settings.face_ort_intra_op_threads,
                settings.face_ort_inter_op_threads,
            ),
        )
    result = build_face_index(
        video_path=context.video_path,
        output_path=str(context.index_dir / "face.npz"),
        model_name=settings.face_model,
        sample_fps=float(options.get("face_sample_fps", settings.face_sample_fps)),
        provider=settings.face_provider,
        device_id=settings.npu_device_id,
        model_root=model_root,
        encoder=encoder,
        decode_height=settings.face_decode_height,
        prefer_ffmpeg=settings.frame_reader == "ffmpeg",
        ort_intra_op_threads=settings.face_ort_intra_op_threads,
        ort_inter_op_threads=settings.face_ort_inter_op_threads,
        milvus_ctx=context.milvus_ctx,
    )
    _write_manifest("face", context, result)
    return result


def _run_asr(context: StageContext) -> dict:
    from app.indexing.asr import build_asr_index, resolve_asr_device

    settings = context.settings
    options = context.options
    sidecar_path = options.get("sidecar_path")
    if sidecar_path:
        sidecar_path = str(settings.resolve_path(sidecar_path))
    result = build_asr_index(
        video_path=context.video_path,
        output_path=str(context.index_dir / "asr.npz"),
        working_dir=str(context.working_dir),
        engine=str(options.get("asr_engine", settings.asr_engine)),
        model_name=str(options.get("asr_model", settings.asr_model)),
        device=resolve_asr_device(
            settings.asr_device,
            settings.cuda_enabled,
            settings.npu_enabled,
            settings.npu_device_id,
        ),
        model_dir=str(settings.app_model_dir / "whisper"),
        language=str(options.get("asr_language", settings.asr_language)),
        sidecar_path=sidecar_path,
        funasr_model=settings.asr_zh_model,
        funasr_model_dir=str(settings.app_model_dir / "funasr"),
        faster_whisper_model_dir=str(settings.app_model_dir / "faster-whisper"),
        model_local_files_only=settings.asr_model_local_files_only,
        semantic_enabled=settings.asr_semantic_enabled,
        semantic_model=settings.asr_semantic_model,
        semantic_device=settings.asr_semantic_device,
        semantic_model_dir=str(settings.app_model_dir / "text-embeddings"),
        semantic_batch_size=settings.asr_semantic_batch_size,
        semantic_local_files_only=settings.asr_semantic_local_files_only,
        debug_artifacts_enabled=bool(options.get("asr_debug_artifacts", settings.asr_debug_artifacts)),
        save_raw_transcript=bool(options.get("asr_save_raw_transcript", settings.asr_save_raw_transcript)),
        vad_strategy=str(options.get("asr_vad_strategy", settings.asr_vad_strategy)),
        milvus_ctx=context.milvus_ctx,
    )
    _write_manifest("asr", context, result)
    if bool(options.get("asr_speaker_enabled", False)):
        result["speaker"] = _run_speaker(context)
    return result


def _run_speaker(context: StageContext) -> dict:
    from app.indexing.speaker import build_speaker_index

    settings = context.settings
    if context.milvus_ctx is not None:
        _pre_delete_modality(context.milvus_ctx, context.video["id"], "speaker")
    asr_path = context.index_dir / "asr.npz"
    if not asr_path.exists():
        raise RuntimeError("Speaker 索引依赖 ASR，请先构建或在同一任务中选择 ASR")
    result = build_speaker_index(
        video_path=context.video_path,
        asr_path=str(asr_path),
        output_path=str(context.index_dir / "speaker.npz"),
        working_dir=str(context.working_dir),
        model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
        model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
        device=settings.speaker_device,
        milvus_ctx=context.milvus_ctx,
    )
    _write_manifest("speaker", context, result)
    return result


def _run_ocr(context: StageContext) -> dict:
    from app.indexing.ocr import build_ocr_index, create_ocr_backend

    settings = context.settings
    options = context.options
    device = settings.ocr_device
    if device == "auto":
        device = "npu" if settings.npu_enabled else "cpu"
    model_root = str(settings.app_model_dir / "rapidocr")
    backend = None
    backend_pool_elapsed = None
    if context.pool is not None:
        key = (
            f"ocr:{settings.ocr_engine}:{settings.ocr_version}:"
            f"{settings.ocr_model_type}:{device}:{settings.npu_device_id}"
        )
        started = time.perf_counter()
        backend = context.pool.get(
            key,
            lambda: create_ocr_backend(
                settings.ocr_engine,
                device=device,
                device_id=settings.npu_device_id,
                model_root=model_root,
                ocr_version=settings.ocr_version,
                det_lang=settings.ocr_det_lang,
                rec_lang=settings.ocr_rec_lang,
                model_type=settings.ocr_model_type,
                npu_self_test=settings.ocr_npu_self_test,
                acl_model_dir=str(settings.app_model_dir / settings.ocr_acl_model_dir),
            ),
        )
        backend_pool_elapsed = time.perf_counter() - started
    result = build_ocr_index(
        video_path=context.video_path,
        output_path=str(context.index_dir / "ocr.npz"),
        working_dir=str(context.working_dir),
        sample_fps=float(options.get("ocr_sample_fps", settings.ocr_sample_fps)),
        decode_height=settings.ocr_decode_height,
        min_confidence=settings.ocr_min_confidence,
        device=device,
        device_id=settings.npu_device_id,
        model_root=model_root,
        ocr_version=settings.ocr_version,
        det_lang=settings.ocr_det_lang,
        rec_lang=settings.ocr_rec_lang,
        model_type=settings.ocr_model_type,
        npu_self_test=settings.ocr_npu_self_test,
        prefer_ffmpeg=settings.frame_reader == "ffmpeg",
        semantic_enabled=settings.ocr_semantic_enabled,
        semantic_model=settings.asr_semantic_model,
        semantic_device=settings.asr_semantic_device,
        semantic_model_dir=str(settings.app_model_dir / "text-embeddings"),
        semantic_batch_size=settings.asr_semantic_batch_size,
        semantic_local_files_only=settings.asr_semantic_local_files_only,
        engine=settings.ocr_engine,
        acl_model_dir=str(settings.app_model_dir / settings.ocr_acl_model_dir),
        backend=backend,
        milvus_ctx=context.milvus_ctx,
    )
    if backend_pool_elapsed is not None:
        result["backend_pool_get_elapsed_seconds"] = round(backend_pool_elapsed, 3)
    _write_manifest("ocr", context, result)
    return result
