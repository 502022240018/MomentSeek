from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.model_pool import ModelPool
from app.core.settings import Settings
from app.indexing.pipeline_manifest import write_stage_manifest

logger = logging.getLogger(__name__)


def require_milvus_writes(settings: Settings) -> None:
    """Reject index production when its only online store is unavailable."""
    if settings.milvus_enabled and settings.milvus_write_enabled:
        return
    raise RuntimeError(
        "Milvus-only indexing requires MILVUS_ENABLED=true and "
        "MILVUS_WRITE_ENABLED=true"
    )


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
    require_milvus_writes(settings)
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
        from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

        # A stage publication updates the shared per-video manifest.  Serialise
        # all Milvus-backed stages of one video, not merely identical stages,
        # so independent rebuild requests cannot overwrite each other's
        # publication pointer.
        lock = video_stage_lock(index_dir, video_id=video["id"], stage="publish")
    with lock:
        milvus_ctx = _setup_milvus_context(video["id"], index_dir, settings)
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
        from app.vector_store.milvus.milvus_asset_version import reserve_next_attempt_version
        from app.vector_store.milvus.milvus_client import get_milvus_client
        from app.vector_store.milvus.milvus_indexer import MilvusWriteContext

        client = get_milvus_client()
        return MilvusWriteContext(
            video_id=video_id,
            # Reserve before writing so an interrupted attempt is never reused.
            # Online readers keep using the per-stage manifest until
            # _write_manifest validates and publishes this version.
            asset_version=reserve_next_attempt_version(index_dir),
            client=client,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Milvus connection failed，索引已中止: video={video_id}: {exc}"
        ) from exc


def _write_manifest(stage: str, context: StageContext, result: dict) -> None:
    if context.milvus_ctx is not None:
        written_rows = result.get("milvus_rows")
        if written_rows is None:
            raise RuntimeError(
                f"Milvus stage={stage} did not report its written row count; refusing publish"
            )
        persisted_rows = context.milvus_ctx.client.count_video_modality_version(
            context.video["id"], stage, context.milvus_ctx.asset_version
        )
        if persisted_rows != int(written_rows):
            raise RuntimeError(
                f"Milvus verification failed stage={stage} video={context.video['id']}: "
                f"expected={written_rows} persisted={persisted_rows}"
            )
        result["milvus_asset_version"] = context.milvus_ctx.asset_version
        result["milvus_row_count"] = persisted_rows
    write_stage_manifest(
        stage,
        index_dir=context.index_dir,
        video=context.video,
        options=context.options,
        settings=context.settings,
        result=result,
    )
    if context.milvus_ctx is not None:
        # The manifest is the per-modality publication pointer.  Only after it
        # points at the fully verified new version can older rows be reclaimed.
        deleted = context.milvus_ctx.client.delete_video_modality_except_version(
            context.video["id"], stage, context.milvus_ctx.asset_version
        )
        if deleted < 0:
            logger.warning(
                "published stage=%s video=%s but deferred old-version cleanup failed",
                stage, context.video["id"],
            )
        from app.vector_store.milvus.milvus_asset_version import publish_asset_version

        publish_asset_version(context.index_dir, context.milvus_ctx.asset_version)


def _run_visual(context: StageContext) -> dict:
    from app.encoders.visual import ClipEncoder, resolve_device
    from app.indexing.modalities.visual.visual import build_visual_index

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
    from app.encoders.face import FaceEncoder
    from app.indexing.modalities.face.faces import build_face_index

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
        gallery_cosine_threshold=settings.face_gallery_cosine_threshold,
        milvus_ctx=context.milvus_ctx,
    )
    _write_manifest("face", context, result)
    return result


def _run_asr(context: StageContext) -> dict:
    from app.indexing.modalities.asr.asr import build_asr_index, resolve_asr_device

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
    from app.indexing.modalities.speaker.speaker import build_speaker_index
    from app.indexing.manifest import load_index_manifest

    settings = context.settings
    asr_asset_version = None
    if context.milvus_ctx is not None:
        manifest = load_index_manifest(context.index_dir) or {}
        asr_channel = (manifest.get("channels") or {}).get("asr") or {}
        asr_asset_version = asr_channel.get("milvus_asset_version")
        if not asr_asset_version:
            raise RuntimeError("Milvus speaker build requires a published ASR stage")
    result = build_speaker_index(
        video_path=context.video_path,
        # The online source of truth is the ASR Milvus collection.  Keep the
        # path only for explicit offline recovery invocations of the builder.
        asr_path=str(context.index_dir / "asr.npz"),
        output_path=str(context.index_dir / "speaker.npz"),
        working_dir=str(context.working_dir),
        model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
        model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
        device=settings.speaker_device,
        milvus_ctx=context.milvus_ctx,
        asr_asset_version=str(asr_asset_version) if asr_asset_version else None,
    )
    _write_manifest("speaker", context, result)
    return result


def _run_ocr(context: StageContext) -> dict:
    from app.indexing.modalities.ocr.ocr import build_ocr_index, create_ocr_backend

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
