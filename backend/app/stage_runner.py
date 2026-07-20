from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db import Catalog
from app.indexing.pipeline_manifest import write_stage_manifest
from app.settings import get_settings


def run(stage: str, job_id: str) -> dict:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    job = catalog.get_job(job_id)
    if not job:
        raise KeyError(f"任务不存在: {job_id}")
    video = catalog.get_video(job["video_id"])
    if not video:
        raise KeyError(f"视频不存在: {job['video_id']}")

    video_index_dir = settings.index_dir / video["id"]
    working_dir = video_index_dir / "work"
    video_index_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    options = job.get("options") or {}
    video_path = str(settings.resolve_path(video["file_path"]))
    sidecar_path = options.get("sidecar_path")
    if sidecar_path:
        sidecar_path = str(settings.resolve_path(sidecar_path))

    if stage == "visual":
        from app.indexing.visual import build_visual_index

        result = build_visual_index(
            video_path=video_path,
            output_path=str(video_index_dir / "visual.npz"),
            model_name=settings.clip_model,
            pretrained=settings.clip_pretrained,
            sample_fps=float(options.get("visual_sample_fps", settings.visual_sample_fps)),
            segment_seconds=float(options.get("visual_segment_seconds", settings.visual_segment_seconds)),
            batch_size=int(options.get("visual_batch_size", settings.visual_batch_size)),
            npu_enabled=settings.npu_enabled,
            npu_device_id=settings.npu_device_id,
            cuda_enabled=settings.cuda_enabled,
            visual_model=str(options.get("visual_model", settings.visual_model)),
            model_cache_dir=str(settings.resolve_path(settings.visual_hf_cache_dir)),
            decode_height=settings.visual_decode_height,
            prefer_ffmpeg=settings.frame_reader == "ffmpeg",
            duration_seconds=float(video.get("duration") or 0),
            segment_strategy=str(options.get("visual_segment_strategy", settings.visual_segment_strategy)),
            min_segment_seconds=float(options.get("visual_min_segment_seconds", settings.visual_min_segment_seconds)),
            max_segment_seconds=float(options.get("visual_max_segment_seconds", settings.visual_max_segment_seconds)),
            shot_detector=str(options.get("visual_shot_detector", settings.visual_shot_detector)),
            shot_detector_threshold=float(options.get("visual_shot_threshold", settings.visual_shot_threshold)),
        )
        write_stage_manifest(stage, index_dir=video_index_dir, video=video, options=options, settings=settings, result=result)
        return result
    if stage == "face":
        from app.indexing.faces import build_face_index

        result = build_face_index(
            video_path=video_path,
            output_path=str(video_index_dir / "face.npz"),
            model_name=settings.face_model,
            sample_fps=float(options.get("face_sample_fps", settings.face_sample_fps)),
            provider=settings.face_provider,
            device_id=settings.npu_device_id,
            model_root=str(settings.app_model_dir / "insightface"),
            decode_height=settings.face_decode_height,
            prefer_ffmpeg=settings.frame_reader == "ffmpeg",
        )
        write_stage_manifest(stage, index_dir=video_index_dir, video=video, options=options, settings=settings, result=result)
        return result
    if stage == "asr":
        from app.indexing.asr import build_asr_index, resolve_asr_device

        result = build_asr_index(
            video_path=video_path,
            output_path=str(video_index_dir / "asr.npz"),
            working_dir=str(working_dir),
            engine=str(options.get("asr_engine", settings.asr_engine)),
            model_name=str(options.get("asr_model", settings.asr_model)),
            device=resolve_asr_device(settings.asr_device, settings.cuda_enabled, settings.npu_enabled, settings.npu_device_id),
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
        )
        write_stage_manifest(stage, index_dir=video_index_dir, video=video, options=options, settings=settings, result=result)
        if bool(options.get("asr_speaker_enabled", False)):
            from app.indexing.speaker import build_speaker_index

            speaker_result = build_speaker_index(
                video_path=video_path,
                asr_path=str(video_index_dir / "asr.npz"),
                output_path=str(video_index_dir / "speaker.npz"),
                working_dir=str(working_dir),
                model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
                model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
                device=settings.speaker_device,
            )
            write_stage_manifest(
                "speaker", index_dir=video_index_dir, video=video,
                options=options, settings=settings, result=speaker_result,
            )
            result["speaker"] = speaker_result
        return result
    if stage == "speaker":
        from app.indexing.speaker import build_speaker_index

        asr_path = video_index_dir / "asr.npz"
        if not asr_path.exists():
            raise RuntimeError("Speaker 索引依赖 ASR，请先构建或在同一任务中选择 ASR")
        result = build_speaker_index(
            video_path=video_path,
            asr_path=str(asr_path),
            output_path=str(video_index_dir / "speaker.npz"),
            working_dir=str(working_dir),
            model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
            model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
            device=settings.speaker_device,
        )
        write_stage_manifest(stage, index_dir=video_index_dir, video=video, options=options, settings=settings, result=result)
        return result
    if stage == "ocr":
        from app.indexing.ocr import build_ocr_index

        device = settings.ocr_device
        if device == "auto":
            device = "npu" if settings.npu_enabled else "cpu"
        result = build_ocr_index(
            video_path=video_path,
            output_path=str(video_index_dir / "ocr.npz"),
            working_dir=str(working_dir),
            sample_fps=float(options.get("ocr_sample_fps", settings.ocr_sample_fps)),
            decode_height=settings.ocr_decode_height,
            min_confidence=settings.ocr_min_confidence,
            device=device,
            device_id=settings.npu_device_id,
            model_root=str(settings.app_model_dir / "rapidocr"),
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
        )
        write_stage_manifest(stage, index_dir=video_index_dir, video=video, options=options, settings=settings, result=result)
        return result
    raise ValueError(f"未知索引阶段: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["visual", "face", "asr", "speaker", "ocr"])
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    result = run(arguments.stage, arguments.job_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
