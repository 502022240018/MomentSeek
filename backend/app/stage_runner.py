from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db import Catalog
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
    thumbnail_dir = settings.thumbnail_dir / video["id"]
    working_dir = video_index_dir / "work"
    video_index_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    options = job.get("options") or {}
    video_path = str(settings.resolve_path(video["file_path"]))
    sidecar_path = options.get("sidecar_path")
    if sidecar_path:
        sidecar_path = str(settings.resolve_path(sidecar_path))

    if stage == "visual":
        from app.indexing.visual import build_visual_index

        return build_visual_index(
            video_path=video_path,
            output_path=str(video_index_dir / "visual.npz"),
            thumbnail_dir=str(thumbnail_dir),
            model_name=settings.clip_model,
            pretrained=settings.clip_pretrained,
            sample_fps=float(options.get("visual_sample_fps", settings.visual_sample_fps)),
            segment_seconds=float(options.get("visual_segment_seconds", settings.visual_segment_seconds)),
            batch_size=int(options.get("visual_batch_size", settings.visual_batch_size)),
            npu_enabled=settings.npu_enabled,
            npu_device_id=settings.npu_device_id,
            cuda_enabled=settings.cuda_enabled,
        )
    if stage == "face":
        from app.indexing.faces import build_face_index

        return build_face_index(
            video_path=video_path,
            output_path=str(video_index_dir / "faces.npz"),
            thumbnail_dir=str(thumbnail_dir),
            model_name=settings.face_model,
            sample_fps=float(options.get("face_sample_fps", settings.face_sample_fps)),
            provider=settings.face_provider,
            device_id=settings.npu_device_id,
            model_root=str(settings.app_model_dir / "insightface"),
        )
    if stage == "asr":
        from app.indexing.asr import build_asr_index, resolve_asr_device

        return build_asr_index(
            video_path=video_path,
            output_path=str(video_index_dir / "asr.json"),
            working_dir=str(working_dir),
            engine=settings.asr_engine,
            model_name=str(options.get("asr_model", settings.asr_model)),
            device=resolve_asr_device(settings.asr_device, settings.cuda_enabled, settings.npu_enabled, settings.npu_device_id),
            model_dir=str(settings.app_model_dir / "whisper"),
            language=str(options.get("asr_language", settings.asr_language)),
            sidecar_path=sidecar_path,
            funasr_model=settings.asr_zh_model,
        )
    raise ValueError(f"未知索引阶段: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["visual", "face", "asr"])
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    result = run(arguments.stage, arguments.job_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
