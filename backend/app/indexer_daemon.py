"""Warm-pool indexer daemon.

A long-lived alternative to the per-job `process_exit` subprocess worker. It polls
the job queue and runs stages in-process, keeping the CLIP / InsightFace models
resident in a ModelPool so back-to-back jobs skip the ~14.5s model load + NPU
kernel compile. Models are released after `indexer_idle_timeout_seconds` of no
work, so on a shared NPU card we only hold the ~2.3GB while actively indexing.

Run instead of relying on launch_job's subprocess fan-out:

    python -m app.indexer_daemon

The API path (create job -> status=queued) is unchanged; this daemon drains it.
"""
from __future__ import annotations

import time
import traceback

from app.db import Catalog
from app.model_pool import ModelPool
from app.settings import Settings, get_settings


def _stage_runner(stage: str, video: dict, options: dict, settings: Settings, pool: ModelPool) -> dict:
    video_index_dir = settings.index_dir / video["id"]
    thumbnail_dir = settings.thumbnail_dir / video["id"]
    working_dir = video_index_dir / "work"
    for directory in (video_index_dir, thumbnail_dir, working_dir):
        directory.mkdir(parents=True, exist_ok=True)
    video_path = str(settings.resolve_path(video["file_path"]))

    if stage == "visual":
        from app.indexing.visual import ClipEncoder, build_visual_index, resolve_device

        device = resolve_device(settings.npu_enabled, settings.npu_device_id, settings.cuda_enabled)
        key = f"clip:{settings.clip_model}:{device}"
        encoder = pool.get(key, lambda: ClipEncoder(settings.clip_model, settings.clip_pretrained, device))
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
            encoder=encoder,
        )
    if stage == "face":
        from app.indexing.faces import FaceEncoder, build_face_index

        root = str(settings.app_model_dir / "insightface")
        key = f"face:{settings.face_model}:{settings.face_provider}:{settings.npu_device_id}"
        encoder = pool.get(key, lambda: FaceEncoder(settings.face_model, settings.face_provider, settings.npu_device_id, root))
        return build_face_index(
            video_path=video_path,
            output_path=str(video_index_dir / "faces.npz"),
            thumbnail_dir=str(thumbnail_dir),
            model_name=settings.face_model,
            sample_fps=float(options.get("face_sample_fps", settings.face_sample_fps)),
            provider=settings.face_provider,
            device_id=settings.npu_device_id,
            model_root=root,
            encoder=encoder,
        )
    if stage == "asr":
        from app.indexing.asr import build_asr_index, resolve_asr_device

        sidecar_path = options.get("sidecar_path")
        if sidecar_path:
            sidecar_path = str(settings.resolve_path(sidecar_path))
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


def execute_job(job_id: str, settings: Settings, catalog: Catalog, pool: ModelPool) -> None:
    job = catalog.get_job(job_id)
    if not job:
        return
    video = catalog.get_video(job["video_id"])
    if not video:
        catalog.update_job(job_id, status="failed", stage="failed", error="视频不存在")
        return

    metrics = {"stages": {}, "total_elapsed_seconds": None}
    job_start = time.perf_counter()
    catalog.update_job(job_id, status="running", stage="starting", progress=0.01, error=None, metrics=metrics)
    catalog.update_video(video["id"], status="indexing")
    completed = set(video.get("indexed_modalities", []))
    options = job.get("options") or {}
    stages = job["modalities"]
    try:
        for index, stage in enumerate(stages):
            stage_start = time.perf_counter()
            catalog.update_job(job_id, stage=stage, progress=round(index / max(1, len(stages)), 3))
            result = _stage_runner(stage, video, options, settings, pool)
            metrics["stages"][stage] = {
                "elapsed_seconds": round(time.perf_counter() - stage_start, 3),
                "status": "completed",
                **result,
            }
            completed.add(stage)
            catalog.update_video(video["id"], indexed_modalities=sorted(completed))
            catalog.update_job(job_id, progress=round((index + 1) / max(1, len(stages)), 3), metrics=metrics)
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
        catalog.update_job(job_id, status="completed", stage="completed", progress=1, error=None, metrics=metrics)
        catalog.update_video(video["id"], status="ready")
    except Exception as exc:
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
        catalog.update_job(job_id, status="failed", stage="failed", error=str(exc), metrics=metrics)
        catalog.update_video(video["id"], status="failed")
        traceback.print_exc()


def next_queued_job(catalog: Catalog) -> dict | None:
    queued = [job for job in catalog.list_jobs() if job.get("status") == "queued"]
    if not queued:
        return None
    queued.sort(key=lambda job: job.get("created_at") or "")  # oldest first
    return queued[0]


def main() -> None:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    pool = ModelPool(idle_timeout=settings.indexer_idle_timeout_seconds)
    print(f"[indexer-daemon] up; idle_timeout={settings.indexer_idle_timeout_seconds}s poll={settings.indexer_poll_seconds}s", flush=True)
    try:
        while True:
            job = next_queued_job(catalog)
            if job is None:
                time.sleep(settings.indexer_poll_seconds)
                continue
            print(f"[indexer-daemon] job {job['id']} stages={job['modalities']} warm={pool.keys()}", flush=True)
            execute_job(job["id"], settings, catalog, pool)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
