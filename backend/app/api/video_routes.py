import uuid

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import IndexRequest, VideoRenameRequest


router = APIRouter()


@router.post("/api/videos", status_code=201)
async def upload_video(
    video: UploadFile = File(...),
    transcript: UploadFile | None = File(default=None),
) -> dict:
    from app import main as runtime

    video_id = uuid.uuid4().hex
    suffix = runtime._safe_suffix(video.filename, ".mp4")
    video_path = runtime.settings.upload_dir / f"{video_id}{suffix}"
    await run_in_threadpool(runtime._save_upload, video, video_path)
    try:
        info = await run_in_threadpool(runtime.probe_video, video_path)
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="无法解析上传的视频") from exc
    sidecar_path = None
    if transcript and transcript.filename:
        transcript_suffix = runtime._safe_suffix(transcript.filename, ".json")
        sidecar_path = runtime.settings.upload_dir / f"{video_id}.transcript{transcript_suffix}"
        await run_in_threadpool(runtime._save_upload, transcript, sidecar_path)
    record = runtime.catalog.create_video({
        "id": video_id,
        "name": video.filename or video_path.name,
        "file_path": str(video_path.resolve()),
        "duration": info.duration,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "status": "uploaded",
    })
    record["sidecar_path"] = str(sidecar_path.resolve()) if sidecar_path else None
    return record


@router.get("/api/videos")
def list_videos() -> list[dict]:
    from app import main as runtime

    videos = runtime.catalog.list_videos()
    for video in videos:
        video["speaker_indexed"] = (
            runtime.settings.index_dir / video["id"] / "speaker.npz"
        ).exists()
    return videos


@router.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    video["jobs"] = runtime.catalog.list_jobs(video_id)
    video["speaker_indexed"] = (runtime.settings.index_dir / video_id / "speaker.npz").exists()
    return video


@router.patch("/api/videos/{video_id}")
def rename_video(video_id: str, request: VideoRenameRequest) -> dict:
    from app import main as runtime

    if not runtime.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    runtime.catalog.update_video(video_id, name=request.name)
    return runtime.catalog.get_video(video_id)


@router.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    jobs = runtime.catalog.list_jobs(video_id)
    if any(job["status"] in {"queued", "running"} for job in jobs):
        raise HTTPException(status_code=409, detail="该视频有索引任务进行中，请等任务结束后再删除")
    runtime._remove_video_files(video, video_id)
    for job in jobs:
        (runtime.settings.app_data_dir / f"job-{job['id']}.log").unlink(missing_ok=True)
    runtime.catalog.delete_video(video_id)
    return {"status": "deleted", "id": video_id}


@router.get("/api/videos/{video_id}/media")
def video_media(video_id: str):
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    video_path = runtime.settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(
        video_path,
        media_type=runtime._video_media_type(video_path, video.get("name")),
        filename=video["name"],
        content_disposition_type="inline",
    )


@router.get("/api/videos/{video_id}/clip")
async def video_clip(
    video_id: str,
    start: float = Query(..., ge=0),
    end: float = Query(..., gt=0),
):
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    if end <= start:
        raise HTTPException(status_code=400, detail="片段结束时间必须大于开始时间")
    video_path = runtime.settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    max_seconds = 45.0
    duration = float(video.get("duration") or 0)
    bounded_end = min(end, start + max_seconds)
    if duration > 0:
        bounded_end = min(bounded_end, duration)
    if bounded_end <= start:
        bounded_end = start + 0.25
    clip_path = runtime._clip_cache_path(video_id, start, bounded_end)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        try:
            await run_in_threadpool(
                runtime.export_preview_clip, video_path, clip_path, start, bounded_end, max_seconds
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=f"{video_id}_{round(start * 1000)}_{round(bounded_end * 1000)}.mp4",
        content_disposition_type="inline",
    )


@router.get("/api/videos/{video_id}/frame")
async def video_frame(video_id: str, time: float = Query(..., ge=0)):
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    video_path = runtime.settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    duration = float(video.get("duration") or 0)
    bounded_time = min(time, duration) if duration > 0 else time
    timestamp_ms = max(0, round(bounded_time * 1000))
    frame_path = runtime._frame_cache_path(video_id, timestamp_ms)
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        try:
            await run_in_threadpool(runtime.extract_video_frame, video_path, frame_path, bounded_time)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _index_options(request: IndexRequest) -> dict:
    values = {
        "visual_model": request.visual_model,
        "visual_sample_fps": request.visual_sample_fps,
        "visual_segment_seconds": request.visual_segment_seconds,
        "visual_segment_strategy": request.visual_segment_strategy,
        "visual_min_segment_seconds": request.visual_min_segment_seconds,
        "visual_max_segment_seconds": request.visual_max_segment_seconds,
        "visual_shot_detector": request.visual_shot_detector,
        "visual_shot_threshold": request.visual_shot_threshold,
        "face_sample_fps": request.face_sample_fps,
        "ocr_sample_fps": request.ocr_sample_fps,
        "asr_engine": request.asr_engine,
        "asr_model": request.asr_model,
        "asr_language": request.asr_language,
        "asr_speaker_enabled": request.asr_speaker_enabled,
    }
    return {key: value for key, value in values.items() if value is not None}


@router.post("/api/videos/{video_id}/index", status_code=202)
def create_index_job(video_id: str, request: IndexRequest = Body(default_factory=IndexRequest)) -> dict:
    from app import main as runtime

    video = runtime.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    running = [
        job for job in runtime.catalog.list_jobs(video_id)
        if job["status"] in {"queued", "running"}
    ]
    if running:
        raise HTTPException(status_code=409, detail="该视频已有索引任务在运行")
    options = _index_options(request)
    for suffix in ("json", "srt", "vtt"):
        sidecar = runtime.settings.upload_dir / f"{video_id}.transcript.{suffix}"
        if sidecar.exists():
            options["sidecar_path"] = str(sidecar)
            break
    job_id = uuid.uuid4().hex
    job = runtime.catalog.create_job({
        "id": job_id,
        "video_id": video_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": request.modalities,
        "options": options,
    })
    if runtime.settings.indexer_mode != "daemon":
        runtime.launch_job(job_id)
    return runtime.catalog.get_job(job_id) or job
