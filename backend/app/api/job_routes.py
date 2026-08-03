from fastapi import APIRouter, HTTPException
from app.platform import context


router = APIRouter()


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:

    job = context.catalog.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    log_path = context.settings.app_data_dir / f"job-{job_id}.log"
    if log_path.exists() and job["status"] == "failed":
        job["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return job


@router.get("/api/jobs")
def list_jobs(video_id: str | None = None) -> list[dict]:

    return context.catalog.list_jobs(video_id)


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel one queued/running job while preserving all other queued work."""

    with context._indexer_daemon_lock:
        job = context.catalog.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        if job["status"] == "cancelled":
            return job
        if job["status"] not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="只有排队中或运行中的任务可以取消")
        previous_status = job["status"]
        context.catalog.update_job(
            job_id, status="cancelled", stage="cancelled", error="用户取消任务", worker_pid=None
        )
        if context.settings.indexer_mode == "daemon":
            if previous_status == "running":
                context._restart_indexer_daemon()
        else:
            context._terminate_process_group(job.get("worker_pid"), expected_job_id=job_id)
        video = context.catalog.get_video(job["video_id"])
        if video:
            context.catalog.update_video(
                video["id"], status="ready" if video.get("indexed_modalities") else "uploaded"
            )
        return context.catalog.get_job(job_id)
