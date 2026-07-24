from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    from app import main as runtime

    job = runtime.catalog.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    log_path = runtime.settings.app_data_dir / f"job-{job_id}.log"
    if log_path.exists() and job["status"] == "failed":
        job["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return job


@router.get("/api/jobs")
def list_jobs(video_id: str | None = None) -> list[dict]:
    from app import main as runtime

    return runtime.catalog.list_jobs(video_id)


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel one queued/running job while preserving all other queued work."""
    from app import main as runtime

    with runtime._indexer_daemon_lock:
        job = runtime.catalog.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        if job["status"] == "cancelled":
            return job
        if job["status"] not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="只有排队中或运行中的任务可以取消")
        previous_status = job["status"]
        runtime.catalog.update_job(
            job_id, status="cancelled", stage="cancelled", error="用户取消任务", worker_pid=None
        )
        if runtime.settings.indexer_mode == "daemon":
            if previous_status == "running":
                runtime._restart_indexer_daemon()
        else:
            runtime._terminate_process_group(job.get("worker_pid"), expected_job_id=job_id)
        video = runtime.catalog.get_video(job["video_id"])
        if video:
            runtime.catalog.update_video(
                video["id"], status="ready" if video.get("indexed_modalities") else "uploaded"
            )
        return runtime.catalog.get_job(job_id)
