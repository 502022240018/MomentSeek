from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from app.db import Catalog
from app.indexer_daemon import execute_job
from app.isolated_stage_workers import IsolatedStageWorkerPool
from app.media import probe_video
from app.settings import get_settings


def _create_job(catalog: Catalog, video_id: str, modalities: list[str], options: dict) -> str:
    job_id = uuid.uuid4().hex
    catalog.create_job({
        "id": job_id,
        "video_id": video_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": modalities,
        "options": options,
    })
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise resident modality workers across ACL/CANN/Torch contexts.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--first-stages", default="visual,face,ocr")
    parser.add_argument("--second-stages", default="visual")
    parser.add_argument("--visual-sample-fps", type=float, default=1.0)
    parser.add_argument("--face-sample-fps", type=float, default=0.5)
    parser.add_argument("--ocr-sample-fps", type=float, default=0.5)
    args = parser.parse_args()

    settings = get_settings()
    if settings.npu_worker_mode.strip().casefold() != "isolated":
        raise RuntimeError("NPU_WORKER_MODE=isolated is required")
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    metadata = probe_video(video_path)
    catalog = Catalog(settings.db_path)
    video_id = f"isolated-smoke-{uuid.uuid4().hex[:12]}"
    catalog.create_video({
        "id": video_id,
        "name": video_path.name,
        "file_path": str(video_path),
        "duration": float(metadata.duration),
        "fps": float(metadata.fps),
        "width": int(metadata.width),
        "height": int(metadata.height),
        "status": "uploaded",
    })
    options = {
        "visual_sample_fps": args.visual_sample_fps,
        "face_sample_fps": args.face_sample_fps,
        "ocr_sample_fps": args.ocr_sample_fps,
    }
    first_stages = [value.strip() for value in args.first_stages.split(",") if value.strip()]
    second_stages = [value.strip() for value in args.second_stages.split(",") if value.strip()]
    pool = IsolatedStageWorkerPool(
        start_timeout_seconds=settings.indexer_worker_start_timeout_seconds,
        max_attempts=settings.indexer_stage_max_attempts,
    )
    try:
        job_ids = []
        for stages in (first_stages, second_stages):
            job_id = _create_job(catalog, video_id, stages, options)
            job_ids.append(job_id)
            execute_job(job_id, settings, catalog, pool)
            job = catalog.get_job(job_id)
            if not job or job["status"] != "completed":
                raise RuntimeError(json.dumps(job, ensure_ascii=False))

        jobs = [catalog.get_job(job_id) for job_id in job_ids]
        first_visual = jobs[0]["metrics"]["stages"].get("visual", {})
        second_visual = jobs[1]["metrics"]["stages"].get("visual", {})
        if first_visual and second_visual:
            if first_visual.get("isolated_worker_pid") != second_visual.get("isolated_worker_pid"):
                raise RuntimeError("visual worker was not resident across jobs")
        report = {
            "success": True,
            "video_id": video_id,
            "worker_keys": pool.keys(),
            "jobs": jobs,
        }
        print(json.dumps(report, ensure_ascii=False))
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
