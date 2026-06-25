from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

from app.db import Catalog
from app.settings import get_settings


@contextmanager
def exclusive_worker_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def execute_job(job_id: str) -> None:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    job = catalog.get_job(job_id)
    if not job:
        raise KeyError(f"任务不存在: {job_id}")
    video = catalog.get_video(job["video_id"])
    if not video:
        raise KeyError(f"视频不存在: {job['video_id']}")

    lock_path = settings.app_data_dir / "index-worker.lock"
    with exclusive_worker_lock(lock_path):
        catalog.update_job(job_id, status="running", stage="starting", progress=0.01, worker_pid=os.getpid())
        catalog.update_video(video["id"], status="indexing")
        completed = set(video.get("indexed_modalities", []))
        stages = job["modalities"]
        try:
            for index, stage in enumerate(stages):
                catalog.update_job(
                    job_id,
                    stage=stage,
                    progress=round(index / max(1, len(stages)), 3),
                )
                environment = os.environ.copy()
                if settings.npu_enabled:
                    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(settings.npu_device_id)
                process = subprocess.run(
                    [sys.executable, "-m", "app.stage_runner", stage, job_id],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                if process.returncode != 0:
                    details = process.stderr.strip() or process.stdout.strip()
                    raise RuntimeError(f"{stage} 阶段失败: {details[-4000:]}")
                completed.add(stage)
                catalog.update_video(video["id"], indexed_modalities=sorted(completed))
                catalog.update_job(
                    job_id,
                    progress=round((index + 1) / max(1, len(stages)), 3),
                )
            catalog.update_job(job_id, status="completed", stage="completed", progress=1, error=None)
            catalog.update_video(video["id"], status="ready")
        except Exception as exc:
            catalog.update_job(job_id, status="failed", stage="failed", error=str(exc))
            catalog.update_video(video["id"], status="failed")
            traceback.print_exc()
            raise


def launch_job(job_id: str) -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "app.worker", job_id],
        cwd=str(backend_dir),
        start_new_session=True,
        stdout=(get_settings().app_data_dir / f"job-{job_id}.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    Catalog(get_settings().db_path).update_job(job_id, worker_pid=process.pid)
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    execute_job(arguments.job_id)


if __name__ == "__main__":
    main()

