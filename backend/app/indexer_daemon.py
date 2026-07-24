"""Serial indexing scheduler with optional persistent per-modality workers."""
from __future__ import annotations

import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from app.db import Catalog
from app.model_pool import ModelPool
from app.settings import Settings, get_settings
from app.stage_executor import execute_stage


@contextmanager
def indexer_singleton_lock(path: Path):
    """Try to become the only indexer daemon for one runtime directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def execute_job(job_id: str, settings: Settings, catalog: Catalog, pool) -> None:
    job = catalog.get_job(job_id)
    if not job:
        return
    if not catalog.claim_queued_job(job_id, worker_pid=os.getpid()):
        return
    video = catalog.get_video(job["video_id"])
    if not video:
        catalog.update_job(job_id, status="failed", stage="failed", error="视频不存在")
        return

    metrics = {"stages": {}, "total_elapsed_seconds": None}
    job_start = time.perf_counter()
    catalog.update_job(job_id, metrics=metrics)
    catalog.update_video(video["id"], status="indexing")
    completed = set(video.get("indexed_modalities", []))
    options = job.get("options") or {}
    stages = job["modalities"]
    try:
        for index, stage in enumerate(stages):
            stage_start = time.perf_counter()
            catalog.update_job(job_id, stage=stage, progress=round(index / max(1, len(stages)), 3))
            if hasattr(pool, "run_stage"):
                result = pool.run_stage(stage, video, options)
            else:
                result = execute_stage(stage, video, options, settings, pool)
            metrics["stages"][stage] = {
                "elapsed_seconds": round(time.perf_counter() - stage_start, 3),
                "status": "completed",
                **result,
            }
            completed.add(stage)
            catalog.update_video(video["id"], indexed_modalities=sorted(completed))
            catalog.update_job(
                job_id,
                progress=round((index + 1) / max(1, len(stages)), 3),
                metrics=metrics,
            )
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
        catalog.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1,
            error=None,
            metrics=metrics,
        )
        catalog.update_video(video["id"], status="ready")
    except Exception as exc:
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
        catalog.update_job(job_id, status="failed", stage="failed", error=str(exc), metrics=metrics)
        catalog.update_video(video["id"], status="failed")
        traceback.print_exc()


def main() -> None:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    with indexer_singleton_lock(settings.app_data_dir / "indexer-daemon.lock") as acquired:
        if not acquired:
            print("[indexer-daemon] another scheduler already owns the runtime; exiting", flush=True)
            return
        if settings.npu_worker_mode == "isolated":
            from app.isolated_stage_workers import IsolatedStageWorkerPool

            pool = IsolatedStageWorkerPool(
                start_timeout_seconds=settings.indexer_worker_start_timeout_seconds,
                max_attempts=settings.indexer_stage_max_attempts,
            )
        else:
            pool = ModelPool(idle_timeout=settings.indexer_idle_timeout_seconds)
        print(
            f"[indexer-daemon] up; worker_mode={settings.npu_worker_mode} "
            f"idle_timeout={settings.indexer_idle_timeout_seconds}s poll={settings.indexer_poll_seconds}s",
            flush=True,
        )
        try:
            while True:
                job = catalog.next_queued_job()
                if job is None:
                    time.sleep(settings.indexer_poll_seconds)
                    continue
                print(
                    f"[indexer-daemon] job {job['id']} stages={job['modalities']} warm={pool.keys()}",
                    flush=True,
                )
                execute_job(job["id"], settings, catalog, pool)
        except KeyboardInterrupt:
            pass
        finally:
            pool.shutdown()


if __name__ == "__main__":
    main()
