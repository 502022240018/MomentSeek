from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from app.db import Catalog
from app.settings import Settings
from app.settings import get_settings


NPU_ENV_KEYS = {
    "NPU_DEVICE_ID",
    "ASCEND_DEVICE_ID",
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
}


@contextmanager
def exclusive_worker_lock(path: Path, poll_seconds: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        # Windows msvcrt LK_LOCK only retries for ~10s before raising EDEADLOCK,
        # unlike fcntl.flock(LOCK_EX) which blocks forever. A long stage on another
        # job (face indexing runs many minutes) would otherwise crash every worker
        # queued behind it and leave the job stuck as "queued". Poll a non-blocking
        # lock instead so we wait indefinitely.
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(poll_seconds)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
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
        if not catalog.claim_queued_job(job_id, worker_pid=os.getpid()):
            return
        metrics = {"stages": {}, "total_elapsed_seconds": None}
        job_start = time.perf_counter()
        catalog.update_job(job_id, metrics=metrics)
        catalog.update_video(video["id"], status="indexing")
        completed = set(video.get("indexed_modalities", []))
        stages = job["modalities"]
        try:
            for index, stage in enumerate(stages):
                stage_start = time.perf_counter()
                catalog.update_job(
                    job_id,
                    stage=stage,
                    progress=round(index / max(1, len(stages)), 3),
                )
                environment = subprocess_environment(settings)
                process = subprocess.run(
                    [sys.executable, "-m", "app.stage_runner", stage, job_id],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=environment,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                )
                elapsed = round(time.perf_counter() - stage_start, 3)
                if process.returncode != 0:
                    metrics["stages"][stage] = {"elapsed_seconds": elapsed, "status": "failed"}
                    metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
                    catalog.update_job(job_id, metrics=metrics)
                    details = process.stderr.strip() or process.stdout.strip()
                    raise RuntimeError(f"{stage} 阶段失败: {details[-4000:]}")
                stage_result = parse_stage_result(process.stdout)
                metrics["stages"][stage] = {
                    "elapsed_seconds": elapsed,
                    "status": "completed",
                    **stage_result,
                }
                completed.add(stage)
                catalog.update_video(video["id"], indexed_modalities=sorted(completed))
                catalog.update_job(
                    job_id,
                    progress=round((index + 1) / max(1, len(stages)), 3),
                    metrics=metrics,
                )
            metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
            catalog.update_job(job_id, status="completed", stage="completed", progress=1, error=None, metrics=metrics)
            catalog.update_video(video["id"], status="ready")
        except Exception as exc:
            metrics["total_elapsed_seconds"] = round(time.perf_counter() - job_start, 3)
            catalog.update_job(job_id, status="failed", stage="failed", error=str(exc), metrics=metrics)
            catalog.update_video(video["id"], status="failed")
            traceback.print_exc()
            raise


def launch_job(job_id: str) -> int:
    settings = get_settings()
    backend_dir = Path(__file__).resolve().parents[1]
    log_path = settings.app_data_dir / f"job-{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "app.worker", job_id],
        cwd=str(backend_dir),
        env=subprocess_environment(settings),
        start_new_session=True,
        stdout=log_path.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    Catalog(settings.db_path).update_job(job_id, worker_pid=process.pid)
    return process.pid


def worker_environment(settings: Settings) -> dict[str, str]:
    environment = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    for name in settings.__class__.model_fields:
        value = getattr(settings, name)
        if value is None:
            continue
        if not settings.npu_enabled and name.upper() in NPU_ENV_KEYS:
            continue
        env_name = name.upper()
        if isinstance(value, bool):
            environment[env_name] = str(value).lower()
        elif isinstance(value, Path):
            if name in {"app_data_dir", "app_model_dir"}:
                environment[env_name] = str(value.resolve())
            else:
                environment[env_name] = str(settings.resolve_path(value))
        else:
            environment[env_name] = str(value)
    return environment


def subprocess_environment(settings: Settings, base_environment: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    if not settings.npu_enabled:
        for key in NPU_ENV_KEYS:
            environment.pop(key, None)
    environment.update(worker_environment(settings))
    if settings.npu_enabled and "ASCEND_RT_VISIBLE_DEVICES" not in environment:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(settings.npu_device_id)
    return environment


def parse_stage_result(output: str) -> dict:
    for line in reversed([item.strip() for item in output.splitlines() if item.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    execute_job(arguments.job_id)


if __name__ == "__main__":
    main()
