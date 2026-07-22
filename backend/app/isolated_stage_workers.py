from __future__ import annotations

import multiprocessing
import os
import time
import traceback
import uuid
from multiprocessing.connection import Connection
from typing import Any

from app.model_pool import ModelPool
from app.settings import get_settings


SUPPORTED_STAGES = {"visual", "face", "asr", "ocr"}
_CONTEXT_ERROR_MARKERS = (
    "error code is 107003",
    "stream is not in the current context",
    "stream is not in current ctx",
)


class IsolatedWorkerError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, remote_traceback: str = "") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.remote_traceback = remote_traceback


def _is_retryable_context_error(message: str) -> bool:
    normalized = message.casefold()
    return any(marker in normalized for marker in _CONTEXT_ERROR_MARKERS)


def _execute_stage(stage: str, video: dict, options: dict, pool: ModelPool) -> dict:
    # Import lazily so the child owns every accelerator import and runtime
    # context created by the stage. The parent scheduler never imports a model.
    from app.stage_executor import execute_stage

    return execute_stage(stage, video, options, get_settings(), pool)


def isolated_stage_worker_main(stage: str, connection: Connection) -> None:
    """Serve one modality in one process for the lifetime of its NPU context."""
    settings = get_settings()
    pool = ModelPool(idle_timeout=settings.indexer_idle_timeout_seconds)
    try:
        connection.send({"type": "ready", "stage": stage, "pid": os.getpid()})
        while True:
            request = connection.recv()
            operation = request.get("operation")
            if operation == "shutdown":
                return
            if operation != "run":
                raise ValueError(f"unsupported isolated worker operation: {operation}")

            request_id = str(request["request_id"])
            started = time.perf_counter()
            try:
                result = _execute_stage(stage, request["video"], request.get("options") or {}, pool)
                connection.send({
                    "type": "result",
                    "request_id": request_id,
                    "stage": stage,
                    "pid": os.getpid(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "result": result,
                })
            except Exception as exc:
                message = str(exc)
                connection.send({
                    "type": "error",
                    "request_id": request_id,
                    "stage": stage,
                    "pid": os.getpid(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "message": message,
                    "retryable": _is_retryable_context_error(message),
                    "traceback": traceback.format_exc(),
                })
                # Any accelerator exception may leave the runtime partially
                # initialized. Exit instead of serving another request from a
                # potentially poisoned context; the supervisor recreates us.
                return
    except (EOFError, BrokenPipeError):
        return
    finally:
        pool.shutdown()
        connection.close()


class IsolatedStageWorker:
    def __init__(
        self,
        stage: str,
        *,
        start_timeout_seconds: float = 30.0,
        process_context: Any | None = None,
    ) -> None:
        if stage not in SUPPORTED_STAGES:
            raise ValueError(f"unsupported isolated worker stage: {stage}")
        self.stage = stage
        self.start_timeout_seconds = max(1.0, float(start_timeout_seconds))
        self._context = process_context or multiprocessing.get_context("spawn")
        self._connection: Connection | None = None
        self._process: Any | None = None

    @property
    def pid(self) -> int | None:
        return int(self._process.pid) if self._process is not None and self._process.pid else None

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def start(self) -> None:
        if self.alive:
            return
        self.stop()
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=isolated_stage_worker_main,
            args=(self.stage, child),
            name=f"momentseek-{self.stage}-worker",
            daemon=False,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        if not parent.poll(self.start_timeout_seconds):
            self.stop()
            raise IsolatedWorkerError(f"{self.stage} worker did not become ready in time", retryable=True)
        response = parent.recv()
        if response.get("type") != "ready":
            self.stop()
            raise IsolatedWorkerError(f"{self.stage} worker returned invalid readiness response", retryable=True)

    def run(self, video: dict, options: dict) -> dict:
        self.start()
        assert self._connection is not None
        assert self._process is not None
        request_id = uuid.uuid4().hex
        try:
            self._connection.send({
                "operation": "run",
                "request_id": request_id,
                "video": video,
                "options": options,
            })
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.stop()
            raise IsolatedWorkerError(f"{self.stage} worker request failed: {exc}", retryable=True) from exc

        while True:
            if self._connection.poll(1.0):
                try:
                    response = self._connection.recv()
                except (EOFError, OSError) as exc:
                    exit_code = self._process.exitcode
                    self.stop()
                    raise IsolatedWorkerError(
                        f"{self.stage} worker exited before returning a result (exit={exit_code})",
                        retryable=True,
                    ) from exc
                break
            if not self._process.is_alive():
                exit_code = self._process.exitcode
                self.stop()
                raise IsolatedWorkerError(
                    f"{self.stage} worker exited before returning a result (exit={exit_code})",
                    retryable=True,
                )

        if response.get("request_id") != request_id:
            self.stop()
            raise IsolatedWorkerError(f"{self.stage} worker returned a mismatched response", retryable=True)
        if response.get("type") == "error":
            error = IsolatedWorkerError(
                f"{self.stage} isolated worker failed: {response.get('message') or 'unknown error'}",
                retryable=bool(response.get("retryable")),
                remote_traceback=str(response.get("traceback") or ""),
            )
            self.stop()
            raise error
        if response.get("type") != "result" or not isinstance(response.get("result"), dict):
            self.stop()
            raise IsolatedWorkerError(f"{self.stage} worker returned an invalid result", retryable=True)

        result = dict(response["result"])
        result["isolated_worker_pid"] = int(response["pid"])
        result["isolated_worker_elapsed_seconds"] = float(response["elapsed_seconds"])
        return result

    def stop(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            if process is not None and process.is_alive():
                try:
                    connection.send({"operation": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        if process is None:
            return
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)


class IsolatedStageWorkerPool:
    """One persistent process and one model pool per accelerator modality."""

    def __init__(self, *, start_timeout_seconds: float = 30.0, max_attempts: int = 2) -> None:
        self.start_timeout_seconds = start_timeout_seconds
        self.max_attempts = max(1, int(max_attempts))
        self._workers: dict[str, IsolatedStageWorker] = {}

    def _worker(self, stage: str) -> IsolatedStageWorker:
        worker = self._workers.get(stage)
        if worker is None:
            worker = IsolatedStageWorker(stage, start_timeout_seconds=self.start_timeout_seconds)
            self._workers[stage] = worker
        return worker

    def run_stage(self, stage: str, video: dict, options: dict) -> dict:
        last_error: IsolatedWorkerError | None = None
        for attempt in range(1, self.max_attempts + 1):
            worker = self._worker(stage)
            try:
                result = worker.run(video, options)
                result["isolated_worker_attempts"] = attempt
                return result
            except IsolatedWorkerError as exc:
                last_error = exc
                worker.stop()
                self._workers.pop(stage, None)
                if not exc.retryable or attempt >= self.max_attempts:
                    details = exc.remote_traceback[-4000:] if exc.remote_traceback else str(exc)
                    raise RuntimeError(details) from exc
                print(
                    f"[indexer-daemon] retrying {stage} in a fresh isolated worker "
                    f"attempt={attempt + 1}/{self.max_attempts}: {exc}",
                    flush=True,
                )
        raise RuntimeError(str(last_error or f"{stage} isolated worker failed"))

    def keys(self) -> list[str]:
        return [f"{stage}:pid={worker.pid}" for stage, worker in sorted(self._workers.items()) if worker.alive]

    def shutdown(self) -> None:
        workers = list(self._workers.values())
        self._workers.clear()
        for worker in workers:
            worker.stop()
