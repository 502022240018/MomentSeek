"""Platform runtime context: process-wide singletons and shared route helpers.

This module owns the objects that used to live in ``app.main``. It never
imports any route module, so routes can import it at the top of the file
without creating an import cycle:

    main -> routes -> context <- main

Tests that need to stub runtime state should monkeypatch attributes here
(e.g. ``monkeypatch.setattr(context, "catalog", fake_catalog)``).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from fastapi import UploadFile

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.execution import worker
from app.integrations.color_grading import ColorGradingManager
from app.media import media
from app.orchestration.retrieval_orchestration import SearchOrchestrator
from app.retrieval.search import SearchEngine

# Media / worker entry points shared by routes (patchable in tests).
probe_video = media.probe_video
export_preview_clip = media.export_preview_clip
extract_video_frame = media.extract_video_frame
extract_frame = media.extract_frame
launch_job = worker.launch_job
subprocess_environment = worker.subprocess_environment

# Process-wide singletons.
settings = get_settings()
catalog = Catalog(settings.db_path)
search_engine = SearchEngine(settings, catalog)
search_orchestrator = SearchOrchestrator(settings, catalog, search_engine)

# Indexer daemon supervision (daemon mode only).
_indexer_daemon_process: subprocess.Popen | None = None
_indexer_daemon_lock = threading.RLock()


def _spawn_indexer_daemon():
    """Start the warm-pool daemon as a child of the API."""
    import sys

    backend_dir = Path(__file__).resolve().parents[2]
    log_path = settings.app_data_dir / "indexer-daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "-m", "app.execution.indexer_daemon"],
        cwd=str(backend_dir),
        env=subprocess_environment(settings),
        start_new_session=True,
        stdout=log_path.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )


def _terminate_process_group(pid: int | None, expected_job_id: str | None = None) -> bool:
    """Terminate one detached worker process group without risking an unrelated PID."""
    if not pid or pid <= 1 or pid == os.getpid():
        return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if expected_job_id and cmdline_path.exists():
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        if "app.execution.worker" not in cmdline or expected_job_id not in cmdline:
            return False
    try:
        process_group = os.getpgid(pid)
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


def _restart_indexer_daemon() -> None:
    """Stop the current daemon process group, then start a fresh queue consumer."""
    global _indexer_daemon_process
    with _indexer_daemon_lock:
        process = _indexer_daemon_process
        if process is not None and process.poll() is None:
            _terminate_process_group(process.pid)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        _indexer_daemon_process = _spawn_indexer_daemon()


def start_indexer_daemon_if_configured() -> None:
    """Lifespan hook: launch the daemon when INDEXER_MODE=daemon."""
    global _indexer_daemon_process
    _indexer_daemon_process = _spawn_indexer_daemon() if settings.indexer_mode == "daemon" else None


def stop_indexer_daemon() -> None:
    """Lifespan hook: terminate the daemon process group on shutdown."""
    global _indexer_daemon_process
    daemon = _indexer_daemon_process
    if daemon is not None and daemon.poll() is None:
        _terminate_process_group(daemon.pid)
    _indexer_daemon_process = None


# Shared route helpers.

def _safe_suffix(filename: str | None, fallback: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix and len(suffix) <= 10 else fallback


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        shutil.copyfileobj(upload.file, target, length=1024 * 1024)


def _remove_video_files(video: dict, video_id: str) -> None:
    files = [settings.resolve_path(video["file_path"])] if video.get("file_path") else []
    files += [settings.upload_dir / f"{video_id}.transcript.{suffix}" for suffix in ("json", "srt", "vtt")]
    for path in files:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    for directory in (
        settings.index_dir / video_id,
        settings.legacy_thumbnail_dir / video_id,
        settings.clip_cache_dir / video_id,
        settings.frame_cache_dir / video_id,
    ):
        shutil.rmtree(directory, ignore_errors=True)


def _video_media_type(path: Path, name: str | None = None) -> str:
    suffix = (Path(name or "").suffix or path.suffix).lower()
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }.get(suffix, "video/mp4")


def _clip_cache_path(video_id: str, start_time: float, end_time: float) -> Path:
    start_ms = max(0, round(start_time * 1000))
    end_ms = max(start_ms + 250, round(end_time * 1000))
    return settings.clip_cache_dir / video_id / f"{start_ms:012d}_{end_ms:012d}.mp4"


def _frame_cache_path(video_id: str, ms: int) -> Path:
    return settings.frame_cache_dir / video_id / f"{max(0, ms):012d}.jpg"


def _color_grading_manager() -> ColorGradingManager:
    return ColorGradingManager(settings, catalog)
