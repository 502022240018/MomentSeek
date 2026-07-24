from __future__ import annotations

import asyncio
import os
import signal
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import entity_routes, job_routes, search_routes, speaker_routes, system_routes, video_routes
from app import media, worker
from app.db import Catalog
from app.retrieval_orchestration import SearchOrchestrator
from app.search import SearchEngine
from app.settings import get_settings


probe_video = media.probe_video
export_preview_clip = media.export_preview_clip
extract_video_frame = media.extract_video_frame
extract_frame = media.extract_frame
launch_job = worker.launch_job
subprocess_environment = worker.subprocess_environment


settings = get_settings()
catalog = Catalog(settings.db_path)
search_engine = SearchEngine(settings, catalog)
search_orchestrator = SearchOrchestrator(settings, catalog, search_engine)
_indexer_daemon_process: subprocess.Popen | None = None
_indexer_daemon_lock = threading.RLock()


def _spawn_indexer_daemon():
    """Start the warm-pool daemon as a child of the API."""
    import sys

    backend_dir = Path(__file__).resolve().parents[1]
    log_path = settings.app_data_dir / "indexer-daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "-m", "app.indexer_daemon"],
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
        if "app.worker" not in cmdline or expected_job_id not in cmdline:
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _indexer_daemon_process
    settings.ensure_dirs()
    if settings.search_prewarm_enabled:
        await asyncio.to_thread(search_engine.prewarm)
    _indexer_daemon_process = _spawn_indexer_daemon() if settings.indexer_mode == "daemon" else None
    try:
        yield
    finally:
        daemon = _indexer_daemon_process
        if daemon is not None and daemon.poll() is None:
            _terminate_process_group(daemon.pid)
        _indexer_daemon_process = None
        search_engine.close()


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


app = FastAPI(
    title="MomentSeek API",
    version=__version__,
    description="Local-first face, visual, ASR and OCR video moment retrieval MVP.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
for route_module in (system_routes, video_routes, speaker_routes, job_routes, entity_routes, search_routes):
    app.include_router(route_module.router)

# Compatibility exports for callers and tests that imported handlers from app.main.
health = system_routes.health
upload_video = video_routes.upload_video
list_videos = video_routes.list_videos
get_video = video_routes.get_video
rename_video = video_routes.rename_video
delete_video = video_routes.delete_video
video_media = video_routes.video_media
video_clip = video_routes.video_clip
video_frame = video_routes.video_frame
create_index_job = video_routes.create_index_job
get_video_speakers = speaker_routes.get_video_speakers
update_video_speaker = speaker_routes.update_video_speaker
update_video_utterance = speaker_routes.update_video_utterance
search_voice = speaker_routes.search_voice
search_voice_upload = speaker_routes.search_voice_upload
get_job = job_routes.get_job
list_jobs = job_routes.list_jobs
cancel_job = job_routes.cancel_job
create_entity = entity_routes.create_entity
list_entities = entity_routes.list_entities
get_entity = entity_routes.get_entity
rename_entity = entity_routes.rename_entity
delete_entity = entity_routes.delete_entity
create_voice_only_entity = entity_routes.create_voice_only_entity
list_entity_voice_samples = entity_routes.list_entity_voice_samples
add_entity_voice_sample = entity_routes.add_entity_voice_sample
entity_reference = entity_routes.entity_reference
search = search_routes.search


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        candidate = static_dir / path
        if path and candidate.is_file() and static_dir in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(static_dir / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse({"name": "MomentSeek", "docs": "/docs", "status": "frontend not built"})
