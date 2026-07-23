from __future__ import annotations

import json
import logging
import os
import os
import signal
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.db import Catalog
from app.deployment import build_deployment_info
from app.media import export_preview_clip, extract_video_frame, probe_video
from app.schemas import (
    EntityUpdateRequest, HealthResponse, IndexRequest, SpeakerUpdateRequest, UtteranceUpdateRequest,
    VideoRenameRequest, VoiceOnlyEntityRequest, VoiceSampleRequest, VoiceSearchRequest,
)
from app.search import SearchEngine
from app.speaker_service import video_speakers, voice_search, voice_search_vectors
from app.settings import get_settings
from app.worker import launch_job, subprocess_environment

# Allow per-deployment log level tuning via APP_LOG_LEVEL env var (default INFO).
# shadow_compare logs at INFO level; APP_LOG_LEVEL=DEBUG surfaces additional diagnostic output.
_app_log_level = getattr(logging, os.getenv("APP_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.getLogger("app").setLevel(_app_log_level)


settings = get_settings()
catalog = Catalog(settings.db_path)
search_engine = SearchEngine(settings, catalog)
_indexer_daemon_process: subprocess.Popen | None = None
_indexer_daemon_lock = threading.RLock()


def _spawn_indexer_daemon():
    """Start the warm-pool daemon as a child of the API (daemon mode only).

    Inherits the container env (incl. Ascend NPU vars) so it uses the same card as
    the proven docker-exec path; does not override ASCEND_RT_VISIBLE_DEVICES.
    """
    import os
    import subprocess
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
    daemon = _spawn_indexer_daemon() if settings.indexer_mode == "daemon" else None
    # Initialise Milvus client (collections are created on first access if absent).
    if settings.milvus_write_enabled:
        try:
            from app.indexing.milvus_client import get_milvus_client
            get_milvus_client()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Milvus init failed: %s — indexing and search will not work", exc,
            )
    _indexer_daemon_process = _spawn_indexer_daemon() if settings.indexer_mode == "daemon" else None
    try:
        yield
    finally:
        daemon = _indexer_daemon_process
        if daemon is not None and daemon.poll() is None:
            _terminate_process_group(daemon.pid)
        _indexer_daemon_process = None


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


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "app_version": __version__,
        **build_deployment_info(settings),
        "npu_enabled": settings.npu_enabled,
        "npu_device_id": settings.npu_device_id if settings.npu_enabled else None,
        "cuda_enabled": settings.cuda_enabled,
        "model_idle_policy": settings.model_idle_policy,
    }


@app.post("/api/videos", status_code=201)
async def upload_video(
    video: UploadFile = File(...),
    transcript: UploadFile | None = File(default=None),
) -> dict:
    video_id = uuid.uuid4().hex
    suffix = _safe_suffix(video.filename, ".mp4")
    video_path = settings.upload_dir / f"{video_id}{suffix}"
    await run_in_threadpool(_save_upload, video, video_path)
    try:
        info = await run_in_threadpool(probe_video, video_path)
    except Exception:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="无法解析上传的视频")

    sidecar_path = None
    if transcript and transcript.filename:
        transcript_suffix = _safe_suffix(transcript.filename, ".json")
        sidecar_path = settings.upload_dir / f"{video_id}.transcript{transcript_suffix}"
        await run_in_threadpool(_save_upload, transcript, sidecar_path)
    record = catalog.create_video({
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


@app.get("/api/videos")
def list_videos() -> list[dict]:
    videos = catalog.list_videos()
    for video in videos:
        # speaker_indexed is determined solely by Milvus (no NPZ files).
        try:
            from app.indexing.milvus_client import get_milvus_client
            col = get_milvus_client().collection_for("speaker")
            video["speaker_indexed"] = bool(
                col.query(
                    expr=f'video_id == "{video["id"]}"',
                    output_fields=["utterance_idx"],
                    limit=1,
                )
            )
        except Exception:
            video["speaker_indexed"] = False
    return videos


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    video["jobs"] = catalog.list_jobs(video_id)
    try:
        from app.indexing.milvus_client import get_milvus_client
        col = get_milvus_client().collection_for("speaker")
        video["speaker_indexed"] = bool(
            col.query(
                expr=f'video_id == "{video_id}"',
                output_fields=["utterance_idx"],
                limit=1,
            )
        )
    except Exception:
        video["speaker_indexed"] = False
    return video


@app.patch("/api/videos/{video_id}")
def rename_video(video_id: str, request: VideoRenameRequest) -> dict:
    if not catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    catalog.update_video(video_id, name=request.name)
    return catalog.get_video(video_id)


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    jobs = catalog.list_jobs(video_id)
    if any(job["status"] in {"queued", "running"} for job in jobs):
        raise HTTPException(status_code=409, detail="该视频有索引任务进行中，请等任务结束后再删除")
    _remove_video_files(video, video_id)
    for job in jobs:
        (settings.app_data_dir / f"job-{job['id']}.log").unlink(missing_ok=True)
    catalog.delete_video(video_id)
    # Remove Milvus vectors so deleted videos never surface in search results.
    if settings.milvus_write_enabled:
        try:
            from app.indexing.milvus_client import get_milvus_client
            get_milvus_client().delete_video(video_id)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Milvus cleanup failed for deleted video=%s: %s", video_id, exc
            )
    return {"status": "deleted", "id": video_id}


@app.get("/api/videos/{video_id}/media")
def video_media(video_id: str):
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    video_path = settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(
        video_path,
        media_type=_video_media_type(video_path, video.get("name")),
        filename=video["name"],
        content_disposition_type="inline",
    )


@app.get("/api/videos/{video_id}/speakers")
def get_video_speakers(video_id: str) -> dict:
    if not catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    try:
        return video_speakers(settings.index_dir, catalog, video_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/videos/{video_id}/speakers/{track_id}")
def update_video_speaker(video_id: str, track_id: int, request: SpeakerUpdateRequest) -> dict:
    if not catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    catalog.upsert_video_speaker(video_id, track_id, **request.model_dump())
    return get_video_speakers(video_id)


@app.patch("/api/videos/{video_id}/utterances/{utterance_index}")
def update_video_utterance(video_id: str, utterance_index: int, request: UtteranceUpdateRequest) -> dict:
    if not catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    catalog.upsert_utterance_override(
        video_id, utterance_index, request.corrected_track_id, request.searchable
    )
    return get_video_speakers(video_id)


@app.post("/api/voice-search")
def search_voice(request: VoiceSearchRequest) -> dict:
    try:
        results = voice_search(
            settings.index_dir, catalog,
            query_video_id=request.query_video_id,
            query_utterance_index=request.query_utterance_index,
            video_ids=request.video_ids,
            limit=request.limit,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(results), "results": results}


@app.post("/api/voice-search/upload")
async def search_voice_upload(
    reference: UploadFile = File(...),
    video_ids: str | None = Form(default=None),
    limit: int = Form(default=50),
) -> dict:
    from app.indexing.speaker import encode_voice_query

    source_path = settings.query_dir / f"{uuid.uuid4().hex}{_safe_suffix(reference.filename, '.wav')}"
    wav_path = source_path.with_suffix(".voice.wav")
    await run_in_threadpool(_save_upload, reference, source_path)
    try:
        process = await run_in_threadpool(
            subprocess.run,
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, text=True,
        )
        if process.returncode != 0:
            raise ValueError(process.stderr.strip() or "无法读取上传声音")
        vectors = await run_in_threadpool(
            encode_voice_query, str(wav_path),
            model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
            model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
            device=settings.speaker_device,
        )
        results = await run_in_threadpool(
            voice_search_vectors, settings.index_dir, catalog,
            query_vectors=vectors,
            video_ids=json.loads(video_ids) if video_ids else None,
            limit=max(1, min(200, limit)),
        )
        return {"query_samples": len(vectors), "count": len(results), "results": results}
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        source_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


@app.get("/api/videos/{video_id}/clip")
async def video_clip(
    video_id: str,
    start: float = Query(..., ge=0),
    end: float = Query(..., gt=0),
):
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    if end <= start:
        raise HTTPException(status_code=400, detail="片段结束时间必须大于开始时间")
    video_path = settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")

    max_seconds = 45.0
    duration = float(video.get("duration") or 0)
    bounded_end = min(end, start + max_seconds)
    if duration > 0:
        bounded_end = min(bounded_end, duration)
    if bounded_end <= start:
        bounded_end = start + 0.25

    clip_path = _clip_cache_path(video_id, start, bounded_end)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        try:
            await run_in_threadpool(export_preview_clip, video_path, clip_path, start, bounded_end, max_seconds)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=f"{video_id}_{round(start * 1000)}_{round(bounded_end * 1000)}.mp4",
        content_disposition_type="inline",
    )


@app.get("/api/videos/{video_id}/frame")
async def video_frame(
    video_id: str,
    time: float = Query(..., ge=0),
):
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    video_path = settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    duration = float(video.get("duration") or 0)
    bounded_time = min(time, duration) if duration > 0 else time
    timestamp_ms = max(0, round(bounded_time * 1000))
    frame_path = _frame_cache_path(video_id, timestamp_ms)
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        try:
            await run_in_threadpool(extract_video_frame, video_path, frame_path, bounded_time)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/videos/{video_id}/index", status_code=202)
def create_index_job(video_id: str, request: IndexRequest = Body(default_factory=IndexRequest)) -> dict:
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    running = [job for job in catalog.list_jobs(video_id) if job["status"] in {"queued", "running"}]
    if running:
        raise HTTPException(status_code=409, detail="该视频已有索引任务在运行")
    options = {
        key: value for key, value in {
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
        }.items() if value is not None
    }
    for suffix in ("json", "srt", "vtt"):
        sidecar = settings.upload_dir / f"{video_id}.transcript.{suffix}"
        if sidecar.exists():
            options["sidecar_path"] = str(sidecar)
            break
    job_id = uuid.uuid4().hex
    job = catalog.create_job({
        "id": job_id,
        "video_id": video_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": request.modalities,
        "options": options,
    })
    # In daemon mode the warm-pool indexer drains the queue itself; otherwise spawn
    # a per-job subprocess worker.
    if settings.indexer_mode != "daemon":
        launch_job(job_id)
    return catalog.get_job(job_id) or job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = catalog.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    log_path = settings.app_data_dir / f"job-{job_id}.log"
    if log_path.exists() and job["status"] == "failed":
        job["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return job


@app.get("/api/jobs")
def list_jobs(video_id: str | None = None) -> list[dict]:
    return catalog.list_jobs(video_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel one queued/running job while preserving all other queued work."""
    with _indexer_daemon_lock:
        job = catalog.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        if job["status"] == "cancelled":
            return job
        if job["status"] not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="只有排队中或运行中的任务可以取消")

        previous_status = job["status"]
        catalog.update_job(
            job_id,
            status="cancelled",
            stage="cancelled",
            error="用户取消任务",
            worker_pid=None,
        )

        if settings.indexer_mode == "daemon":
            # A queued job needs no process action. A running job executes inside
            # the daemon, so restart that one queue consumer; other queued jobs
            # remain in SQLite and are picked up by the fresh daemon.
            if previous_status == "running":
                _restart_indexer_daemon()
        else:
            # Detached subprocess workers may be running or waiting on the
            # single-worker lock. Terminate only the process group recorded for
            # this job; PID/cmdline validation avoids killing an unrelated PID.
            _terminate_process_group(job.get("worker_pid"), expected_job_id=job_id)

        video = catalog.get_video(job["video_id"])
        if video:
            catalog.update_video(
                video["id"],
                status="ready" if video.get("indexed_modalities") else "uploaded",
            )
        return catalog.get_job(job_id)


@app.post("/api/entities", status_code=201)
async def create_entity(name: str = Form(...), reference: UploadFile = File(...)) -> dict:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="人物名称不能为空")
    entity_id = uuid.uuid4().hex
    reference_path = settings.app_data_dir / "entities" / f"{entity_id}{_safe_suffix(reference.filename, '.jpg')}"
    await run_in_threadpool(_save_upload, reference, reference_path)
    try:
        vector = await run_in_threadpool(search_engine._face().encode_reference, str(reference_path))
    except Exception as exc:
        reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    embedding_path = reference_path.with_suffix(".npz")
    np.savez_compressed(embedding_path, embedding=vector.astype(np.float32))

    # Phase 3: Store embedding in database as BLOB (dual-write)
    face_embedding_blob = vector.astype(np.float32).tobytes()

    try:
        return catalog.create_entity({
            "id": entity_id,
            "name": name,
            "reference_path": str(reference_path),
            "embedding_path": str(embedding_path),
            "face_embedding": face_embedding_blob,
        })
    except sqlite3.IntegrityError:
        reference_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="该人物名称已存在")


@app.get("/api/entities")
def list_entities() -> list[dict]:
    return catalog.list_entities()


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:
    entity = catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    entity["voice_samples"] = catalog.list_voice_samples(entity_id)
    return entity


@app.patch("/api/entities/{entity_id}")
def rename_entity(entity_id: str, request: EntityUpdateRequest) -> dict:
    try:
        if not catalog.rename_entity(entity_id, request.name):
            raise HTTPException(status_code=404, detail="人物不存在")
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc
    return get_entity(entity_id)


@app.delete("/api/entities/{entity_id}")
def delete_entity(entity_id: str) -> dict:
    entity = catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    paths = [entity.get("reference_path"), entity.get("embedding_path")]
    paths.extend(sample.get("embedding_path") for sample in catalog.list_voice_samples(entity_id))
    if not catalog.delete_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    for value in paths:
        if value:
            Path(value).unlink(missing_ok=True)
    shutil.rmtree(settings.app_data_dir / "entities" / entity_id, ignore_errors=True)
    return {"status": "deleted", "id": entity_id}


@app.post("/api/entities/voice-only", status_code=201)
def create_voice_only_entity(request: VoiceOnlyEntityRequest) -> dict:
    try:
        return catalog.create_entity({
            "id": uuid.uuid4().hex, "name": request.name,
            "reference_path": "", "embedding_path": None,
            "face_embedding": None,
        })
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc


@app.get("/api/entities/{entity_id}/voice-samples")
def list_entity_voice_samples(entity_id: str) -> list[dict]:
    if not catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    samples = catalog.list_voice_samples(entity_id)
    for sample in samples:
        if sample.get("source_video_id") is not None and sample.get("source_utterance_index") is not None:
            try:
                view = video_speakers(settings.index_dir, catalog, sample["source_video_id"])
                utterance = next(
                    (item for item in view["utterances"] if item["index"] == int(sample["source_utterance_index"])),
                    None,
                )
                if utterance:
                    sample["clip_url"] = utterance["clip_url"]
                    sample["text"] = utterance["text"]
            except (FileNotFoundError, IndexError, ValueError):
                pass
    return samples


@app.post("/api/entities/{entity_id}/voice-samples", status_code=201)
def add_entity_voice_sample(entity_id: str, request: VoiceSampleRequest) -> dict:
    if not catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    path = settings.index_dir / request.video_id / "speaker.npz"
    try:
        from app.speaker_service import _load_speaker_data
        data = _load_speaker_data(path, request.video_id)
        vector = data["utterance_embeddings"][request.utterance_index].astype(np.float32)
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="声音片段不存在") from exc
    sample_id = uuid.uuid4().hex
    embedding_path = settings.app_data_dir / "entities" / entity_id / "voice" / f"{sample_id}.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embedding_path, embedding=vector)

    # Phase 3: Store voice embedding in database as BLOB (dual-write)
    voice_embedding_blob = vector.tobytes()

    sample = catalog.create_voice_sample({
        "id": sample_id, "entity_id": entity_id, "source_type": "video_utterance",
        "source_video_id": request.video_id, "source_utterance_index": request.utterance_index,
        "audio_path": None, "embedding_path": str(embedding_path),
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
        "voice_embedding": voice_embedding_blob,
    })
    if request.bind_track_id is not None:
        catalog.bind_speaker_identity(request.video_id, request.bind_track_id, entity_id)
    return sample


@app.get("/api/entities/{entity_id}/reference")
def entity_reference(entity_id: str):
    entity = catalog.get_entity(entity_id)
    if not entity or not entity.get("reference_path") or not Path(entity["reference_path"]).is_file():
        raise HTTPException(status_code=404, detail="人物参考图不存在")
    return FileResponse(entity["reference_path"], content_disposition_type="inline")


@app.post("/api/search")
async def search(
    query_text: str | None = Form(default=None),
    query_image: UploadFile | None = File(default=None),
    modalities: str = Form(default="visual,face,asr,ocr"),
    video_ids: str | None = Form(default=None),
    alpha: float = Form(default=0.5),
    limit: int = Form(default=24),
) -> dict:
    selected_modalities = [item.strip() for item in modalities.split(",") if item.strip()]
    if not query_text and not query_image:
        raise HTTPException(status_code=422, detail="请提供查询文字或参考图")
    if any(item not in {"visual", "face", "asr", "ocr"} for item in selected_modalities):
        raise HTTPException(status_code=422, detail="检索通道不合法")
    image_path = None
    if query_image and query_image.filename:
        image_path = settings.query_dir / f"{uuid.uuid4().hex}{_safe_suffix(query_image.filename, '.jpg')}"
        await run_in_threadpool(_save_upload, query_image, image_path)
    try:
        started = time.perf_counter()
        results = await run_in_threadpool(
            search_engine.search,
            query_text.strip() if query_text else None,
            str(image_path) if image_path else None,
            selected_modalities,
            json.loads(video_ids) if video_ids else None,
            max(0, min(1, alpha)),
            max(1, min(100, limit)),
        )
        elapsed_seconds = round(time.perf_counter() - started, 3)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)
    return {
        "query": query_text,
        "modalities": selected_modalities,
        "count": len(results),
        "above_count": sum(1 for item in results if item.get("above_threshold")),
        "elapsed_seconds": elapsed_seconds,
        "results": results,
    }


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
