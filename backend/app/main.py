from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.db import Catalog
from app.media import probe_video
from app.schemas import HealthResponse, IndexRequest, VideoRenameRequest
from app.search import SearchEngine
from app.settings import get_settings
from app.worker import launch_job


settings = get_settings()
catalog = Catalog(settings.db_path)
search_engine = SearchEngine(settings, catalog)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(
    title="MomentSeek API",
    version=__version__,
    description="Local-first face, visual and ASR video moment retrieval MVP.",
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
    for directory in (settings.index_dir / video_id, settings.thumbnail_dir / video_id):
        shutil.rmtree(directory, ignore_errors=True)


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
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
    return catalog.list_videos()


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    video["jobs"] = catalog.list_jobs(video_id)
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
    return {"status": "deleted", "id": video_id}


@app.get("/api/videos/{video_id}/media")
def video_media(video_id: str):
    video = catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    video_path = settings.resolve_path(video["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(video_path, filename=video["name"], content_disposition_type="inline")


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
            "visual_sample_fps": request.visual_sample_fps,
            "visual_segment_seconds": request.visual_segment_seconds,
            "face_sample_fps": request.face_sample_fps,
            "asr_model": request.asr_model,
            "asr_language": request.asr_language,
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
    try:
        return catalog.create_entity({
            "id": entity_id,
            "name": name,
            "reference_path": str(reference_path),
            "embedding_path": str(embedding_path),
        })
    except sqlite3.IntegrityError:
        reference_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="该人物名称已存在")


@app.get("/api/entities")
def list_entities() -> list[dict]:
    return catalog.list_entities()


@app.get("/api/entities/{entity_id}/reference")
def entity_reference(entity_id: str):
    entity = catalog.get_entity(entity_id)
    if not entity or not Path(entity["reference_path"]).exists():
        raise HTTPException(status_code=404, detail="人物参考图不存在")
    return FileResponse(entity["reference_path"], content_disposition_type="inline")


@app.post("/api/search")
async def search(
    query_text: str | None = Form(default=None),
    query_image: UploadFile | None = File(default=None),
    modalities: str = Form(default="visual,face,asr"),
    video_ids: str | None = Form(default=None),
    alpha: float = Form(default=0.5),
    limit: int = Form(default=24),
) -> dict:
    selected_modalities = [item.strip() for item in modalities.split(",") if item.strip()]
    if not query_text and not query_image:
        raise HTTPException(status_code=422, detail="请提供查询文字或参考图")
    if any(item not in {"visual", "face", "asr"} for item in selected_modalities):
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


@app.get("/api/thumbnails/{video_id}/{filename}")
def thumbnail(video_id: str, filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="文件名不合法")
    path = settings.thumbnail_dir / video_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="缩略图不存在")
    return FileResponse(path, content_disposition_type="inline")


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
