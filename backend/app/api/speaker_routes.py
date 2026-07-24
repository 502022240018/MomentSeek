import json
import subprocess
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import SpeakerUpdateRequest, UtteranceUpdateRequest, VoiceSearchRequest
from app.speaker_service import video_speakers, voice_search, voice_search_vectors


router = APIRouter()


@router.get("/api/videos/{video_id}/speakers")
def get_video_speakers(video_id: str) -> dict:
    from app import main as runtime

    if not runtime.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    try:
        return video_speakers(runtime.settings.index_dir, runtime.catalog, video_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/videos/{video_id}/speakers/{track_id}")
def update_video_speaker(video_id: str, track_id: int, request: SpeakerUpdateRequest) -> dict:
    from app import main as runtime

    if not runtime.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    runtime.catalog.upsert_video_speaker(video_id, track_id, **request.model_dump())
    return get_video_speakers(video_id)


@router.patch("/api/videos/{video_id}/utterances/{utterance_index}")
def update_video_utterance(video_id: str, utterance_index: int, request: UtteranceUpdateRequest) -> dict:
    from app import main as runtime

    if not runtime.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    runtime.catalog.upsert_utterance_override(
        video_id, utterance_index, request.corrected_track_id, request.searchable
    )
    return get_video_speakers(video_id)


@router.post("/api/voice-search")
def search_voice(request: VoiceSearchRequest) -> dict:
    from app import main as runtime

    try:
        results = voice_search(
            runtime.settings.index_dir,
            runtime.catalog,
            query_video_id=request.query_video_id,
            query_utterance_index=request.query_utterance_index,
            video_ids=request.video_ids,
            limit=request.limit,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(results), "results": results}


@router.post("/api/voice-search/upload")
async def search_voice_upload(
    reference: UploadFile = File(...),
    video_ids: str | None = Form(default=None),
    limit: int = Form(default=50),
) -> dict:
    from app import main as runtime
    from app.indexing.speaker import encode_voice_query

    settings = runtime.settings
    source_path = settings.query_dir / f"{uuid.uuid4().hex}{runtime._safe_suffix(reference.filename, '.wav')}"
    wav_path = source_path.with_suffix(".voice.wav")
    await run_in_threadpool(runtime._save_upload, reference, source_path)
    try:
        process = await run_in_threadpool(
            subprocess.run,
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise ValueError(process.stderr.strip() or "无法读取上传声音")
        vectors = await run_in_threadpool(
            encode_voice_query,
            str(wav_path),
            model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
            model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
            device=settings.speaker_device,
        )
        results = await run_in_threadpool(
            voice_search_vectors,
            settings.index_dir,
            runtime.catalog,
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
