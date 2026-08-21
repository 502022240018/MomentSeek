import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import SpeakerUpdateRequest, UtteranceUpdateRequest, VoiceSearchRequest
from app.identity.speaker_service import (
    SpeakerMilvusCoverageError,
    encode_voice_reference_file,
    video_speakers,
    voice_search,
    voice_search_vectors,
)
from app.platform import context


router = APIRouter()


@router.get("/api/videos/{video_id}/speakers")
def get_video_speakers(video_id: str) -> dict:

    if not context.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    try:
        return video_speakers(context.catalog, video_id)
    except SpeakerMilvusCoverageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/videos/{video_id}/speakers/{track_id}")
def update_video_speaker(video_id: str, track_id: int, request: SpeakerUpdateRequest) -> dict:

    if not context.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    context.catalog.upsert_video_speaker(video_id, track_id, **request.model_dump())
    return get_video_speakers(video_id)


@router.patch("/api/videos/{video_id}/utterances/{utterance_index}")
def update_video_utterance(video_id: str, utterance_index: int, request: UtteranceUpdateRequest) -> dict:

    if not context.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    context.catalog.upsert_utterance_override(
        video_id, utterance_index, request.corrected_track_id, request.searchable
    )
    return get_video_speakers(video_id)


@router.post("/api/voice-search")
def search_voice(request: VoiceSearchRequest) -> dict:

    try:
        results = voice_search(
            context.catalog,
            query_video_id=request.query_video_id,
            query_utterance_index=request.query_utterance_index,
            video_ids=request.video_ids,
            limit=request.limit,
        )
    except SpeakerMilvusCoverageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Catch-all: Milvus exceptions (MilvusServiceError, MilvusException),
        # RuntimeError from index-type drift check, etc. — surfaces as 503
        # rather than a raw 500 with no body.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"count": len(results), "results": results}


@router.post("/api/voice-search/upload")
async def search_voice_upload(
    reference: UploadFile = File(...),
    video_ids: str | None = Form(default=None),
    limit: int = Form(default=50),
) -> dict:
    settings = context.settings
    source_path = settings.query_dir / f"{uuid.uuid4().hex}{context._safe_suffix(reference.filename, '.wav')}"
    await run_in_threadpool(context._save_upload, reference, source_path)
    try:
        vectors = await run_in_threadpool(
            encode_voice_reference_file,
            settings,
            source_path,
        )
        results = await run_in_threadpool(
            voice_search_vectors,
            context.catalog,
            query_vectors=vectors,
            video_ids=json.loads(video_ids) if video_ids else None,
            limit=max(1, min(200, limit)),
        )
        return {"query_samples": len(vectors), "count": len(results), "results": results}
    except SpeakerMilvusCoverageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Milvus service failures (MilvusServiceError, MilvusException),
        # index-type drift check, RuntimeError from encoder, etc. —
        # surfaces as 503 not 400, mirroring the /api/voice-search route.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        source_path.unlink(missing_ok=True)
