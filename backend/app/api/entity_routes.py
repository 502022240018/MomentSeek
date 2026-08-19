import shutil
import sqlite3
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.schemas import EntityUpdateRequest, VoiceOnlyEntityRequest, VoiceSampleRequest
from app.identity.speaker_service import SpeakerMilvusCoverageError, video_speakers
from app.identity.face_gallery_service import delete_entity_face_samples
from app.vector_store.milvus.milvus_client import get_milvus_client
from app.vector_store.milvus.milvus_schema import entity_face_sample_pk
from app.platform import context


router = APIRouter()


@router.post("/api/entities", status_code=201)
async def create_entity(name: str = Form(...), reference: UploadFile = File(...)) -> dict:

    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="人物名称不能为空")
    entity_id = uuid.uuid4().hex
    reference_path = context.settings.app_data_dir / "entities" / (
        f"{entity_id}{context._safe_suffix(reference.filename, '.jpg')}"
    )
    await run_in_threadpool(context._save_upload, reference, reference_path)
    try:
        vector = await run_in_threadpool(context.search_engine._face().encode_reference, str(reference_path))
    except Exception as exc:
        reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sample_id = uuid.uuid4().hex
    try:
        entity = context.catalog.create_entity({
            "id": entity_id,
            "name": name,
            "reference_path": str(reference_path),
            "embedding_path": None,
        })
        collection = get_milvus_client().collection("entity_face_samples")
        collection.upsert([{
            "pk": entity_face_sample_pk(entity_id, sample_id),
            "entity_id": entity_id, "sample_id": sample_id,
            "source_video_id": "", "source_asset_version": "",
            "source_group_idx": -1, "quality": 1.0,
            "embedding": vector.astype(np.float32).tolist(),
        }])
        collection.flush()
        return entity
    except sqlite3.IntegrityError as exc:
        reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc
    except Exception:
        context.catalog.delete_entity(entity_id)
        reference_path.unlink(missing_ok=True)
        raise


@router.get("/api/entities")
def list_entities() -> list[dict]:

    return context.catalog.list_entities()


@router.get("/api/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:

    entity = context.catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    entity["voice_samples"] = context.catalog.list_voice_samples(entity_id)
    return entity


@router.patch("/api/entities/{entity_id}")
def rename_entity(entity_id: str, request: EntityUpdateRequest) -> dict:

    try:
        if not context.catalog.rename_entity(entity_id, request.name):
            raise HTTPException(status_code=404, detail="人物不存在")
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc
    return get_entity(entity_id)


@router.delete("/api/entities/{entity_id}")
def delete_entity(entity_id: str) -> dict:

    entity = context.catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    paths = [entity.get("reference_path"), entity.get("embedding_path")]
    paths.extend(sample.get("embedding_path") for sample in context.catalog.list_voice_samples(entity_id))
    try:
        delete_entity_face_samples(entity_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Milvus 人脸样本删除失败: {exc}") from exc
    if not context.catalog.delete_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    for value in paths:
        if value:
            Path(value).unlink(missing_ok=True)
    shutil.rmtree(context.settings.app_data_dir / "entities" / entity_id, ignore_errors=True)
    return {"status": "deleted", "id": entity_id}


@router.post("/api/entities/voice-only", status_code=201)
def create_voice_only_entity(request: VoiceOnlyEntityRequest) -> dict:

    try:
        return context.catalog.create_entity({
            "id": uuid.uuid4().hex, "name": request.name,
            "reference_path": "", "embedding_path": None,
        })
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc


@router.get("/api/entities/{entity_id}/voice-samples")
def list_entity_voice_samples(entity_id: str) -> list[dict]:

    if not context.catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    samples = context.catalog.list_voice_samples(entity_id)
    # Multiple samples routinely share one source video; build each video's
    # speaker panel at most once (avoid the previous per-sample N+1 rebuild).
    # Cache failures as None too, so a failing video is not retried per sample.
    panels: dict[str, dict | None] = {}
    for sample in samples:
        if sample.get("source_video_id") is None or sample.get("source_utterance_index") is None:
            continue
        video_id = sample["source_video_id"]
        if video_id not in panels:
            try:
                panels[video_id] = video_speakers(context.catalog, video_id)
            except (SpeakerMilvusCoverageError, FileNotFoundError, IndexError, ValueError):
                panels[video_id] = None
        view = panels[video_id]
        if view is None:
            continue
        utterance = next(
            (item for item in view["utterances"] if item["index"] == int(sample["source_utterance_index"])),
            None,
        )
        if utterance:
            sample["clip_url"] = utterance["clip_url"]
            sample["text"] = utterance["text"]
    return samples


@router.post("/api/entities/{entity_id}/voice-samples", status_code=201)
def add_entity_voice_sample(entity_id: str, request: VoiceSampleRequest) -> dict:
    from app.identity.speaker_service import speaker_utterance_embedding

    if not context.catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    try:
        vector = speaker_utterance_embedding(
            context.catalog,
            request.video_id,
            request.utterance_index,
        )
    except SpeakerMilvusCoverageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="声音片段不存在") from exc
    sample_id = uuid.uuid4().hex
    voice_embedding = np.asarray(vector, dtype=np.float32).tobytes(order="C")
    sample = context.catalog.create_voice_sample({
        "id": sample_id, "entity_id": entity_id, "source_type": "video_utterance",
        "source_video_id": request.video_id, "source_utterance_index": request.utterance_index,
        "audio_path": None, "embedding_path": "", "voice_embedding": voice_embedding,
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
    })
    if request.bind_track_id is not None:
        context.catalog.bind_speaker_identity(request.video_id, request.bind_track_id, entity_id)
    return sample


@router.get("/api/entities/{entity_id}/reference")
def entity_reference(entity_id: str):

    entity = context.catalog.get_entity(entity_id)
    if not entity or not entity.get("reference_path") or not Path(entity["reference_path"]).is_file():
        raise HTTPException(status_code=404, detail="人物参考图不存在")
    return FileResponse(entity["reference_path"], content_disposition_type="inline")
