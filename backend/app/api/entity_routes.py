import shutil
import sqlite3
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import EntityUpdateRequest, VoiceOnlyEntityRequest, VoiceSampleRequest
from app.speaker_service import video_speakers


router = APIRouter()


@router.post("/api/entities", status_code=201)
async def create_entity(name: str = Form(...), reference: UploadFile = File(...)) -> dict:
    from app import main as runtime

    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="人物名称不能为空")
    entity_id = uuid.uuid4().hex
    reference_path = runtime.settings.app_data_dir / "entities" / (
        f"{entity_id}{runtime._safe_suffix(reference.filename, '.jpg')}"
    )
    await run_in_threadpool(runtime._save_upload, reference, reference_path)
    try:
        vector = await run_in_threadpool(runtime.search_engine._face().encode_reference, str(reference_path))
    except Exception as exc:
        reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    embedding_path = reference_path.with_suffix(".npz")
    np.savez_compressed(embedding_path, embedding=vector.astype(np.float32))
    try:
        return runtime.catalog.create_entity({
            "id": entity_id,
            "name": name,
            "reference_path": str(reference_path),
            "embedding_path": str(embedding_path),
        })
    except sqlite3.IntegrityError as exc:
        reference_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc


@router.get("/api/entities")
def list_entities() -> list[dict]:
    from app import main as runtime

    return runtime.catalog.list_entities()


@router.get("/api/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:
    from app import main as runtime

    entity = runtime.catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    entity["voice_samples"] = runtime.catalog.list_voice_samples(entity_id)
    return entity


@router.patch("/api/entities/{entity_id}")
def rename_entity(entity_id: str, request: EntityUpdateRequest) -> dict:
    from app import main as runtime

    try:
        if not runtime.catalog.rename_entity(entity_id, request.name):
            raise HTTPException(status_code=404, detail="人物不存在")
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc
    return get_entity(entity_id)


@router.delete("/api/entities/{entity_id}")
def delete_entity(entity_id: str) -> dict:
    from app import main as runtime

    entity = runtime.catalog.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="人物不存在")
    paths = [entity.get("reference_path"), entity.get("embedding_path")]
    paths.extend(sample.get("embedding_path") for sample in runtime.catalog.list_voice_samples(entity_id))
    if not runtime.catalog.delete_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    for value in paths:
        if value:
            Path(value).unlink(missing_ok=True)
    shutil.rmtree(runtime.settings.app_data_dir / "entities" / entity_id, ignore_errors=True)
    return {"status": "deleted", "id": entity_id}


@router.post("/api/entities/voice-only", status_code=201)
def create_voice_only_entity(request: VoiceOnlyEntityRequest) -> dict:
    from app import main as runtime

    try:
        return runtime.catalog.create_entity({
            "id": uuid.uuid4().hex, "name": request.name,
            "reference_path": "", "embedding_path": None,
        })
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc


@router.get("/api/entities/{entity_id}/voice-samples")
def list_entity_voice_samples(entity_id: str) -> list[dict]:
    from app import main as runtime

    if not runtime.catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    samples = runtime.catalog.list_voice_samples(entity_id)
    for sample in samples:
        if sample.get("source_video_id") is None or sample.get("source_utterance_index") is None:
            continue
        try:
            view = video_speakers(runtime.settings.index_dir, runtime.catalog, sample["source_video_id"])
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


@router.post("/api/entities/{entity_id}/voice-samples", status_code=201)
def add_entity_voice_sample(entity_id: str, request: VoiceSampleRequest) -> dict:
    from app import main as runtime
    from app.indexing.speaker import load_speaker_index

    if not runtime.catalog.get_entity(entity_id):
        raise HTTPException(status_code=404, detail="人物不存在")
    path = runtime.settings.index_dir / request.video_id / "speaker.npz"
    try:
        data = load_speaker_index(path)
        vector = data["utterance_embeddings"][request.utterance_index].astype(np.float32)
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="声音片段不存在") from exc
    sample_id = uuid.uuid4().hex
    embedding_path = runtime.settings.app_data_dir / "entities" / entity_id / "voice" / f"{sample_id}.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embedding_path, embedding=vector)
    sample = runtime.catalog.create_voice_sample({
        "id": sample_id, "entity_id": entity_id, "source_type": "video_utterance",
        "source_video_id": request.video_id, "source_utterance_index": request.utterance_index,
        "audio_path": None, "embedding_path": str(embedding_path),
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
    })
    if request.bind_track_id is not None:
        runtime.catalog.bind_speaker_identity(request.video_id, request.bind_track_id, entity_id)
    return sample


@router.get("/api/entities/{entity_id}/reference")
def entity_reference(entity_id: str):
    from app import main as runtime

    entity = runtime.catalog.get_entity(entity_id)
    if not entity or not entity.get("reference_path") or not Path(entity["reference_path"]).is_file():
        raise HTTPException(status_code=404, detail="人物参考图不存在")
    return FileResponse(entity["reference_path"], content_disposition_type="inline")
