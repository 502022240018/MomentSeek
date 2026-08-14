from __future__ import annotations

import sqlite3
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.schemas import FaceGroupLibraryRequest
from app.identity.face_gallery_service import (
    attach_group_to_entity,
    copy_thumbnail_as_reference,
    ensure_group_thumbnail,
    get_face_group,
    published_face_version,
    video_face_groups,
)
from app.platform import context


router = APIRouter()


@router.get("/api/videos/{video_id}/face-gallery")
def get_video_face_gallery(video_id: str) -> dict:
    if not context.catalog.get_video(video_id):
        raise HTTPException(status_code=404, detail="视频不存在")
    try:
        return video_face_groups(
            context.settings.index_dir,
            context.catalog,
            video_id,
            context.settings.face_gallery_cosine_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Milvus 人脸分组不可用: {exc}") from exc


@router.get("/api/videos/{video_id}/face-gallery/{group_idx}/thumbnail")
async def get_face_group_thumbnail(video_id: str, group_idx: int, asset_version: str):
    video = context.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    try:
        if published_face_version(context.settings.index_dir, video_id) != asset_version:
            raise HTTPException(status_code=409, detail="人脸索引已更新，请刷新页面")
        group = get_face_group(video_id, asset_version, group_idx)
        if not group:
            raise HTTPException(status_code=404, detail="人脸分组不存在")
        path = await run_in_threadpool(
            ensure_group_thumbnail, context.settings, video, asset_version, group
        )
        return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline", headers={"Cache-Control": "public, max-age=86400"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/videos/{video_id}/face-gallery/{group_idx}/library", status_code=201)
async def add_face_group_to_library(video_id: str, group_idx: int, request: FaceGroupLibraryRequest) -> dict:
    video = context.catalog.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    if bool(request.entity_id) == bool(request.new_entity_name):
        raise HTTPException(status_code=422, detail="请选择现有人物或填写一个新人物名称")
    created_entity_id: str | None = None
    created_reference_path = None
    try:
        if published_face_version(context.settings.index_dir, video_id) != request.asset_version:
            raise HTTPException(status_code=409, detail="人脸索引已更新，请刷新页面")
        entity_id = request.entity_id
        if request.new_entity_name:
            group = get_face_group(video_id, request.asset_version, group_idx)
            if not group:
                raise HTTPException(status_code=404, detail="人脸分组不存在")
            entity_id = uuid.uuid4().hex
            reference_path = context.settings.app_data_dir / "entities" / f"{entity_id}.jpg"
            created_reference_path = reference_path
            thumbnail = await run_in_threadpool(
                ensure_group_thumbnail, context.settings, video, request.asset_version, group
            )
            await run_in_threadpool(copy_thumbnail_as_reference, thumbnail, reference_path)
            context.catalog.create_entity({
                "id": entity_id, "name": request.new_entity_name,
                "reference_path": str(reference_path), "embedding_path": None,
            })
            created_entity_id = entity_id
            created_reference_path = reference_path
        result = attach_group_to_entity(
            context.catalog, video_id, request.asset_version, group_idx, str(entity_id)
        )
        return result
    except HTTPException:
        raise
    except sqlite3.IntegrityError as exc:
        if created_reference_path:
            created_reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="该人物名称已存在") from exc
    except KeyError as exc:
        if created_entity_id:
            context.catalog.delete_entity(created_entity_id)
            created_reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if created_entity_id:
            context.catalog.delete_entity(created_entity_id)
            created_reference_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"写入人物库失败: {exc}") from exc
