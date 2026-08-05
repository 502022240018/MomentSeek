import sqlite3

from fastapi import APIRouter, HTTPException

from app.api.schemas import FolderCreateRequest, FolderRenameRequest, VideoFolderMembershipRequest
from app.catalog.db import Catalog
from app.platform import context


router = APIRouter()


def _reject_default_name(name: str) -> None:
    if name.casefold() == Catalog.DEFAULT_FOLDER_NAME.casefold():
        raise HTTPException(status_code=409, detail="默认文件夹为系统保留名称")


@router.get("/api/folders")
def list_folders() -> list[dict]:
    return context.catalog.list_folders()


@router.post("/api/folders", status_code=201)
def create_folder(request: FolderCreateRequest) -> dict:
    _reject_default_name(request.name)
    try:
        return context.catalog.create_folder(request.name)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="已存在同名文件夹") from exc


@router.patch("/api/folders/{folder_id}")
def rename_folder(folder_id: str, request: FolderRenameRequest) -> dict:
    if folder_id == Catalog.DEFAULT_FOLDER_ID:
        raise HTTPException(status_code=409, detail="默认文件夹不能重命名")
    _reject_default_name(request.name)
    try:
        changed = context.catalog.rename_folder(folder_id, request.name)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="已存在同名文件夹") from exc
    if not changed:
        raise HTTPException(status_code=404, detail="文件夹不存在")
    return context.catalog.get_folder(folder_id)


@router.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str) -> dict:
    if folder_id == Catalog.DEFAULT_FOLDER_ID:
        raise HTTPException(status_code=409, detail="默认文件夹不能删除")
    released_count = context.catalog.delete_folder(folder_id)
    if released_count is None:
        raise HTTPException(status_code=404, detail="文件夹不存在")
    return {"status": "deleted", "id": folder_id, "released_video_count": released_count}


@router.post("/api/videos/folders")
def update_video_folders(request: VideoFolderMembershipRequest) -> dict:
    try:
        context.catalog.update_video_folders(request.video_ids, request.folder_ids, request.operation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "ok", "operation": request.operation}
