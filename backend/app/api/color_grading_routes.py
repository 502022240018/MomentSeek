from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.color_grading import MAX_REFERENCE_IMAGE_BYTES, ColorGradingError

router = APIRouter()


def _save_reference_image(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    try:
        with destination.open("wb") as target:
            while chunk := upload.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_REFERENCE_IMAGE_BYTES:
                    raise ValueError("参考图片不能超过 25 MB")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


@router.get("/api/color-grading/status")
async def color_grading_status() -> dict:
    from app import main as runtime

    return await run_in_threadpool(runtime._color_grading_manager().capability)


@router.post("/api/color-grading/tasks", status_code=202)
async def create_color_grading_task(
    input_video_id: str = Form(...),
    reference_type: str = Form(...),
    ncc: bool = Form(default=False),
    ref_video_id: str | None = Form(default=None),
    ref_image: UploadFile | None = File(default=None),  # noqa: B008
) -> dict:
    from app import main as runtime

    manager = runtime._color_grading_manager()
    capability = await run_in_threadpool(manager.capability)
    if not capability["enabled"]:
        raise HTTPException(status_code=503, detail="当前部署未启用视频仿色")
    if not capability["available"]:
        raise HTTPException(
            status_code=503,
            detail=capability["reason"] or "仿色服务尚未就绪",
        )
    if not runtime.catalog.get_video(input_video_id):
        raise HTTPException(status_code=404, detail="原视频不存在")

    reference_type = reference_type.strip().lower()
    if reference_type not in {"image", "video"}:
        raise HTTPException(
            status_code=422,
            detail="reference_type 只能是 image 或 video",
        )
    task_id = uuid.uuid4().hex
    reference_video_id = None
    reference_image_path = None
    if reference_type == "image":
        if ref_image is None or not ref_image.filename:
            raise HTTPException(status_code=422, detail="请选择参考图片")
        if ref_video_id:
            raise HTTPException(
                status_code=422,
                detail="参考图片和参考视频不能同时提交",
            )
        suffix = runtime._safe_suffix(ref_image.filename, ".jpg")
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(
                status_code=422,
                detail="参考图片只支持 JPG、PNG 或 WebP",
            )
        reference_path = (
            runtime.settings.color_grading_reference_dir
            / task_id
            / f"reference{suffix}"
        )
        try:
            await run_in_threadpool(
                _save_reference_image,
                ref_image,
                reference_path,
            )
            await run_in_threadpool(manager.validate_reference_image, reference_path)
        except ValueError as error:
            shutil.rmtree(reference_path.parent, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        reference_image_path = str(reference_path.resolve())
    else:
        if ref_image is not None and ref_image.filename:
            raise HTTPException(
                status_code=422,
                detail="参考图片和参考视频不能同时提交",
            )
        if not ref_video_id:
            raise HTTPException(status_code=422, detail="请选择参考视频")
        if ref_video_id == input_video_id:
            raise HTTPException(
                status_code=422,
                detail="原视频和参考视频不能是同一个文件",
            )
        if not runtime.catalog.get_video(ref_video_id):
            raise HTTPException(status_code=404, detail="参考视频不存在")
        reference_video_id = ref_video_id

    runtime.catalog.create_color_grading_task(
        {
            "id": task_id,
            "input_video_id": input_video_id,
            "reference_type": reference_type,
            "reference_video_id": reference_video_id,
            "reference_image_path": reference_image_path,
            "ncc": ncc,
            "status": "submitting",
            "stage": "submitting",
        }
    )
    return await run_in_threadpool(manager.submit, task_id)


@router.get("/api/color-grading/tasks")
async def list_color_grading_tasks() -> list[dict]:
    from app import main as runtime

    return await run_in_threadpool(runtime._color_grading_manager().list)


@router.get("/api/color-grading/tasks/{task_id}")
async def get_color_grading_task(task_id: str) -> dict:
    from app import main as runtime

    try:
        return await run_in_threadpool(runtime._color_grading_manager().sync, task_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error


@router.get("/api/color-grading/tasks/{task_id}/media")
def color_grading_media(task_id: str):
    from app import main as runtime

    try:
        path = runtime._color_grading_manager().media_path(task_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error
    except (ColorGradingError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
        content_disposition_type="inline",
    )


@router.get("/api/color-grading/tasks/{task_id}/lut")
def color_grading_lut(task_id: str):
    from app import main as runtime

    try:
        path = runtime._color_grading_manager().lut_path(task_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error
    except (ColorGradingError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{task_id}.cube",
    )


@router.get("/api/color-grading/tasks/{task_id}/reference")
def color_grading_reference(task_id: str):
    from app import main as runtime

    try:
        path = runtime._color_grading_manager().reference_path(task_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error
    except (ColorGradingError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@router.post("/api/color-grading/tasks/{task_id}/import", status_code=201)
async def import_color_grading_result(task_id: str) -> dict:
    from app import main as runtime

    try:
        return await run_in_threadpool(
            runtime._color_grading_manager().import_result,
            task_id,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error
    except (ColorGradingError, ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
