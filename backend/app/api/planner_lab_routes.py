from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.orchestration.retrieval_orchestration import OrchestrationError
from app.orchestration.snapmind_lab import CandidatePlan, SnapMindPlannerLab
from app.platform import context


router = APIRouter(prefix="/api/planner-lab", tags=["planner-lab"])
planner_lab = SnapMindPlannerLab(context.search_orchestrator)


def _parse_ids(value: str | None, field_name: str) -> list[str] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} 必须是 JSON 字符串数组") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise HTTPException(status_code=422, detail=f"{field_name} 必须是非空字符串数组")
    return list(dict.fromkeys(item.strip() for item in parsed)) or None


def _scope(video_ids: str | None, folder_ids: str | None) -> tuple[list[str] | None, dict]:
    requested_videos = _parse_ids(video_ids, "video_ids")
    requested_folders = _parse_ids(folder_ids, "folder_ids")
    try:
        resolved = context.catalog.resolve_video_scope(requested_videos, requested_folders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return resolved, {
        "video_ids": requested_videos or [],
        "folder_ids": requested_folders or [],
        "resolved_video_count": len(resolved) if resolved is not None else len(context.catalog.list_videos()),
    }


def _ensure_enabled() -> None:
    if not getattr(context.settings, "planner_lab_enabled", True):
        raise HTTPException(status_code=404, detail="Planner Lab 未启用")


@router.get("/capabilities")
def capabilities() -> dict:
    return planner_lab.capabilities()


@router.post("/plans")
async def propose_plans(
    query_text: str = Form(...),
    query_image: UploadFile | None = File(default=None),
    video_ids: str | None = Form(default=None),
    folder_ids: str | None = Form(default=None),
    mode: str = Form(default="assist"),
    orchestration_profile: str | None = Form(default=None),
) -> dict:
    _ensure_enabled()
    query = query_text.strip()
    if not query:
        raise HTTPException(status_code=422, detail="请输入查询文字")
    if mode not in {"guide", "assist", "auto"}:
        raise HTTPException(status_code=422, detail="mode 必须是 guide、assist 或 auto")
    resolved, scope = _scope(video_ids, folder_ids)
    try:
        outcome = await run_in_threadpool(
            planner_lab.propose,
            query,
            mode,
            resolved,
            bool(query_image and query_image.filename),
            orchestration_profile,
        )
    except OrchestrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {**outcome, "scope": scope}


@router.post("/execute")
async def execute_plan(
    query_text: str = Form(...),
    plan: str = Form(...),
    query_image: UploadFile | None = File(default=None),
    video_ids: str | None = Form(default=None),
    folder_ids: str | None = Form(default=None),
    max_steps: int | None = Form(default=None),
) -> dict:
    _ensure_enabled()
    query = query_text.strip()
    if not query:
        raise HTTPException(status_code=422, detail="请输入查询文字")
    try:
        candidate = CandidatePlan.model_validate_json(plan)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"计划格式不合法: {exc.errors()[0]['msg']}") from exc
    resolved, scope = _scope(video_ids, folder_ids)
    image_path = None
    if query_image and query_image.filename:
        image_path = context.settings.query_dir / (
            f"{uuid.uuid4().hex}{context._safe_suffix(query_image.filename, '.jpg')}"
        )
        await run_in_threadpool(context._save_upload, query_image, image_path)
    try:
        outcome = await run_in_threadpool(
            planner_lab.execute,
            query,
            str(image_path) if image_path else None,
            candidate,
            resolved,
            max(1, min(max_steps, len(candidate.steps))) if max_steps is not None else None,
        )
    except OrchestrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)
    return {**outcome, "scope": scope}
