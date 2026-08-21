from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.orchestration.retrieval_orchestration import OrchestrationError
from app.identity.speaker_service import (
    SpeakerMilvusCoverageError,
    entity_voice_embeddings,
    resolve_voice_reference_vectors,
)
from app.orchestration.snapmind_lab import CandidatePlan, SnapMindPlannerLab, VoiceReference
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


def _voice_reference(
    value: str | None,
    upload: UploadFile | None,
    *,
    require_upload: bool,
) -> VoiceReference | None:
    if value is None:
        if upload and upload.filename:
            return VoiceReference(kind="upload", label=upload.filename)
        return None
    try:
        reference = VoiceReference.model_validate_json(value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"声音引用格式不合法: {exc.errors()[0]['msg']}",
        ) from exc
    has_upload = bool(upload and upload.filename)
    if reference.kind == "upload" and require_upload and not has_upload:
        raise HTTPException(status_code=422, detail="upload 声音引用必须附带音频文件")
    if reference.kind != "upload" and has_upload:
        raise HTTPException(status_code=422, detail="一次只能选择一种声音引用")
    if not getattr(context.settings, "planner_voice_search_enabled", True):
        raise HTTPException(status_code=404, detail="Planner 声纹工具未启用")
    if reference.kind == "entity":
        try:
            entity_voice_embeddings(context.catalog, reference.entity_id or "")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    elif reference.kind == "utterance":
        publication = context.catalog.get_modality_publication(
            reference.video_id or "",
            "speaker",
        )
        if not publication or publication.get("status") != "ready":
            raise HTTPException(status_code=422, detail="所选视频没有可用的说话人索引")
    return reference


def _voice_exclude(reference: VoiceReference | None) -> tuple[str, int] | None:
    if (
        reference is None
        or reference.kind != "utterance"
        or reference.utterance_index is None
    ):
        return None
    return reference.video_id or "", reference.utterance_index


@router.get("/capabilities")
def capabilities() -> dict:
    return planner_lab.capabilities()


@router.post("/plans")
async def propose_plans(
    query_text: str = Form(...),
    query_image: UploadFile | None = File(default=None),
    query_audio: UploadFile | None = File(default=None),
    voice_reference: str | None = Form(default=None),
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
    trusted_voice_reference = _voice_reference(
        voice_reference,
        query_audio,
        require_upload=False,
    )
    try:
        outcome = await run_in_threadpool(
            planner_lab.propose,
            query=query,
            mode=mode,
            video_ids=resolved,
            has_query_image=bool(query_image and query_image.filename),
            profile_name=orchestration_profile,
            voice_reference=trusted_voice_reference,
        )
    except OrchestrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {**outcome, "scope": scope}


@router.post("/execute")
async def execute_plan(
    query_text: str = Form(...),
    plan: str = Form(...),
    query_image: UploadFile | None = File(default=None),
    query_audio: UploadFile | None = File(default=None),
    voice_reference: str | None = Form(default=None),
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
    trusted_voice_reference = _voice_reference(
        voice_reference,
        query_audio,
        require_upload=True,
    )
    image_path = None
    audio_path = None
    if query_image and query_image.filename:
        image_path = context.settings.query_dir / (
            f"{uuid.uuid4().hex}{context._safe_suffix(query_image.filename, '.jpg')}"
        )
        await run_in_threadpool(context._save_upload, query_image, image_path)
    if query_audio and query_audio.filename:
        audio_path = context.settings.query_dir / (
            f"{uuid.uuid4().hex}{context._safe_suffix(query_audio.filename, '.wav')}"
        )
        await run_in_threadpool(context._save_upload, query_audio, audio_path)
    try:
        voice_vectors = None
        if trusted_voice_reference is not None:
            voice_vectors = await run_in_threadpool(
                resolve_voice_reference_vectors,
                context.catalog,
                context.settings,
                trusted_voice_reference.model_dump(),
                upload_path=audio_path,
            )
        outcome = await run_in_threadpool(
            planner_lab.execute,
            query=query,
            image_path=str(image_path) if image_path else None,
            plan=candidate,
            video_ids=resolved,
            max_steps=(
                max(1, min(max_steps, len(candidate.steps)))
                if max_steps is not None
                else None
            ),
            voice_vectors=voice_vectors,
            voice_exclude=_voice_exclude(trusted_voice_reference),
        )
    except (OrchestrationError, SpeakerMilvusCoverageError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)
        if audio_path:
            audio_path.unlink(missing_ok=True)
    return {**outcome, "scope": scope}
