import json
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.orchestration.retrieval_orchestration import OrchestrationError
from app.platform import context


router = APIRouter()


def _parse_id_list(value: str | None, field_name: str) -> list[str] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} 必须是 JSON 字符串数组") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise HTTPException(status_code=422, detail=f"{field_name} 必须是非空字符串数组")
    return list(dict.fromkeys(item.strip() for item in parsed))


@router.get("/api/orchestration/profiles")
def orchestration_profiles() -> dict:

    return context.search_orchestrator.profiles()


@router.post("/api/search")
async def search(
    query_text: str | None = Form(default=None),
    query_image: UploadFile | None = File(default=None),
    modalities: str = Form(default="visual,face,asr,ocr"),
    video_ids: str | None = Form(default=None),
    folder_ids: str | None = Form(default=None),
    alpha: float = Form(default=0.5),
    limit: int = Form(default=24),
    orchestration_profile: str | None = Form(default=None),
    planner_mode: str = Form(default="auto"),
    reranker_mode: str = Form(default="auto"),
) -> dict:

    selected_modalities = [item.strip() for item in modalities.split(",") if item.strip()]
    if not query_text and not query_image:
        raise HTTPException(status_code=422, detail="请提供查询文字或参考图")
    if any(item not in {"visual", "face", "asr", "ocr"} for item in selected_modalities):
        raise HTTPException(status_code=422, detail="检索通道不合法")
    if planner_mode not in {"auto", "off", "force"}:
        raise HTTPException(status_code=422, detail="planner_mode 必须是 auto、off 或 force")
    if reranker_mode not in {"auto", "off", "force"}:
        raise HTTPException(status_code=422, detail="reranker_mode 必须是 auto、off 或 force")
    requested_video_ids = _parse_id_list(video_ids, "video_ids")
    requested_folder_ids = _parse_id_list(folder_ids, "folder_ids")
    try:
        resolved_video_ids = context.catalog.resolve_video_scope(requested_video_ids, requested_folder_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_path = None
    if query_image and query_image.filename:
        image_path = context.settings.query_dir / (
            f"{uuid.uuid4().hex}{context._safe_suffix(query_image.filename, '.jpg')}"
        )
        await run_in_threadpool(context._save_upload, query_image, image_path)
    try:
        started = time.perf_counter()
        outcome = await run_in_threadpool(
            context.search_orchestrator.search,
            query_text.strip() if query_text else None,
            str(image_path) if image_path else None,
            selected_modalities,
            resolved_video_ids,
            max(0, min(1, alpha)),
            max(1, min(100, limit)),
            profile_name=orchestration_profile,
            planner_mode=planner_mode,
            reranker_mode=reranker_mode,
        )
        results = outcome["results"]
        elapsed_seconds = round(time.perf_counter() - started, 3)
    except OrchestrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)
    return {
        "query": query_text,
        "modalities": selected_modalities,
        "count": len(results),
        "above_count": sum(1 for item in results if item.get("above_threshold")),
        "elapsed_seconds": elapsed_seconds,
        "execution": outcome["execution"],
        "scope": {"folder_ids": requested_folder_ids or [], "video_ids": requested_video_ids or [],
                  "resolved_video_count": len(resolved_video_ids) if resolved_video_ids is not None else len(context.catalog.list_videos())},
        "results": results,
    }
