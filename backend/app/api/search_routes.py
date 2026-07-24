import json
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.retrieval_orchestration import OrchestrationError


router = APIRouter()


@router.get("/api/orchestration/profiles")
def orchestration_profiles() -> dict:
    from app import main as runtime

    return runtime.search_orchestrator.profiles()


@router.post("/api/search")
async def search(
    query_text: str | None = Form(default=None),
    query_image: UploadFile | None = File(default=None),
    modalities: str = Form(default="visual,face,asr,ocr"),
    video_ids: str | None = Form(default=None),
    alpha: float = Form(default=0.5),
    limit: int = Form(default=24),
    orchestration_profile: str | None = Form(default=None),
    planner_mode: str = Form(default="auto"),
    reranker_mode: str = Form(default="auto"),
) -> dict:
    from app import main as runtime

    selected_modalities = [item.strip() for item in modalities.split(",") if item.strip()]
    if not query_text and not query_image:
        raise HTTPException(status_code=422, detail="请提供查询文字或参考图")
    if any(item not in {"visual", "face", "asr", "ocr"} for item in selected_modalities):
        raise HTTPException(status_code=422, detail="检索通道不合法")
    if planner_mode not in {"auto", "off", "force"}:
        raise HTTPException(status_code=422, detail="planner_mode 必须是 auto、off 或 force")
    if reranker_mode not in {"auto", "off", "force"}:
        raise HTTPException(status_code=422, detail="reranker_mode 必须是 auto、off 或 force")
    image_path = None
    if query_image and query_image.filename:
        image_path = runtime.settings.query_dir / (
            f"{uuid.uuid4().hex}{runtime._safe_suffix(query_image.filename, '.jpg')}"
        )
        await run_in_threadpool(runtime._save_upload, query_image, image_path)
    try:
        started = time.perf_counter()
        outcome = await run_in_threadpool(
            runtime.search_orchestrator.search,
            query_text.strip() if query_text else None,
            str(image_path) if image_path else None,
            selected_modalities,
            json.loads(video_ids) if video_ids else None,
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
        "results": results,
    }
