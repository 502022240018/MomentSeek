from fastapi import APIRouter

from app import __version__
from app.deployment import build_deployment_info
from app.schemas import HealthResponse


router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    from app import main as runtime

    settings = runtime.settings
    return {
        "status": "ok",
        "version": __version__,
        "app_version": __version__,
        **build_deployment_info(settings),
        "npu_enabled": settings.npu_enabled,
        "npu_device_id": settings.npu_device_id if settings.npu_enabled else None,
        "cuda_enabled": settings.cuda_enabled,
        "model_idle_policy": settings.model_idle_policy,
        "indexer_mode": settings.indexer_mode,
        "npu_worker_mode": settings.npu_worker_mode,
        "orchestration_enabled": settings.orchestration_enabled,
        "orchestration_profile": settings.orchestration_profile,
        "milvus_enabled": settings.milvus_enabled,
        "milvus_primary": (
            settings.milvus_enabled
            and settings.milvus_read_enabled
            and settings.milvus_rollout_percent == 100
        ),
        "milvus_fallback_enabled": settings.milvus_fallback_enabled,
    }
