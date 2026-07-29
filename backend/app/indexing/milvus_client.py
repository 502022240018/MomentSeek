"""Milvus client: lifecycle, collection initialisation, and index management.

Instantiated once per process via get_milvus_client() and shared across all
indexers and search calls.  Callers must NOT create MilvusClient inline inside
build functions — pass it via MilvusWriteContext instead.
"""
from __future__ import annotations

import logging
import socket
import threading

from pymilvus import Collection, CollectionSchema, connections, utility

from app.settings import get_settings

from .milvus_schema import (
    create_asr_schema,
    create_face_schema,
    create_ocr_schema,
    create_speaker_schema,
    create_visual_schema,
)

logger = logging.getLogger(__name__)


def _get_visual_index_config() -> dict:
    """Get visual index config dynamically (HNSW or DiskANN based on settings).

    Called at runtime to support dynamic configuration changes.
    """
    settings = get_settings()
    if settings.visual_use_diskann:
        return {
            "index_type": "DISKANN",
            "metric_type": "COSINE",
            "params": {
                "max_degree": 56,              # 图的最大度数（必需）
                "search_list_size": 128,       # 构建时搜索列表大小（必需）
                "pq_code_budget_gb": 0.125,    # PQ压缩预算（可选）
                "build_dram_budget_gb": 32.0,  # 构建时内存预算（可选）
            },
        }
    else:
        # 默认使用HNSW
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }


# Static index configs for non-visual modalities
_STATIC_INDEX_CONFIGS: dict[str, dict] = {
    "asr_embeddings": {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    },
    "ocr_embeddings": {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    },
    "face_embeddings": {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    },
    "speaker_embeddings": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
}


def get_collection_index_config(collection_name: str) -> dict:
    """Get index config for a collection (dynamic for visual, static for others).

    This function is called at runtime to support dynamic configuration changes,
    particularly for visual_embeddings which can switch between DISKANN and HNSW.

    Args:
        collection_name: Collection name (e.g., "visual_embeddings")

    Returns:
        Index configuration dict with index_type, metric_type, and params
    """
    if collection_name == "visual_embeddings":
        return _get_visual_index_config()
    return _STATIC_INDEX_CONFIGS[collection_name]


# Collection name → (schema_factory, index_params)
# Note: For visual_embeddings, use get_collection_index_config() at runtime
_COLLECTION_CONFIGS: dict[str, dict] = {
    "visual_embeddings": {
        "schema": create_visual_schema,
        "index": None,  # Placeholder - use get_collection_index_config() at runtime
    },
    "asr_embeddings": {
        "schema": create_asr_schema,
        "index": _STATIC_INDEX_CONFIGS["asr_embeddings"],
    },
    "ocr_embeddings": {
        "schema": create_ocr_schema,
        "index": _STATIC_INDEX_CONFIGS["ocr_embeddings"],
    },
    "face_embeddings": {
        "schema": create_face_schema,
        "index": _STATIC_INDEX_CONFIGS["face_embeddings"],
    },
    "speaker_embeddings": {
        "schema": create_speaker_schema,
        "index": _STATIC_INDEX_CONFIGS["speaker_embeddings"],
    },
}

_COLLECTION_FOR_MODALITY: dict[str, str] = {
    "visual":  "visual_embeddings",
    "asr":     "asr_embeddings",
    "ocr":     "ocr_embeddings",
    "face":    "face_embeddings",
    "speaker": "speaker_embeddings",
}


def ensure_milvus_reachable() -> None:
    """Fail fast before PyMilvus enters its longer gRPC reconnect loop."""
    settings = get_settings()
    timeout = min(0.5, settings.milvus_query_timeout_seconds)
    try:
        with socket.create_connection(
            (settings.milvus_host, settings.milvus_port),
            timeout=timeout,
        ):
            return
    except OSError as exc:
        raise ConnectionError(
            f"Milvus is unreachable at "
            f"{settings.milvus_host}:{settings.milvus_port}: {exc}"
        ) from exc


class MilvusClient:
    """Application-scoped Milvus client.

    Use get_milvus_client() to obtain the singleton; do NOT instantiate directly
    inside indexing workers or search handlers.
    """

    _instance: MilvusClient | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> MilvusClient:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:  # double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        s = get_settings()
        host = s.milvus_host
        port = str(s.milvus_port)
        logger.info("Connecting to Milvus at %s:%s", host, port)
        connections.connect(
            alias="default",
            host=host,
            port=port,
            timeout=s.milvus_query_timeout_seconds,
        )
        self._ready = True
        self._init_collections()
        logger.info("MilvusClient ready — %d collections", len(_COLLECTION_CONFIGS))

    # ------------------------------------------------------------------
    # Collection init
    # ------------------------------------------------------------------

    def _init_collections(self) -> None:
        for name, config in _COLLECTION_CONFIGS.items():
            if not utility.has_collection(name):
                logger.info("Creating collection: %s", name)
                schema: CollectionSchema = config["schema"]()
                col = Collection(name=name, schema=schema, consistency_level="Strong")
                # Get index config dynamically for runtime support
                index_config = get_collection_index_config(name) if name == "visual_embeddings" else config["index"]
                col.create_index(field_name="embedding", index_params=index_config)
                col.load()
                logger.info("Collection %s created and loaded", name)
            else:
                col = Collection(name)
                load_state = utility.load_state(name)
                if load_state.name != "Loaded":
                    logger.info("Loading existing collection: %s", name)
                    col.load()
                else:
                    logger.debug("Collection %s already loaded", name)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def collection(self, name: str) -> Collection:
        return Collection(name)

    def collection_for(self, modality: str) -> Collection:
        name = _COLLECTION_FOR_MODALITY[modality]
        return Collection(name)

    def stats(self, name: str) -> dict:
        col = Collection(name)
        load_state = utility.load_state(name)
        return {
            "name": name,
            "num_entities": col.num_entities,
            "loaded": load_state.name == "Loaded",
        }

    def health_check(self) -> bool:
        try:
            utility.list_collections()
            return True
        except Exception as exc:
            logger.error("Milvus health check failed: %s", exc)
            return False

    def delete_video(self, video_id: str) -> dict[str, int]:
        """Delete all records for a video across every collection.

        Returns a dict of {collection_name: deleted_count}.
        Safe to call even when the video has no records (no-op).
        """
        counts: dict[str, int] = {}
        expr = f'video_id == "{video_id}"'
        for name in _COLLECTION_CONFIGS:
            col = Collection(name)
            try:
                result = col.delete(expr)
                col.flush()
                counts[name] = getattr(result, "delete_count", 0)
            except Exception as exc:
                logger.warning("delete_video %s from %s failed: %s", video_id, name, exc)
                counts[name] = -1
        return counts

    def delete_video_version(self, video_id: str, asset_version: str) -> dict[str, int]:
        """Delete only records for a specific (video_id, asset_version) pair.

        Used by the safe version-switch flow: write new version → validate →
        call delete_video_version(old_asset_ver) → switch current version pointer.
        """
        counts: dict[str, int] = {}
        expr = f'video_id == "{video_id}" and asset_version == "{asset_version}"'
        for name in _COLLECTION_CONFIGS:
            col = Collection(name)
            try:
                result = col.delete(expr)
                col.flush()
                counts[name] = getattr(result, "delete_count", 0)
            except Exception as exc:
                logger.warning(
                    "delete_video_version %s@%s from %s failed: %s",
                    video_id, asset_version, name, exc,
                )
                counts[name] = -1
        return counts

    def delete_video_modality(self, video_id: str, modality: str) -> int:
        """Delete all records for a video from a single modality's collection.

        Used before re-indexing a specific stage to prevent orphan records when
        the new index produces fewer rows than the original (e.g. shorter video,
        different sample rate, or model change).  Safe to call on an empty
        collection — returns 0 without error.

        **Deletion scope**: Removes records for *all* ``asset_version`` values
        associated with this ``video_id`` in the given modality.  If you need
        version-scoped cleanup (e.g. keeping version "2" while removing version
        "1"), use ``delete_video_version()`` instead.

        Returns:
            Number of records deleted, or -1 on failure.
        """
        name = _COLLECTION_FOR_MODALITY[modality]
        col = Collection(name)
        expr = f'video_id == "{video_id}"'
        try:
            result = col.delete(expr)
            col.flush()
            count = getattr(result, "delete_count", 0)
            logger.info(
                "delete_video_modality video=%s modality=%s deleted=%d",
                video_id, modality, count,
            )
            return count
        except Exception as exc:
            logger.warning(
                "delete_video_modality %s/%s failed: %s", video_id, modality, exc
            )
            return -1

    def count_video_modality(self, video_id: str, modality: str) -> int:
        """Return the persisted row count for one video and modality."""
        name = _COLLECTION_FOR_MODALITY[modality]
        rows = Collection(name).query(
            expr=f'video_id == "{video_id}"',
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_client: MilvusClient | None = None
_client_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """Return the process-wide MilvusClient, initialising it on first call.

    Thread-safe: uses double-checked locking so concurrent callers do not
    race to create duplicate connections.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-checked locking
                _client = MilvusClient()
    return _client


def reset_milvus_client() -> None:
    """Force re-initialisation (used in tests and after config changes)."""
    global _client
    MilvusClient._instance = None
    _client = None
    get_settings.cache_clear()
