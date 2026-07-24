"""Milvus client: lifecycle, collection initialisation, and index management.

Instantiated once per process via get_milvus_client() and shared across all
indexers and search calls.  Callers must NOT create MilvusClient inline inside
build functions — pass it via MilvusWriteContext instead.
"""
from __future__ import annotations

import logging
from typing import Optional

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

# Collection name → (schema_factory, index_params)
_COLLECTION_CONFIGS: dict[str, dict] = {
    "visual_embeddings": {
        "schema": create_visual_schema,
        "index": {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    },
    "asr_embeddings": {
        "schema": create_asr_schema,
        "index": {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200},
        },
    },
    "ocr_embeddings": {
        "schema": create_ocr_schema,
        "index": {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200},
        },
    },
    "face_embeddings": {
        "schema": create_face_schema,
        "index": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        },
    },
    "speaker_embeddings": {
        "schema": create_speaker_schema,
        "index": {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    },
}

_COLLECTION_FOR_MODALITY: dict[str, str] = {
    "visual":  "visual_embeddings",
    "asr":     "asr_embeddings",
    "ocr":     "ocr_embeddings",
    "face":    "face_embeddings",
    "speaker": "speaker_embeddings",
}


class MilvusClient:
    """Application-scoped Milvus client.

    Use get_milvus_client() to obtain the singleton; do NOT instantiate directly
    inside indexing workers or search handlers.
    """

    _instance: Optional["MilvusClient"] = None

    def __new__(cls) -> "MilvusClient":
        if cls._instance is None:
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
        connections.connect(alias="default", host=host, port=port)
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
                col.create_index(field_name="embedding", index_params=config["index"])
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


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Return the process-wide MilvusClient, initialising it on first call."""
    global _client
    if _client is None:
        _client = MilvusClient()
    return _client


def reset_milvus_client() -> None:
    """Force re-initialisation (used in tests and after config changes)."""
    global _client
    MilvusClient._instance = None
    _client = None
    get_settings.cache_clear()
