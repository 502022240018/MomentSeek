"""Milvus client: lifecycle, collection initialisation, and index management.

Instantiated once per process via get_milvus_client() and shared across all
indexers and search calls.  Callers must NOT create MilvusClient inline inside
build functions — pass it via MilvusWriteContext instead.
"""
from __future__ import annotations

import logging
import socket
import threading
import uuid
from collections.abc import Iterable

from pymilvus import Collection, CollectionSchema, connections, utility

from app.core.settings import get_settings

from .milvus_schema import (
    create_asr_schema,
    create_face_schema,
    create_face_group_schema,
    create_entity_face_sample_schema,
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
# Note: For collections with multiple indexes (like asr_embeddings / ocr_embeddings), this
# dict only represents the primary dense vector index for informational purposes.
# The actual index creation uses _COLLECTION_CONFIGS["indexes"] (multi-index path).
_STATIC_INDEX_CONFIGS: dict[str, dict] = {
    "asr_embeddings": {
        # DiskANN dense index — actual creation uses _COLLECTION_CONFIGS["asr_embeddings"]["indexes"]
        "index_type": "DISKANN",
        "metric_type": "IP",
        "params": {
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": 0.125,
            "build_dram_budget_gb": 32.0,
        },
    },
    "ocr_embeddings": {
        "index_type": "DISKANN",
        "metric_type": "IP",
        "params": {
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": 0.125,
            "build_dram_budget_gb": 32.0,
        },
    },
    "face_embeddings": {
        # Migrated IVF_FLAT → DISKANN for 千万级 scale (disk-resident vectors +
        # PQ in memory). Face is the highest-dimension modality (512), so the
        # memory saving is the largest of all modalities. COSINE retained: face
        # embeddings are unit-normalised ArcFace vectors (faces.py), and
        # visual/speaker proved DiskANN supports COSINE in this stack.
        "index_type": "DISKANN",
        "metric_type": "COSINE",
        "params": {
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": 0.125,
            "build_dram_budget_gb": 32.0,
        },
    },
    "face_groups": {
        "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 256},
    },
    "entity_face_samples": {
        "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 256},
    },
    "speaker_embeddings": {
        # Migrated HNSW → DISKANN for 千万级 scale (disk-resident vectors +
        # PQ in memory). COSINE retained: speaker embeddings are normalised
        # unit vectors, and visual proved DiskANN supports COSINE in this stack.
        "index_type": "DISKANN",
        "metric_type": "COSINE",
        "params": {
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": 0.125,
            "build_dram_budget_gb": 32.0,
        },
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
        "indexes": {
            # Dense index: DiskANN for semantic search
            "embedding": {
                "index_type": "DISKANN",
                "metric_type": "IP",
                "params": {
                    "max_degree": 56,
                    "search_list_size": 128,
                    "pq_code_budget_gb": 0.125,
                    "build_dram_budget_gb": 32.0,
                },
            },
            # Sparse index: BM25 function output requires BM25 metric type
            "sparse_embedding": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"drop_ratio_build": 0.2},
            },
        },
    },
    "ocr_embeddings": {
        "schema": create_ocr_schema,
        "indexes": {
            # Dense index: DiskANN for semantic search
            "embedding": {
                "index_type": "DISKANN",
                "metric_type": "IP",
                "params": {
                    "max_degree": 56,
                    "search_list_size": 128,
                    "pq_code_budget_gb": 0.125,
                    "build_dram_budget_gb": 32.0,
                },
            },
            # Sparse index: BM25 function output requires BM25 metric type
            "sparse_embedding": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"drop_ratio_build": 0.2},
            },
        },
    },
    "face_embeddings": {
        "schema": create_face_schema,
        "index": _STATIC_INDEX_CONFIGS["face_embeddings"],
        "video_scoped": True,
    },
    "face_groups": {
        "schema": create_face_group_schema,
        "index": _STATIC_INDEX_CONFIGS["face_groups"],
        "video_scoped": True,
    },
    "entity_face_samples": {
        "schema": create_entity_face_sample_schema,
        "index": _STATIC_INDEX_CONFIGS["entity_face_samples"],
        "video_scoped": False,
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


def _runtime_collection_layout(settings) -> tuple[dict[str, str], dict[str, dict]]:
    """Resolve deployment-local collection names without changing defaults."""
    modality_names = dict(_COLLECTION_FOR_MODALITY)
    asr_name = str(settings.milvus_asr_collection)
    if asr_name in _COLLECTION_CONFIGS and asr_name != "asr_embeddings":
        raise ValueError(
            f"MILVUS_ASR_COLLECTION conflicts with reserved collection: {asr_name}"
        )
    modality_names["asr"] = asr_name
    configs = {
        name: config
        for name, config in _COLLECTION_CONFIGS.items()
        if name != "asr_embeddings"
    }
    configs[asr_name] = _COLLECTION_CONFIGS["asr_embeddings"]
    return modality_names, configs

_OCR_V2_REQUIRED_FIELDS = frozenset({
    "text",
    "embedding",
    "sparse_embedding",
    "has_embedding",
})
_OCR_V2_REQUIRED_INDEX_FIELDS = frozenset({"embedding", "sparse_embedding"})

_ASR_V2_REQUIRED_FIELDS = frozenset({
    "text",
    "embedding",
    "sparse_embedding",
    "has_embedding",
})
_ASR_V2_REQUIRED_INDEX_FIELDS = frozenset({"embedding", "sparse_embedding"})


def _validate_existing_ocr_collection(col: Collection) -> None:
    """Fail fast when an existing OCR collection predates hybrid search.

    Adding a BM25 function or sparse-vector field is not an in-place Milvus
    schema migration.  Starting successfully against the legacy collection
    would defer the failure to a user's first OCR query, so require operators
    to rebuild it before this release serves traffic.
    """
    schema = col.schema
    fields = {field.name for field in schema.fields}
    function_names = {
        function.name
        for function in (getattr(schema, "functions", None) or [])
    }
    index_fields = {
        index.field_name
        for index in col.indexes
        if getattr(index, "field_name", None)
    }

    missing_fields = sorted(_OCR_V2_REQUIRED_FIELDS - fields)
    missing_functions = sorted({"bm25_ocr"} - function_names)
    missing_indexes = sorted(_OCR_V2_REQUIRED_INDEX_FIELDS - index_fields)
    if not (missing_fields or missing_functions or missing_indexes):
        return

    details: list[str] = []
    if missing_fields:
        details.append(f"fields={missing_fields}")
    if missing_functions:
        details.append(f"functions={missing_functions}")
    if missing_indexes:
        details.append(f"indexes={missing_indexes}")
    raise RuntimeError(
        "Milvus collection 'ocr_embeddings' uses the legacy OCR schema "
        f"({', '.join(details)}). Drop and rebuild the OCR Milvus index "
        "before deploying hybrid OCR search."
    )


def _validate_existing_asr_collection(col: Collection) -> None:
    """Fail fast when an existing ASR collection predates hybrid search.

    Adding a BM25 function or sparse-vector field is not an in-place Milvus
    schema migration.  Starting successfully against the legacy collection
    would defer the failure to a user's first ASR query, so require operators
    to rebuild it before this release serves traffic.
    """
    schema = col.schema
    fields = {field.name for field in schema.fields}
    function_names = {
        function.name
        for function in (getattr(schema, "functions", None) or [])
    }
    index_fields = {
        index.field_name
        for index in col.indexes
        if getattr(index, "field_name", None)
    }

    missing_fields = sorted(_ASR_V2_REQUIRED_FIELDS - fields)
    missing_functions = sorted({"bm25_asr"} - function_names)
    missing_indexes = sorted(_ASR_V2_REQUIRED_INDEX_FIELDS - index_fields)
    if not (missing_fields or missing_functions or missing_indexes):
        return

    details: list[str] = []
    if missing_fields:
        details.append(f"fields={missing_fields}")
    if missing_functions:
        details.append(f"functions={missing_functions}")
    if missing_indexes:
        details.append(f"indexes={missing_indexes}")
    raise RuntimeError(
        "Milvus collection 'asr_embeddings' uses the legacy ASR schema "
        f"({', '.join(details)}). Drop and rebuild the ASR Milvus index "
        "before deploying hybrid ASR search."
    )


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
        self._collection_for_modality, self._collection_configs = (
            _runtime_collection_layout(s)
        )
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
        logger.info(
            "MilvusClient ready — %d collections", len(self._collection_configs)
        )

    # ------------------------------------------------------------------
    # Collection init
    # ------------------------------------------------------------------

    def _init_collections(self) -> None:
        for name, config in self._collection_configs.items():
            if not utility.has_collection(name):
                logger.info("Creating collection: %s", name)
                schema: CollectionSchema = config["schema"]()
                col = Collection(name=name, schema=schema, consistency_level="Strong")

                # Handle different index configuration formats
                if "indexes" in config:
                    # Multiple indexes (OCR uses dense + sparse retrieval).
                    for field_name, index_params in config["indexes"].items():
                        col.create_index(field_name=field_name, index_params=index_params)
                        logger.info("Created index on %s.%s: %s", name, field_name, index_params["index_type"])
                else:
                    # Single index (legacy format)
                    index_config = get_collection_index_config(name) if name == "visual_embeddings" else config["index"]
                    col.create_index(field_name="embedding", index_params=index_config)

                col.load()
                logger.info("Collection %s created and loaded", name)
            else:
                col = Collection(name)
                if name == "ocr_embeddings":
                    _validate_existing_ocr_collection(col)
                elif name == self._collection_for_modality["asr"]:
                    _validate_existing_asr_collection(col)
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
        name = self._collection_for_modality[modality]
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
        for name, config in self._collection_configs.items():
            if not config.get("video_scoped", True):
                continue
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
        for name, config in self._collection_configs.items():
            if not config.get("video_scoped", True):
                continue
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
        name = self._collection_for_modality[modality]
        try:
            # Check if collection exists first
            if not utility.has_collection(name):
                logger.info(
                    "delete_video_modality video=%s modality=%s: collection %s does not exist, skipping",
                    video_id, modality, name
                )
                return 0  # No records to delete

            col = Collection(name)
            expr = f'video_id == "{video_id}"'
            result = col.delete(expr)
            col.flush()
            count = getattr(result, "delete_count", 0)
            logger.info(
                "delete_video_modality video=%s modality=%s deleted=%d",
                video_id, modality, count,
            )
            if modality == "face" and utility.has_collection("face_groups"):
                groups = Collection("face_groups")
                groups.delete(expr)
                groups.flush()
            return count
        except Exception as exc:
            logger.warning(
                "delete_video_modality %s/%s failed: %s", video_id, modality, exc
            )
            return -1

    def delete_video_modality_except_version(
        self, video_id: str, modality: str, keep_asset_version: str
    ) -> int:
        """Remove superseded rows only after *keep_asset_version* is published."""
        name = self._collection_for_modality[modality]
        try:
            if not utility.has_collection(name):
                return 0
            col = Collection(name)
            expr = (
                f'video_id == "{video_id}" and '
                f'asset_version != "{keep_asset_version}"'
            )
            result = col.delete(expr)
            col.flush()
            if modality == "face" and utility.has_collection("face_groups"):
                groups = Collection("face_groups")
                groups.delete(expr)
                groups.flush()
            return int(getattr(result, "delete_count", 0))
        except Exception as exc:
            logger.warning(
                "superseded-version cleanup failed video=%s modality=%s keep=%s: %s",
                video_id, modality, keep_asset_version, exc,
            )
            return -1

    def count_video_modality_version(
        self, video_id: str, modality: str, asset_version: str
    ) -> int:
        """Return the persisted rows for one published modality version."""
        name = self._collection_for_modality[modality]
        rows = Collection(name).query(
            expr=(f'video_id == "{video_id}" and asset_version == "{asset_version}"'),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    def count_video_modality(self, video_id: str, modality: str) -> int:
        """Return the persisted row count for one video and modality."""
        name = self._collection_for_modality[modality]
        rows = Collection(name).query(
            expr=f'video_id == "{video_id}"',
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    def count_face_groups_version(
        self,
        video_id: str,
        asset_version: str,
        group_model_version: str,
    ) -> int:
        """Return rows for one immutable Face group generation."""
        rows = self.collection("face_groups").query(
            expr=(
                f'video_id == "{video_id}" and '
                f'asset_version == "{asset_version}" and '
                f'model_version == "{group_model_version}"'
            ),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0


class ExistingMilvusCollectionsClient:
    """Maintenance-only client for an explicit set of existing collections.

    Unlike :class:`MilvusClient`, this client never creates collections and
    never validates unrelated modality schemas.  It uses an isolated PyMilvus
    connection alias so constructing it cannot weaken or poison the
    application singleton.  Operational migrations should request the exact
    collections they need and fail if any are absent.
    """

    def __init__(self, required_collections: Iterable[str]) -> None:
        names = tuple(dict.fromkeys(str(name).strip() for name in required_collections))
        if not names or any(not name for name in names):
            raise ValueError("required_collections must contain non-empty names")
        settings = get_settings()
        self._collection_for_modality, _ = _runtime_collection_layout(settings)
        self._alias = f"maintenance_{uuid.uuid4().hex}"
        self._required_collections = frozenset(names)
        connections.connect(
            alias=self._alias,
            host=settings.milvus_host,
            port=str(settings.milvus_port),
            timeout=settings.milvus_query_timeout_seconds,
        )
        try:
            missing = [
                name
                for name in names
                if not utility.has_collection(name, using=self._alias)
            ]
            if missing:
                raise RuntimeError(
                    "required Milvus collections do not exist: "
                    + ", ".join(sorted(missing))
                )
        except Exception:
            connections.disconnect(self._alias)
            raise

    def close(self) -> None:
        connections.disconnect(self._alias)

    def collection(self, name: str) -> Collection:
        if name not in self._required_collections:
            raise ValueError(f"collection was not authorized for maintenance: {name}")
        return Collection(name, using=self._alias)

    def collection_for(self, modality: str) -> Collection:
        try:
            name = self._collection_for_modality[modality]
        except KeyError as exc:
            raise ValueError(f"unknown Milvus modality: {modality}") from exc
        return self.collection(name)

    def count_video_modality_version(
        self, video_id: str, modality: str, asset_version: str
    ) -> int:
        rows = self.collection_for(modality).query(
            expr=(f'video_id == "{video_id}" and asset_version == "{asset_version}"'),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    def count_face_groups_version(
        self,
        video_id: str,
        asset_version: str,
        group_model_version: str,
    ) -> int:
        rows = self.collection("face_groups").query(
            expr=(
                f'video_id == "{video_id}" and '
                f'asset_version == "{asset_version}" and '
                f'model_version == "{group_model_version}"'
            ),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

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
