#!/usr/bin/env python3
"""One-off recreation of the ASR Milvus collection for hybrid retrieval.

The ASR DiskANN + BM25 schema cannot be added to an existing collection in
place.  This command intentionally bypasses ``MilvusClient``: that client
correctly rejects a legacy collection, while this maintenance operation must
connect before the replacement collection exists.

Run from the application container.  It only recreates the collection schema;
rebuild ASR rows afterwards with ``app.maintenance.reindex_milvus_only``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly places its script directory, not ``backend/``, on
# sys.path (the container command is ``python scripts/recreate_...py``).
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pymilvus import Collection, connections, utility  # noqa: E402

from app.core.settings import get_settings  # noqa: E402
from app.vector_store.milvus.milvus_client import _COLLECTION_CONFIGS  # noqa: E402
from app.vector_store.milvus.milvus_schema import create_asr_schema  # noqa: E402


COLLECTION_NAME = "asr_embeddings"
CONNECTION_ALIAS = "asr_schema_migration"
REQUIRED_FIELDS = frozenset({"text", "embedding", "sparse_embedding", "has_embedding"})
REQUIRED_FUNCTIONS = frozenset({"bm25_asr"})
REQUIRED_INDEXES = frozenset({"embedding", "sparse_embedding"})


def _collection_state() -> dict:
    """Describe the current collection without changing it."""
    if not utility.has_collection(COLLECTION_NAME, using=CONNECTION_ALIAS):
        return {"exists": False, "ready": False}

    collection = Collection(COLLECTION_NAME, using=CONNECTION_ALIAS)
    fields = {field.name for field in collection.schema.fields}
    functions = {
        function.name
        for function in (getattr(collection.schema, "functions", None) or [])
    }
    indexes = {
        index.field_name
        for index in collection.indexes
        if getattr(index, "field_name", None)
    }
    return {
        "exists": True,
        "ready": (
            REQUIRED_FIELDS.issubset(fields)
            and REQUIRED_FUNCTIONS.issubset(functions)
            and REQUIRED_INDEXES.issubset(indexes)
        ),
        "missing_fields": sorted(REQUIRED_FIELDS - fields),
        "missing_functions": sorted(REQUIRED_FUNCTIONS - functions),
        "missing_indexes": sorted(REQUIRED_INDEXES - indexes),
        "row_count": collection.num_entities,
    }


def _create_collection() -> None:
    """Create and load the exact ASR schema/indexes used by the application."""
    collection = Collection(
        name=COLLECTION_NAME,
        schema=create_asr_schema(),
        consistency_level="Strong",
        using=CONNECTION_ALIAS,
    )
    for field_name, index_params in _COLLECTION_CONFIGS[COLLECTION_NAME]["indexes"].items():
        collection.create_index(field_name=field_name, index_params=index_params)
    collection.load()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一次性重建 ASR DiskANN + BM25 Milvus collection（不重建视频数据）"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行删除旧 collection 并创建新 collection；默认只检查",
    )
    parser.add_argument(
        "--confirm-drop-asr-embeddings",
        action="store_true",
        help="确认允许删除 asr_embeddings 的所有现有行（与 --execute 同时必需）",
    )
    args = parser.parse_args()
    if args.execute != args.confirm_drop_asr_embeddings:
        parser.error("执行迁移必须同时传入 --execute --confirm-drop-asr-embeddings")

    settings = get_settings()
    connections.connect(
        alias=CONNECTION_ALIAS,
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        timeout=settings.milvus_query_timeout_seconds,
    )
    try:
        before = _collection_state()
        if not args.execute:
            print(json.dumps({
                "status": "dry_run",
                "collection": COLLECTION_NAME,
                "current": before,
                "action": (
                    "no_action_needed" if before["ready"]
                    else "will_drop_and_recreate_then_reindex_asr"
                ),
            }, ensure_ascii=False))
            return 0

        if before["ready"]:
            print(json.dumps({
                "status": "already_compatible",
                "collection": COLLECTION_NAME,
                "current": before,
                "next": "run ASR reindex only when rows need rebuilding",
            }, ensure_ascii=False))
            return 0

        if before["exists"]:
            utility.drop_collection(COLLECTION_NAME, using=CONNECTION_ALIAS)
        _create_collection()
        after = _collection_state()
        if not after["ready"]:
            raise RuntimeError(f"ASR collection verification failed: {after}")
        print(json.dumps({
            "status": "recreated",
            "collection": COLLECTION_NAME,
            "previous": before,
            "current": after,
            "next": "run reindex_milvus_only with --modalities asr --execute",
        }, ensure_ascii=False))
        return 0
    finally:
        connections.disconnect(CONNECTION_ALIAS)


if __name__ == "__main__":
    raise SystemExit(main())
