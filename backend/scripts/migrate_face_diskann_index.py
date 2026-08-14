#!/usr/bin/env python3
"""One-time migration of the Face IVF_FLAT/L2 index to DiskANN/COSINE.

The Face collection schema and vectors are unchanged. Rebuilding only the
``embedding`` index preserves all ``face_embeddings`` rows and published
manifest pointers; no NPZ data is read or written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pymilvus import Collection, connections, utility  # noqa: E402
from pymilvus.exceptions import IndexNotExistException  # noqa: E402

from app.core.settings import get_settings  # noqa: E402
from app.vector_store.milvus.milvus_client import get_collection_index_config  # noqa: E402


COLLECTION_NAME = "face_embeddings"
CONNECTION_ALIAS = "face_diskann_migration"
EXPECTED_INDEX_TYPE = "DISKANN"
EXPECTED_METRIC_TYPE = "COSINE"


def _index_details(collection: Collection) -> dict[str, str | None]:
    try:
        index = collection.index()
    except IndexNotExistException:
        return {"index_type": None, "metric_type": None}
    params = getattr(index, "params", {}) or {}
    return {
        "index_type": params.get("index_type"),
        "metric_type": params.get("metric_type"),
    }


def _collection_state() -> dict:
    if not utility.has_collection(COLLECTION_NAME, using=CONNECTION_ALIAS):
        return {"exists": False, "ready": False, "row_count": 0}

    collection = Collection(COLLECTION_NAME, using=CONNECTION_ALIAS)
    index = _index_details(collection)
    return {
        "exists": True,
        "ready": (
            index["index_type"] == EXPECTED_INDEX_TYPE
            and index["metric_type"] == EXPECTED_METRIC_TYPE
        ),
        "row_count": int(collection.num_entities),
        **index,
    }


def _replace_vector_index(collection: Collection) -> None:
    """Replace only the embedding index; collection rows and schema survive."""
    collection.release()
    collection.drop_index()
    collection.create_index(
        field_name="embedding",
        index_params=get_collection_index_config(COLLECTION_NAME),
    )
    collection.load()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一次性将 face_embeddings 索引迁移为 DiskANN/COSINE（保留全部行）"
    )
    parser.add_argument("--execute", action="store_true", help="执行索引替换；默认只检查")
    parser.add_argument(
        "--confirm-rebuild-face-index",
        action="store_true",
        help="确认允许 release 并重建 face_embeddings 向量索引（不会删除行）",
    )
    args = parser.parse_args()
    if args.execute != args.confirm_rebuild_face_index:
        parser.error("执行迁移必须同时传入 --execute --confirm-rebuild-face-index")

    settings = get_settings()
    connections.connect(
        alias=CONNECTION_ALIAS,
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        timeout=settings.milvus_query_timeout_seconds,
    )
    try:
        before = _collection_state()
        if not before["exists"]:
            print(json.dumps({
                "status": "collection_absent",
                "collection": COLLECTION_NAME,
                "current": before,
                "next": "start the application once to create the empty DiskANN collection",
            }, ensure_ascii=False))
            return 0
        if not args.execute:
            print(json.dumps({
                "status": "dry_run",
                "collection": COLLECTION_NAME,
                "current": before,
                "action": "no_action_needed" if before["ready"] else "will_replace_vector_index",
            }, ensure_ascii=False))
            return 0
        if before["ready"]:
            print(json.dumps({
                "status": "already_compatible",
                "collection": COLLECTION_NAME,
                "current": before,
            }, ensure_ascii=False))
            return 0

        collection = Collection(COLLECTION_NAME, using=CONNECTION_ALIAS)
        _replace_vector_index(collection)
        after = _collection_state()
        if not after["ready"]:
            raise RuntimeError(f"Face DiskANN index verification failed: {after}")
        if after["row_count"] != before["row_count"]:
            raise RuntimeError(
                "Face index migration changed row count unexpectedly: "
                f"before={before['row_count']} after={after['row_count']}"
            )
        print(json.dumps({
            "status": "migrated",
            "collection": COLLECTION_NAME,
            "previous": before,
            "current": after,
            "next": "remove this one-time script after every formal environment is migrated",
        }, ensure_ascii=False))
        return 0
    finally:
        connections.disconnect(CONNECTION_ALIAS)


if __name__ == "__main__":
    raise SystemExit(main())
