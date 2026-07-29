#!/usr/bin/env python3
"""Rebuild only the Visual vector index when switching HNSW/DiskANN.

The collection and its embeddings are preserved. Pause indexing writes before
running this script and keep NPZ fallback enabled until the rebuild finishes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from pymilvus import Collection, connections, utility  # noqa: E402

from app.indexing.milvus_client import get_collection_index_config  # noqa: E402
from app.settings import get_settings  # noqa: E402

COLLECTION_NAME = "visual_embeddings"
INDEX_FIELD = "embedding"
INDEX_BUILD_TIMEOUT_SECONDS = 1800


def _index_state(collection: Collection) -> tuple[str, str, dict[str, Any]]:
    index = collection.index()
    if index is None:
        return "NONE", "", {}
    params = dict(index.params or {})
    return (
        str(params.get("index_type", "UNKNOWN")),
        str(getattr(index, "index_name", "") or ""),
        params,
    )


def _create_index(
    collection: Collection,
    index_params: dict[str, Any],
    index_name: str,
) -> None:
    kwargs: dict[str, Any] = {}
    if index_name:
        kwargs["index_name"] = index_name
    collection.create_index(
        field_name=INDEX_FIELD,
        index_params=index_params,
        **kwargs,
    )
    utility.wait_for_index_building_complete(
        COLLECTION_NAME,
        index_name=index_name,
        timeout=INDEX_BUILD_TIMEOUT_SECONDS,
    )


def _drop_index(collection: Collection, index_name: str) -> None:
    kwargs: dict[str, Any] = {}
    if index_name:
        kwargs["index_name"] = index_name
    collection.drop_index(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the Visual embedding index without deleting vectors"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Allow the index rebuild after an interactive confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the current and target index without changing Milvus",
    )
    args = parser.parse_args()
    settings = get_settings()

    print(f"Connecting to Milvus: {settings.milvus_host}:{settings.milvus_port}")
    try:
        connections.connect(
            host=settings.milvus_host,
            port=settings.milvus_port,
            timeout=5,
        )
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return 1

    target_params = get_collection_index_config(COLLECTION_NAME)
    target_type = str(target_params["index_type"])

    if not utility.has_collection(COLLECTION_NAME):
        print(f"{COLLECTION_NAME} does not exist.")
        print(f"The application will create it with {target_type} on next startup.")
        return 0

    collection = Collection(COLLECTION_NAME)
    collection.flush()
    num_entities_before = collection.num_entities
    try:
        current_type, index_name, current_params = _index_state(collection)
    except Exception as exc:
        print(f"Failed to read the current index: {exc}")
        return 1

    print("\nCurrent state")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Entities:   {num_entities_before:,}")
    print(f"  Index:      {current_type}")
    print(f"  Target:     {target_type}")

    if current_type == target_type:
        print(f"\nThe Visual index is already {target_type}; no rebuild is needed.")
        return 0

    if args.dry_run:
        print("\nDry run only; Milvus was not modified.")
        print(f"Run: python {Path(__file__).name} --confirm")
        return 0

    if not settings.milvus_fallback_enabled:
        print("\nRefusing to rebuild while MILVUS_FALLBACK_ENABLED=false.")
        print("Enable the global NPZ fallback first so retrieval remains available.")
        return 1

    if not args.confirm:
        print("\nPass --confirm after reviewing the dry-run output.")
        return 1

    print("\nPause Visual indexing writes before continuing.")
    print("Only the Milvus index definition will be replaced; vectors are preserved.")
    response = input("Type 'DELETE' to replace the current index definition: ")
    if response != "DELETE":
        print("Cancelled.")
        return 1

    old_params = current_params
    dropped_old_index = current_type != "NONE"
    try:
        collection.release()
        if dropped_old_index:
            _drop_index(collection, index_name)
        print(f"Building {target_type} index...")
        _create_index(collection, target_params, index_name)
        collection.load()
        collection.flush()
    except Exception as exc:
        print(f"Index rebuild failed: {exc}")
        if dropped_old_index and old_params:
            print(f"Attempting to restore the previous {current_type} index...")
            try:
                try:
                    _drop_index(collection, index_name)
                except Exception:
                    pass
                _create_index(collection, old_params, index_name)
                collection.load()
                print("Previous index restored.")
            except Exception as restore_exc:
                print(f"Automatic restore failed: {restore_exc}")
                print("Keep NPZ fallback enabled and restore the index manually.")
        return 1

    rebuilt_type, _, _ = _index_state(collection)
    num_entities_after = collection.num_entities
    if rebuilt_type != target_type:
        print(f"Verification failed: expected {target_type}, found {rebuilt_type}.")
        return 1
    if num_entities_after != num_entities_before:
        print(
            "Verification failed: entity count changed "
            f"from {num_entities_before:,} to {num_entities_after:,}."
        )
        return 1

    print("\nIndex rebuild completed.")
    print(f"  Index:    {rebuilt_type}")
    print(f"  Entities: {num_entities_after:,} (unchanged)")
    print("Restart the application process if its environment variables changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
