#!/usr/bin/env python3
"""Create / verify the ASR collection with DiskANN + BM25 hybrid search support.

The collection is named ``asr_embeddings`` (in-place upgrade, no _v2 suffix),
mirroring the OCR approach.  This script can be used for:
  * Manual creation / verification in a fresh environment
  * Confirming the schema after a drop-and-recreate migration

Usage:
    # Inside the app container:
    python backend/scripts/create_asr_v2_collection.py

    # With explicit video_id check:
    python backend/scripts/create_asr_v2_collection.py --check-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from pymilvus import utility


def main(check_only: bool = False) -> int:
    """Create and verify the ASR DiskANN + BM25 collection."""
    from app.vector_store.milvus.milvus_client import MilvusClient

    print("ASR collection setup (DiskANN + BM25 hybrid search)")
    print("=" * 55)

    try:
        client = MilvusClient()

        if utility.has_collection("asr_embeddings"):
            print("✓ asr_embeddings already exists")

            col = client.collection_for("asr")
            schema = col.schema
            schema_fields = {f.name for f in schema.fields}
            function_names = {
                fn.name
                for fn in (getattr(schema, "functions", None) or [])
            }
            index_fields = {
                idx.field_name
                for idx in col.indexes
                if getattr(idx, "field_name", None)
            }

            required_fields = {"sparse_embedding", "has_embedding", "embedding", "text"}
            missing_fields = required_fields - schema_fields
            if missing_fields:
                print(f"✗ Schema incomplete — missing fields: {sorted(missing_fields)}")
                print("  Drop the collection and restart the app to recreate it:")
                print("  python backend/scripts/create_asr_v2_collection.py --drop")
                return 1

            if "bm25_asr" not in function_names:
                print("✗ BM25 function 'bm25_asr' not found — collection uses legacy schema")
                return 1

            if "sparse_embedding" not in index_fields:
                print("✗ SPARSE_INVERTED_INDEX on sparse_embedding not found")
                return 1

            print("✓ Schema verified:")
            print(f"  Dense field  : embedding ({schema_fields})")
            print(f"  Sparse field : sparse_embedding (BM25 auto-computed)")
            print(f"  BM25 function: bm25_asr (chinese analyzer)")
            print(f"  Text field   : text (enable_analyzer=True)")
            print(f"  Indexes      : {sorted(index_fields)}")

            load_state = utility.load_state("asr_embeddings")
            if load_state.name == "Loaded":
                print(f"✓ Collection loaded, entities: {col.num_entities}")
            else:
                print(f"⚠ Collection not loaded ({load_state.name}), loading...")
                col.load()
                print("✓ Loaded")

            return 0

        if check_only:
            print("✗ asr_embeddings does not exist")
            return 1

        # Collection absent — the MilvusClient constructor already called
        # _init_collections(), so the collection should have been created.
        # If we reach here something went wrong.
        print("✗ asr_embeddings was not created during MilvusClient init")
        print("  Check logs for errors in _init_collections()")
        return 1

    except RuntimeError as exc:
        # _validate_existing_asr_collection raised — need migration
        print(f"✗ {exc}")
        print()
        print("Migration required:")
        print("  1. Drop the old collection:")
        print("     python backend/scripts/create_asr_v2_collection.py --drop")
        print("  2. Restart the app (will auto-recreate with new schema)")
        return 1

    except Exception as exc:
        print(f"✗ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def drop_and_recreate() -> int:
    """Drop the legacy asr_embeddings collection so the app can recreate it."""
    print("Dropping legacy asr_embeddings collection...")
    try:
        from pymilvus import connections, utility
        from app.core.settings import get_settings

        s = get_settings()
        connections.connect(host=s.milvus_host, port=str(s.milvus_port))

        if utility.has_collection("asr_embeddings"):
            count_before = 0
            try:
                from pymilvus import Collection
                count_before = Collection("asr_embeddings").num_entities
            except Exception:
                pass
            utility.drop_collection("asr_embeddings")
            print(f"✓ Dropped asr_embeddings ({count_before} entities deleted)")
        else:
            print("⚠ asr_embeddings does not exist — nothing to drop")

        print()
        print("Next step: restart the app (./deploy_0829.sh or docker restart)")
        print("  The app will auto-create asr_embeddings with DiskANN + BM25 schema.")
        return 0

    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/verify ASR hybrid collection")
    parser.add_argument("--check-only", action="store_true",
                        help="Only verify the existing collection, do not create")
    parser.add_argument("--drop", action="store_true",
                        help="Drop the legacy collection so the app can recreate it")
    args = parser.parse_args()

    if args.drop:
        sys.exit(drop_and_recreate())
    else:
        sys.exit(main(check_only=args.check_only))
