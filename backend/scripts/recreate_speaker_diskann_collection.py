#!/usr/bin/env python3
"""Recreate speaker_embeddings collection with DiskANN index.

The speaker modality was previously using HNSW but the config now specifies
DISKANN. This script drops the old collection and triggers recreation with
the correct index type via the standard collection initialization.

Usage:
    python3 backend/scripts/recreate_speaker_diskann_collection.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.core.settings import get_settings
from pymilvus import utility, connections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.milvus_enabled:
        print("❌ Milvus is disabled in config")
        sys.exit(1)

    print("=" * 70)
    print("Speaker DiskANN Collection Migration")
    print("=" * 70)

    # Connect to Milvus
    alias = "default"
    if not connections.has_connection(alias):
        connections.connect(
            alias=alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
        )

    collection_name = "speaker_embeddings"

    # Check current state
    if utility.has_collection(collection_name, using=alias):
        print(f"\n📋 Collection '{collection_name}' exists")

        # Inspect current index
        from pymilvus import Collection
        coll = Collection(collection_name, using=alias)

        try:
            coll.load()
        except Exception:
            pass  # May already be loaded

        indexes = coll.indexes
        current_index_type = None
        if indexes:
            idx = indexes[0]
            current_index_type = idx.params.get('index_type', '?')
            print(f"   Current index: {current_index_type}")
            print(f"   Metric: {idx.params.get('metric_type', '?')}")
        else:
            print("   No index found")

        row_count = coll.num_entities
        print(f"   Rows: {row_count:,}")

        # Check if migration is needed
        if current_index_type == "DISKANN":
            print(f"\n✅ Collection already uses DISKANN — no migration needed")
            return

        if args.dry_run:
            print(f"\n🔍 DRY RUN: Would drop collection '{collection_name}'")
            print(f"            Would recreate with DISKANN index")
            print(f"            Would trigger reindex for all videos")
            return

        # Drop existing collection
        print(f"\n🗑️  Dropping existing collection...")
        utility.drop_collection(collection_name, using=alias)
        print(f"   ✅ Collection dropped")
    else:
        print(f"\nℹ️  Collection '{collection_name}' does not exist")
        if args.dry_run:
            print(f"   DRY RUN: Would be created with DISKANN on next backend start")
            return

    print(f"\n✅ Migration complete")
    print(f"\n📝 Next steps:")
    print(f"   1. Restart backend to recreate collection with DISKANN:")
    print(f"      docker restart momentseek-0829-platform")
    print(f"   2. Wait for backend to be healthy (check /api/health)")
    print(f"   3. Trigger reindex for all videos with speaker data:")
    print(f"      docker exec momentseek-0829-platform python3 -m app.indexing.cli rebuild-all --modality speaker")
    print("=" * 70)


if __name__ == "__main__":
    main()
