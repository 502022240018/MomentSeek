#!/usr/bin/env python3
"""Milvus schema migration script.

Detects collections created with older schema versions (missing fields such as
``has_embedding``) and recreates them with the current schema.

Usage
-----
# Dry run — report outdated collections without changing anything:
    python scripts/migrate_milvus_schema.py --dry-run

# Live migration — drops outdated collections and recreates them:
    python scripts/migrate_milvus_schema.py

WARNING: This script drops collections.  All indexed vector data is deleted.
         Re-index every affected video after running this script.

Required environment variables (same as the backend service):
    MILVUS_HOST, MILVUS_PORT  (defaults: localhost, 19530)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root or from backend/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # backend/

# ---------------------------------------------------------------------------
# Canonical field sets — derived from milvus_schema.py
# ---------------------------------------------------------------------------

# Fields that MUST be present in each collection for the current code to work.
_REQUIRED_FIELDS: dict[str, set[str]] = {
    "visual":  {"pk", "video_id", "asset_version", "model_version",
                "frame_idx", "timestamp_ms", "segment_id",
                "segment_start_ms", "segment_end_ms", "embedding"},
    "asr":     {"pk", "video_id", "asset_version", "model_version",
                "segment_idx", "start_ms", "end_ms", "text",
                "has_embedding", "embedding"},
    "ocr":     {"pk", "video_id", "asset_version", "model_version",
                "frame_idx", "region_idx", "frame_ms", "text",
                "start_ms", "end_ms", "avg_box_score",
                "has_embedding", "embedding"},
    "face":    {"pk", "video_id", "asset_version", "model_version",
                "track_idx", "start_ms", "end_ms", "best_ms", "embedding"},
    "speaker": {"pk", "video_id", "asset_version", "model_version",
                "utterance_idx", "start_ms", "end_ms",
                "asr_chunk_idx", "track_id", "embedding"},
}


def _check_collections(client) -> dict[str, set[str]]:
    """Return {modality: missing_fields} for every outdated collection."""
    outdated: dict[str, set[str]] = {}
    for modality, required in _REQUIRED_FIELDS.items():
        try:
            col = client.collection_for(modality)
            actual = {f.name for f in col.schema.fields}
            missing = required - actual
            if missing:
                outdated[modality] = missing
        except Exception as exc:
            print(f"  [WARN] Could not inspect '{modality}' collection: {exc}")
    return outdated


def _migrate(client, outdated: dict[str, set[str]], dry_run: bool) -> None:
    from pymilvus import utility

    for modality, missing in outdated.items():
        col_name = client.collection_for(modality).name
        print(f"\n[{'DRY RUN' if dry_run else 'MIGRATE'}] {modality} ({col_name})")
        print(f"  Missing fields: {sorted(missing)}")

        if dry_run:
            print(f"  → Would drop and recreate '{col_name}'")
            continue

        print(f"  Dropping '{col_name}' …", end=" ", flush=True)
        utility.drop_collection(col_name)
        print("done")

        print(f"  Recreating '{col_name}' …", end=" ", flush=True)
        # Re-import here to get the factory for this modality.
        from app.indexing.milvus_client import _COLLECTION_CONFIGS
        cfg = _COLLECTION_CONFIGS[modality]
        from pymilvus import Collection, CollectionSchema

        schema: CollectionSchema = cfg["schema_factory"]()
        col = Collection(name=col_name, schema=schema)

        for index_cfg in cfg.get("indexes", []):
            col.create_index(
                field_name=index_cfg["field"],
                index_params={
                    "metric_type": index_cfg["metric_type"],
                    "index_type":  index_cfg["index_type"],
                    "params":      index_cfg.get("params", {}),
                },
            )
        col.load()
        print("done")
        print(f"  ✓ '{col_name}' recreated with correct schema")
        print(f"  ⚠  Re-index all videos that used this modality!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and migrate outdated Milvus collection schemas."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report outdated collections without making any changes.",
    )
    args = parser.parse_args()

    from app.indexing.milvus_client import get_milvus_client

    print("Connecting to Milvus …")
    try:
        client = get_milvus_client()
    except Exception as exc:
        print(f"ERROR: Cannot connect to Milvus: {exc}")
        sys.exit(1)
    print("Connected.\n")

    print("Inspecting collection schemas …")
    outdated = _check_collections(client)

    if not outdated:
        print("\n✓ All collections have up-to-date schemas. No migration needed.")
        return

    print(f"\nFound {len(outdated)} outdated collection(s):")
    for modality, missing in outdated.items():
        print(f"  {modality:8s}  missing: {sorted(missing)}")

    if args.dry_run:
        print("\n[dry-run] No changes made. Re-run without --dry-run to migrate.")
    else:
        print("\nWARNING: Migration will DROP these collections (all data lost).")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    _migrate(client, outdated, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nMigration complete.")
        print("Next steps:")
        print("  1. Restart the backend service.")
        print("  2. Re-index every video that was previously indexed.")


if __name__ == "__main__":
    main()
