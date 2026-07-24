#!/usr/bin/env python3
"""Milvus schema migration script.

Detects collections created with older schema/index versions and recreates
them with the current runtime configuration.

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
# Canonical schema/index comparison
# ---------------------------------------------------------------------------

def _field_signature(field) -> tuple:
    params = getattr(field, "params", {}) or {}
    return (
        field.dtype,
        params.get("dim"),
        params.get("max_length"),
        bool(getattr(field, "is_primary", False)),
    )


def _check_collections(client) -> dict[str, list[str]]:
    """Return incompatibility reasons for every outdated collection."""
    from app.indexing.milvus_client import (
        _COLLECTION_CONFIGS,
        _COLLECTION_FOR_MODALITY,
    )

    outdated: dict[str, list[str]] = {}
    for modality, collection_name in _COLLECTION_FOR_MODALITY.items():
        reasons: list[str] = []
        try:
            col = client.collection_for(modality)
            config = _COLLECTION_CONFIGS[collection_name]
            expected_fields = {
                field.name: _field_signature(field)
                for field in config["schema"]().fields
            }
            actual_fields = {
                field.name: _field_signature(field)
                for field in col.schema.fields
            }
            for name in sorted(expected_fields.keys() - actual_fields.keys()):
                reasons.append(f"missing field: {name}")
            for name in sorted(expected_fields.keys() & actual_fields.keys()):
                if actual_fields[name] != expected_fields[name]:
                    reasons.append(
                        f"field mismatch: {name} "
                        f"actual={actual_fields[name]} expected={expected_fields[name]}"
                    )

            embedding_indexes = [
                index for index in col.indexes
                if getattr(index, "field_name", None) == "embedding"
            ]
            if not embedding_indexes:
                reasons.append("missing embedding index")
            else:
                actual_index = getattr(embedding_indexes[0], "params", {}) or {}
                expected_index = config["index"]
                for key in ("index_type", "metric_type"):
                    actual_value = str(actual_index.get(key, "")).upper()
                    expected_value = str(expected_index[key]).upper()
                    if actual_value != expected_value:
                        reasons.append(
                            f"index {key} mismatch: "
                            f"actual={actual_value or '<missing>'} expected={expected_value}"
                        )
        except Exception as exc:
            reasons.append(f"inspection failed: {exc}")
        if reasons:
            outdated[modality] = reasons
    return outdated


def _migrate(client, outdated: dict[str, list[str]], dry_run: bool) -> None:
    from pymilvus import utility

    for modality, reasons in outdated.items():
        col_name = client.collection_for(modality).name
        print(f"\n[{'DRY RUN' if dry_run else 'MIGRATE'}] {modality} ({col_name})")
        for reason in reasons:
            print(f"  - {reason}")

        if dry_run:
            print(f"  → Would drop and recreate '{col_name}'")
            continue

        print(f"  Dropping '{col_name}' …", end=" ", flush=True)
        utility.drop_collection(col_name)
        print("done")

        print(f"  Recreating '{col_name}' …", end=" ", flush=True)
        from app.indexing.milvus_client import _COLLECTION_CONFIGS
        from pymilvus import Collection

        cfg = _COLLECTION_CONFIGS[col_name]
        col = Collection(
            name=col_name,
            schema=cfg["schema"](),
            consistency_level="Strong",
        )
        col.create_index(field_name="embedding", index_params=cfg["index"])
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
    parser.add_argument(
        "--yes", action="store_true",
        help="Confirm destructive migration non-interactively.",
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
    for modality, reasons in outdated.items():
        print(f"  {modality:8s}  issues: {'; '.join(reasons)}")

    if args.dry_run:
        print("\n[dry-run] No changes made. Re-run without --dry-run to migrate.")
    else:
        print("\nWARNING: Migration will DROP these collections (all data lost).")
        if not args.yes:
            confirm = input("Type 'yes' to proceed: ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                sys.exit(0)

    _migrate(client, outdated, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nMigration complete.")
        print("Next steps:")
        print("  1. Restart the backend service.")
        print("  2. Run scripts/backfill_milvus.py against the retained NPZ indexes.")


if __name__ == "__main__":
    main()
