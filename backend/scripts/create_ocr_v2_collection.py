#!/usr/bin/env python3
"""Create OCR v2 collection with DiskANN + BM25 hybrid search support.

This script creates the ocr_embeddings_v2 collection if it doesn't exist.
The collection will be automatically created when the application starts,
but this script can be used for manual creation or verification.

Usage:
    python backend/scripts/create_ocr_v2_collection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.indexing.milvus_client import get_milvus_client
from pymilvus import utility


def main() -> int:
    """Create and verify OCR v2 collection."""
    print("Creating OCR v2 collection (DiskANN + BM25)...")

    try:
        client = get_milvus_client()

        # Check if v2 already exists
        if utility.has_collection("ocr_embeddings_v2"):
            print("✓ ocr_embeddings_v2 already exists")

            # Verify schema
            col = client.collection_for_name("ocr_embeddings_v2")
            schema_fields = {f.name for f in col.schema.fields}

            required_fields = {"sparse_embedding", "has_embedding", "embedding", "text"}
            missing_fields = required_fields - schema_fields

            if missing_fields:
                print(f"✗ Schema incomplete, missing fields: {missing_fields}")
                return 1

            print("✓ Schema verified:")
            print(f"  - Dense field: embedding (384d)")
            print(f"  - Sparse field: sparse_embedding (BM25)")
            print(f"  - Text field: text (with Chinese analyzer)")
            print(f"  - Compatibility field: has_embedding")

            # Check if collection is loaded
            load_state = utility.load_state("ocr_embeddings_v2")
            if load_state.name == "Loaded":
                print("✓ Collection is loaded and ready")
            else:
                print(f"⚠ Collection exists but not loaded: {load_state.name}")
                print("  Loading collection...")
                col.load()
                print("✓ Collection loaded")

            return 0

        # Create v2 collection (will be created by _init_collections)
        print("Creating ocr_embeddings_v2 collection...")
        client._init_collections()

        if utility.has_collection("ocr_embeddings_v2"):
            print("✓ ocr_embeddings_v2 created successfully")

            # Verify
            col = client.collection_for_name("ocr_embeddings_v2")
            schema_fields = {f.name for f in col.schema.fields}
            assert "sparse_embedding" in schema_fields
            assert "has_embedding" in schema_fields
            assert "embedding" in schema_fields
            print("✓ Schema verified")

            return 0
        else:
            print("✗ Failed to create ocr_embeddings_v2")
            return 1

    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
