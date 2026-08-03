#!/usr/bin/env python3
"""End-to-end test for OCR hybrid search (DiskANN + BM25).

This test:
1. Creates a test collection with OCR schema
2. Creates DiskANN and BM25 indexes
3. Inserts test data
4. Performs hybrid search
5. Validates results
6. Cleans up
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from pymilvus import Collection, connections, utility, AnnSearchRequest, WeightedRanker
from app.indexing.milvus_schema import create_ocr_schema


def test_ocr_e2e():
    """Test OCR v2: schema → insert → hybrid search → cleanup."""

    print("="*70)
    print("OCR Hybrid Search E2E Test")
    print("="*70)

    # Connect to Milvus
    connections.connect(host="localhost", port=19531)
    print("\n✓ Connected to Milvus")

    # Clean up any existing test collection
    collection_name = "ocr_embeddings_e2e_test"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"✓ Dropped existing test collection")

    # Create collection with OCR schema
    schema = create_ocr_schema()
    col = Collection(name=collection_name, schema=schema)
    print(f"✓ Created collection: {collection_name}")

    # Create indexes
    print("\nCreating indexes...")

    # DiskANN index for dense vectors
    col.create_index("embedding", {
        "index_type": "DISKANN",
        "metric_type": "IP",
        "params": {
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": 0.125,
            "build_dram_budget_gb": 32.0,
        }
    })
    print("  ✓ Created DISKANN index on embedding field")

    # BM25 index for sparse vectors
    col.create_index("sparse_embedding", {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {"drop_ratio_build": 0.2}
    })
    print("  ✓ Created SPARSE_INVERTED_INDEX on sparse_embedding field")

    # Load collection
    col.load()
    print("✓ Collection loaded")

    # Insert test data
    print("\nInserting test data...")
    test_video_id = "test_video_e2e"
    test_data = [
        {
            "pk": str(uuid.uuid4()),
            "video_id": test_video_id,
            "asset_version": "v1",
            "model_version": "ocr_v2",
            "frame_idx": 0,
            "region_idx": 0,
            "frame_ms": 0,
            "start_ms": 0,
            "end_ms": 1000,
            "avg_box_score": 0.95,
            "text": "人工智能和机器学习技术的发展",
            "has_embedding": True,
            "embedding": np.random.rand(384).tolist(),
        },
        {
            "pk": str(uuid.uuid4()),
            "video_id": test_video_id,
            "asset_version": "v1",
            "model_version": "ocr_v2",
            "frame_idx": 1,
            "region_idx": 0,
            "frame_ms": 1000,
            "start_ms": 1000,
            "end_ms": 2000,
            "avg_box_score": 0.92,
            "text": "深度学习神经网络算法研究",
            "has_embedding": True,
            "embedding": np.random.rand(384).tolist(),
        },
        {
            "pk": str(uuid.uuid4()),
            "video_id": test_video_id,
            "asset_version": "v1",
            "model_version": "ocr_v2",
            "frame_idx": 2,
            "region_idx": 0,
            "frame_ms": 2000,
            "start_ms": 2000,
            "end_ms": 3000,
            "avg_box_score": 0.88,
            "text": "计算机视觉图像识别系统",
            "has_embedding": False,  # Test lexical-only frame
            "embedding": [0.0] * 384,
        },
        {
            "pk": str(uuid.uuid4()),
            "video_id": test_video_id,
            "asset_version": "v1",
            "model_version": "ocr_v2",
            "frame_idx": 3,
            "region_idx": 0,
            "frame_ms": 3000,
            "start_ms": 3000,
            "end_ms": 4000,
            "avg_box_score": 0.91,
            "text": "自然语言处理技术应用",
            "has_embedding": True,
            "embedding": np.random.rand(384).tolist(),
        }
    ]

    col.insert(test_data)
    col.flush()
    print(f"✓ Inserted {len(test_data)} test records")

    # Test 1: Hybrid search (semantic + lexical)
    print("\n--- Test 1: Hybrid Search ---")
    query_text = "机器学习技术"
    query_embedding = np.random.rand(384).tolist()

    print(f"Query: {query_text}")

    # Dense request (DiskANN)
    dense_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"search_list": 100}},
        limit=10,
        expr=f'video_id == "{test_video_id}" AND has_embedding == True'
    )

    # Sparse request (BM25)
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=10,
        expr=f'video_id == "{test_video_id}"'
    )

    # Hybrid search with weighted ranker (lexical 70%, semantic 30%)
    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=WeightedRanker(0.3, 0.7),
        limit=3,
        output_fields=["text", "frame_idx", "frame_ms", "has_embedding"]
    )

    print(f"✓ Hybrid search completed")
    print(f"  Results: {len(results[0])} hits\n")

    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] Score: {hit.score:.4f}")
        print(f"      Text: {hit.entity.get('text')}")
        print(f"      Frame: {hit.entity.get('frame_idx')} @ {hit.entity.get('frame_ms')}ms")
        print(f"      Has embedding: {hit.entity.get('has_embedding')}")

    assert len(results[0]) > 0, "Should return at least one result"

    # Test 2: BM25-only search
    print("\n--- Test 2: BM25-Only Search ---")
    results_bm25 = col.search(
        data=[query_text],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=3,
        expr=f'video_id == "{test_video_id}"',
        output_fields=["text", "frame_idx", "has_embedding"]
    )

    print(f"✓ BM25-only search completed")
    print(f"  Results: {len(results_bm25[0])} hits\n")

    for i, hit in enumerate(results_bm25[0]):
        print(f"  [{i+1}] Score: {hit.score:.4f}")
        print(f"      Text: {hit.entity.get('text')}")
        print(f"      Has embedding: {hit.entity.get('has_embedding')}")

    # Test 3: Dense-only search
    print("\n--- Test 3: Dense-Only Search ---")
    results_dense = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"search_list": 100}},
        limit=3,
        expr=f'video_id == "{test_video_id}" AND has_embedding == True',
        output_fields=["text", "frame_idx", "has_embedding"]
    )

    print(f"✓ Dense-only search completed")
    print(f"  Results: {len(results_dense[0])} hits\n")

    for i, hit in enumerate(results_dense[0]):
        print(f"  [{i+1}] Score: {hit.score:.4f}")
        print(f"      Text: {hit.entity.get('text')}")

    # Clean up
    utility.drop_collection(collection_name)
    print(f"\n✓ Cleaned up test collection")

    print("\n" + "="*70)
    print("✓ All E2E tests passed!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_ocr_e2e()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
