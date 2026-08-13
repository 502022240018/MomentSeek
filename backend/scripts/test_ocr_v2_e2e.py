#!/usr/bin/env python3
"""End-to-end test for OCR v2 hybrid search (DiskANN + BM25)."""

import sys
import uuid
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import connections, utility

from app.indexing.milvus_client import MilvusClient
from app.indexing.milvus_schema import create_ocr_schema


def test_ocr_v2_e2e():
    """Test OCR v2: schema → insert → hybrid search."""

    # Connect to Milvus
    connections.connect(host="localhost", port=19531)

    # Clean up any existing test collection
    collection_name = "ocr_embeddings_v2_test"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"✓ Dropped existing test collection: {collection_name}")

    # Create collection with OCR schema
    from pymilvus import Collection
    schema = create_ocr_schema()
    col = Collection(name=collection_name, schema=schema)
    print(f"✓ Created collection: {collection_name}")

    # Create indexes
    col.create_index("embedding", {
        "index_type": "DISKANN",
        "metric_type": "IP",
        "params": {"search_list": 200}
    })
    print("✓ Created DISKANN index on embedding field")

    col.create_index("sparse_embedding", {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {"drop_ratio_build": 0.2}
    })
    print("✓ Created SPARSE_INVERTED_INDEX on sparse_embedding field")

    # Load collection
    col.load()
    print("✓ Collection loaded")

    # Insert test data
    test_video_id = "test_video_001"
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
            "has_embedding": False,
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

    # Test hybrid search
    from pymilvus import AnnSearchRequest, WeightedRanker

    query_text = "机器学习技术"
    query_embedding = np.random.rand(384).tolist()

    print(f"\n--- Hybrid Search Test ---")
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

    print(f"\n✓ Hybrid search completed")
    print(f"  Results: {len(results[0])} hits\n")

    for i, hit in enumerate(results[0]):
        print(f"  [{i+1}] Score: {hit.score:.4f}")
        print(f"      Text: {hit.entity.get('text')}")
        print(f"      Frame: {hit.entity.get('frame_idx')} @ {hit.entity.get('frame_ms')}ms")
        print(f"      Has embedding: {hit.entity.get('has_embedding')}")
        print()

    # Clean up
    utility.drop_collection(collection_name)
    print(f"✓ Cleaned up test collection: {collection_name}")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_ocr_v2_e2e()
