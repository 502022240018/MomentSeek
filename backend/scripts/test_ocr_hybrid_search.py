#!/usr/bin/env python3
"""Test OCR hybrid search functionality.

Usage:
    python backend/scripts/test_ocr_hybrid_search.py [video_id] [query_text]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_search import milvus_ocr_candidates
from app.settings import get_settings


def test_ocr_hybrid_search(video_id: str = "test_video", query_text: str = "测试文字") -> int:
    """Test OCR hybrid search with a sample query."""
    print(f"Testing OCR hybrid search...")
    print(f"  Video ID: {video_id}")
    print(f"  Query text: {query_text}")

    try:
        # Get client and settings
        client = get_milvus_client()
        settings = get_settings()

        # Check if v2 collection exists
        try:
            col_v2 = client.collection_for_name("ocr_embeddings_v2")
            print(f"✓ Using ocr_embeddings_v2 (hybrid search)")
            print(f"  Recall size: {settings.ocr_hybrid_recall_size}")
            print(f"  Lexical weight: {settings.ocr_lexical_weight}")
            print(f"  Semantic weight: {1.0 - settings.ocr_lexical_weight}")
        except:
            print(f"⚠ ocr_embeddings_v2 not found, using legacy v1")

        # Generate random query embedding (384d for OCR)
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Perform search
        print(f"\nPerforming search...")
        candidates = milvus_ocr_candidates(
            client=client,
            video_id=video_id,
            query_text=query_text,
            query_embedding=query_embedding,
            limit=20,
        )

        print(f"\n✓ Search completed")
        print(f"  Results: {len(candidates)} candidates")

        # Display top 5 results
        if candidates:
            print(f"\nTop 5 results:")
            for i, c in enumerate(candidates[:5], 1):
                print(f"  {i}. {c.start_time:.1f}s - {c.end_time:.1f}s")
                print(f"     Score: {c.score:.4f}")
                print(f"     Text: {c.text[:80] if c.text else '(no text)'}...")
                if c.features:
                    if "hybrid_score" in c.features:
                        print(f"     Hybrid score: {c.features['hybrid_score']:.4f}")
                        print(f"     OCR confidence: {c.features.get('ocr_confidence', 0):.4f}")
                        print(f"     Source: {c.features.get('source', 'legacy')}")
        else:
            print(f"\n⚠ No results found for this query")

        return 0

    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    video_id = sys.argv[1] if len(sys.argv) > 1 else "test_video"
    query_text = sys.argv[2] if len(sys.argv) > 2 else "测试文字"

    return test_ocr_hybrid_search(video_id, query_text)


if __name__ == "__main__":
    sys.exit(main())
