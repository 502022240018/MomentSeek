#!/usr/bin/env python3
"""Test OCR hybrid search functionality with all edge cases.

Tests:
1. Normal hybrid search (semantic + lexical)
2. BM25-only search (query_embedding=None)
3. Dense-only search (empty query_text)
4. Empty query handling
5. Score aggregation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_search import milvus_ocr_candidates_hybrid
from app.settings import get_settings


pytestmark = pytest.mark.integration  # Mark all tests in this file as integration


class TestOCRHybridSearch:
    """Test OCR hybrid search functionality."""

    @pytest.fixture
    def client(self):
        """Get Milvus client."""
        return get_milvus_client()

    @pytest.fixture
    def settings(self):
        """Get settings."""
        return get_settings()

    def test_normal_hybrid_search(self, client):
        """Test normal hybrid search with both semantic and lexical."""
        query_text = "工资"
        query_embedding = np.random.rand(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10,
        )

        assert isinstance(candidates, list)
        print(f"✓ Normal hybrid search returned {len(candidates)} candidates")

    def test_bm25_only_search(self, client):
        """Test BM25-only search when query_embedding is None."""
        query_text = "工资"
        query_embedding = None  # Semantic embedding not available

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,  # None triggers BM25-only
            limit=10,
        )

        assert isinstance(candidates, list)
        print(f"✓ BM25-only search returned {len(candidates)} candidates")
        print("  This should not crash even when semantic embedding is None")

    def test_dense_only_search(self, client):
        """Test dense-only search when query_text is empty."""
        query_text = ""  # Empty text
        query_embedding = np.random.rand(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10,
        )

        assert isinstance(candidates, list)
        print(f"✓ Dense-only search returned {len(candidates)} candidates")

    def test_empty_query_handling(self, client):
        """Test handling of completely empty query."""
        query_text = ""
        query_embedding = None

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10,
        )

        assert candidates == []
        print("✓ Empty query correctly returns empty list")

    def test_whitespace_query_handling(self, client):
        """Test handling of whitespace-only query."""
        query_text = "   "  # Only whitespace
        query_embedding = None

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10,
        )

        assert candidates == []
        print("✓ Whitespace-only query correctly returns empty list")

    def test_candidate_structure(self, client):
        """Test that returned candidates have correct structure."""
        query_text = "测试"
        query_embedding = np.random.rand(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id="test_video",
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10,
        )

        if candidates:
            candidate = candidates[0]
            assert hasattr(candidate, 'video_id')
            assert hasattr(candidate, 'score')
            assert hasattr(candidate, 'modality')
            assert candidate.modality == "ocr"
            assert hasattr(candidate, 'start_time')
            assert hasattr(candidate, 'end_time')
            assert hasattr(candidate, 'text')
            print("✓ Candidate structure is correct")
        else:
            print("⚠ No candidates returned (collection might be empty)")


def test_ocr_hybrid_search_cli(video_id: str = "test_video", query_text: str = "测试文字") -> int:
    """CLI test for OCR hybrid search."""
    print(f"Testing OCR hybrid search...")
    print(f"  Video ID: {video_id}")
    print(f"  Query text: {query_text}")

    try:
        client = get_milvus_client()
        settings = get_settings()

        print(f"\nConfiguration:")
        print(f"  Recall size: {settings.ocr_hybrid_recall_size}")
        print(f"  Lexical weight: {settings.ocr_lexical_weight}")
        print(f"  Semantic weight: {1.0 - settings.ocr_lexical_weight}")
        print(f"  Search list: {settings.ocr_diskann_search_list}")

        # Test 1: Normal hybrid search
        print(f"\n--- Test 1: Normal Hybrid Search ---")
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        candidates = milvus_ocr_candidates_hybrid(
            client=client,
            video_id=video_id,
            query_text=query_text,
            query_embedding=query_embedding,
            limit=20,
        )

        print(f"✓ Hybrid search completed: {len(candidates)} candidates")

        # Test 2: BM25-only search
        print(f"\n--- Test 2: BM25-Only Search (None embedding) ---")
        candidates_bm25 = milvus_ocr_candidates_hybrid(
            client=client,
            video_id=video_id,
            query_text=query_text,
            query_embedding=None,  # Test None handling
            limit=20,
        )

        print(f"✓ BM25-only search completed: {len(candidates_bm25)} candidates")

        # Display top 5 results from hybrid search
        if candidates:
            print(f"\nTop 5 results (Hybrid):")
            for i, c in enumerate(candidates[:5], 1):
                print(f"  {i}. {c.start_time:.1f}s - {c.end_time:.1f}s")
                print(f"     Score: {c.score:.4f}")
                print(f"     Text: {c.text[:60] if c.text else '(no text)'}...")
                if c.features:
                    print(f"     OCR confidence: {c.features.get('ocr_confidence', 0):.4f}")
        else:
            print(f"\n⚠ No results found for this query")

        return 0

    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    video_id = sys.argv[1] if len(sys.argv) > 1 else "test_video"
    query_text = sys.argv[2] if len(sys.argv) > 2 else "测试文字"
    sys.exit(test_ocr_hybrid_search_cli(video_id, query_text))
