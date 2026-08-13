#!/usr/bin/env python3
"""Test script for simplified visual ANN search.

Tests:
1. Syntax and imports
2. Single query
3. Multi-query aggregation
4. Index type verification
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
from app.indexing.milvus_search_visual_v2 import (
    milvus_visual_candidates_ann,
    MilvusVisualSearchError,
    _normalize,
    _aggregate_by_segment,
)

def test_normalize():
    """Test vector normalization."""
    vec = np.array([3.0, 4.0])
    normalized = _normalize(vec)
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 1e-6, f"Expected norm=1.0, got {norm}"
    print("✓ Vector normalization works")

def test_multi_query_aggregation():
    """Test multi-query aggregation logic."""
    # Mock ANN results: 2 queries, 2 frames in same segment
    ann_results = [
        # Frame 1, query 0
        {"query_idx": 0, "frame_idx": 1, "segment_id": 1, "timestamp_ms": 1000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.8},
        # Frame 1, query 1
        {"query_idx": 1, "frame_idx": 1, "segment_id": 1, "timestamp_ms": 1000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.6},
        # Frame 2, query 0
        {"query_idx": 0, "frame_idx": 2, "segment_id": 1, "timestamp_ms": 2000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.7},
        # Frame 2, query 1
        {"query_idx": 1, "frame_idx": 2, "segment_id": 1, "timestamp_ms": 2000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.5},
    ]

    candidates = _aggregate_by_segment(
        ann_results,
        video_id="test_video",
        limit=10,
        profile="balanced",
        n_queries=2,
    )

    assert len(candidates) == 1, f"Expected 1 segment candidate, got {len(candidates)}"

    # Frame 1: 0.65 * 0.7 + 0.35 * 0.6 = 0.665
    # Frame 2: 0.65 * 0.6 + 0.35 * 0.5 = 0.565
    # Segment: mean([0.665, 0.565]) = 0.615
    expected_score = (0.665 + 0.565) / 2
    actual_score = candidates[0].raw_score

    assert abs(actual_score - expected_score) < 0.01, \
        f"Expected segment score ~{expected_score:.3f}, got {actual_score:.3f}"

    print(f"✓ Multi-query aggregation correct: {actual_score:.3f}")

def test_single_query_aggregation():
    """Test single query aggregation (no cross-query logic)."""
    ann_results = [
        {"query_idx": 0, "frame_idx": 1, "segment_id": 1, "timestamp_ms": 1000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.8},
        {"query_idx": 0, "frame_idx": 2, "segment_id": 1, "timestamp_ms": 2000,
         "segment_start_ms": 0, "segment_end_ms": 5000, "cosine": 0.7},
    ]

    candidates = _aggregate_by_segment(
        ann_results,
        video_id="test_video",
        limit=10,
        profile="balanced",
        n_queries=1,
    )

    assert len(candidates) == 1
    # Single query: use scores directly, then segment mean
    expected_score = (0.8 + 0.7) / 2
    actual_score = candidates[0].raw_score

    assert abs(actual_score - expected_score) < 0.01, \
        f"Expected segment score ~{expected_score:.3f}, got {actual_score:.3f}"

    print(f"✓ Single-query aggregation correct: {actual_score:.3f}")

def test_imports():
    """Test that all required modules can be imported."""
    from app.indexing.milvus_client import MilvusClient
    from app.indexing.milvus_search import milvus_visual_candidates
    from app.search import Candidate, visual_confidence
    print("✓ All imports successful")

def main():
    print("Testing simplified visual ANN search...\n")

    try:
        test_imports()
        test_normalize()
        test_single_query_aggregation()
        test_multi_query_aggregation()

        print("\n✅ All tests passed!")
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
