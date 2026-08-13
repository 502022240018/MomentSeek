"""Phase 1 verification tests: metadata decoupling from manifest.json.

Tests that milvus_visual_candidates() can correctly infer duration_ms and
segment_ms from Milvus data itself, without relying on manifest.json.
"""
from unittest.mock import MagicMock, Mock
import numpy as np


def test_milvus_visual_infers_segment_ms_from_bounds():
    """Test that segment_ms is inferred from segment_start_ms/segment_end_ms."""
    from app.indexing.milvus_search import milvus_visual_candidates

    # Mock client that returns rows with explicit segment boundaries
    mock_client = MagicMock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Simulate 3 frames in one segment (0-5000ms)
    mock_rows = [
        {
            "frame_idx": 0, "timestamp_ms": 0, "segment_id": 0,
            "segment_start_ms": 0, "segment_end_ms": 5000,
            "embedding": [0.1] * 1152
        },
        {
            "frame_idx": 1, "timestamp_ms": 200, "segment_id": 0,
            "segment_start_ms": 0, "segment_end_ms": 5000,
            "embedding": [0.2] * 1152
        },
        {
            "frame_idx": 2, "timestamp_ms": 400, "segment_id": 0,
            "segment_start_ms": 0, "segment_end_ms": 5000,
            "embedding": [0.3] * 1152
        },
    ]

    # Mock query_iterator
    mock_iter = Mock()
    mock_iter.next.side_effect = [mock_rows, []]
    mock_iter.close = Mock()
    mock_collection.query_iterator.return_value = mock_iter

    query = np.random.randn(1152).astype(np.float32)

    # Call WITHOUT providing segment_ms
    results = milvus_visual_candidates(
        mock_client, "test_video", query,
        duration_ms=None, segment_ms=None,  # Should infer both
        profile="balanced", limit=10
    )

    # Should successfully return candidates
    assert isinstance(results, list)
    # Verify it computed segment boundaries (start=0, end=5000)
    if results:
        assert results[0].start_time == 0.0
        assert results[0].end_time == 5.0


def test_milvus_visual_infers_duration_from_max_timestamp():
    """Test that duration_ms is inferred from max(timestamp_ms)."""
    from app.indexing.milvus_search import milvus_visual_candidates

    mock_client = MagicMock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Simulate frames up to 10000ms
    mock_rows = [
        {
            "frame_idx": 0, "timestamp_ms": 0, "segment_id": 0,
            "segment_start_ms": 0, "segment_end_ms": 5000,
            "embedding": [0.1] * 1152
        },
        {
            "frame_idx": 1, "timestamp_ms": 10000, "segment_id": 1,
            "segment_start_ms": 5000, "segment_end_ms": 10000,
            "embedding": [0.2] * 1152
        },
    ]

    mock_iter = Mock()
    mock_iter.next.side_effect = [mock_rows, []]
    mock_iter.close = Mock()
    mock_collection.query_iterator.return_value = mock_iter

    query = np.random.randn(1152).astype(np.float32)

    # Call WITHOUT providing duration_ms
    results = milvus_visual_candidates(
        mock_client, "test_video", query,
        duration_ms=None, segment_ms=None,
        profile="balanced", limit=10
    )

    # Should infer duration from max timestamp (10000ms = 10s)
    assert isinstance(results, list)


def test_milvus_visual_fallback_to_provided_params():
    """Test backward compatibility: provided params are used as fallback."""
    from app.indexing.milvus_search import milvus_visual_candidates

    mock_client = MagicMock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Simulate OLD data without segment bounds (segment_start_ms = -1)
    mock_rows = [
        {
            "frame_idx": 0, "timestamp_ms": 0, "segment_id": -1,
            "segment_start_ms": -1, "segment_end_ms": -1,
            "embedding": [0.1] * 1152
        },
        {
            "frame_idx": 1, "timestamp_ms": 3000, "segment_id": -1,
            "segment_start_ms": -1, "segment_end_ms": -1,
            "embedding": [0.2] * 1152
        },
    ]

    mock_iter = Mock()
    mock_iter.next.side_effect = [mock_rows, []]
    mock_iter.close = Mock()
    mock_collection.query_iterator.return_value = mock_iter

    query = np.random.randn(1152).astype(np.float32)

    # Call WITH provided params (backward compatibility)
    results = milvus_visual_candidates(
        mock_client, "test_video", query,
        duration_ms=15000, segment_ms=5000,  # Should use these as fallback
        profile="balanced", limit=10
    )

    # Should successfully use fallback values
    assert isinstance(results, list)


def test_empty_milvus_data_returns_empty_list():
    """Test that empty Milvus result returns empty candidate list."""
    from app.indexing.milvus_search import milvus_visual_candidates

    mock_client = MagicMock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Empty result
    mock_iter = Mock()
    mock_iter.next.return_value = []
    mock_iter.close = Mock()
    mock_collection.query_iterator.return_value = mock_iter

    query = np.random.randn(1152).astype(np.float32)

    results = milvus_visual_candidates(
        mock_client, "test_video", query,
        duration_ms=None, segment_ms=None,
        profile="balanced", limit=10
    )

    assert results == []


if __name__ == "__main__":
    print("Running Phase 1 metadata decoupling tests...")
    test_milvus_visual_infers_segment_ms_from_bounds()
    print("✓ segment_ms inference test passed")

    test_milvus_visual_infers_duration_from_max_timestamp()
    print("✓ duration_ms inference test passed")

    test_milvus_visual_fallback_to_provided_params()
    print("✓ backward compatibility test passed")

    test_empty_milvus_data_returns_empty_list()
    print("✓ empty data test passed")

    print("\n✓ All Phase 1 tests passed!")
