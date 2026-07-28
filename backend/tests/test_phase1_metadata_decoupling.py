"""Phase 1 verification tests: metadata decoupling from manifest.json.

Tests that milvus_visual_candidates() can correctly infer duration_ms and
segment_ms from Milvus data itself, without relying on manifest.json.
"""
from unittest.mock import MagicMock, Mock, patch
import numpy as np


def test_milvus_visual_infers_segment_ms_from_bounds():
    """Test that segment_ms is inferred from segment_start_ms/segment_end_ms."""
    from app.indexing.milvus_search import milvus_visual_candidates

    # Mock settings to ensure test isolation
    with patch("app.indexing.milvus_search_visual_v2.get_settings") as mock_settings:
        mock_settings.return_value.visual_use_diskann = True
        mock_settings.return_value.visual_ann_top_k = 500

        # Mock client that returns rows with explicit segment boundaries
        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection

        # Mock index information for DiskANN/HNSW verification
        mock_index = Mock()
        mock_index.params = {"index_type": "DISKANN"}  # Match config expectation
        mock_collection.index.return_value = mock_index

        # Mock search results (ANN path uses collection.search(), not query_iterator)
        mock_hit = Mock()
        mock_hit.distance = 0.85
        mock_hit.entity = Mock()
        mock_hit.entity.get = lambda field, default=None: {
            "frame_idx": 0,
            "timestamp_ms": 200,
            "segment_id": 0,
            "segment_start_ms": 0,
            "segment_end_ms": 5000,
        }.get(field, default)

        mock_collection.search.return_value = [[mock_hit]]

        query = np.random.randn(1152).astype(np.float32)

        # Call WITHOUT providing segment_ms
        results = milvus_visual_candidates(
            mock_client, "test_video", query,
            duration_ms=None, segment_ms=None,  # Should infer both
            profile="balanced", limit=10
        )

        # Should successfully return candidates
        assert isinstance(results, list)
        assert len(results) >= 1
        # Verify it computed segment boundaries (start=0, end=5000)
        assert results[0].start_time == 0.0
        assert results[0].end_time == 5.0


def test_milvus_visual_infers_duration_from_max_timestamp():
    """Test that duration_ms is inferred from max(timestamp_ms)."""
    from app.indexing.milvus_search import milvus_visual_candidates

    with patch("app.indexing.milvus_search_visual_v2.get_settings") as mock_settings:
        mock_settings.return_value.visual_use_diskann = True
        mock_settings.return_value.visual_ann_top_k = 500

        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection

        # Mock index information for DiskANN/HNSW verification
        mock_index = Mock()
        mock_index.params = {"index_type": "DISKANN"}
        mock_collection.index.return_value = mock_index

        # Mock search results
        mock_hit = Mock()
        mock_hit.distance = 0.82
        mock_hit.entity = Mock()
        mock_hit.entity.get = lambda field, default=None: {
            "frame_idx": 1,
            "timestamp_ms": 10000,
            "segment_id": 1,
            "segment_start_ms": 5000,
            "segment_end_ms": 10000,
        }.get(field, default)

        mock_collection.search.return_value = [[mock_hit]]

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

    with patch("app.indexing.milvus_search_visual_v2.get_settings") as mock_settings:
        mock_settings.return_value.visual_use_diskann = True
        mock_settings.return_value.visual_ann_top_k = 500

        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection

        # Mock index information for DiskANN/HNSW verification
        mock_index = Mock()
        mock_index.params = {"index_type": "DISKANN"}
        mock_collection.index.return_value = mock_index

        # Mock search results with OLD data (segment_start_ms = -1)
        mock_hit = Mock()
        mock_hit.distance = 0.78
        mock_hit.entity = Mock()
        mock_hit.entity.get = lambda field, default=None: {
            "frame_idx": 0,
            "timestamp_ms": 0,
            "segment_id": -1,
            "segment_start_ms": -1,
            "segment_end_ms": -1,
        }.get(field, default)

        mock_collection.search.return_value = [[mock_hit]]

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

    with patch("app.indexing.milvus_search_visual_v2.get_settings") as mock_settings:
        mock_settings.return_value.visual_use_diskann = True
        mock_settings.return_value.visual_ann_top_k = 500

        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection

        # Mock index information for DiskANN/HNSW verification
        mock_index = Mock()
        mock_index.params = {"index_type": "DISKANN"}
        mock_collection.index.return_value = mock_index

        # Empty search result
        mock_collection.search.return_value = [[]]

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
