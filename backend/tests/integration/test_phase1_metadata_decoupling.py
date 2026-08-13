"""Phase 1 verification tests: metadata decoupling from manifest.json.

Tests that milvus_visual_candidates() works correctly with the ANN-based v2
implementation. The v2 path uses collection.search() (not query_iterator) and
infers all segment metadata directly from the Milvus hit fields.

NOTE: duration_ms / segment_ms / rows parameters are accepted for backward
compatibility but are NOT used by the ANN implementation.  Tests verify that
the function still returns correct results regardless of whether those params
are supplied.
"""
from unittest.mock import Mock, MagicMock, patch

import numpy as np

import pytest


def _make_mock_client(search_return, index_type: str = "DISKANN"):
    """Return a mock MilvusClient whose collection.search() yields search_return."""
    mock_client = MagicMock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Mock col.index() for _verify_index_type
    mock_index = Mock()
    mock_index.params = {"index_type": index_type}
    mock_collection.index.return_value = mock_index

    mock_collection.search.return_value = search_return
    return mock_client, mock_collection


def _make_mock_settings(use_diskann: bool = True, ann_top_k: int = 500):
    """Return a configured settings Mock."""
    s = Mock()
    s.visual_use_diskann = use_diskann
    s.visual_ann_top_k = ann_top_k
    s.milvus_query_timeout_seconds = 10.0
    return s


def _make_hit(
    frame_idx: int,
    timestamp_ms: int,
    segment_id: int,
    segment_start_ms: int,
    segment_end_ms: int,
    distance: float = 0.85,
) -> Mock:
    """Create a mock Milvus ANN hit."""
    data = {
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "segment_id": segment_id,
        "segment_start_ms": segment_start_ms,
        "segment_end_ms": segment_end_ms,
    }
    hit = Mock()
    hit.distance = distance
    hit.entity = Mock()
    hit.entity.get = lambda field, default=None: data.get(field, default)
    return hit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_milvus_visual_infers_segment_ms_from_bounds():
    """ANN path uses segment_start/end_ms from hit metadata directly."""
    from app.indexing.milvus_search import milvus_visual_candidates
    from app.indexing.milvus_search_visual_v2 import _reset_index_verification

    _reset_index_verification()

    hit = _make_hit(
        frame_idx=0, timestamp_ms=200,
        segment_id=0, segment_start_ms=0, segment_end_ms=5000,
        distance=0.85,
    )
    mock_client, mock_collection = _make_mock_client([[hit]])

    with patch("app.settings.get_settings", return_value=_make_mock_settings()):
        results = milvus_visual_candidates(
            mock_client, "test_video",
            np.random.randn(1152).astype(np.float32),
            duration_ms=None, segment_ms=None,
            profile="balanced", limit=10,
        )

    assert isinstance(results, list)
    assert len(results) >= 1
    # Segment boundary comes from the hit's segment_start/end_ms fields
    assert results[0].start_time == pytest.approx(0.0)
    assert results[0].end_time == pytest.approx(5.0)
    mock_collection.search.assert_called_once()


def test_milvus_visual_infers_duration_from_max_timestamp():
    """ANN path works correctly even when only max timestamp is in the hits."""
    from app.indexing.milvus_search import milvus_visual_candidates
    from app.indexing.milvus_search_visual_v2 import _reset_index_verification

    _reset_index_verification()

    hit = _make_hit(
        frame_idx=1, timestamp_ms=10000,
        segment_id=1, segment_start_ms=5000, segment_end_ms=10000,
        distance=0.82,
    )
    mock_client, mock_collection = _make_mock_client([[hit]])

    with patch("app.settings.get_settings", return_value=_make_mock_settings()):
        results = milvus_visual_candidates(
            mock_client, "test_video",
            np.random.randn(1152).astype(np.float32),
            duration_ms=None, segment_ms=None,
            profile="balanced", limit=10,
        )

    assert isinstance(results, list)
    assert len(results) >= 1
    mock_collection.search.assert_called_once()


def test_deprecated_params_accepted_without_error(caplog):
    """duration_ms/segment_ms/rows are accepted for backward compat; a warning is logged."""
    import logging
    from app.indexing.milvus_search import milvus_visual_candidates
    from app.indexing.milvus_search_visual_v2 import _reset_index_verification

    _reset_index_verification()

    hit = _make_hit(
        frame_idx=0, timestamp_ms=0,
        segment_id=0, segment_start_ms=0, segment_end_ms=5000,
        distance=0.78,
    )
    mock_client, mock_collection = _make_mock_client([[hit]])

    with patch("app.settings.get_settings", return_value=_make_mock_settings()):
        with caplog.at_level(logging.WARNING, logger="app.indexing.milvus_search"):
            results = milvus_visual_candidates(
                mock_client, "test_video",
                np.random.randn(1152).astype(np.float32),
                duration_ms=15000, segment_ms=5000,
                profile="balanced", limit=10,
            )

    # Function must still return valid results
    assert isinstance(results, list)
    # Deprecation warning must be emitted
    assert any("not used" in rec.message for rec in caplog.records), (
        "Expected a deprecation warning for duration_ms/segment_ms usage"
    )
    mock_collection.search.assert_called_once()


def test_empty_milvus_data_returns_empty_list():
    """Empty ANN result set → empty candidate list; no exception raised."""
    from app.indexing.milvus_search import milvus_visual_candidates
    from app.indexing.milvus_search_visual_v2 import _reset_index_verification

    _reset_index_verification()

    mock_client, mock_collection = _make_mock_client([[]])  # empty hit list

    with patch("app.settings.get_settings", return_value=_make_mock_settings()):
        results = milvus_visual_candidates(
            mock_client, "test_video",
            np.random.randn(1152).astype(np.float32),
            profile="balanced", limit=10,
        )

    assert results == []
    mock_collection.search.assert_called_once()
