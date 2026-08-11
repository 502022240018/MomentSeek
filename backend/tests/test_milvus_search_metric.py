"""Unit tests for milvus_search metric-type lookup and face COSINE scoring.

These tests do NOT require a running Milvus instance — they verify the static
lookup tables and that face candidate scoring trusts the COSINE distance
returned by Milvus (post IVF_FLAT/L2 → DISKANN/COSINE migration).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.vector_store.milvus.milvus_client import _COLLECTION_FOR_MODALITY
from app.vector_store.milvus.milvus_search import (
    _MODALITY_METRIC,
    get_modality_index_type,
    milvus_face_candidates,
)
from app.vector_store.milvus.milvus_search_visual_v2 import _aggregate_by_segment

# ---------------------------------------------------------------------------
# 1. Metric-type table completeness and consistency
# ---------------------------------------------------------------------------

def test_modality_metric_covers_all_modalities():
    """Every modality known to the client has an entry in _MODALITY_METRIC."""
    assert set(_MODALITY_METRIC) == set(_COLLECTION_FOR_MODALITY)


def test_modality_index_type_covers_all_modalities():
    """Every modality known to the client has an entry accessible via get_modality_index_type()."""
    # Verify all modalities can be queried
    for modality in _COLLECTION_FOR_MODALITY:
        index_type = get_modality_index_type(modality)
        assert index_type in ["DISKANN", "HNSW", "IVF_FLAT"], (
            f"modality '{modality}' returned unexpected index type: {index_type}"
        )


def test_modality_metric_matches_collection_configs():
    """_MODALITY_METRIC must be in sync with _COLLECTION_CONFIGS (the index definition)."""
    from app.vector_store.milvus.milvus_client import get_collection_index_config

    for modality, collection_name in _COLLECTION_FOR_MODALITY.items():
        # Get index config dynamically for visual, statically for others
        index_config = get_collection_index_config(collection_name)
        expected = index_config["metric_type"]
        actual   = _MODALITY_METRIC[modality]
        assert actual == expected, (
            f"modality '{modality}': _MODALITY_METRIC={actual!r} "
            f"but collection config says {expected!r}"
        )


def test_modality_index_type_matches_collection_configs():
    """get_modality_index_type() must return values matching collection configs."""
    from app.vector_store.milvus.milvus_client import get_collection_index_config

    for modality, collection_name in _COLLECTION_FOR_MODALITY.items():
        # Get index config dynamically
        index_config = get_collection_index_config(collection_name)
        expected = index_config["index_type"]
        actual = get_modality_index_type(modality)
        assert actual == expected, (
            f"modality '{modality}': get_modality_index_type()={actual!r} "
            f"but collection config says {expected!r}"
        )


# ---------------------------------------------------------------------------
# 2. Specific per-modality values (guards against accidental regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modality,expected_metric,expected_index", [
    ("visual",  "COSINE",   None),  # Visual uses dynamic config (DISKANN or HNSW)
    ("asr",     "IP",       "DISKANN"),  # ASR now uses DISKANN
    ("ocr",     "IP",       "DISKANN"),  # OCR now uses DISKANN
    ("face",    "COSINE",   "DISKANN"),  # migrated IVF_FLAT/L2 → DISKANN/COSINE for 千万级 scale
    ("speaker", "COSINE",   "DISKANN"),  # migrated HNSW → DISKANN for 千万级 scale
])
def test_per_modality_metric_and_index(modality, expected_metric, expected_index):
    assert _MODALITY_METRIC[modality] == expected_metric
    if expected_index is not None:  # Skip index check for visual (dynamic config)
        assert get_modality_index_type(modality) == expected_index


def test_visual_ann_uses_supported_retrieval_profiler_api():
    """Visual ANN must use span/increment, not the removed mark() API."""
    from app.retrieval.retrieval_metrics import RetrievalProfiler
    from app.vector_store.milvus.milvus_search_visual_v2 import (
        _reset_index_verification,
        milvus_visual_candidates_ann,
    )

    settings = MagicMock(
        visual_use_diskann=True,
        visual_ann_top_k=10,
        visual_ann_segment_top_n=1,
    )
    client = MagicMock()
    collection = MagicMock()
    client.collection_for.return_value = collection
    collection.index.return_value = MagicMock(params={"index_type": "DISKANN"})
    hit = MagicMock()
    hit.distance = 0.85
    hit.entity.get.side_effect = lambda field, default=None: {
        "frame_idx": 0,
        "timestamp_ms": 200,
        "segment_id": 0,
        "segment_start_ms": 0,
        "segment_end_ms": 5_000,
    }.get(field, default)
    collection.search.return_value = [[hit]]

    _reset_index_verification()
    try:
        with patch("app.core.settings.get_settings", return_value=settings):
            profiler = RetrievalProfiler()
            results = milvus_visual_candidates_ann(
                client,
                "test-video",
                "7",
                [np.ones(1152, dtype=np.float32)],
                limit=10,
                profiler=profiler,
            )
    finally:
        _reset_index_verification()

    snapshot = profiler.snapshot()
    assert results
    assert snapshot["timing"]["milvus_rpc"]["visual"] >= 0
    assert snapshot["counters"]["milvus"]["visual_requests"] == 1
    assert snapshot["counters"]["milvus"]["visual_rows"] == 1


# ---------------------------------------------------------------------------
# 3. Face trusted-COSINE distance (post IVF_FLAT/L2 → DISKANN/COSINE migration)
# ---------------------------------------------------------------------------

def test_face_candidates_trusts_cosine_distance():
    """milvus_face_candidates must trust Milvus' COSINE _distance directly.

    Post-migration, face_embeddings uses DISKANN + COSINE on unit vectors, so
    Milvus returns ``_distance`` that IS the exact cosine similarity. The former
    two-phase L2→cosine re-score was removed; this test mocks .search() to yield a
    known cosine value and verifies Candidate.raw_score equals it (no conversion,
    and no dependency on an ``embedding`` output field).
    """
    from unittest.mock import MagicMock

    # COSINE metric returns the cosine similarity directly as the distance.
    cosine_expected = 0.72

    fake_hit = MagicMock()
    fake_hit.distance = cosine_expected
    fake_hit.entity.get = lambda field, default=None: {
        "track_idx": 0,
        "start_ms":  0,
        "end_ms":    5000,
        "best_ms":   1000,
    }.get(field, default)

    fake_results = [[fake_hit]]

    fake_col = MagicMock()
    fake_col.search.return_value = fake_results

    fake_client = MagicMock()
    fake_client.collection_for.return_value = fake_col

    query = np.ones(512, dtype=np.float32)
    query /= np.linalg.norm(query)

    candidates = milvus_face_candidates(
        fake_client, "vid-test", query, "7", limit=5, threshold=0.35
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert abs(candidate.raw_score - cosine_expected) < 1e-5, (
        f"Expected raw_score≈{cosine_expected}, got {candidate.raw_score}"
    )
    # cosine 0.72 > threshold 0.35 → above_threshold must be True
    assert candidate.above_threshold is True
    assert candidate.decision == "absolute_hit"


def test_visual_segment_top_n_controls_frame_aggregation():
    """Top-1 keeps the peak frame while Top-3 averages the three best frames."""
    ann_results = [
        {
            "query_idx": 0,
            "frame_idx": frame_idx,
            "timestamp_ms": frame_idx * 1000,
            "segment_id": 7,
            "segment_start_ms": 0,
            "segment_end_ms": 5000,
            "cosine": cosine,
        }
        for frame_idx, cosine in enumerate((0.9, 0.6, 0.3))
    ]

    top_1 = _aggregate_by_segment(
        ann_results, "video-test", 1, "balanced", 1, segment_top_n=1
    )
    top_3 = _aggregate_by_segment(
        ann_results, "video-test", 1, "balanced", 1, segment_top_n=3
    )

    assert top_1[0].raw_score == pytest.approx(0.9)
    assert top_3[0].raw_score == pytest.approx(0.6)
