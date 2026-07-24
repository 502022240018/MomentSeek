"""Unit tests for milvus_search metric-type lookup and L2→cosine conversion.

These tests do NOT require a running Milvus instance — they verify the static
lookup tables and conversion math that were incorrect before the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.indexing.milvus_client import _COLLECTION_CONFIGS, _COLLECTION_FOR_MODALITY
from app.indexing.milvus_search import (
    _MODALITY_INDEX_TYPE,
    _MODALITY_METRIC,
    milvus_face_candidates,
)


# ---------------------------------------------------------------------------
# 1. Metric-type table completeness and consistency
# ---------------------------------------------------------------------------

def test_modality_metric_covers_all_modalities():
    """Every modality known to the client has an entry in _MODALITY_METRIC."""
    assert set(_MODALITY_METRIC) == set(_COLLECTION_FOR_MODALITY)


def test_modality_index_type_covers_all_modalities():
    """Every modality known to the client has an entry in _MODALITY_INDEX_TYPE."""
    assert set(_MODALITY_INDEX_TYPE) == set(_COLLECTION_FOR_MODALITY)


def test_modality_metric_matches_collection_configs():
    """_MODALITY_METRIC must be in sync with _COLLECTION_CONFIGS (the index definition)."""
    for modality, collection_name in _COLLECTION_FOR_MODALITY.items():
        expected = _COLLECTION_CONFIGS[collection_name]["index"]["metric_type"]
        actual   = _MODALITY_METRIC[modality]
        assert actual == expected, (
            f"modality '{modality}': _MODALITY_METRIC={actual!r} "
            f"but _COLLECTION_CONFIGS says {expected!r}"
        )


def test_modality_index_type_matches_collection_configs():
    """_MODALITY_INDEX_TYPE must be in sync with _COLLECTION_CONFIGS."""
    for modality, collection_name in _COLLECTION_FOR_MODALITY.items():
        expected = _COLLECTION_CONFIGS[collection_name]["index"]["index_type"]
        actual   = _MODALITY_INDEX_TYPE[modality]
        assert actual == expected, (
            f"modality '{modality}': _MODALITY_INDEX_TYPE={actual!r} "
            f"but _COLLECTION_CONFIGS says {expected!r}"
        )


# ---------------------------------------------------------------------------
# 2. Specific per-modality values (guards against accidental regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modality,expected_metric,expected_index", [
    ("visual",  "COSINE",   "HNSW"),
    ("asr",     "IP",       "HNSW"),
    ("ocr",     "IP",       "HNSW"),
    ("face",    "L2",       "IVF_FLAT"),
    ("speaker", "COSINE",   "HNSW"),
])
def test_per_modality_metric_and_index(modality, expected_metric, expected_index):
    assert _MODALITY_METRIC[modality]     == expected_metric
    assert _MODALITY_INDEX_TYPE[modality] == expected_index


# ---------------------------------------------------------------------------
# 3. L2 → cosine conversion (face modality)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cosine", [-1.0, -0.5, 0.0, 0.35, 0.5, 0.8, 0.9, 1.0])
def test_l2_cosine_round_trip(cosine):
    """Milvus squared-L2 converts to cosine exactly for unit vectors."""
    # Milvus L2 metric returns the squared Euclidean distance.
    l2 = max(0.0, 2.0 * (1.0 - cosine))
    recovered = 1.0 - l2 / 2.0
    assert abs(recovered - cosine) < 1e-6, (
        f"cosine={cosine} → L2={l2:.6f} → recovered={recovered:.6f}"
    )


def test_face_candidates_l2_to_cosine_conversion():
    """milvus_face_candidates must convert L2 distances to cosine before scoring.

    We mock the _search internals by patching the client's collection_for to
    return an object whose .search() method yields known L2 values, then verify
    that the resulting Candidate.raw_score equals the expected cosine.
    """
    from unittest.mock import MagicMock, patch

    # Known cosine value we want to recover
    cosine_expected = 0.72
    # Squared L2 distance corresponding to that cosine for normalized vectors.
    l2_dist = 2.0 * (1.0 - cosine_expected)

    # Build a fake Milvus hit object
    fake_hit = MagicMock()
    fake_hit.distance = l2_dist
    fake_hit.entity.get = lambda field, default=None: {
        "track_idx": 0,
        "start_ms":  0,
        "end_ms":    5000,
        "best_ms":   1000,
    }.get(field, default)

    fake_results = [[fake_hit]]

    # Fake collection whose .search() returns our synthetic L2 result
    fake_col = MagicMock()
    fake_col.search.return_value = fake_results

    fake_client = MagicMock()
    fake_client.collection_for.return_value = fake_col

    query = np.ones(512, dtype=np.float32)
    query /= np.linalg.norm(query)

    candidates = milvus_face_candidates(fake_client, "vid-test", query, limit=5, threshold=0.35)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert abs(candidate.raw_score - cosine_expected) < 1e-5, (
        f"Expected raw_score≈{cosine_expected}, got {candidate.raw_score}"
    )
    # cosine 0.72 > threshold 0.35 → above_threshold must be True
    assert candidate.above_threshold is True
    assert candidate.decision == "absolute_hit"
