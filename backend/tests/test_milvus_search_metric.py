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
    get_modality_metric_type,
    milvus_asr_candidates_hybrid,
    milvus_face_candidates,
    milvus_ocr_candidates_hybrid,
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
        actual = get_modality_metric_type(modality)
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
    collection.index.return_value = MagicMock(
        params={"index_type": "DISKANN", "metric_type": "COSINE"}
    )
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


def test_visual_ann_passes_configured_milvus_timeout():
    from app.vector_store.milvus.milvus_search_visual_v2 import (
        _ann_recall_multi_query,
    )

    collection = MagicMock()
    collection.search.return_value = [[]]
    client = MagicMock()
    client.collection_for.return_value = collection
    settings = MagicMock(milvus_query_timeout_seconds=7.5)

    with patch("app.core.settings.get_settings", return_value=settings):
        assert _ann_recall_multi_query(
            client,
            "video-test",
            "asset-test",
            np.ones((1, 1152), dtype=np.float32),
            top_k=10,
            use_diskann=False,
            profiler=None,
        ) == []

    assert collection.search.call_args.kwargs["timeout"] == 7.5


def test_visual_index_transient_introspection_failure_is_not_cached():
    from app.vector_store.milvus.milvus_search_visual_v2 import (
        _reset_index_verification,
        _verify_index_type_once,
    )

    collection = MagicMock()
    collection.index.side_effect = [
        RuntimeError("temporary timeout"),
        MagicMock(params={"index_type": "DISKANN", "metric_type": "COSINE"}),
    ]
    client = MagicMock()
    client.collection_for.return_value = collection

    _reset_index_verification()
    try:
        _verify_index_type_once(client, True)
        _verify_index_type_once(client, True)
    finally:
        _reset_index_verification()

    assert collection.index.call_count == 2


def test_visual_index_rejects_non_cosine_metric():
    from app.vector_store.milvus.milvus_search_visual_v2 import (
        MilvusVisualSearchError,
        _reset_index_verification,
        _verify_index_type_once,
    )

    collection = MagicMock()
    collection.index.return_value = MagicMock(
        params={"index_type": "DISKANN", "metric_type": "IP"}
    )
    client = MagicMock()
    client.collection_for.return_value = collection

    _reset_index_verification()
    try:
        with pytest.raises(MilvusVisualSearchError, match="expects COSINE"):
            _verify_index_type_once(client, True)
    finally:
        _reset_index_verification()


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


def test_face_legacy_ivf_l2_profile_exactly_rescores_embeddings():
    """Legacy IVF/L2 recall must be re-ranked with exact ArcFace cosine."""
    from app.core.settings import Settings
    from app.vector_store.milvus.milvus_search import _reset_index_verification

    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0
    weaker = np.zeros(512, dtype=np.float32)
    weaker[0] = 0.6
    weaker[1] = 0.8
    stronger = query.copy()

    def face_hit(track_idx: int, l2: float, embedding: np.ndarray):
        hit = MagicMock()
        hit.distance = l2
        hit.entity.get.side_effect = lambda field, default=None: {
            "track_idx": track_idx,
            "start_ms": track_idx * 1000,
            "end_ms": (track_idx + 1) * 1000,
            "best_ms": track_idx * 1000,
            "embedding": embedding.tolist(),
        }.get(field, default)
        return hit

    collection = MagicMock()
    collection.index.return_value = MagicMock(
        params={"index_type": "IVF_FLAT", "metric_type": "L2"}
    )
    # Deliberately return the weaker cosine hit first. Exact re-scoring must
    # put track 1 first regardless of mocked ANN order/L2 values.
    collection.search.return_value = [[
        face_hit(0, 0.1, weaker),
        face_hit(1, 0.2, stronger),
    ]]
    client = MagicMock()
    client.collection_for.return_value = collection
    settings = Settings(
        milvus_face_ann_profile="ivf_flat_l2",
        face_ivf_nprobe=37,
        face_recall_multiplier=1,
    )

    _reset_index_verification()
    try:
        with patch(
            "app.vector_store.milvus.milvus_search.get_settings",
            return_value=settings,
        ):
            candidates = milvus_face_candidates(
                client,
                "vid-legacy",
                query,
                "legacy-version",
                limit=2,
                threshold=0.35,
            )
    finally:
        _reset_index_verification()

    assert [candidate.unit_id for candidate in candidates] == [1, 0]
    assert [candidate.raw_score for candidate in candidates] == pytest.approx([1.0, 0.6])
    search_kwargs = collection.search.call_args.kwargs
    assert search_kwargs["param"] == {
        "metric_type": "L2",
        "params": {"nprobe": 37},
    }
    assert search_kwargs["limit"] == 4
    assert "embedding" in search_kwargs["output_fields"]


def test_face_collection_config_follows_explicit_legacy_profile():
    from app.core.settings import Settings
    from app.vector_store.milvus.milvus_client import get_collection_index_config

    settings = Settings(milvus_face_ann_profile="ivf_flat_l2")
    with patch(
        "app.vector_store.milvus.milvus_client.get_settings",
        return_value=settings,
    ):
        config = get_collection_index_config("face_embeddings")

    assert config == {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    }


def _candidate_hit(fields: dict, *, score: float = 0.8, distance: float = 0.8):
    hit = MagicMock()
    hit.score = score
    hit.distance = distance
    hit.entity.get.side_effect = lambda field, default=None: fields.get(field, default)
    return hit


def test_asr_candidates_require_explicit_valid_time_window(caplog):
    valid = _candidate_hit({
        "segment_idx": 0,
        "start_ms": 0,
        "end_ms": 1000,
        "text": "hello",
        "has_embedding": False,
    })
    missing = _candidate_hit({
        "segment_idx": 1,
        "start_ms": 1000,
        "text": "missing end",
        "has_embedding": False,
    })
    invalid = _candidate_hit({
        "segment_idx": 2,
        "start_ms": 2000,
        "end_ms": 2000,
        "text": "empty window",
        "has_embedding": False,
    })
    collection = MagicMock()
    collection.search.return_value = [[valid, missing, invalid]]
    client = MagicMock()
    client.collection_for.return_value = collection

    candidates = milvus_asr_candidates_hybrid(
        client,
        "video-test",
        "asset-test",
        "hello",
        None,
        10,
    )

    assert [(item.start_time, item.end_time) for item in candidates] == [(0.0, 1.0)]
    assert "ASR search dropped 2" in caplog.text


def test_ocr_candidates_require_explicit_valid_time_window(caplog):
    valid = _candidate_hit({
        "frame_idx": 0,
        "frame_ms": 700,
        "start_ms": 0,
        "end_ms": 900,
        "text": "cash register",
        "avg_box_score": 0.9,
        "has_embedding": False,
    })
    missing = _candidate_hit({
        "frame_idx": 1,
        "frame_ms": 1000,
        "start_ms": 900,
        "text": "missing end",
        "has_embedding": False,
    })
    invalid = _candidate_hit({
        "frame_idx": 2,
        "frame_ms": 1500,
        "start_ms": 1600,
        "end_ms": 1500,
        "text": "reversed window",
        "has_embedding": False,
    })
    collection = MagicMock()
    collection.search.return_value = [[valid, missing, invalid]]
    client = MagicMock()
    client.collection_for.return_value = collection

    candidates = milvus_ocr_candidates_hybrid(
        client,
        "video-test",
        "asset-test",
        "cash register",
        None,
        10,
    )

    # A legitimate zero start is preserved, not treated as a legacy sentinel and
    # re-inferred from frame_ms.
    assert [(item.start_time, item.end_time) for item in candidates] == [(0.0, 0.9)]
    assert "OCR search dropped 2" in caplog.text


def test_face_candidates_require_explicit_valid_time_window(caplog):
    valid = _candidate_hit({
        "track_idx": 0,
        "start_ms": 0,
        "end_ms": 5000,
        "best_ms": 0,
    }, distance=0.72)
    missing = _candidate_hit({
        "track_idx": 1,
        "start_ms": 5000,
        "best_ms": 5200,
    }, distance=0.71)
    invalid = _candidate_hit({
        "track_idx": 2,
        "start_ms": 6000,
        "end_ms": 6000,
        "best_ms": 6000,
    }, distance=0.70)
    collection = MagicMock()
    collection.search.return_value = [[valid, missing, invalid]]
    client = MagicMock()
    client.collection_for.return_value = collection
    query = np.ones(512, dtype=np.float32)

    with patch(
        "app.vector_store.milvus.milvus_search._verify_ann_index_type_once"
    ):
        candidates = milvus_face_candidates(
            client,
            "video-test",
            query,
            "asset-test",
            limit=10,
            threshold=0.35,
        )

    assert [(item.start_time, item.end_time, item.best_ms) for item in candidates] == [
        (0.0, 5.0, 0)
    ]
    assert "FACE search dropped 2" in caplog.text


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


def test_visual_aggregation_drops_zero_length_time_bounds():
    invalid = [{
        "query_idx": 0,
        "frame_idx": 0,
        "timestamp_ms": 0,
        "segment_id": 0,
        "segment_start_ms": 0,
        "segment_end_ms": 0,
        "cosine": 0.9,
    }]

    assert _aggregate_by_segment(invalid, "video-test", 10, "balanced", 1) == []


def test_visual_aggregation_drops_entire_segment_with_conflicting_bounds():
    inconsistent = [
        {
            "query_idx": 0,
            "frame_idx": 0,
            "timestamp_ms": 500,
            "segment_id": 3,
            "segment_start_ms": 0,
            "segment_end_ms": 1000,
            "cosine": 0.9,
        },
        {
            "query_idx": 0,
            "frame_idx": 1,
            "timestamp_ms": 1500,
            "segment_id": 3,
            "segment_start_ms": 1000,
            "segment_end_ms": 2000,
            "cosine": 0.8,
        },
    ]

    assert _aggregate_by_segment(inconsistent, "video-test", 10, "balanced", 1) == []


def test_visual_ann_does_not_default_missing_time_fields_to_zero():
    from app.vector_store.milvus.milvus_search_visual_v2 import _ann_recall_multi_query

    hit = MagicMock()
    hit.distance = 0.9
    hit.entity.get.side_effect = lambda field: {
        "frame_idx": 0,
        "timestamp_ms": 100,
        "segment_id": 0,
        "segment_start_ms": 0,
        # segment_end_ms intentionally absent
    }.get(field)
    collection = MagicMock()
    collection.search.return_value = [[hit]]
    client = MagicMock()
    client.collection_for.return_value = collection

    results = _ann_recall_multi_query(
        client,
        "video-test",
        "1",
        np.ones((1, 1152), dtype=np.float32),
        top_k=10,
        use_diskann=False,
        profiler=None,
    )

    assert results == []
