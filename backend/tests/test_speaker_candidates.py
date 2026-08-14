"""Unit tests for milvus_speaker_candidates after the P0/P1 optimization.

No running Milvus required — the collection is mocked. These lock in the three
behavioural guarantees of the single-phase (no re-score) DiskANN path:

1. ``output_fields`` no longer requests ``embedding`` (network/serialisation win).
2. The DiskANN ``search_list`` is ``max(ann_limit, setting)`` so it never
   violates DiskANN's ``search_list >= limit`` hard constraint.
3. ``Candidate.score`` is the trusted Milvus COSINE ``_distance`` (no Python
   re-score), and threshold=None resolves to the configured default while
   threshold=-1.0 keeps every candidate (voice-search semantics).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.core.settings import get_settings
from app.vector_store.milvus.milvus_search import (
    _reset_index_verification,
    milvus_speaker_candidates,
)

_VIDEO_ID = "spk-video"
_ASSET_VERSION = "7"


def _hit(distance: float, utterance_idx: int, track_id: int = 0) -> MagicMock:
    hit = MagicMock()
    hit.distance = distance
    entity = {
        "utterance_idx": utterance_idx,
        "start_ms": utterance_idx * 1000,
        "end_ms": utterance_idx * 1000 + 500,
        "track_id": track_id,
        "asr_chunk_idx": utterance_idx,
    }
    hit.entity.get.side_effect = entity.get
    return hit


def _make_client(hits: list[MagicMock]) -> tuple[MagicMock, MagicMock]:
    collection = MagicMock()
    collection.search.return_value = [hits]
    # _verify_ann_index_type_once introspects col.index() — present the expected
    # Speaker index type and metric.
    collection.index.return_value.params = {
        "index_type": "DISKANN",
        "metric_type": "COSINE",
    }
    client = MagicMock()
    client.collection_for.return_value = collection
    return client, collection


def _unit_query() -> np.ndarray:
    q = np.zeros(192, dtype=np.float32)
    q[0] = 1.0
    return q


def test_output_fields_exclude_embedding():
    """The re-score removal means embedding is never fetched back."""
    _reset_index_verification()
    client, collection = _make_client([_hit(0.9, 0)])

    milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(), _ASSET_VERSION, limit=10
    )

    collection.search.assert_called_once()
    output_fields = collection.search.call_args.kwargs["output_fields"]
    assert "embedding" not in output_fields
    assert set(output_fields) == {
        "utterance_idx", "start_ms", "end_ms", "track_id", "asr_chunk_idx",
    }


def test_search_list_respects_diskann_constraint():
    """search_list = max(ann_limit, setting); never below the requested limit."""
    _reset_index_verification()
    settings = get_settings()
    # Choose a limit whose ann_limit exceeds the configured search_list so the
    # max() actually raises it (guards the DiskANN search_list >= limit rule).
    big_limit = settings.speaker_diskann_search_list + 50
    ann_limit = big_limit * settings.speaker_recall_multiplier
    client, collection = _make_client([])

    milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(), _ASSET_VERSION, limit=big_limit
    )

    param = collection.search.call_args.kwargs["param"]
    assert param["metric_type"] == "COSINE"
    assert param["params"]["search_list"] == max(
        ann_limit, settings.speaker_diskann_search_list
    )
    # The ANN limit itself is forwarded to search().
    assert collection.search.call_args.kwargs["limit"] == ann_limit


def test_score_is_trusted_distance_and_sorted():
    """score == Milvus _distance (no re-score) and results are sorted desc."""
    _reset_index_verification()
    client, _ = _make_client([_hit(0.4, 0), _hit(0.95, 1), _hit(0.7, 2)])

    candidates = milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(), _ASSET_VERSION, limit=10, threshold=-1.0
    )

    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert scores == [0.95, 0.7, 0.4]
    # raw_score mirrors the trusted cosine.
    assert all(c.raw_score == c.score for c in candidates)


def test_threshold_none_uses_setting_default():
    """threshold=None resolves to settings.speaker_identity_threshold."""
    _reset_index_verification()
    default = get_settings().speaker_identity_threshold
    # One hit above, one below the configured default.
    client, _ = _make_client([_hit(default + 0.1, 0), _hit(default - 0.1, 1)])

    candidates = milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(), _ASSET_VERSION, limit=10
    )

    by_unit = {c.unit_id: c for c in candidates}
    assert by_unit[0].above_threshold is True
    assert by_unit[0].decision == "absolute_hit"
    assert by_unit[1].above_threshold is False
    assert by_unit[1].decision == "weak"


def test_threshold_negative_keeps_all_above():
    """Voice-search passes threshold=-1.0 → every candidate is above_threshold."""
    _reset_index_verification()
    client, _ = _make_client([_hit(0.1, 0), _hit(0.05, 1)])

    candidates = milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(), _ASSET_VERSION, limit=10, threshold=-1.0
    )

    assert all(c.above_threshold for c in candidates)
    assert all(c.decision == "absolute_hit" for c in candidates)
