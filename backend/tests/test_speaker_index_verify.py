"""Unit tests for the per-modality fail-fast index-type verification.

A HNSW→DISKANN config change does not rebuild an existing collection, so a stale
collection would silently break DiskANN search. ``_verify_ann_index_type_once``
surfaces that drift explicitly. These tests also guard the key correctness
property: the check is keyed per modality, so face (IVF_FLAT) is never judged
against speaker's expected DISKANN type.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.vector_store.milvus.milvus_search import (
    MilvusServiceError,
    _reset_index_verification,
    milvus_face_candidates,
    milvus_speaker_candidates,
)

_VIDEO_ID = "verify-video"
_ASSET_VERSION = "1"


def _make_client(actual_index_type: str) -> tuple[MagicMock, MagicMock]:
    collection = MagicMock()
    collection.search.return_value = [[]]
    collection.index.return_value.params = {"index_type": actual_index_type}
    client = MagicMock()
    client.collection_for.return_value = collection
    return client, collection


def _unit_query(dim: int) -> np.ndarray:
    q = np.zeros(dim, dtype=np.float32)
    q[0] = 1.0
    return q


def test_speaker_stale_hnsw_collection_fails_fast():
    """Speaker config expects DISKANN; a live HNSW collection must raise."""
    _reset_index_verification()
    client, _ = _make_client("HNSW")

    with pytest.raises(MilvusServiceError, match="Index type mismatch"):
        milvus_speaker_candidates(
            client, _VIDEO_ID, _unit_query(192), _ASSET_VERSION, limit=10
        )


def test_speaker_diskann_collection_passes():
    """Matching DISKANN collection passes verification (no raise)."""
    _reset_index_verification()
    client, collection = _make_client("DISKANN")

    milvus_speaker_candidates(
        client, _VIDEO_ID, _unit_query(192), _ASSET_VERSION, limit=10
    )
    collection.search.assert_called_once()


def test_face_ivf_flat_not_judged_against_speaker_type():
    """Face expects IVF_FLAT — verification must use face's own expected type,
    not speaker's DISKANN. An IVF_FLAT face collection must pass cleanly."""
    _reset_index_verification()
    client, collection = _make_client("IVF_FLAT")

    milvus_face_candidates(
        client, _VIDEO_ID, _unit_query(512), _ASSET_VERSION, limit=10
    )
    collection.search.assert_called_once()
