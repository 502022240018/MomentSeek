"""Unit tests asserting hybrid retrieval passes a query timeout to Milvus.

These tests do NOT require a running Milvus instance. They mock the collection
returned by ``client.collection_for(...)`` and verify that all three retrieval
paths (bm25-only / dense-only / hybrid) forward
``timeout=settings.milvus_query_timeout_seconds`` to ``search``/``hybrid_search``
for both the ASR and OCR hybrid entry points.

Without the timeout kwarg a stalled Milvus would block retrieval indefinitely,
so ``MILVUS_QUERY_TIMEOUT_SECONDS`` (default 3.0s) would have no effect.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.core.settings import get_settings
from app.vector_store.milvus.milvus_search import (
    _reset_index_verification,
    milvus_asr_candidates_hybrid,
    milvus_ocr_candidates_hybrid,
    milvus_speaker_candidates,
)

_VIDEO_ID = "timeout-video"
_ASSET_VERSION = "1"
_DIM = 384


def _make_client() -> tuple[MagicMock, MagicMock]:
    """Return (client, collection) where every search call yields no hits."""
    collection = MagicMock()
    # ``for hit in results[0]`` must iterate over an empty hit list.
    collection.search.return_value = [[]]
    collection.hybrid_search.return_value = [[]]
    # Speaker/face go through _ann_search → _verify_ann_index_type_once, which
    # introspects col.index(); present the expected DISKANN type for speaker.
    collection.index.return_value.params = {"index_type": "DISKANN"}
    client = MagicMock()
    client.collection_for.return_value = collection
    return client, collection


def _unit_embedding() -> np.ndarray:
    embedding = np.zeros(_DIM, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

def test_asr_bm25_only_passes_timeout():
    """query_embedding=None → BM25-only path forwards the query timeout."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_asr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "人工智能", None, limit=20
    )

    collection.search.assert_called_once()
    assert collection.search.call_args.kwargs["timeout"] == expected
    collection.hybrid_search.assert_not_called()


def test_asr_dense_only_passes_timeout():
    """Empty query_text with an embedding → dense-only path forwards the timeout."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_asr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "   ", _unit_embedding(), limit=20
    )

    collection.search.assert_called_once()
    assert collection.search.call_args.kwargs["timeout"] == expected
    collection.hybrid_search.assert_not_called()


def test_asr_hybrid_passes_timeout():
    """Text + embedding → hybrid path forwards the timeout to hybrid_search."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_asr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "人工智能", _unit_embedding(), limit=20
    )

    collection.hybrid_search.assert_called_once()
    assert collection.hybrid_search.call_args.kwargs["timeout"] == expected
    collection.search.assert_not_called()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def test_ocr_bm25_only_passes_timeout():
    """query_embedding=None → BM25-only path forwards the query timeout."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_ocr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "发票", None, limit=20
    )

    collection.search.assert_called_once()
    assert collection.search.call_args.kwargs["timeout"] == expected
    collection.hybrid_search.assert_not_called()


def test_ocr_dense_only_passes_timeout():
    """Empty query_text with an embedding → dense-only path forwards the timeout."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_ocr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "   ", _unit_embedding(), limit=20
    )

    collection.search.assert_called_once()
    assert collection.search.call_args.kwargs["timeout"] == expected
    collection.hybrid_search.assert_not_called()


def test_ocr_hybrid_passes_timeout():
    """Text + embedding → hybrid path forwards the timeout to hybrid_search."""
    expected = get_settings().milvus_query_timeout_seconds
    client, collection = _make_client()

    milvus_ocr_candidates_hybrid(
        client, _VIDEO_ID, _ASSET_VERSION, "发票", _unit_embedding(), limit=20
    )

    collection.hybrid_search.assert_called_once()
    assert collection.hybrid_search.call_args.kwargs["timeout"] == expected
    collection.search.assert_not_called()


# ---------------------------------------------------------------------------
# Speaker (single-phase ANN via _ann_search)
# ---------------------------------------------------------------------------

def test_speaker_passes_timeout():
    """Speaker ANN search forwards the query timeout to search."""
    expected = get_settings().milvus_query_timeout_seconds
    _reset_index_verification()
    client, collection = _make_client()

    # Speaker embeddings are 192-dim (CAM++ 3D-Speaker schema), not the
    # 384-dim used by ASR/OCR semantic encoders.
    speaker_query = np.zeros(192, dtype=np.float32)
    speaker_query[0] = 1.0

    milvus_speaker_candidates(
        client, _VIDEO_ID, speaker_query, _ASSET_VERSION, limit=20
    )

    collection.search.assert_called_once()
    assert collection.search.call_args.kwargs["timeout"] == expected
