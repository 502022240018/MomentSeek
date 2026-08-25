"""Live Milvus coverage for ASR DiskANN + BM25 hybrid retrieval.

Mirrors test_ocr_hybrid_search.py: an isolated collection is created with the
same schema and indexes as production, seeded with a handful of rows, and the
three retrieval paths (hybrid / dense-only / bm25-only) are exercised against
milvus_asr_candidates_hybrid.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pytest
from pymilvus import Collection, connections, utility

from app.core.settings import get_settings
from app.vector_store.milvus.milvus_client import _COLLECTION_CONFIGS
from app.vector_store.milvus.milvus_schema import EMBEDDING_DIMS, create_asr_schema
from app.vector_store.milvus.milvus_search import milvus_asr_candidates_hybrid

pytestmark = pytest.mark.integration

_DIM = EMBEDDING_DIMS["asr"]  # 384
_VIDEO_ID = "asr-hybrid-video"
_ASSET_VERSION = "1"


@dataclass
class _AsrOnlyClient:
    collection: Collection

    def collection_for(self, modality: str) -> Collection:
        assert modality == "asr"
        return self.collection


@pytest.fixture(scope="module")
def asr_collection():
    """Create an isolated collection using the same schema and indexes as production."""
    settings = get_settings()
    alias = f"asr-hybrid-test-{uuid4().hex}"
    collection_name = f"asr_hybrid_test_{uuid4().hex}"
    try:
        connections.connect(
            alias=alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
            timeout=settings.milvus_query_timeout_seconds,
        )
    except Exception as exc:
        pytest.skip(f"Cannot connect to Milvus: {exc}")
    try:
        col = Collection(
            name=collection_name,
            schema=create_asr_schema(),
            consistency_level="Strong",
            using=alias,
        )
        for field_name, index_params in _COLLECTION_CONFIGS["asr_embeddings"]["indexes"].items():
            col.create_index(field_name=field_name, index_params=index_params)
        col.load()

        # Two orthogonal unit vectors so dense recall is unambiguous.
        dense = np.zeros(_DIM, dtype=np.float32)
        dense[0] = 1.0
        other = np.zeros(_DIM, dtype=np.float32)
        other[1] = 1.0

        col.insert([
            {
                "pk": "asr-hybrid-semantic",
                "video_id": _VIDEO_ID,
                "asset_version": "1",
                "model_version": "test",
                "segment_idx": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "text": "语义匹配样本",
                "has_embedding": True,
                "embedding": dense.tolist(),
            },
            {
                "pk": "asr-hybrid-lexical",
                "video_id": _VIDEO_ID,
                "asset_version": "1",
                "model_version": "test",
                "segment_idx": 1,
                "start_ms": 1000,
                "end_ms": 2000,
                "text": "今天我们讨论人工智能的未来",
                "has_embedding": False,
                "embedding": [0.0] * _DIM,
            },
            {
                "pk": "asr-hybrid-other",
                "video_id": _VIDEO_ID,
                "asset_version": "1",
                "model_version": "test",
                "segment_idx": 2,
                "start_ms": 2000,
                "end_ms": 3000,
                "text": "无关内容",
                "has_embedding": True,
                "embedding": other.tolist(),
            },
        ])
        col.flush()
        yield _AsrOnlyClient(col)
    finally:
        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)
        connections.disconnect(alias)


def test_bm25_only_retrieves_lexical_only_chunk(asr_collection):
    """query_embedding=None → BM25-only; a lexical-only chunk must surface."""
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "人工智能",
        None,
        limit=10,
    )

    assert any(c.text == "今天我们讨论人工智能的未来" for c in candidates)
    assert any(not c.features["has_embedding"] for c in candidates)
    for c in candidates:
        assert c.modality == "asr"
        assert c.unit_type == "chunk"
        assert c.features["source"] == "milvus_hybrid"


def test_dense_only_excludes_lexical_only_chunk(asr_collection):
    """Empty query_text → dense-only; lexical-only chunks are filtered out."""
    query = np.zeros(_DIM, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "",
        query,
        limit=10,
    )

    assert candidates
    assert all(c.features["has_embedding"] for c in candidates)
    texts = {c.text for c in candidates}
    # The dense semantic chunk surfaces; the lexical-only chunk is filtered out.
    assert "语义匹配样本" in texts
    assert "今天我们讨论人工智能的未来" not in texts


def test_hybrid_search_combines_dense_and_bm25(asr_collection):
    """Both signals present → hybrid returns both the semantic and lexical hits."""
    query = np.zeros(_DIM, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "人工智能",
        query,
        limit=10,
    )

    texts = {c.text for c in candidates}
    assert "语义匹配样本" in texts
    assert "今天我们讨论人工智能的未来" in texts


def test_empty_query_and_no_embedding_returns_empty(asr_collection):
    """No text and no embedding is a valid empty answer, not an error."""
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "",
        None,
        limit=10,
    )

    assert candidates == []


def test_candidates_are_above_threshold_before_global_pass(asr_collection):
    """The hybrid function leaves above_threshold=True; search.py finalises it."""
    query = np.zeros(_DIM, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "人工智能",
        query,
        limit=20,
    )

    assert candidates
    assert all(c.above_threshold for c in candidates)
    assert all("hybrid_score" in (c.features or {}) for c in candidates)


def test_limit_is_respected(asr_collection):
    """limit=1 must cap results to 1 even when multiple chunks match."""
    query = np.zeros(_DIM, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "人工智能",
        query,
        limit=1,
    )

    assert len(candidates) == 1


def test_row_without_has_embedding_defaults_to_true(asr_collection):
    """A row inserted without has_embedding (schema default_value=True) must appear
    in dense-only results because the dense filter is ``has_embedding == True``."""
    col = asr_collection.collection

    default_vec = np.zeros(_DIM, dtype=np.float32)
    default_vec[0] = 1.0  # same direction as the semantic query used in other tests

    col.insert([{
        "pk": "asr-hybrid-nohas",
        "video_id": _VIDEO_ID,
        "asset_version": _ASSET_VERSION,
        "model_version": "test",
        "segment_idx": 99,
        "start_ms": 9000,
        "end_ms": 10000,
        "text": "无has_embedding字段的样本",
        "embedding": default_vec.tolist(),
        # has_embedding intentionally absent — schema default_value=True must apply
    }])
    col.flush()

    candidates = milvus_asr_candidates_hybrid(
        asr_collection,
        _VIDEO_ID,
        _ASSET_VERSION,
        "",        # empty text → dense-only path
        default_vec,
        limit=10,
    )

    texts = {c.text for c in candidates}
    assert "无has_embedding字段的样本" in texts, (
        "Row written without has_embedding should surface in dense-only search; "
        "schema default_value=True was not applied by Milvus"
    )
