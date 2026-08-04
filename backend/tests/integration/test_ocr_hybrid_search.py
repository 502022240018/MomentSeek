"""Live Milvus coverage for OCR DiskANN + BM25 hybrid retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pytest
from pymilvus import Collection, connections, utility

from app.core.settings import get_settings
from app.vector_store.milvus.milvus_client import _COLLECTION_CONFIGS
from app.vector_store.milvus.milvus_schema import create_ocr_schema
from app.vector_store.milvus.milvus_search import milvus_ocr_candidates_hybrid

pytestmark = pytest.mark.integration


@dataclass
class _OcrOnlyClient:
    collection: Collection

    def collection_for(self, modality: str) -> Collection:
        assert modality == "ocr"
        return self.collection


@pytest.fixture(scope="module")
def ocr_collection():
    """Create an isolated collection using the same schema and indexes as production."""
    settings = get_settings()
    alias = f"ocr-hybrid-test-{uuid4().hex}"
    collection_name = f"ocr_hybrid_test_{uuid4().hex}"
    connections.connect(
        alias=alias,
        host=settings.milvus_host,
        port=settings.milvus_port,
        timeout=settings.milvus_query_timeout_seconds,
    )
    try:
        col = Collection(
            name=collection_name,
            schema=create_ocr_schema(),
            consistency_level="Strong",
            using=alias,
        )
        for field_name, index_params in _COLLECTION_CONFIGS["ocr_embeddings"]["indexes"].items():
            col.create_index(field_name=field_name, index_params=index_params)
        col.load()

        dense = np.zeros(384, dtype=np.float32)
        dense[0] = 1.0
        other = np.zeros(384, dtype=np.float32)
        other[1] = 1.0
        col.insert([
            {
                "pk": "ocr-hybrid-semantic",
                "video_id": "ocr-hybrid-video",
                "asset_version": "1",
                "model_version": "test",
                "frame_idx": 0,
                "region_idx": 0,
                "frame_ms": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "avg_box_score": 0.9,
                "text": "语义匹配样本",
                "has_embedding": True,
                "embedding": dense.tolist(),
            },
            {
                "pk": "ocr-hybrid-lexical",
                "video_id": "ocr-hybrid-video",
                "asset_version": "1",
                "model_version": "test",
                "frame_idx": 1,
                "region_idx": 0,
                "frame_ms": 1000,
                "start_ms": 1000,
                "end_ms": 2000,
                "avg_box_score": 0.8,
                "text": "工资到账通知",
                "has_embedding": False,
                "embedding": [0.0] * 384,
            },
            {
                "pk": "ocr-hybrid-other",
                "video_id": "ocr-hybrid-video",
                "asset_version": "1",
                "model_version": "test",
                "frame_idx": 2,
                "region_idx": 0,
                "frame_ms": 2000,
                "start_ms": 2000,
                "end_ms": 3000,
                "avg_box_score": 0.7,
                "text": "无关文本",
                "has_embedding": True,
                "embedding": other.tolist(),
            },
        ])
        col.flush()
        yield _OcrOnlyClient(col)
    finally:
        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)
        connections.disconnect(alias)


def test_bm25_only_retrieves_lexical_only_frame(ocr_collection):
    candidates = milvus_ocr_candidates_hybrid(
        ocr_collection,
        "ocr-hybrid-video",
        "1",
        "工资",
        None,
        limit=10,
    )

    assert any(candidate.text == "工资到账通知" for candidate in candidates)
    assert any(not candidate.features["has_embedding"] for candidate in candidates)


def test_dense_only_excludes_lexical_only_frame(ocr_collection):
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_ocr_candidates_hybrid(
        ocr_collection,
        "ocr-hybrid-video",
        "1",
        "",
        query,
        limit=10,
    )

    assert candidates
    assert all(candidate.features["has_embedding"] for candidate in candidates)
    assert candidates[0].text == "语义匹配样本"


def test_hybrid_search_combines_dense_and_bm25(ocr_collection):
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1.0
    candidates = milvus_ocr_candidates_hybrid(
        ocr_collection,
        "ocr-hybrid-video",
        "1",
        "工资",
        query,
        limit=10,
    )

    texts = {candidate.text for candidate in candidates}
    assert "语义匹配样本" in texts
    assert "工资到账通知" in texts
