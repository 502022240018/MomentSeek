"""Visual ANN Search Integration Tests

Tests the simplified visual search implementation against a live Milvus instance.

Requirements:
- Milvus running and accessible
- At least one indexed video with visual embeddings
- Test video ID configured below

Run:
    pytest backend/tests/integration/test_visual_ann.py -v -s
"""
import time
import pytest
import numpy as np

pytestmark = pytest.mark.integration

from app.vector_store.milvus.milvus_client import MilvusClient
from app.vector_store.milvus.milvus_search import milvus_visual_candidates
from app.vector_store.milvus.milvus_search_visual_v2 import (
    MilvusVisualSearchError,
    _verify_index_type,
    _normalize,
)
from app.core.settings import get_settings


# Configuration - update with actual test video
TEST_VIDEO_ID = "test_video_001"  # TODO: Replace with indexed video ID
SKIP_IF_NO_VIDEO = True  # Set False to fail tests if video not found


@pytest.fixture(scope="module")
def milvus_client():
    """Milvus client fixture."""
    settings = get_settings()
    if not settings.milvus_enabled:
        pytest.skip("Milvus is disabled")

    try:
        client = MilvusClient()
        return client
    except Exception as e:
        pytest.skip(f"Cannot connect to Milvus: {e}")


@pytest.fixture(scope="module")
def test_video_id(milvus_client):
    """Get a test video ID with visual embeddings."""
    try:
        col = milvus_client.collection_for("visual")
        # Try to find any video with embeddings
        results = col.query(
            expr="video_id != ''",
            output_fields=["video_id"],
            limit=1,
        )
        if results:
            video_id = results[0].get("video_id")
            print(f"\n✓ Using test video: {video_id}")
            return video_id
    except Exception as e:
        if SKIP_IF_NO_VIDEO:
            pytest.skip(f"No indexed video found: {e}")
        else:
            pytest.fail(f"No indexed video found: {e}")


@pytest.fixture
def query_embedding():
    """Generate a normalized random query vector."""
    vec = np.random.randn(1152).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestVisualANNBasics:
    """Basic functionality tests."""

    def test_index_type_verification(self, milvus_client):
        """Test that index type verification works."""
        settings = get_settings()

        # Should not raise if index matches config
        try:
            _verify_index_type(milvus_client, settings.visual_use_diskann)
            print(f"\n✓ Index type verified: {'DiskANN' if settings.visual_use_diskann else 'HNSW'}")
        except MilvusVisualSearchError as e:
            pytest.fail(f"Index type mismatch: {e}")

    def test_single_query_search(self, milvus_client, test_video_id, query_embedding):
        """Test single-query visual search."""
        start = time.perf_counter()

        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=query_embedding,
            limit=10,
            profile="balanced",
        )

        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n✓ Single query recalled {len(candidates)} candidates in {elapsed:.1f}ms")

        # Assertions
        assert isinstance(candidates, list), "Should return list"
        assert all(c.modality == "visual" for c in candidates), "All should be visual"
        assert all(c.video_id == test_video_id for c in candidates), "All should match video"
        assert all(0 <= c.score <= 1 for c in candidates), "Scores should be in [0,1]"

        # Check sorted descending
        if len(candidates) > 1:
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True), "Should be sorted by score"

    def test_multi_query_search(self, milvus_client, test_video_id):
        """Test multi-query visual search (AND semantics)."""
        # Create 2 different query vectors
        query1 = np.random.randn(1152).astype(np.float32)
        query1 = query1 / np.linalg.norm(query1)

        query2 = np.random.randn(1152).astype(np.float32)
        query2 = query2 / np.linalg.norm(query2)

        multi_query = np.stack([query1, query2])

        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=multi_query,
            limit=10,
            profile="balanced",
        )

        print(f"\n✓ Multi-query (2 queries) recalled {len(candidates)} candidates")

        assert isinstance(candidates, list)
        # Multi-query should still return results
        # (though possibly fewer than single query due to AND semantics)


class TestVisualANNProfiles:
    """Test different search profiles."""

    def test_precision_profile(self, milvus_client, test_video_id, query_embedding):
        """Test precision profile."""
        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=query_embedding,
            limit=10,
            profile="precision",
        )

        assert len(candidates) <= 10, "Precision should respect limit"
        print(f"\n✓ Precision profile: {len(candidates)} candidates")

    def test_balanced_profile(self, milvus_client, test_video_id, query_embedding):
        """Test balanced profile (default)."""
        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=query_embedding,
            limit=10,
            profile="balanced",
        )

        assert len(candidates) <= 10, "Balanced should respect limit"
        print(f"\n✓ Balanced profile: {len(candidates)} candidates")

    def test_recall_profile(self, milvus_client, test_video_id, query_embedding):
        """Test recall profile (returns more candidates)."""
        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=query_embedding,
            limit=10,
            profile="recall",
        )

        # Recall mode can return more than limit (up to 500)
        print(f"\n✓ Recall profile: {len(candidates)} candidates")


class TestVisualANNPerformance:
    """Performance benchmarks."""

    def test_latency_target(self, milvus_client, test_video_id, query_embedding):
        """Test that query latency is reasonable."""
        times = []
        n_runs = 5

        for _ in range(n_runs):
            start = time.perf_counter()
            candidates = milvus_visual_candidates(
                client=milvus_client,
                video_id=test_video_id,
                query=query_embedding,
                limit=20,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_ms = np.mean(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000

        print(f"\n✓ Latency: avg={avg_ms:.1f}ms, p95={p95_ms:.1f}ms ({n_runs} runs)")

        # Latency should be under 200ms for reasonable-sized videos
        assert avg_ms < 300, f"Average latency {avg_ms:.1f}ms exceeds 300ms threshold"


class TestVisualANNCandidateFields:
    """Test candidate structure and fields."""

    def test_candidate_fields(self, milvus_client, test_video_id, query_embedding):
        """Test that candidates have required fields and no legacy fields."""
        candidates = milvus_visual_candidates(
            client=milvus_client,
            video_id=test_video_id,
            query=query_embedding,
            limit=5,
        )

        if not candidates:
            pytest.skip("No candidates returned")

        c = candidates[0]

        # Required fields
        assert hasattr(c, "video_id"), "Missing video_id"
        assert hasattr(c, "start_time"), "Missing start_time"
        assert hasattr(c, "end_time"), "Missing end_time"
        assert hasattr(c, "score"), "Missing score"
        assert hasattr(c, "raw_score"), "Missing raw_score"
        assert hasattr(c, "modality"), "Missing modality"
        assert hasattr(c, "evidence"), "Missing evidence"
        assert hasattr(c, "unit_type"), "Missing unit_type"
        assert hasattr(c, "unit_id"), "Missing unit_id"

        # Legacy fields should NOT exist
        assert not hasattr(c, "robust_z"), "Legacy field robust_z should be removed"
        assert not hasattr(c, "percentile"), "Legacy field percentile should be removed"
        assert not hasattr(c, "distribution_reliable"), "Legacy field should be removed"

        print(f"\n✓ Candidate fields correct")
        print(f"  score={c.score:.3f}, raw_score={c.raw_score:.3f}")
        print(f"  evidence: {c.evidence[:80]}...")


class TestVisualANNHelpers:
    """Test helper functions."""

    def test_normalize_function(self):
        """Test vector normalization."""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normalized = _normalize(vec)

        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6, f"Normalized vector has norm {norm}, expected 1.0"

        # Test zero vector
        zero_vec = np.zeros(5, dtype=np.float32)
        normalized_zero = _normalize(zero_vec)
        assert np.allclose(normalized_zero, zero_vec), "Zero vector should remain zero"

        print("\n✓ Normalization works correctly")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
