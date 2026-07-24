"""
Test script to verify Milvus connection and collection initialization.

Run from the project root:
    python backend/tests/test_milvus_connection.py

Or from inside the container:
    python -u /app/backend/tests/test_milvus_connection.py
使用示例：
docker exec momentseek-mvp-app-cuda bash -c "PYTHONPATH=/app/backend python -u /app/backend/tests/test_milvus_connection.py 2>&1"
"""

import sys
import logging
from pathlib import Path
import uuid

# Add backend to path so `app.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))  # /app/backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

_ALL_COLLECTIONS = [
    "visual_embeddings",
    "asr_embeddings",
    "face_embeddings",
    "ocr_embeddings",
    "speaker_embeddings",
]


def _check_milvus_connection() -> bool:
    """Test Milvus connection, health check, and collection status."""
    try:
        from app.indexing.milvus_client import MilvusClient
        from app.indexing.milvus_schema import EMBEDDING_DIMS

        logger.info("=" * 60)
        logger.info("Testing Milvus Connection")
        logger.info("=" * 60)

        client = MilvusClient()

        # 1. Health check
        logger.info("\n1. Health Check")
        is_healthy = client.health_check()
        logger.info("   Milvus health: %s", "✓ OK" if is_healthy else "✗ FAILED")
        if not is_healthy:
            logger.error("Milvus is not healthy. Check the service.")
            return False

        # 2. Collection status — all 5 collections
        logger.info("\n2. Collection Status")
        for coll_name in _ALL_COLLECTIONS:
            stats = client.stats(coll_name)
            logger.info("   %s:", coll_name)
            logger.info("     - Loaded:   %s", stats["loaded"])
            logger.info("     - Entities: %d", stats["num_entities"])

        # 3. Embedding dimensions
        logger.info("\n3. Embedding Dimensions")
        for modality, dim in EMBEDDING_DIMS.items():
            logger.info("   %-10s %dd", modality + ":", dim)

        logger.info("\n" + "=" * 60)
        logger.info("✓ Connection test passed!")
        logger.info("=" * 60)
        return True

    except Exception as exc:
        logger.error("\n✗ Connection test failed: %s", exc, exc_info=True)
        return False


def test_milvus_connection() -> None:
    """Expose the live connection check as a real pytest assertion."""
    assert _check_milvus_connection()


def _check_batch_buffer() -> bool:
    """Test BatchBuffer upsert using the current MilvusClient + visual_pk API.

    A uniquely named temporary collection is used so this test can never drop
    or rewrite the production visual_embeddings collection.
    """
    temporary_collection = f"test_visual_embeddings_{uuid.uuid4().hex}"
    try:
        import numpy as np
        from pymilvus import Collection
        from app.indexing.milvus_client import MilvusClient, _COLLECTION_CONFIGS
        from app.indexing.batch_buffer import BatchBuffer
        from app.indexing.milvus_schema import visual_pk, MODEL_VERSIONS

        logger.info("\n" + "=" * 60)
        logger.info("Testing Batch Buffer")
        logger.info("=" * 60)

        client = MilvusClient()

        logger.info("\n0. Create isolated collection %s", temporary_collection)
        config = _COLLECTION_CONFIGS["visual_embeddings"]
        schema = config["schema"]()
        col_new = Collection(
            name=temporary_collection,
            schema=schema,
            consistency_level="Strong",
        )
        col_new.create_index(field_name="embedding", index_params=config["index"])
        col_new.load()
        logger.info("   Temporary collection ready")

        collection = client.collection(temporary_collection)

        model_ver = MODEL_VERSIONS["visual"]
        test_video = "connection_test_video"
        asset_ver  = "0"

        # pk_generator receives the data dict and must return a deterministic string
        def pk_gen(data: dict) -> str:
            return visual_pk(
                data["video_id"],
                data["asset_version"],
                data["frame_idx"],
                data["model_version"],
            )

        # 1. Insert 5 mock frames
        logger.info("\n1. Inserting 5 mock frames (batch_size=3 → 2 auto-flushes)")
        with BatchBuffer(collection=collection, batch_size=3, pk_generator=pk_gen) as buf:
            for i in range(5):
                buf.add({
                    "video_id":      test_video,
                    "asset_version": asset_ver,
                    "frame_idx":     i,
                    "timestamp_ms":  i * 1000,
                    "model_version": model_ver,
                    "embedding":     np.random.rand(1152).tolist(),
                })
                logger.info("   Added frame %d, pending: %d", i, buf.pending_count)

        logger.info("   Total flushed: %d", buf.total_flushed)

        stats_after = client.stats(temporary_collection)
        logger.info("\n2. temporary collection now has %d entities", stats_after["num_entities"])

        # 2. Idempotent re-insert: same PKs → entity count must not grow
        logger.info("\n3. Idempotent re-insert (same 5 frames, different embeddings)")
        count_before = stats_after["num_entities"]

        with BatchBuffer(collection=collection, batch_size=10, pk_generator=pk_gen) as buf:
            for i in range(5):
                buf.add({
                    "video_id":      test_video,
                    "asset_version": asset_ver,
                    "frame_idx":     i,
                    "timestamp_ms":  i * 1000,
                    "model_version": model_ver,
                    "embedding":     np.random.rand(1152).tolist(),
                })

        count_after = client.stats(temporary_collection)["num_entities"]
        delta = count_after - count_before
        logger.info("   Entities before: %d", count_before)
        logger.info("   Entities after:  %d", count_after)
        logger.info("   Delta: %d (expected 0 — upsert is idempotent)", delta)

        if delta != 0:
            raise AssertionError(
                f"Upsert is not idempotent: entity count changed by {delta}"
            )

        logger.info("\n" + "=" * 60)
        logger.info("✓ Batch buffer test passed!")
        logger.info("=" * 60)
        return True

    except Exception as exc:
        logger.error("\n✗ Batch buffer test failed: %s", exc, exc_info=True)
        return False
    finally:
        try:
            from pymilvus import Collection, utility

            if utility.has_collection(temporary_collection):
                Collection(temporary_collection).drop()
                logger.info("Dropped temporary collection %s", temporary_collection)
        except Exception as cleanup_exc:
            logger.warning(
                "Failed to drop temporary collection %s: %s",
                temporary_collection,
                cleanup_exc,
            )


def test_batch_buffer() -> None:
    """Expose the isolated batch-buffer check as a real pytest assertion."""
    assert _check_batch_buffer()


if __name__ == "__main__":
    ok = _check_milvus_connection()
    if ok:
        ok = _check_batch_buffer()
    sys.exit(0 if ok else 1)
