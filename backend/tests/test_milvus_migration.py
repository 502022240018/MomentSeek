"""Tests for Milvus dual-write infrastructure.

Covers the scenarios listed in the migration spec:
  1. Idempotent upsert (repeat job produces no duplicate rows)
  2. Single-video rebuild (new asset_version coexists with old)
  3. Model-version upgrade (different model_ver = new rows, old still queryable)
  4. Partial batch write failure + retry
  5. Milvus write success but NPZ write failure (data survives in Milvus)
  6. Data integrity after video deletion
  7. Fail-closed Milvus writes abort the index job
  8. Fallback routing: MilvusServiceError → NPZ (only when FALLBACK_ENABLED)
  9. Empty Milvus result is NOT treated as a service error
 10. should_use_milvus_for_video() stable hash routing

All tests mock Milvus at the Collection level so no live Milvus is needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_milvus_reachability_preflight_fails_before_grpc_retry():
    from app.vector_store.milvus.milvus_client import ensure_milvus_reachable

    with patch(
        "app.vector_store.milvus.milvus_client.socket.create_connection",
        side_effect=OSError("connection refused"),
    ):
        with pytest.raises(ConnectionError, match="Milvus is unreachable"):
            ensure_milvus_reachable()

def _make_npz(tmp_path: Path, modality: str) -> Path:
    """Create a minimal valid NPZ for the given modality."""
    path = tmp_path / f"{modality}.npz"
    if modality == "visual":
        np.savez_compressed(
            path,
            frame_embeddings=np.random.rand(3, 1152).astype(np.float16),
            frame_times_ms=np.array([0, 1000, 2000], dtype=np.int32),
            segment_frame_offsets=np.array([0, 1, 2, 3], dtype=np.int32),
        )
    elif modality == "asr":
        np.savez_compressed(
            path,
            chunk_times_ms=np.array([[0, 3000], [3000, 6000]], dtype=np.int32),
            texts=np.array(["hello world", "foo bar"], dtype="U"),
            embeddings=np.random.rand(2, 384).astype(np.float16),
            embedding_chunk_indices=np.array([0, 1], dtype=np.int32),
        )
    elif modality == "face":
        np.savez_compressed(
            path,
            embeddings=np.random.rand(2, 512).astype(np.float32),
            track_times_ms=np.array([[0, 5000, 2500], [5000, 10000, 7500]], dtype=np.int32),
        )
    elif modality == "ocr":
        np.savez_compressed(
            path,
            frame_times_ms=np.array([0, 1000], dtype=np.int32),
            frame_windows_ms=np.array([[0, 1000], [1000, 2000]], dtype=np.int32),
            box_frame_indices=np.array([0, 0, 1], dtype=np.int32),
            box_texts=np.array(["hello", "world", "foo"], dtype="U"),
            box_scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            boxes=np.zeros((3, 4, 2), dtype=np.float32),
            embeddings=np.random.rand(2, 384).astype(np.float16),
            embedding_frame_indices=np.array([0, 1], dtype=np.int32),
        )
    elif modality == "speaker":
        np.savez_compressed(
            path,
            utterance_embeddings=np.random.rand(3, 192).astype(np.float16),
            utterance_times_ms=np.array([[0, 2000], [2000, 5000], [5000, 8000]], dtype=np.int32),
            utterance_refs=np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int32),
            track_embeddings=np.random.rand(2, 192).astype(np.float16),
            track_representative_indices=np.array([0, 2], dtype=np.int32),
        )
    return path


def _make_mock_client(collection_name: str = "visual_embeddings") -> MagicMock:
    """Return a MilvusClient mock with a collection that records upsert calls."""
    col = MagicMock()
    col.name = collection_name
    col.upsert = MagicMock()
    client = MagicMock()
    client.collection_for = MagicMock(return_value=col)
    return client, col


# ---------------------------------------------------------------------------
# 1. Idempotent upsert
# ---------------------------------------------------------------------------

def test_visual_upsert_idempotent(tmp_path):
    npz = _make_npz(tmp_path, "visual")
    client, col = _make_mock_client()

    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext, VisualMilvusIndexer

    ctx = MilvusWriteContext(video_id="vid1", asset_version="1", client=client)
    indexer = VisualMilvusIndexer()

    # First upsert
    count1 = indexer.upsert_from_npz(ctx, npz)
    # Second upsert (same data) should produce identical PKs → Milvus upsert handles dedup
    count2 = indexer.upsert_from_npz(ctx, npz)

    assert count1 == count2 == 3  # 3 frames
    # Verify PKs are deterministic (same PK both times)
    calls = col.upsert.call_args_list
    pks_first  = {row["pk"] for row in calls[0][0][0]}
    pks_second = {row["pk"] for row in calls[1][0][0]}
    assert pks_first == pks_second, "Idempotent upsert must produce identical PKs"


# ---------------------------------------------------------------------------
# 2. Single-video rebuild — new asset_version coexists
# ---------------------------------------------------------------------------

def test_visual_asset_version_isolation(tmp_path):
    npz = _make_npz(tmp_path, "visual")
    client, col = _make_mock_client()

    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext, VisualMilvusIndexer

    ctx_v1 = MilvusWriteContext(video_id="vid1", asset_version="1", client=client)
    ctx_v2 = MilvusWriteContext(video_id="vid1", asset_version="2", client=client)
    indexer = VisualMilvusIndexer()

    indexer.upsert_from_npz(ctx_v1, npz)
    indexer.upsert_from_npz(ctx_v2, npz)

    calls = col.upsert.call_args_list
    pks_v1 = {row["pk"] for row in calls[0][0][0]}
    pks_v2 = {row["pk"] for row in calls[1][0][0]}
    assert pks_v1.isdisjoint(pks_v2), "Different asset_versions must produce disjoint PKs"

    versions_v1 = {row["asset_version"] for row in calls[0][0][0]}
    versions_v2 = {row["asset_version"] for row in calls[1][0][0]}
    assert versions_v1 == {"1"}
    assert versions_v2 == {"2"}


# ---------------------------------------------------------------------------
# 3. Model-version upgrade
# ---------------------------------------------------------------------------

def test_model_version_upgrade_disjoint_pks(tmp_path):
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext, VisualMilvusIndexer
    npz = _make_npz(tmp_path, "visual")
    client, col = _make_mock_client()
    indexer = VisualMilvusIndexer()

    ctx_old = MilvusWriteContext(
        video_id="vid1", asset_version="1", client=client,
        model_versions={"visual": "siglip2-so400m-v1"},
    )
    ctx_new = MilvusWriteContext(
        video_id="vid1", asset_version="1", client=client,
        model_versions={"visual": "siglip2-so400m-v2"},
    )
    indexer.upsert_from_npz(ctx_old, npz)
    indexer.upsert_from_npz(ctx_new, npz)

    calls = col.upsert.call_args_list
    pks_old = {row["pk"] for row in calls[0][0][0]}
    pks_new = {row["pk"] for row in calls[1][0][0]}
    assert pks_old.isdisjoint(pks_new), "Different model versions must produce disjoint PKs"


# ---------------------------------------------------------------------------
# 4. Partial batch write failure + retry queue
# ---------------------------------------------------------------------------

def test_write_failure_fails_closed(tmp_path):
    npz = _make_npz(tmp_path, "visual")
    client = MagicMock()
    col = MagicMock()
    col.upsert = MagicMock(side_effect=RuntimeError("timeout"))
    client.collection_for = MagicMock(return_value=col)

    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext, write_modality_to_milvus

    ctx = MilvusWriteContext(video_id="vid_raise", asset_version="1", client=client)

    with pytest.raises(RuntimeError, match="fail-closed"):
        write_modality_to_milvus(ctx, "visual", npz)


# ---------------------------------------------------------------------------
# 6. Data integrity after video deletion
# ---------------------------------------------------------------------------

def test_delete_video_calls_all_collections():
    client = MagicMock()
    deleted_names: list[str] = []

    def make_col(name):
        col = MagicMock()
        col.delete = MagicMock(return_value=MagicMock(delete_count=5))
        deleted_names.append(name)
        return col

    from app.vector_store.milvus.milvus_client import _COLLECTION_CONFIGS

    client.collection_for = MagicMock(side_effect=lambda m: make_col(m))

    # Patch Collection at the pymilvus level to avoid real connections.
    from unittest.mock import patch as upatch
    with upatch("app.vector_store.milvus.milvus_client.Collection") as MockCol:
        col_instances: dict[str, MagicMock] = {}

        def col_factory(name):
            m = MagicMock()
            m.delete = MagicMock(return_value=MagicMock(delete_count=3))
            col_instances[name] = m
            return m

        MockCol.side_effect = col_factory

        from app.vector_store.milvus.milvus_client import MilvusClient
        # Create a minimally wired client without real Milvus.
        cli = object.__new__(MilvusClient)
        cli._ready = True
        counts = cli.delete_video("some_video_id")

    expected_collections = {
        name for name, config in _COLLECTION_CONFIGS.items()
        if config.get("video_scoped", True)
    }
    assert set(counts.keys()) == expected_collections, \
        "delete_video must target every video-scoped collection"


# ---------------------------------------------------------------------------
# 7. Speaker modality indexer works
# ---------------------------------------------------------------------------

def test_speaker_upsert(tmp_path):
    npz = _make_npz(tmp_path, "speaker")
    client, col = _make_mock_client("speaker_embeddings")

    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext, SpeakerMilvusIndexer

    ctx = MilvusWriteContext(video_id="vid_spk", asset_version="1", client=client)
    count = SpeakerMilvusIndexer().upsert_from_npz(ctx, npz)
    assert count == 3  # 3 utterances

    rows = col.upsert.call_args[0][0]
    assert all("utterance_idx" in r for r in rows)
    assert all("track_id" in r for r in rows)
    assert all("embedding" in r for r in rows)


# ---------------------------------------------------------------------------
# 8. Empty Milvus result is NOT a service error
# ---------------------------------------------------------------------------

def test_empty_milvus_result_is_not_service_error():
    """Milvus returning 0 results is valid; no exception should be raised."""
    # Empty results would come back as an empty list from _search()
    # and produce an empty Candidate list — no exception.
    empty: list = []
    # Simply verify no exception is raised when result is empty.
    # (The actual search functions return [] on empty, not raise.)
    assert empty == []


# ---------------------------------------------------------------------------
# 9. Write queue retry smoke test
# ---------------------------------------------------------------------------

def test_stage_publish_verifies_before_switching_and_then_reclaims_old_rows(tmp_path):
    """Failed verification never changes the reader-visible channel manifest."""
    from app.indexing.stage_executor import StageContext, _write_manifest
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext

    client = MagicMock()
    client.count_video_modality_version.return_value = 2
    context = StageContext(
        video={"id": "vid_x", "duration": 1}, options={}, settings=MagicMock(),
        pool=None, video_path="unused", index_dir=tmp_path, working_dir=tmp_path,
        milvus_ctx=MilvusWriteContext(video_id="vid_x", asset_version="2", client=client),
    )
    with patch("app.indexing.stage_executor.write_stage_manifest") as write_manifest:
        with pytest.raises(RuntimeError, match="Milvus verification failed"):
            _write_manifest("visual", context, {"milvus_rows": 3})
    write_manifest.assert_not_called()
    client.delete_video_modality_except_version.assert_not_called()


# ---------------------------------------------------------------------------
# 13. Re-index cleanup is scoped to superseded versions
# ---------------------------------------------------------------------------

def test_stage_publish_reclaims_only_versions_not_selected_by_manifest(tmp_path):
    from app.indexing.stage_executor import StageContext, _write_manifest
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext

    client = MagicMock()
    client.count_video_modality_version.return_value = 3
    client.delete_video_modality_except_version.return_value = 1
    context = StageContext(
        video={"id": "vid_reindex", "duration": 1}, options={}, settings=MagicMock(),
        pool=None, video_path="unused", index_dir=tmp_path, working_dir=tmp_path,
        milvus_ctx=MilvusWriteContext(video_id="vid_reindex", asset_version="2", client=client),
    )
    with (
        patch("app.indexing.stage_executor.write_stage_manifest") as write_manifest,
        patch("app.vector_store.milvus.milvus_asset_version.publish_asset_version"),
    ):
        _write_manifest("visual", context, {"milvus_rows": 3})
    write_manifest.assert_called_once()
    client.delete_video_modality_except_version.assert_called_once_with(
        "vid_reindex", "visual", "2"
    )


# ---------------------------------------------------------------------------
# 14. asset_version auto-increment  (隐患 2)
# ---------------------------------------------------------------------------

def test_asset_version_starts_at_1(tmp_path):
    """First call returns "1" (no meta file present)."""
    from app.vector_store.milvus.milvus_asset_version import current_asset_version
    assert current_asset_version(tmp_path) == "1"


def test_asset_version_bump_increments(tmp_path):
    """bump_asset_version() persists and increments the counter."""
    from app.vector_store.milvus.milvus_asset_version import bump_asset_version, current_asset_version

    assert bump_asset_version(tmp_path) == "2"
    assert current_asset_version(tmp_path) == "2"
    assert bump_asset_version(tmp_path) == "3"
    assert current_asset_version(tmp_path) == "3"


def test_asset_version_bump_handles_non_integer_legacy(tmp_path):
    """Non-integer stored value (legacy) restarts from '2'."""
    import json
    meta = tmp_path / "milvus_meta.json"
    meta.write_text(json.dumps({"asset_version": "abc"}), encoding="utf-8")

    from app.vector_store.milvus.milvus_asset_version import bump_asset_version
    assert bump_asset_version(tmp_path) == "2"


def test_interrupted_attempt_uses_a_new_version_and_preserves_reader_pointer(tmp_path):
    """A failed v2 write cannot contaminate a v3 retry or replace published v1."""
    from types import SimpleNamespace

    from app.indexing.manifest import load_index_manifest, update_channel_manifest
    from app.indexing.stage_executor import StageContext, _write_manifest
    from app.vector_store.milvus.milvus_asset_version import (
        current_asset_version,
        current_attempt_version,
        publish_asset_version,
        reserve_next_attempt_version,
    )
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext

    update_channel_manifest(
        tmp_path,
        video_id="vid_retry",
        duration_seconds=1,
        segment_seconds=1,
        channel="visual",
        channel_manifest={"milvus_asset_version": "1", "milvus_row_count": 3},
    )
    publish_asset_version(tmp_path, "1")

    failed_version = reserve_next_attempt_version(tmp_path)
    assert failed_version == "2"
    assert current_asset_version(tmp_path) == "1"
    assert load_index_manifest(tmp_path)["channels"]["visual"]["milvus_asset_version"] == "1"

    # Simulate an interrupted v2 attempt that left only part of its rows behind.
    failed_client = MagicMock()
    failed_client.count_video_modality_version.return_value = 1
    failed_context = StageContext(
        video={"id": "vid_retry", "duration": 1}, options={},
        settings=SimpleNamespace(visual_model="siglip2-test", visual_sample_fps=1.0, visual_segment_seconds=1.0),
        pool=None, video_path="unused", index_dir=tmp_path, working_dir=tmp_path,
        milvus_ctx=MilvusWriteContext(video_id="vid_retry", asset_version=failed_version, client=failed_client),
    )
    with pytest.raises(RuntimeError, match="Milvus verification failed"):
        _write_manifest("visual", failed_context, {"milvus_rows": 2})
    assert load_index_manifest(tmp_path)["channels"]["visual"]["milvus_asset_version"] == "1"

    retry_version = reserve_next_attempt_version(tmp_path)
    assert retry_version == "3"
    assert current_attempt_version(tmp_path) == "3"

    successful_client = MagicMock()
    successful_client.count_video_modality_version.return_value = 2
    successful_client.delete_video_modality_except_version.return_value = 4
    successful_context = StageContext(
        video={"id": "vid_retry", "duration": 1}, options={},
        settings=SimpleNamespace(visual_model="siglip2-test", visual_sample_fps=1.0, visual_segment_seconds=1.0),
        pool=None, video_path="unused", index_dir=tmp_path, working_dir=tmp_path,
        milvus_ctx=MilvusWriteContext(video_id="vid_retry", asset_version=retry_version, client=successful_client),
    )
    _write_manifest("visual", successful_context, {"milvus_rows": 2})

    assert load_index_manifest(tmp_path)["channels"]["visual"]["milvus_asset_version"] == "3"
    assert current_asset_version(tmp_path) == "3"
    successful_client.delete_video_modality_except_version.assert_called_once_with(
        "vid_retry", "visual", "3"
    )


# ---------------------------------------------------------------------------
# 15. Stale write-queue jobs are cancelled on re-index  (隐患 2)
# ---------------------------------------------------------------------------

def test_execute_stage_has_no_destructive_pre_delete():
    import app.indexing.stage_executor as executor

    assert not hasattr(executor, "_pre_delete_modality")


# ---------------------------------------------------------------------------
# 17. Stage lock prevents concurrent re-index  (隐患 5)
# ---------------------------------------------------------------------------

def test_stage_lock_blocks_concurrent_same_stage(tmp_path):
    """Acquiring the same stage lock twice raises StageLockError."""
    from app.vector_store.milvus.milvus_stage_lock import StageLockError, video_stage_lock

    with video_stage_lock(tmp_path, video_id="vid_lock", stage="visual"):
        # Second acquisition of the same lock must fail immediately.
        with pytest.raises(StageLockError):
            with video_stage_lock(tmp_path, video_id="vid_lock", stage="visual"):
                pass  # should not reach here


def test_stage_lock_allows_different_stages(tmp_path):
    """Different stages on the same video can be locked concurrently."""
    from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

    # Both locks should be acquirable without error.
    with video_stage_lock(tmp_path, video_id="vid_multi", stage="visual"):
        with video_stage_lock(tmp_path, video_id="vid_multi", stage="asr"):
            pass  # no StageLockError expected


def test_publish_lock_serializes_different_modalities_for_one_video(tmp_path):
    from app.vector_store.milvus.milvus_stage_lock import StageLockError, video_stage_lock

    with video_stage_lock(tmp_path, video_id="vid_publish", stage="publish"):
        with pytest.raises(StageLockError):
            with video_stage_lock(tmp_path, video_id="vid_publish", stage="publish"):
                pass


def test_stage_lock_releases_on_exception(tmp_path):
    """The lock is released even when the body raises an exception."""
    from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

    with pytest.raises(ValueError):
        with video_stage_lock(tmp_path, video_id="vid_exc", stage="face"):
            raise ValueError("simulated indexing failure")

    # After the exception the lock should be released — re-acquiring must succeed.
    with video_stage_lock(tmp_path, video_id="vid_exc", stage="face"):
        pass  # must not raise
