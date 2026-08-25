from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.settings import Settings
from app.retrieval.retrieval_metrics import RetrievalProfiler
from app.vector_store.milvus.milvus_search_visual_v2 import (
    MilvusVisualSearchError,
    _aggregate_by_segment,
    _aggregate_global_by_segment,
    _ann_recall_global_multi_query,
    milvus_visual_candidates_ann,
    milvus_visual_candidates_global_ann,
)


def _hit(
    video_id: str,
    asset_version: str,
    *,
    score: float = 0.8,
    segment_id: int = 0,
    frame_idx: int = 0,
    start_ms: int = 0,
    end_ms: int = 5_000,
    timestamp_ms: int = 1_000,
):
    hit = MagicMock()
    hit.distance = score
    fields = {
        "video_id": video_id,
        "asset_version": asset_version,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "segment_id": segment_id,
        "segment_start_ms": start_ms,
        "segment_end_ms": end_ms,
    }
    hit.entity.get.side_effect = lambda field: fields.get(field)
    return hit


def _client_with_results(result_sets):
    collection = MagicMock()
    collection.search.return_value = result_sets
    client = MagicMock()
    client.collection_for.return_value = collection
    return client, collection


def _row(
    video_id: str,
    asset_version: str,
    *,
    query_idx: int = 0,
    segment_id: int = 0,
    frame_idx: int = 0,
    start_ms: int = 0,
    end_ms: int = 1_000,
    timestamp_ms: int = 500,
    cosine: float = 0.8,
) -> dict:
    return {
        "video_id": video_id,
        "asset_version": asset_version,
        "query_idx": query_idx,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "segment_id": segment_id,
        "segment_start_ms": start_ms,
        "segment_end_ms": end_ms,
        "cosine": cosine,
    }


def test_global_visual_ann_batches_queries_and_publications_into_one_rpc():
    special_video = 'video-"a\\b'
    scope = {special_video: 'version-"1', "video-b": "2"}
    client, collection = _client_with_results([
        [_hit(special_video, 'version-"1', score=0.9)],
        [_hit("video-b", "2", score=0.8)],
    ])
    profiler = RetrievalProfiler()
    settings = MagicMock(milvus_query_timeout_seconds=4.5)

    with patch("app.core.settings.get_settings", return_value=settings):
        rows = _ann_recall_global_multi_query(
            client,
            scope,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            top_k=2_000,
            use_diskann=True,
            profiler=profiler,
        )

    collection.search.assert_called_once()
    kwargs = collection.search.call_args.kwargs
    assert len(kwargs["data"]) == 2
    assert kwargs["limit"] == 2_000
    assert kwargs["param"]["params"]["search_list"] == 2_000
    assert kwargs["timeout"] == 4.5
    assert {"video_id", "asset_version"} <= set(kwargs["output_fields"])
    assert "embedding" not in kwargs["output_fields"]
    assert kwargs["expr"] == " or ".join(
        f"(video_id == {json.dumps(video_id, ensure_ascii=False)} and "
        f"asset_version == {json.dumps(asset_version, ensure_ascii=False)})"
        for video_id, asset_version in sorted(scope.items())
    )
    assert {(row["video_id"], row["asset_version"]) for row in rows} == set(
        scope.items()
    )
    snapshot = profiler.snapshot()
    assert snapshot["counters"]["milvus"]["visual_requests"] == 1
    assert snapshot["counters"]["milvus"]["visual_rpc_attempted"] == 1
    assert snapshot["counters"]["milvus"]["visual_rpc_succeeded"] == 1
    assert "visual_rpc_failed" not in snapshot["counters"]["milvus"]
    assert snapshot["counters"]["milvus"]["visual_rows"] == 2
    assert snapshot["counters"]["visual_scope"]["publication_pair_count"] == 2
    assert snapshot["counters"]["visual_scope"]["expr_utf8_bytes"] == len(
        kwargs["expr"].encode("utf-8")
    )
    assert snapshot["timing_stats"]["visual_scope"]["expr_build"]["count"] == 1


def test_global_visual_ann_drops_stale_and_out_of_scope_hits():
    client, _collection = _client_with_results([[
        _hit("video-a", "stale", score=0.99),
        _hit("video-b", "2", score=0.9),
        _hit("video-outside", "1", score=0.95),
    ]])
    settings = MagicMock(milvus_query_timeout_seconds=3.0)

    with patch("app.core.settings.get_settings", return_value=settings):
        rows = _ann_recall_global_multi_query(
            client,
            {"video-a": "current", "video-b": "2"},
            np.ones((1, 2), dtype=np.float32),
            top_k=10,
            use_diskann=False,
            profiler=None,
        )

    assert [(row["video_id"], row["asset_version"]) for row in rows] == [
        ("video-b", "2")
    ]


def test_global_visual_aggregation_uses_publication_composite_keys():
    rows = [
        _row("video-a", "1", frame_idx=0, start_ms=0, end_ms=1_000),
        _row(
            "video-a",
            "1",
            frame_idx=1,
            start_ms=1_000,
            end_ms=2_000,
            timestamp_ms=1_500,
        ),
        _row("video-b", "2", cosine=0.7),
    ]

    candidates = _aggregate_global_by_segment(
        rows,
        {"video-a": "1", "video-b": "2"},
        limit=10,
        profile="balanced",
        n_queries=1,
    )

    # The inconsistent video-a segment is rejected without removing the same
    # segment id from another publication.
    assert [(item.video_id, item.unit_id) for item in candidates] == [
        ("video-b", 0)
    ]


def test_global_visual_ann_rejects_partial_multiquery_result_sets():
    client, _collection = _client_with_results([[]])
    settings = MagicMock(milvus_query_timeout_seconds=3.0)

    with patch("app.core.settings.get_settings", return_value=settings):
        with pytest.raises(MilvusVisualSearchError) as error:
            _ann_recall_global_multi_query(
                client,
                {"video-a": "1"},
                np.ones((2, 2), dtype=np.float32),
                top_k=10,
                use_diskann=False,
                profiler=None,
            )

    assert isinstance(error.value.__cause__, ValueError)
    assert "expected=2 actual=1" in str(error.value.__cause__)


def test_global_visual_ann_counts_an_rpc_that_raises_as_attempted_and_failed():
    client, collection = _client_with_results([])
    collection.search.side_effect = RuntimeError("milvus unavailable")
    settings = MagicMock(milvus_query_timeout_seconds=3.0)
    profiler = RetrievalProfiler()

    with patch("app.core.settings.get_settings", return_value=settings):
        with pytest.raises(MilvusVisualSearchError, match="publication cohort"):
            _ann_recall_global_multi_query(
                client,
                {"video-a": "1"},
                np.ones((1, 2), dtype=np.float32),
                top_k=10,
                use_diskann=False,
                profiler=profiler,
            )

    counters = profiler.snapshot()["counters"]["milvus"]
    assert counters["visual_requests"] == 1
    assert counters["visual_rpc_attempted"] == 1
    assert counters["visual_rpc_failed"] == 1
    assert "visual_rpc_succeeded" not in counters


def test_single_publication_aggregation_matches_legacy_fields_and_order():
    legacy_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"video_id", "asset_version"}
        }
        for row in [
            _row("video-a", "1", segment_id=4, cosine=0.8),
            _row(
                "video-a",
                "1",
                segment_id=7,
                frame_idx=2,
                start_ms=2_000,
                end_ms=3_000,
                timestamp_ms=2_500,
                cosine=0.9,
            ),
        ]
    ]
    global_rows = [
        {**row, "video_id": "video-a", "asset_version": "1"}
        for row in legacy_rows
    ]

    legacy = _aggregate_by_segment(
        legacy_rows, "video-a", 10, "balanced", 1
    )
    global_candidates = _aggregate_global_by_segment(
        global_rows, {"video-a": "1"}, 10, "balanced", 1
    )

    assert global_candidates == legacy


def test_single_publication_public_wrapper_matches_global_api():
    query = [np.ones(2, dtype=np.float32)]
    settings = MagicMock(
        visual_ann_top_k=2_000,
        visual_ann_segment_top_n=3,
        visual_use_diskann=True,
        milvus_query_timeout_seconds=3.0,
    )
    result_sets = [[_hit("video-a", "1", score=0.9)]]
    single_client, _ = _client_with_results(result_sets)
    global_client, _ = _client_with_results(result_sets)

    with (
        patch("app.core.settings.get_settings", return_value=settings),
        patch(
            "app.vector_store.milvus.milvus_search_visual_v2._verify_index_type_once"
        ),
    ):
        single = milvus_visual_candidates_ann(
            single_client, "video-a", "1", query, 10, "balanced"
        )
        global_candidates = milvus_visual_candidates_global_ann(
            global_client, {"video-a": "1"}, query, 10, "balanced"
        )

    assert single == global_candidates


def test_visual_ann_global_recall_default_is_2000():
    assert Settings.model_fields["visual_ann_top_k"].default == 2_000
