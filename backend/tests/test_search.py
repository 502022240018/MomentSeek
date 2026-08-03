import json

import numpy as np
import pytest

from app.catalog.db import Catalog
from app.retrieval.search import (
    Candidate,
    SearchEngine,
    SearchResult,
    _fuse_candidate_groups,
    _groups,
    _reserve_asr_lexical_results,
    lexical_score,
)
from app.retrieval.retrieval_metrics import RetrievalProfiler
from app.core.settings import Settings


def _settings(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    return settings


def _create_video(settings, catalog, video_id="video-1", name="video.mp4", duration=60):
    video_path = settings.upload_dir / f"{video_id}.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": video_id,
        "name": name,
        "file_path": str(video_path),
        "duration": duration,
        "fps": 25,
        "width": 1280,
        "height": 720,
        "status": "ready",
    })
    return settings.index_dir / video_id


def _write_manifest(index_dir, video_id, channels, duration_ms=60000, segment_ms=5000):
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_manifest.json").write_text(
        json.dumps({
            "schema_version": 3,
            "video_id": video_id,
            "duration_ms": duration_ms,
            "segment_ms": segment_ms,
            "channels": channels,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_visual_adjacent_segments_remain_separate():
    candidates = [
        Candidate("video-1", 0, 5, 0.95, "visual"),
        Candidate("video-1", 5, 10, 0.94, "visual"),
        Candidate("video-1", 10, 15, 0.93, "visual"),
    ]

    groups = _groups(candidates, gap=2, max_duration=15)

    assert [(group[0].start_time, group[-1].end_time) for group in groups] == [(0, 5), (5, 10), (10, 15)]


def test_asr_adjacent_segments_can_merge():
    candidates = [
        Candidate("video-1", 10, 13, 1.0, "asr"),
        Candidate("video-1", 14, 17, 1.0, "asr"),
    ]

    groups = _groups(candidates, gap=2, max_duration=15)

    assert len(groups) == 1
    assert min(item.start_time for item in groups[0]) == 10
    assert max(item.end_time for item in groups[0]) == 17


def test_cjk_lexical_score_keeps_bigram_coverage_on_entity_extension():
    text = "说实话,我们天山不好进的,一般都去昆仑。"

    assert lexical_score("昆仑山", text) == pytest.approx(1 / 2)
    assert lexical_score("昆仑山", "今天去昆明旅游") == 0


def test_asr_lexical_pool_preserves_primary_top3_and_reserves_next_slot():
    def result(name: str, score: float, lexical: float) -> SearchResult:
        return SearchResult(
            video_id=name,
            video_name=name,
            start_time=0,
            end_time=1,
            score=score,
            modalities=["asr"],
            thumbnail_url=None,
            media_url="",
            clip_url="",
            decision="semantic_hit",
            evidence=[{"modality": "asr", "lexical_score": lexical}],
        )

    primary = [
        result("lexical-top", 0.99, 0.5),
        result("semantic-1", 0.98, 0.0),
        result("semantic-2", 0.97, 0.0),
        result("semantic-3", 0.96, 0.0),
        result("weak-lexical", 0.95, 0.4),
        result("lexical-reserved", 0.50, 0.5),
    ]

    reranked = _reserve_asr_lexical_results(primary, limit=5)

    assert [item.video_id for item in reranked[:4]] == [
        "lexical-top",
        "semantic-1",
        "semantic-2",
        "lexical-reserved",
    ]
    assert reranked.index(primary[3]) < reranked.index(primary[4])


def test_visual_priority_orders_visual_evidence_before_auxiliary_candidates():
    candidates = [
        Candidate("video-1", 0, 1, 0.70, "visual"),
        Candidate("video-1", 10, 11, 0.99, "asr"),
    ]
    videos = [{"id": "video-1", "name": "video.mp4"}]

    default_results = _fuse_candidate_groups(
        candidates, videos, merge_gap=0, max_result_seconds=5
    )
    priority_results = _fuse_candidate_groups(
        candidates,
        videos,
        merge_gap=0,
        max_result_seconds=5,
        primary_modality="visual",
    )

    assert default_results[0].modalities == ["asr"]
    assert priority_results[0].modalities == ["visual"]
    assert priority_results[1].modalities == ["asr"]


def test_visual_priority_does_not_cross_threshold_tiers():
    candidates = [
        Candidate("video-1", 0, 1, 0.95, "visual", above_threshold=False),
        Candidate("video-1", 10, 11, 0.40, "ocr", above_threshold=True),
        Candidate("video-1", 20, 21, 0.90, "face", above_threshold=False),
    ]
    videos = [{"id": "video-1", "name": "video.mp4"}]

    results = _fuse_candidate_groups(
        candidates,
        videos,
        merge_gap=0,
        max_result_seconds=5,
        primary_modality="visual",
    )

    assert [item.modalities for item in results] == [["ocr"], ["visual"], ["face"]]


def test_search_rejects_legacy_index_without_v3_manifest(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog)
    catalog.update_video("video-1", indexed_modalities=["visual"])
    index_dir.mkdir(parents=True)
    np.savez_compressed(
        index_dir / "visual.npz",
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        start_times=np.asarray([0], dtype=np.float32),
        end_times=np.asarray([5], dtype=np.float32),
    )

    engine = SearchEngine(settings, catalog)

    with pytest.raises(ValueError, match="索引版本过旧"):
        engine.search("football", None, ["visual"], ["video-1"])


def test_visual_query_subqueries_use_one_batched_encoder_call(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(settings, Catalog(settings.db_path))
    calls = []

    class StubClip:
        def encode_queries(self, texts, image_path, alpha):
            calls.append((texts, image_path, alpha))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    engine._clip = lambda _model=None: StubClip()  # type: ignore[method-assign]

    vectors = engine._encode_visual_queries(
        "siglip2-so400m-384",
        ["players holding cartons", "players clinking cartons"],
        None,
        0.5,
        None,
    )

    assert vectors.shape == (2, 2)
    assert calls == [(
        ["players holding cartons", "players clinking cartons"],
        None,
        0.5,
    )]


def test_semantic_query_is_encoded_once_per_model_per_request(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(settings, Catalog(settings.db_path))
    calls = []
    engine._encode_asr_query = lambda text, model: (  # type: ignore[method-assign]
        calls.append((text, model)) or np.asarray([1.0, 0.0], dtype=np.float32)
    )
    cache = {}
    manifest = {
        "semantic_model_key": "semantic-a",
        "semantic_status": "complete",
    }
    embeddings = np.ones((1, 2), dtype=np.float32)

    first = engine._semantic_query("hello", manifest, embeddings, cache, None)
    second = engine._semantic_query("hello", manifest, embeddings, cache, None)

    assert np.array_equal(first, second)
    assert calls == [("hello", "semantic-a")]


def test_retrieval_profiler_accumulates_timings_and_counters():
    profiler = RetrievalProfiler()

    with profiler.span("query_encode", "visual"):
        pass
    profiler.increment("milvus", "visual_rows", 12)
    profiler.increment("milvus", "visual_rows", 3)

    snapshot = profiler.snapshot()
    assert snapshot["timing"]["query_encode"]["visual"] >= 0
    assert snapshot["counters"]["milvus"]["visual_rows"] == 15


def test_prewarm_loads_visual_default_and_reports_resident_models(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        asr_semantic_enabled=True,
    )
    engine = SearchEngine(settings, Catalog(settings.db_path))
    visual_calls = []

    class StubVisual:
        def encode_queries(self, texts, image_path, alpha=0.5):
            visual_calls.append((texts, image_path, alpha))
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    engine._clip = lambda model=None, profiler=None: (  # type: ignore[method-assign]
        engine._clip_encoders.setdefault(model, StubVisual())
    )

    engine._encode_asr_query = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no semantic-complete index requires a text model")
        )
    )

    status = engine.prewarm()

    assert status["status"] == "ready"
    assert status["resident"] is True
    assert status["visual_models"] == [settings.visual_model]
    assert status["text_models"] == []
    assert status["requested_text_models"] == []
    assert len(visual_calls) == 1


def test_prewarm_scans_deduplicated_manifest_model_keys(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        asr_semantic_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    visual_dir = _create_video(
        settings,
        catalog,
        video_id="visual-model",
    )
    catalog.update_video("visual-model", indexed_modalities=["visual"])
    _write_manifest(visual_dir, "visual-model", {
        "visual": {
            "file": "visual.npz",
            "model_key": "chinese-clip-vit-b16",
        }
    })
    np.savez_compressed(visual_dir / "visual.npz", value=np.asarray([1]))

    asr_dir = _create_video(settings, catalog, video_id="semantic-model")
    catalog.update_video("semantic-model", indexed_modalities=["asr"])
    _write_manifest(asr_dir, "semantic-model", {
        "asr": {
            "file": "asr.npz",
            "semantic_model_key": "semantic-from-manifest",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(asr_dir / "asr.npz", value=np.asarray([1]))

    engine = SearchEngine(settings, catalog)
    visual_calls = []
    text_calls = []

    class StubVisual:
        def encode_queries(self, texts, image_path, alpha=0.5):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    def fake_clip(model, profiler=None):
        visual_calls.append(model)
        return engine._clip_encoders.setdefault(model, StubVisual())

    def fake_text(text, model, profiler=None):
        text_calls.append(model)
        engine._text_encoders[(model, "cpu")] = object()
        return np.asarray([1.0, 0.0], dtype=np.float32)

    engine._clip = fake_clip  # type: ignore[method-assign]
    engine._encode_asr_query = fake_text  # type: ignore[method-assign]

    status = engine.prewarm()

    assert status["status"] == "ready"
    assert set(visual_calls) == {
        settings.visual_model,
        "chinese-clip-vit-b16",
    }
    assert set(text_calls) == {"semantic-from-manifest"}
    assert status["requested_visual_models"] == sorted(set(visual_calls))
    assert status["requested_text_models"] == sorted(set(text_calls))


def test_prewarm_ignores_disabled_and_failed_semantic_models(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        search_prewarm_required=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    for channel, semantic_status in (
        ("asr", "disabled"),
        ("ocr", "failed"),
    ):
        video_id = f"{channel}-{semantic_status}"
        index_dir = _create_video(settings, catalog, video_id=video_id)
        catalog.update_video(video_id, indexed_modalities=[channel])
        _write_manifest(index_dir, video_id, {
            channel: {
                "file": f"{channel}.npz",
                "semantic_model_key": f"unused-{channel}-model",
                "semantic_status": semantic_status,
            }
        })
        np.savez_compressed(
            index_dir / f"{channel}.npz",
            value=np.asarray([1]),
        )

    engine = SearchEngine(settings, catalog)

    class StubVisual:
        def encode_queries(self, texts, image_path, alpha=0.5):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    engine._clip = (  # type: ignore[method-assign]
        lambda model=None, profiler=None: engine._clip_encoders.setdefault(
            model,
            StubVisual(),
        )
    )
    text_calls = []
    engine._encode_asr_query = (  # type: ignore[method-assign]
        lambda text, model, profiler=None: text_calls.append(model)
    )

    status = engine.prewarm()

    assert status["status"] == "ready"
    assert status["requested_text_models"] == []
    assert status["text_models"] == []
    assert text_calls == []


def test_prewarm_discovers_complete_ocr_model_when_asr_semantic_is_disabled(
    tmp_path,
):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        asr_semantic_enabled=False,
        ocr_semantic_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, video_id="ocr-semantic")
    catalog.update_video("ocr-semantic", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "ocr-semantic", {
        "ocr": {
            "file": "ocr.npz",
            "semantic_model_key": "ocr-semantic-model",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(index_dir / "ocr.npz", value=np.asarray([1]))
    engine = SearchEngine(settings, catalog)

    class StubVisual:
        def encode_queries(self, texts, image_path, alpha=0.5):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    engine._clip = (  # type: ignore[method-assign]
        lambda model=None, profiler=None: engine._clip_encoders.setdefault(
            model,
            StubVisual(),
        )
    )

    def fake_text(text, model, profiler=None):
        engine._text_encoders[(model, "cpu")] = object()
        return np.asarray([1.0, 0.0], dtype=np.float32)

    engine._encode_asr_query = fake_text  # type: ignore[method-assign]

    status = engine.prewarm()

    assert status["status"] == "ready"
    assert status["requested_text_models"] == ["ocr-semantic-model"]
    assert status["text_models"] == ["ocr-semantic-model"]


def test_required_prewarm_fails_for_missing_complete_semantic_model(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        search_prewarm_required=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, video_id="asr-semantic")
    catalog.update_video("asr-semantic", indexed_modalities=["asr"])
    _write_manifest(index_dir, "asr-semantic", {
        "asr": {
            "file": "asr.npz",
            "semantic_model_key": "missing-semantic-model",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(index_dir / "asr.npz", value=np.asarray([1]))
    engine = SearchEngine(settings, catalog)

    class StubVisual:
        def encode_queries(self, texts, image_path, alpha=0.5):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    engine._clip = (  # type: ignore[method-assign]
        lambda model=None, profiler=None: engine._clip_encoders.setdefault(
            model,
            StubVisual(),
        )
    )
    engine._encode_asr_query = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing semantic model")
        )
    )

    with pytest.raises(RuntimeError, match="missing-semantic-model"):
        engine.prewarm()

    assert engine.query_model_status()["status"] == "error"


def test_query_model_status_reads_encoder_maps_under_lock(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(
        settings,
        Catalog(settings.db_path),
    )

    class LockCheckingDict(dict):
        def __iter__(self):
            assert engine._encoder_lock._is_owned()  # type: ignore[attr-defined]
            return super().__iter__()

    engine._clip_encoders = LockCheckingDict({"visual-a": object()})
    engine._text_encoders = LockCheckingDict({("text-a", "cpu"): object()})

    status = engine.query_model_status()

    assert status["visual_models"] == ["visual-a"]
    assert status["text_models"] == ["text-a"]


def _make_visual_index(settings, catalog, video_id="v-shadow"):
    """Create a minimal v3 visual index for shadow_compare tests."""
    index_dir = _create_video(settings, catalog, video_id=video_id, duration=20)
    catalog.update_video(video_id, indexed_modalities=["visual"])
    _write_manifest(index_dir, video_id, {
        "visual": {
            "file": "visual.npz",
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 5.0,
            "decode_status": "complete",
            "milvus_asset_version": "1",
        }
    }, duration_ms=20000)
    np.savez_compressed(
        index_dir / "visual.npz",
        frame_embeddings=np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float16),
        frame_times_ms=np.asarray([1000, 6000], dtype=np.int32),
        segment_frame_offsets=np.asarray([0, 1, 2], dtype=np.int32),
    )
    return video_id


def test_milvus_batches_are_scored_before_next_batch_is_loaded(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    settings.milvus_search_video_batch_size = 2
    catalog = Catalog(settings.db_path)
    video_ids = [
        _make_visual_index(settings, catalog, video_id=f"batch-{index}")
        for index in range(3)
    ]
    engine = SearchEngine(settings, catalog)
    events = []

    def fake_query(_client, modality, batch_ids, _fields, _profiler):
        events.append(("query", modality, tuple(batch_ids)))
        return {
            video_id: [{"video_id": video_id}]
            for video_id in batch_ids
        }

    def fake_candidates(video, **_kwargs):
        events.append(("score", video["id"]))
        return [Candidate(video["id"], 0.0, 1.0, 0.8, "visual")]

    with (
        patch.object(engine, "_prepare_query_vectors"),
        patch.object(engine, "_get_milvus_client", return_value=object()),
        patch.object(
            engine,
            "_query_rows_for_videos",
            side_effect=fake_query,
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            side_effect=fake_candidates,
        ),
    ):
        engine.search("football", None, ["visual"])

    # After the fix (Visual removed from BULK_QUERY_FIELDS), _query_rows_for_videos
    # is never called for visual modality. Visual uses ANN search directly without
    # pre-fetching rows. Only scoring events should occur.
    assert sorted(events) == sorted(("score", video_id) for video_id in video_ids)


def test_query_encoding_finishes_before_local_candidate_scoring(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(
        settings,
        catalog,
        video_id="timing-order",
    )
    engine = SearchEngine(settings, catalog)
    profiler = RetrievalProfiler()
    events = []

    class StubClip:
        def encode_queries(self, texts, image_path, alpha):
            events.append("encode")
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    engine._clip = (  # type: ignore[method-assign]
        lambda _model=None, _profiler=None: StubClip()
    )

    def fake_candidates(video, **_kwargs):
        events.append("score")
        return [Candidate(video["id"], 0.0, 1.0, 0.8, "visual")]

    with (
        patch.object(engine, "_get_milvus_client", return_value=object()),
        patch.object(
            engine,
            "_query_rows_for_videos",
            return_value={video_id: [{"video_id": video_id}]},
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            side_effect=fake_candidates,
        ),
    ):
        engine.search(
            "football",
            None,
            ["visual"],
            [video_id],
            profiler=profiler,
        )

    assert events == ["encode", "score"]
    timing = profiler.snapshot()["timing"]
    assert "visual" in timing["query_encode"]
    assert "visual_scoring" in timing["local_processing"]


def test_milvus_is_primary_and_npz_is_not_read_on_success(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog, video_id="v-primary")
    engine = SearchEngine(settings, catalog)
    milvus_hit = Candidate(
        video_id=video_id,
        start_time=5.0,
        end_time=10.0,
        score=0.9,
        modality="visual",
        evidence="[milvus] hit",
    )

    with (
        patch.object(engine, "_prepare_query_vectors"),
        patch.object(engine, "_get_milvus_client", return_value=object()),
        patch.object(
            engine,
            "_query_rows_for_videos",
            return_value={video_id: []},
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            return_value=[milvus_hit],
        ) as milvus_search,
        patch("app.retrieval.search.np.load") as npz_load,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    milvus_search.assert_called_once()
    npz_load.assert_not_called()
    assert results[0]["start_time"] == 5.0
    assert results[0]["evidence"][0]["features"] == {}


def test_visual_ann_does_not_bulk_fetch_rows(tmp_path):
    """Regression: visual must NOT appear in the BULK_QUERY_FIELDS pre-fetch loop.

    The v2 ANN implementation (milvus_visual_candidates_ann) issues its own
    collection.search() call and never consumes pre-fetched rows.  Before the
    fix, the full query_iterator traversal ran for every visual-indexed video,
    reading all frame embeddings from Milvus before the ANN search — a no-op
    fetch that wasted significant I/O.

    This test guards against that regression by asserting that
    _query_rows_for_videos is never called with modality="visual" during a
    real Milvus-routed visual search.
    """
    from unittest.mock import MagicMock, patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog, video_id="v-ann-no-bulk")
    engine = SearchEngine(settings, catalog)

    milvus_hit = Candidate(video_id, 2.0, 7.0, 0.88, "visual")
    query_rows_calls: list[str] = []

    def spy_query_rows(_client, modality, _video_ids, _fields, _profiler):
        query_rows_calls.append(modality)
        return {video_id: []}

    with (
        patch.object(engine, "_prepare_query_vectors"),
        patch.object(engine, "_get_milvus_client", return_value=MagicMock()),
        patch.object(engine, "_query_rows_for_videos", side_effect=spy_query_rows),
        patch.object(engine, "_milvus_candidates_for_video", return_value=[milvus_hit]),
    ):
        engine.search("football", None, ["visual"], [video_id])

    assert "visual" not in query_rows_calls, (
        "_query_rows_for_videos must never be called with modality='visual'; "
        "visual uses ANN (collection.search) directly and pre-fetching all "
        "frame embeddings is a costly no-op.  "
        f"Actual calls: {query_rows_calls}"
    )
