import numpy as np
import pytest

from app.catalog.db import Catalog
from app.retrieval.search import (
    Candidate,
    SearchEngine,
    _fuse_candidate_groups,
    _groups,
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


def _publish(catalog, video_id, channels, **_unused):
    for modality, raw in channels.items():
        channel = dict(raw)
        asset_version = str(channel.pop("milvus_asset_version", "1"))
        row_count = int(channel.pop("milvus_row_count", 1))
        channel.pop("file", None)
        catalog.publish_modality(
            video_id,
            modality,
            asset_version=asset_version,
            row_count=row_count,
            metadata=channel,
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


def test_visual_priority_uses_support_to_rank_primary_backed_results():
    candidates = [
        Candidate("video-1", 0, 1, 0.90, "visual"),
        Candidate("video-1", 0, 1, 0.10, "ocr"),
        Candidate("video-1", 10, 11, 0.80, "visual"),
        Candidate("video-1", 10, 11, 1.00, "ocr"),
        Candidate("video-1", 20, 21, 0.99, "asr"),
    ]
    videos = [{"id": "video-1", "name": "video.mp4"}]

    results = _fuse_candidate_groups(
        candidates,
        videos,
        merge_gap=0,
        max_result_seconds=5,
        primary_modality="visual",
    )

    assert [(item.start_time, item.modalities) for item in results] == [
        (10, ["ocr", "visual"]),
        (0, ["ocr", "visual"]),
        (20, ["asr"]),
    ]
    assert results[0].score > results[1].score


def test_pure_ocr_groups_respect_max_result_seconds():
    candidates = [
        Candidate("video-1", 0, 1, 0.95, "ocr"),
        Candidate("video-1", 1, 2, 0.99, "ocr"),
        Candidate("video-1", 2, 3, 0.98, "ocr"),
        Candidate("video-1", 3, 4, 0.97, "ocr"),
    ]

    groups = _groups(candidates, gap=2, max_duration=2)

    assert [(min(item.start_time for item in group), max(item.end_time for item in group)) for group in groups] == [
        (0, 2),
        (2, 4),
    ]
    assert all(max(item.end_time for item in group) - min(item.start_time for item in group) <= 2 for group in groups)


def test_pure_ocr_group_can_reach_exact_max_result_seconds():
    candidates = [
        Candidate("video-1", 0, 1, 0.98, "ocr"),
        Candidate("video-1", 1, 2, 0.99, "ocr"),
    ]

    groups = _groups(candidates, gap=2, max_duration=2)

    assert len(groups) == 1
    assert min(item.start_time for item in groups[0]) == 0
    assert max(item.end_time for item in groups[0]) == 2


def test_mixed_ocr_groups_respect_max_result_seconds_before_auxiliary_merge():
    ocr_early = Candidate("video-1", 0, 1, 0.95, "ocr")
    ocr_seed = Candidate("video-1", 1, 2, 0.99, "ocr")
    ocr_late = Candidate("video-1", 2, 3, 0.98, "ocr")
    face = Candidate("video-1", 2.2, 2.8, 0.90, "face", raw_score=0.72)

    groups = _groups([ocr_early, ocr_seed, ocr_late, face], gap=2, max_duration=2)

    face_group = next(group for group in groups if face in group)
    assert ocr_late in face_group
    assert ocr_seed not in face_group
    assert all(max(item.end_time for item in group) - min(item.start_time for item in group) <= 2 for group in groups)


def test_fuse_candidate_groups_caps_ocr_only_result_windows():
    candidates = [
        Candidate("video-1", 0, 1, 0.95, "ocr"),
        Candidate("video-1", 1, 2, 0.99, "ocr"),
        Candidate("video-1", 2, 3, 0.98, "ocr"),
    ]
    videos = [{"id": "video-1", "name": "video.mp4"}]

    results = _fuse_candidate_groups(candidates, videos, merge_gap=2, max_result_seconds=2)

    assert [(item.start_time, item.end_time) for item in results] == [(0, 2), (2, 3)]
    assert all((item.end_time - item.start_time) <= 2 for item in results)


def test_search_ignores_legacy_npz_without_catalog_publication(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog)
    # Simulate a pre-publication database that claimed visual availability.
    # The public Catalog API deliberately cannot create this stale state.
    with catalog.connect() as connection:
        connection.execute(
            "UPDATE videos SET indexed_modalities=? WHERE id=?",
            ('["visual"]', "video-1"),
        )
    index_dir.mkdir(parents=True)
    np.savez_compressed(
        index_dir / "visual.npz",
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        start_times=np.asarray([0], dtype=np.float32),
        end_times=np.asarray([5], dtype=np.float32),
    )

    engine = SearchEngine(settings, catalog)

    # Ready publications, not the stale compatibility flag or cold NPZ, define
    # online visibility. With no publication there is simply no visual channel
    # to query.
    assert engine.search("football", None, ["visual"], ["video-1"]) == []


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


def test_optional_face_channel_does_not_block_non_face_image_search(tmp_path):
    engine = SearchEngine(_settings(tmp_path), Catalog(_settings(tmp_path).db_path))

    class NoFaceEncoder:
        def encode_reference(self, _path):
            raise ValueError("参考图中未检测到人脸")

    engine._face_encoder = NoFaceEncoder()

    assert engine._resolve_face_query(None, "product.jpg", optional=True) is None


def test_face_only_image_search_reports_missing_face(tmp_path):
    engine = SearchEngine(_settings(tmp_path), Catalog(_settings(tmp_path).db_path))

    class NoFaceEncoder:
        def encode_reference(self, _path):
            raise ValueError("参考图中未检测到人脸")

    engine._face_encoder = NoFaceEncoder()

    with pytest.raises(ValueError, match="未检测到人脸"):
        engine._resolve_face_query(None, "product.jpg", optional=False)


def test_semantic_query_is_encoded_once_per_model_per_request(tmp_path):
    settings = _settings(tmp_path)
    engine = SearchEngine(settings, Catalog(settings.db_path))
    calls = []
    engine._encode_asr_query = lambda text, model: (  # type: ignore[method-assign]
        calls.append((text, model)) or np.asarray([1.0, 0.0], dtype=np.float32)
    )
    cache = {}
    publication = {
        "semantic_model_key": "semantic-a",
        "semantic_status": "complete",
    }
    embeddings = np.ones((1, 2), dtype=np.float32)

    first = engine._semantic_query("hello", publication, embeddings, cache, None)
    second = engine._semantic_query("hello", publication, embeddings, cache, None)

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


def test_prewarm_scans_deduplicated_publication_model_keys(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        search_prewarm_enabled=True,
        asr_semantic_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    _create_video(
        settings,
        catalog,
        video_id="visual-model",
    )
    _publish(catalog, "visual-model", {
        "visual": {
            "model_key": "chinese-clip-vit-b16",
        }
    })
    _create_video(settings, catalog, video_id="semantic-model")
    _publish(catalog, "semantic-model", {
        "asr": {
            "semantic_model_key": "semantic-from-publication",
            "semantic_status": "complete",
        }
    })
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
    assert set(text_calls) == {"semantic-from-publication"}
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
        _create_video(settings, catalog, video_id=video_id)
        _publish(catalog, video_id, {
            channel: {
                "semantic_model_key": f"unused-{channel}-model",
                "semantic_status": semantic_status,
            }
        })

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
    _create_video(settings, catalog, video_id="ocr-semantic")
    _publish(catalog, "ocr-semantic", {
        "ocr": {
            "semantic_model_key": "ocr-semantic-model",
            "semantic_status": "complete",
        }
    })
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
    _create_video(settings, catalog, video_id="asr-semantic")
    _publish(catalog, "asr-semantic", {
        "asr": {
            "semantic_model_key": "missing-semantic-model",
            "semantic_status": "complete",
        }
    })
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
    """Create a minimal published visual index for Milvus search tests."""
    _create_video(settings, catalog, video_id=video_id, duration=20)
    _publish(catalog, video_id, {
        "visual": {
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 5.0,
            "decode_status": "complete",
            "milvus_asset_version": "1",
        }
    }, duration_ms=20000)
    return video_id


def test_milvus_search_scores_each_selected_video_once(tmp_path):
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

    def fake_candidates(video, **_kwargs):
        events.append(("score", video["id"]))
        return [Candidate(video["id"], 0.0, 1.0, 0.8, "visual")]

    with (
        patch.object(engine, "_prepare_query_vectors"),
        patch.object(engine, "_get_milvus_client", return_value=object()),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            side_effect=fake_candidates,
        ),
    ):
        engine.search("football", None, ["visual"])

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


def test_milvus_publication_is_the_online_retrieval_source(tmp_path):
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
            "_milvus_candidates_for_video",
            return_value=[milvus_hit],
        ) as milvus_search,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    milvus_search.assert_called_once()
    assert results[0]["start_time"] == 5.0
    assert results[0]["evidence"][0]["features"] == {}
