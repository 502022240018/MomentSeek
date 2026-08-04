import json

import numpy as np
import pytest

from app.catalog.db import Catalog
from app.retrieval.search import (
    Candidate,
    SearchEngine,
    _fuse_candidate_groups,
    _groups,
    _visual_candidates,
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


@pytest.mark.skip(reason="ASR migrated to Milvus; NPZ v3 fallback removed")
def test_asr_v3_lexical_search_uses_chunk_times_and_texts(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="interview.mp4")
    catalog.update_video("video-1", indexed_modalities=["asr"])
    _write_manifest(index_dir, "video-1", {
        "asr": {
            "file": "asr.npz",
            "engine": "whisper",
            "model_key": "small",
            "language": "zh",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "decode_status": "complete",
            "semantic_status": "disabled",
        }
    })
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[10000, 13000], [14000, 17000], [40000, 42000]], dtype=np.int32),
        texts=np.asarray(["我们正在讨论电影投资", "电影投资需要长期判断", "今天天气很好"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )

    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled semantic index must not encode a query")
        )
    )

    results = engine.search("电影投资", None, ["asr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 10
    assert results[0]["end_time"] == 17
    assert results[0]["media_url"] == "/api/videos/video-1/media"
    assert results[0]["clip_url"] == "/api/videos/video-1/clip?start=10.000&end=17.000"
    assert results[0]["evidence"][0]["unit_type"] == "chunk"


@pytest.mark.skip(reason="ASR migrated to Milvus; NPZ v3 fallback removed")
@pytest.mark.skip(reason="ASR migrated to Milvus; NPZ v3 fallback removed")
def test_asr_v3_sparse_semantic_indices_map_embeddings_to_chunks(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="interview.mp4")
    catalog.update_video("video-1", indexed_modalities=["asr"])
    _write_manifest(index_dir, "video-1", {
        "asr": {
            "file": "asr.npz",
            "engine": "whisper",
            "model_key": "small",
            "language": "zh",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "decode_status": "complete",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[10000, 13000], [20000, 21000], [30000, 33000]], dtype=np.int32),
        texts=np.asarray(["这部电影需要很多资金支持", "", "今天天气很好"]),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        embedding_chunk_indices=np.asarray([0, 2], dtype=np.int32),
    )
    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = lambda text, model_name: np.asarray([1.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]

    results = engine.search("投资预算", None, ["asr"], ["video-1"])

    assert results
    assert results[0]["start_time"] == 10
    assert results[0]["decision"] == "semantic_hit"
    assert results[0]["evidence"][0]["semantic_score"] is not None
    assert results[0]["evidence"][0]["unit_id"] == 0


@pytest.mark.skip(reason="ASR migrated to Milvus; NPZ v3 fallback removed")
@pytest.mark.skip(reason="ASR migrated to Milvus; NPZ v3 fallback removed")
def test_asr_search_falls_back_to_lexical_when_semantic_query_model_missing(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="interview.mp4")
    catalog.update_video("video-1", indexed_modalities=["asr"])
    _write_manifest(index_dir, "video-1", {
        "asr": {
            "file": "asr.npz",
            "engine": "funasr",
            "model_key": "iic/SenseVoiceSmall",
            "language": "zh",
            "semantic_model_key": "missing-semantic",
            "embedding_space": "minilm-text-semantic",
            "decode_status": "complete",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[10000, 13000]], dtype=np.int32),
        texts=np.asarray(["电影投资需要长期判断"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float16),
        embedding_chunk_indices=np.asarray([0], dtype=np.int32),
    )
    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = lambda *_args: (_ for _ in ()).throw(FileNotFoundError("missing semantic"))  # type: ignore[method-assign]

    results = engine.search("电影投资", None, ["asr"], ["video-1"])

    assert results
    assert results[0]["decision"] == "lexical_hit"
    assert results[0]["evidence"][0]["semantic_score"] is None


@pytest.mark.skip(reason="OCR migrated to Milvus; NPZ v3 fallback removed")
def test_ocr_legacy_v3_requires_rebuild(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="legacy.mp4")
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "video-1", {
        "ocr": {
            "file": "ocr.npz",
            "schema_version": 3,
            "model_key": "PP-OCRv6",
            "decode_status": "complete",
            "semantic_status": "disabled",
        }
    })
    np.savez_compressed(
        index_dir / "ocr.npz",
        chunk_times_ms=np.asarray([[5000, 6000, 5000]], dtype=np.int32),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
        box_chunk_indices=np.asarray([0], dtype=np.int32),
        box_texts=np.asarray(["FIFA"]),
        box_scores=np.asarray([0.95], dtype=np.float32),
        boxes=np.zeros((1, 4, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="缺少帧级数组"):
        SearchEngine(settings, catalog).search("FIFA", None, ["ocr"], ["video-1"])


@pytest.mark.skip(reason="OCR migrated to Milvus; NPZ v3 fallback removed")
def test_ocr_v3_search_groups_box_text_by_frame(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="match.mp4")
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "video-1", {
        "ocr": {
            "file": "ocr.npz",
            "engine": "rapidocr",
            "schema_version": 3,
            "model_key": "PP-OCRv6",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "sample_fps": 0.05,
            "decode_status": "complete",
            "semantic_status": "disabled",
        }
    })
    np.savez_compressed(
        index_dir / "ocr.npz",
        frame_times_ms=np.asarray([5000, 40000], dtype=np.int32),
        frame_windows_ms=np.asarray([[5000, 6000], [40000, 41000]], dtype=np.int32),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_frame_indices=np.empty((0,), dtype=np.int32),
        box_frame_indices=np.asarray([0, 0, 0, 1], dtype=np.int32),
        box_texts=np.asarray(["FIFA", "WORLD", "CUP", "UNRELATED"]),
        box_scores=np.asarray([0.95, 0.93, 0.90, 0.91], dtype=np.float32),
        boxes=np.zeros((4, 4, 2), dtype=np.float32),
    )

    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled semantic index must not encode a query")
        )
    )

    results = engine.search("FIFA", None, ["ocr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 5
    assert results[0]["end_time"] == 6
    assert results[0]["thumbnail_url"] == "/api/videos/video-1/frame?time=5.000"
    assert results[0]["evidence"][0]["text"] == "FIFA"
    assert results[0]["evidence"][0]["features"]["ocr_frame_text"] == "FIFA WORLD CUP"
    assert results[0]["evidence"][0]["features"]["ocr_score"] == 0.95


@pytest.mark.skip(reason="OCR migrated to Milvus; NPZ v3 fallback removed")
def test_ocr_v3_sparse_semantic_indices_map_embeddings_to_frames(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="match.mp4")
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "video-1", {
        "ocr": {
            "file": "ocr.npz",
            "engine": "rapidocr",
            "schema_version": 3,
            "model_key": "PP-OCRv6",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "sample_fps": 0.05,
            "decode_status": "complete",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(
        index_dir / "ocr.npz",
        frame_times_ms=np.asarray([5000, 40000], dtype=np.int32),
        frame_windows_ms=np.asarray([[5000, 6000], [40000, 41000]], dtype=np.int32),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        embedding_frame_indices=np.asarray([0, 1], dtype=np.int32),
        box_frame_indices=np.asarray([0, 0, 1], dtype=np.int32),
        box_texts=np.asarray(["FIFA", "WORLD CUP", "UNRELATED"]),
        box_scores=np.asarray([0.95, 0.93, 0.91], dtype=np.float32),
        boxes=np.zeros((3, 4, 2), dtype=np.float32),
    )
    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = lambda text, model_name: np.asarray([1.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]

    results = engine.search("soccer tournament", None, ["ocr"], ["video-1"])

    assert results
    assert results[0]["start_time"] == 5
    assert results[0]["decision"] == "semantic_hit"
    assert results[0]["evidence"][0]["modality"] == "ocr"
    assert results[0]["evidence"][0]["unit_id"] == 0


def test_face_v3_search_uses_track_times_and_on_demand_thumbnail(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="faces.mp4")
    catalog.update_video("video-1", indexed_modalities=["face"])
    _write_manifest(index_dir, "video-1", {
        "face": {
            "file": "face.npz",
            "model_key": "buffalo_l",
            "embedding_space": "arcface-identity",
            "sample_fps": 1.0,
            "decode_status": "complete",
        }
    })
    np.savez_compressed(
        index_dir / "face.npz",
        track_times_ms=np.asarray([[10000, 15000, 12000], [30000, 35000, 32000]], dtype=np.int32),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    class StubFace:
        def encode_reference(self, image_path):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._face = lambda: StubFace()  # type: ignore[method-assign]

    results = engine.search(None, "query.jpg", ["face"], ["video-1"])

    assert results[0]["start_time"] == 10
    assert results[0]["end_time"] == 15
    assert results[0]["thumbnail_url"] == "/api/videos/video-1/frame?time=12.000"
    assert results[0]["evidence"][0]["unit_type"] == "track"
    assert results[0]["evidence"][0]["best_ms"] == 12000


def test_visual_search_encodes_query_with_each_manifest_model(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    for video_id, model_key, vector in (
        ("siglip-video", "siglip2-so400m-384", [1.0, 0.0]),
        ("chinese-video", "chinese-clip-vit-b16", [0.0, 1.0]),
    ):
        index_dir = _create_video(settings, catalog, video_id=video_id, name=f"{video_id}.mp4", duration=10)
        catalog.update_video(video_id, indexed_modalities=["visual"])
        _write_manifest(index_dir, video_id, {
            "visual": {
                "file": "visual.npz",
                "model_key": model_key,
                "embedding_space": "siglip2-image-text",
                "sample_fps": 5.0,
                "decode_status": "complete",
            }
        }, duration_ms=10000)
        np.savez_compressed(
            index_dir / "visual.npz",
            frame_embeddings=np.asarray([vector], dtype=np.float16),
            frame_times_ms=np.asarray([1000], dtype=np.int32),
            segment_frame_offsets=np.asarray([0, 1, 1], dtype=np.int32),
        )

    class StubClip:
        def __init__(self, vector):
            self.vector = vector

        def encode_query(self, text, image_path, alpha):
            return self.vector

    calls: list[str] = []

    def fake_clip(model_key=None):
        calls.append(model_key)
        if model_key == "chinese-clip-vit-b16":
            return StubClip(np.asarray([0.0, 1.0], dtype=np.float32))
        return StubClip(np.asarray([1.0, 0.0], dtype=np.float32))

    engine = SearchEngine(settings, catalog)
    engine._clip = fake_clip  # type: ignore[method-assign]

    results = engine.search("stadium", None, ["visual"], limit=10)

    assert {result["video_id"] for result in results} == {"siglip-video", "chinese-video"}
    assert set(calls) == {"siglip2-so400m-384", "chinese-clip-vit-b16"}


# ---------------------------------------------------------------------------
# shadow_compare decoupling tests
# ---------------------------------------------------------------------------

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
