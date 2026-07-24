import json

import numpy as np
import pytest

from app.db import Catalog
from app.search import (
    Candidate,
    SearchEngine,
    SearchResult,
    _groups,
    _reserve_asr_lexical_results,
    _visual_candidates,
    lexical_score,
)
from app.settings import Settings


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


def test_visual_v3_frame_offsets_skip_empty_decode_bucket(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, duration=20)
    catalog.update_video("video-1", indexed_modalities=["visual"])
    _write_manifest(index_dir, "video-1", {
        "visual": {
            "file": "visual.npz",
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 5.0,
            "decode_status": "partial",
        }
    }, duration_ms=20000)
    np.savez_compressed(
        index_dir / "visual.npz",
        frame_embeddings=np.asarray([
            [0.80, 0.60],
            [0.20, 0.98],
            [1.00, 0.00],
        ], dtype=np.float16),
        frame_times_ms=np.asarray([1000, 6000, 16000], dtype=np.int32),
        segment_frame_offsets=np.asarray([0, 1, 2, 2, 3], dtype=np.int32),
    )

    class StubClip:
        def encode_query(self, text, image_path, alpha):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._clip = lambda model_key=None: StubClip()  # type: ignore[method-assign]

    results = engine.search("football", None, ["visual"], ["video-1"], limit=3)

    assert results[0]["start_time"] == 15
    assert results[0]["end_time"] == 20
    assert results[0]["thumbnail_url"] == "/api/videos/video-1/frame?time=16.000"
    evidence = results[0]["evidence"][0]
    assert evidence["unit_type"] == "segment"
    assert evidence["unit_id"] == 3
    assert evidence["best_ms"] == 16000
    assert evidence["features"]["visual_top1"] == 1.0


def test_visual_v3_optional_segment_times_override_fixed_bucket_times(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, duration=20)
    catalog.update_video("video-1", indexed_modalities=["visual"])
    _write_manifest(index_dir, "video-1", {
        "visual": {
            "file": "visual.npz",
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 5.0,
            "decode_status": "complete",
            "segment_strategy": "shot",
            "segment_times": "explicit",
        }
    }, duration_ms=20000)
    np.savez_compressed(
        index_dir / "visual.npz",
        frame_embeddings=np.asarray([
            [0.40, 0.91],
            [1.00, 0.00],
        ], dtype=np.float16),
        frame_times_ms=np.asarray([2100, 7800], dtype=np.int32),
        segment_frame_offsets=np.asarray([0, 1, 2], dtype=np.int32),
        segment_times_ms=np.asarray([[0, 2830], [2830, 9410]], dtype=np.int32),
    )

    class StubClip:
        def encode_query(self, text, image_path, alpha):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._clip = lambda model_key=None: StubClip()  # type: ignore[method-assign]

    results = engine.search("close-up", None, ["visual"], ["video-1"], limit=2)

    assert results[0]["start_time"] == 2.83
    assert results[0]["end_time"] == 9.41
    evidence = results[0]["evidence"][0]
    assert evidence["unit_id"] == 1
    assert evidence["features"]["segment_time_source"] == "explicit"
    assert evidence["features"]["segment_strategy"] == "shot"


def test_visual_v3_rejects_invalid_optional_segment_times(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, duration=20)
    catalog.update_video("video-1", indexed_modalities=["visual"])
    _write_manifest(index_dir, "video-1", {
        "visual": {
            "file": "visual.npz",
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 5.0,
            "decode_status": "complete",
        }
    }, duration_ms=20000)
    np.savez_compressed(
        index_dir / "visual.npz",
        frame_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        frame_times_ms=np.asarray([1000, 6000], dtype=np.int32),
        segment_frame_offsets=np.asarray([0, 1, 2], dtype=np.int32),
        segment_times_ms=np.asarray([[0, 4000]], dtype=np.int32),
    )

    class StubClip:
        def encode_query(self, text, image_path, alpha):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._clip = lambda model_key=None: StubClip()  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="segment_times_ms"):
        engine.search("close-up", None, ["visual"], ["video-1"], limit=2)


def test_visual_ranking_score_prefers_cross_video_raw_similarity_over_local_percentile():
    class VisualData:
        def __init__(self, scores):
            self._values = {
                "frame_embeddings": np.asarray(
                    [[score, np.sqrt(max(0.0, 1.0 - score * score))] for score in scores],
                    dtype=np.float32,
                ),
                "frame_times_ms": np.asarray([index * 5000 + 1000 for index in range(len(scores))], dtype=np.int32),
                "segment_frame_offsets": np.arange(len(scores) + 1, dtype=np.int32),
            }
            self.files = tuple(self._values)

        def __getitem__(self, key):
            return self._values[key]

    query = np.asarray([1.0, 0.0], dtype=np.float32)
    relevant = _visual_candidates(
        VisualData([0.55, 0.50, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39]),
        query,
        "relevant-video",
        duration_ms=40000,
        segment_ms=5000,
        profile="balanced",
        limit=8,
    )
    unrelated = _visual_candidates(
        VisualData([0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]),
        query,
        "unrelated-video",
        duration_ms=40000,
        segment_ms=5000,
        profile="balanced",
        limit=8,
    )

    relevant_mid = next(item for item in relevant if item.unit_id == 1)
    unrelated_best = next(item for item in unrelated if item.unit_id == 0)

    assert relevant_mid.raw_score > unrelated_best.raw_score
    assert relevant_mid.percentile < unrelated_best.percentile
    assert relevant_mid.score > unrelated_best.score


def test_visual_subquery_fusion_prefers_constraint_coverage():
    class VisualData:
        def __init__(self):
            self._values = {
                "frame_embeddings": np.asarray(
                    [[1.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5)]],
                    dtype=np.float32,
                ),
                "frame_times_ms": np.asarray([1000, 6000], dtype=np.int32),
                "segment_frame_offsets": np.asarray([0, 1, 2], dtype=np.int32),
            }
            self.files = tuple(self._values)

        def __getitem__(self, key):
            return self._values[key]

    candidates = _visual_candidates(
        VisualData(),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "video-1",
        duration_ms=10000,
        segment_ms=5000,
        profile="balanced",
        limit=2,
    )

    assert candidates[0].unit_id == 1
    assert candidates[0].features["visual_subquery_count"] == 2
    assert candidates[0].features["visual_subquery_scores"] == pytest.approx(
        [np.sqrt(0.5), np.sqrt(0.5)]
    )


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

    results = SearchEngine(settings, catalog).search("电影投资", None, ["asr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 10
    assert results[0]["end_time"] == 17
    assert results[0]["media_url"] == "/api/videos/video-1/media"
    assert results[0]["clip_url"] == "/api/videos/video-1/clip?start=10.000&end=17.000"
    assert results[0]["evidence"][0]["unit_type"] == "chunk"


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

    results = SearchEngine(settings, catalog).search("FIFA", None, ["ocr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 5
    assert results[0]["end_time"] == 6
    assert results[0]["thumbnail_url"] == "/api/videos/video-1/frame?time=5.000"
    assert results[0]["evidence"][0]["text"] == "FIFA"
    assert results[0]["evidence"][0]["features"]["ocr_frame_text"] == "FIFA WORLD CUP"
    assert results[0]["evidence"][0]["features"]["ocr_score"] == 0.95


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
        }
    }, duration_ms=20000)
    np.savez_compressed(
        index_dir / "visual.npz",
        frame_embeddings=np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float16),
        frame_times_ms=np.asarray([1000, 6000], dtype=np.int32),
        segment_frame_offsets=np.asarray([0, 1, 2], dtype=np.int32),
    )
    return video_id


def test_shadow_compare_fires_without_milvus_read_routing(tmp_path):
    """shadow_compare_log is called even when reads are served from NPZ
    (MILVUS_READ_ENABLED=false / MILVUS_ROLLOUT_PERCENT=0)."""
    from unittest.mock import patch, MagicMock

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog)

    class StubClip:
        def encode_query(self, text, image_path, alpha):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._clip = lambda model_key=None: StubClip()  # type: ignore[method-assign]

    with (
        patch("app.indexing.milvus_flags.milvus_read_enabled", return_value=False),
        patch("app.indexing.milvus_flags.should_use_milvus_for_video", return_value=False),
        patch("app.indexing.milvus_flags.milvus_shadow_compare_enabled", return_value=True),
        patch("app.indexing.milvus_flags.milvus_fallback_enabled", return_value=True),
        patch("app.indexing.milvus_client.ensure_milvus_reachable"),
        patch("app.indexing.milvus_client.get_milvus_client", return_value=MagicMock()),
        patch("app.indexing.milvus_search.milvus_visual_candidates", return_value=[]),
        patch("app.indexing.milvus_search.shadow_compare_log") as mock_shadow_log,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    # shadow_compare_log must be called with the correct video_id and modality
    mock_shadow_log.assert_called_once()
    args = mock_shadow_log.call_args[0]
    assert args[0] == video_id
    assert args[1] == "visual"

    # NPZ results are returned — reads are not affected by shadow mode
    assert len(results) > 0


def test_shadow_compare_milvus_error_silenced_in_shadow_only_mode(tmp_path):
    """A MilvusServiceError in shadow-only mode is swallowed; NPZ results are returned."""
    from unittest.mock import patch, MagicMock
    from app.indexing.milvus_search import MilvusServiceError

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog)

    class StubClip:
        def encode_query(self, text, image_path, alpha):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    engine = SearchEngine(settings, catalog)
    engine._clip = lambda model_key=None: StubClip()  # type: ignore[method-assign]

    with (
        patch("app.indexing.milvus_flags.milvus_read_enabled", return_value=False),
        patch("app.indexing.milvus_flags.should_use_milvus_for_video", return_value=False),
        patch("app.indexing.milvus_flags.milvus_shadow_compare_enabled", return_value=True),
        patch("app.indexing.milvus_flags.milvus_fallback_enabled", return_value=True),
        patch("app.indexing.milvus_client.ensure_milvus_reachable"),
        patch("app.indexing.milvus_client.get_milvus_client", return_value=MagicMock()),
        patch("app.indexing.milvus_search.milvus_visual_candidates",
              side_effect=MilvusServiceError("connection refused")),
    ):
        # Must not raise — shadow error is never surfaced to the caller
        results = engine.search("football", None, ["visual"], [video_id])

    # NPZ results are still served correctly
    assert len(results) > 0


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
        patch(
            "app.indexing.milvus_flags.should_use_milvus_for_video",
            return_value=True,
        ),
        patch(
            "app.indexing.milvus_flags.milvus_shadow_compare_enabled",
            return_value=False,
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            return_value=[milvus_hit],
        ) as milvus_search,
        patch.object(engine, "_candidates_for_video") as npz_search,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    milvus_search.assert_called_once()
    npz_search.assert_not_called()
    assert results[0]["start_time"] == 5.0
    assert results[0]["evidence"][0]["features"] == {}


def test_milvus_service_failure_falls_back_to_npz(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog, video_id="v-fallback")
    engine = SearchEngine(settings, catalog)
    npz_hit = Candidate(
        video_id=video_id,
        start_time=0.0,
        end_time=5.0,
        score=0.8,
        modality="visual",
    )

    with (
        patch(
            "app.indexing.milvus_flags.should_use_milvus_for_video",
            return_value=True,
        ),
        patch(
            "app.indexing.milvus_flags.milvus_shadow_compare_enabled",
            return_value=False,
        ),
        patch(
            "app.indexing.milvus_flags.milvus_fallback_enabled",
            return_value=True,
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            side_effect=ConnectionError("Milvus unavailable"),
        ),
        patch.object(
            engine,
            "_candidates_for_video",
            return_value=[npz_hit],
        ) as npz_search,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    npz_search.assert_called_once()
    assert results[0]["start_time"] == 0.0


def test_milvus_service_failure_is_not_retried_for_every_video(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    _make_visual_index(settings, catalog, video_id="v-fallback-1")
    _make_visual_index(settings, catalog, video_id="v-fallback-2")
    engine = SearchEngine(settings, catalog)

    with (
        patch(
            "app.indexing.milvus_flags.should_use_milvus_for_video",
            return_value=True,
        ),
        patch(
            "app.indexing.milvus_flags.milvus_shadow_compare_enabled",
            return_value=False,
        ),
        patch(
            "app.indexing.milvus_flags.milvus_fallback_enabled",
            return_value=True,
        ),
        patch.object(
            engine,
            "_milvus_candidates_for_video",
            side_effect=ConnectionError("Milvus unavailable"),
        ) as milvus_search,
        patch.object(
            engine,
            "_candidates_for_video",
            return_value=[],
        ) as npz_search,
    ):
        engine.search("football", None, ["visual"])

    assert milvus_search.call_count == 1
    assert npz_search.call_count == 2


def test_milvus_empty_channel_falls_back_to_npz(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog, video_id="v-coverage-gap")
    engine = SearchEngine(settings, catalog)
    npz_hit = Candidate(
        video_id=video_id,
        start_time=3.0,
        end_time=8.0,
        score=0.85,
        modality="visual",
    )

    with (
        patch("app.indexing.milvus_flags.should_use_milvus_for_video", return_value=True),
        patch("app.indexing.milvus_flags.milvus_shadow_compare_enabled", return_value=False),
        patch("app.indexing.milvus_flags.milvus_fallback_enabled", return_value=True),
        patch.object(engine, "_milvus_candidates_for_video", return_value=[]),
        patch.object(
            engine, "_candidates_for_video", return_value=[npz_hit]
        ) as npz_search,
    ):
        results = engine.search("football", None, ["visual"], [video_id])

    npz_search.assert_called_once()
    assert results[0]["start_time"] == 3.0


def test_milvus_partial_coverage_only_recovers_missing_channel(tmp_path):
    from unittest.mock import patch

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    video_id = _make_visual_index(settings, catalog, video_id="v-partial")
    video = catalog.get_video(video_id)
    catalog.update_video(
        video_id,
        indexed_modalities=[*video["indexed_modalities"], "asr"],
    )
    engine = SearchEngine(settings, catalog)
    milvus_hit = Candidate(video_id, 5.0, 10.0, 0.9, "visual")
    stale_npz_visual = Candidate(video_id, 0.0, 5.0, 0.99, "visual")
    npz_asr = Candidate(video_id, 6.0, 9.0, 0.7, "asr")

    with (
        patch("app.indexing.milvus_flags.should_use_milvus_for_video", return_value=True),
        patch("app.indexing.milvus_flags.milvus_shadow_compare_enabled", return_value=False),
        patch("app.indexing.milvus_flags.milvus_fallback_enabled", return_value=True),
        patch.object(
            engine, "_milvus_candidates_for_video", return_value=[milvus_hit]
        ),
        patch.object(
            engine,
            "_candidates_for_video",
            return_value=[stale_npz_visual, npz_asr],
        ),
    ):
        results = engine.search("football", None, ["visual", "asr"], [video_id])

    evidence_modalities = {
        evidence["modality"]
        for result in results
        for evidence in result["evidence"]
    }
    assert evidence_modalities == {"visual", "asr"}
    assert all(result["start_time"] != 0.0 for result in results)
