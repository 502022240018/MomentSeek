import json

import numpy as np
import pytest

from app.db import Catalog
from app.search import Candidate, SearchEngine, _groups
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
    assert results[0]["thumbnail_url"] == "/api/thumbnails/video-1/visual_000003.jpg"
    evidence = results[0]["evidence"][0]
    assert evidence["unit_type"] == "segment"
    assert evidence["unit_id"] == 3
    assert evidence["best_ms"] == 16000
    assert evidence["features"]["visual_top1"] == 1.0


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


def test_ocr_v3_search_groups_box_text_by_chunk(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="match.mp4")
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "video-1", {
        "ocr": {
            "file": "ocr.npz",
            "engine": "rapidocr",
            "model_key": "PP-OCRv4",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "sample_fps": 0.05,
            "decode_status": "complete",
            "semantic_status": "disabled",
        }
    })
    np.savez_compressed(
        index_dir / "ocr.npz",
        chunk_times_ms=np.asarray([[5000, 6000, 5000], [40000, 41000, 40000]], dtype=np.int32),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
        box_chunk_indices=np.asarray([0, 0, 0, 1], dtype=np.int32),
        box_texts=np.asarray(["FIFA", "WORLD", "CUP", "UNRELATED"]),
        box_scores=np.asarray([0.95, 0.93, 0.90, 0.91], dtype=np.float32),
        boxes=np.zeros((4, 4, 2), dtype=np.float32),
    )

    results = SearchEngine(settings, catalog).search("FIFA", None, ["ocr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 5
    assert results[0]["end_time"] == 6
    assert results[0]["thumbnail_url"] == "/api/thumbnails/video-1/ocr_000000.jpg"
    assert results[0]["evidence"][0]["text"] == "FIFA WORLD CUP"
    assert results[0]["evidence"][0]["features"]["ocr_score"] == 0.95


def test_ocr_v3_sparse_semantic_indices_map_embeddings_to_chunks(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    index_dir = _create_video(settings, catalog, name="match.mp4")
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    _write_manifest(index_dir, "video-1", {
        "ocr": {
            "file": "ocr.npz",
            "engine": "rapidocr",
            "model_key": "PP-OCRv4",
            "semantic_model_key": "fake-semantic",
            "embedding_space": "minilm-text-semantic",
            "sample_fps": 0.05,
            "decode_status": "complete",
            "semantic_status": "complete",
        }
    })
    np.savez_compressed(
        index_dir / "ocr.npz",
        chunk_times_ms=np.asarray([[5000, 6000, 5000], [40000, 41000, 40000]], dtype=np.int32),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        embedding_chunk_indices=np.asarray([0, 1], dtype=np.int32),
        box_chunk_indices=np.asarray([0, 0, 1], dtype=np.int32),
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


def test_face_v3_search_uses_track_times_and_row_thumbnail(tmp_path):
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
    assert results[0]["thumbnail_url"] == "/api/thumbnails/video-1/face_000000.jpg"
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
    assert len(calls) == 2
