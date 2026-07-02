import json

import numpy as np

from app.db import Catalog
from app.search import Candidate, SearchEngine, _groups, _visual_candidates, _visual_model_key_from_index
from app.settings import Settings


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


def test_asr_search_returns_merged_playable_moment(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": "video-1", "name": "interview.mp4", "file_path": str(video_path),
        "duration": 60, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["asr"])
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "asr.json").write_text(json.dumps({
        "chunks": [
            {"start_time": 10, "end_time": 13, "text": "我们正在讨论电影投资"},
            {"start_time": 14, "end_time": 17, "text": "电影投资需要长期判断"},
            {"start_time": 40, "end_time": 42, "text": "今天天气很好"},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    results = SearchEngine(settings, catalog).search("电影投资", None, ["asr"], ["video-1"])
    assert len(results) == 1
    assert results[0]["start_time"] == 10
    assert results[0]["end_time"] == 17
    assert results[0]["media_url"] == "/api/videos/video-1/media"
    assert results[0]["clip_url"] == "/api/videos/video-1/clip?start=10.000&end=17.000"


def test_asr_semantic_search_recalls_non_lexical_match(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": "video-1", "name": "interview.mp4", "file_path": str(video_path),
        "duration": 60, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["asr"])
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "asr.json").write_text(json.dumps({
        "chunks": [
            {"start_time": 10, "end_time": 13, "text": "这部电影需要很多资金支持"},
            {"start_time": 30, "end_time": 33, "text": "今天天气很好"},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        index_dir / "asr_semantic.npz",
        schema_version=np.asarray([1], dtype=np.int16),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunk_indices=np.asarray([0, 1], dtype=np.int32),
        model=np.asarray(["fake-semantic"]),
    )

    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = lambda text, model_name: np.asarray([1.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]

    results = engine.search("投资预算", None, ["asr"], ["video-1"])

    assert results
    assert results[0]["start_time"] == 10
    assert results[0]["decision"] == "semantic_hit"
    assert results[0]["evidence"][0]["semantic_score"] is not None
    assert results[0]["evidence"][0]["lexical_score"] < 0.25


def test_ocr_search_returns_text_overlay_moment_with_thumbnail(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": "video-1", "name": "match.mp4", "file_path": str(video_path),
        "duration": 60, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "ocr.json").write_text(json.dumps({
        "chunks": [
            {"start_time": 5, "end_time": 6, "text": "FIFA WORLD CUP QATAR", "thumbnail": "ocr_000005.jpg"},
            {"start_time": 40, "end_time": 41, "text": "UNRELATED SCOREBOARD", "thumbnail": "ocr_000040.jpg"},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    results = SearchEngine(settings, catalog).search("FIFA", None, ["ocr"], ["video-1"])

    assert len(results) == 1
    assert results[0]["start_time"] == 5
    assert results[0]["end_time"] == 6
    assert results[0]["modalities"] == ["ocr"]
    assert results[0]["thumbnail_url"] == "/api/thumbnails/video-1/ocr_000005.jpg"
    assert results[0]["clip_url"] == "/api/videos/video-1/clip?start=5.000&end=6.000"
    assert results[0]["evidence"][0]["modality"] == "ocr"


def test_ocr_semantic_search_recalls_non_lexical_text(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": "video-1", "name": "match.mp4", "file_path": str(video_path),
        "duration": 60, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["ocr"])
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "ocr.json").write_text(json.dumps({
        "chunks": [
            {"start_time": 5, "end_time": 6, "text": "FIFA WORLD CUP QATAR", "thumbnail": "ocr_000005.jpg"},
            {"start_time": 40, "end_time": 41, "text": "UNRELATED SCOREBOARD", "thumbnail": "ocr_000040.jpg"},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        index_dir / "ocr_semantic.npz",
        schema_version=np.asarray([1], dtype=np.int16),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunk_indices=np.asarray([0, 1], dtype=np.int32),
        model=np.asarray(["fake-semantic"]),
    )

    engine = SearchEngine(settings, catalog)
    engine._encode_asr_query = lambda text, model_name: np.asarray([1.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]

    results = engine.search("soccer tournament", None, ["ocr"], ["video-1"])

    assert results
    assert results[0]["start_time"] == 5
    assert results[0]["decision"] == "semantic_hit"
    assert results[0]["evidence"][0]["semantic_score"] is not None
    assert results[0]["evidence"][0]["modality"] == "ocr"


def test_visual_v2_frame_maxsim_can_beat_segment_mean(tmp_path):
    index_path = tmp_path / "visual.npz"
    np.savez_compressed(
        index_path,
        schema_version=np.asarray([2], dtype=np.int16),
        segment_ids=np.asarray([10, 20], dtype=np.int32),
        embeddings=np.asarray([
            [0.80, 0.60],
            [0.20, 0.98],
        ], dtype=np.float32),
        start_times=np.asarray([0, 5], dtype=np.float32),
        end_times=np.asarray([5, 10], dtype=np.float32),
        thumbnails=np.asarray(["visual_000010.jpg", "visual_000020.jpg"]),
        frame_embeddings=np.asarray([
            [0.80, 0.60],
            [0.79, 0.61],
            [0.78, 0.62],
            [1.00, 0.00],
            [0.95, 0.31],
            [0.90, 0.44],
        ], dtype=np.float32),
        frame_times=np.asarray([0.0, 1.0, 2.0, 5.5, 6.0, 6.5], dtype=np.float32),
        frame_segment_ids=np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int32),
        model=np.asarray(["test"]),
    )

    with np.load(index_path, allow_pickle=False) as data:
        candidates = _visual_candidates(data, np.asarray([1.0, 0.0], dtype=np.float32), "video-1", limit=2)

    assert candidates[0].start_time == 5
    assert candidates[0].best_time == 5.5
    assert candidates[0].raw_score == 1.0
    assert candidates[0].visual_top1 == 1.0
    assert candidates[0].visual_mean < candidates[1].visual_mean
    assert "best_frame=5.50s" in (candidates[0].evidence or "")


def test_visual_model_key_prefers_index_metadata(tmp_path):
    index_path = tmp_path / "visual.npz"
    np.savez_compressed(
        index_path,
        schema_version=np.asarray([2], dtype=np.int16),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        start_times=np.asarray([0], dtype=np.float32),
        end_times=np.asarray([5], dtype=np.float32),
        thumbnails=np.asarray(["visual_000000.jpg"]),
        visual_model=np.asarray(["chinese-clip-vit-b16"]),
        model=np.asarray(["ChineseCLIP ViT-B/16"]),
    )

    with np.load(index_path, allow_pickle=False) as data:
        assert _visual_model_key_from_index(data, "siglip2-so400m-384") == "chinese-clip-vit-b16"


def test_visual_search_encodes_query_with_each_index_model(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    for video_id in ("siglip-video", "chinese-video"):
        video_path = settings.upload_dir / f"{video_id}.mp4"
        video_path.write_bytes(b"not-needed-for-search")
        catalog.create_video({
            "id": video_id, "name": f"{video_id}.mp4", "file_path": str(video_path),
            "duration": 10, "fps": 25, "width": 1280, "height": 720, "status": "ready",
        })
        catalog.update_video(video_id, indexed_modalities=["visual"])
        (settings.index_dir / video_id).mkdir(parents=True)

    np.savez_compressed(
        settings.index_dir / "siglip-video" / "visual.npz",
        schema_version=np.asarray([2], dtype=np.int16),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        start_times=np.asarray([0], dtype=np.float32),
        end_times=np.asarray([5], dtype=np.float32),
        thumbnails=np.asarray(["visual_000000.jpg"]),
        visual_model=np.asarray(["siglip2-so400m-384"]),
    )
    np.savez_compressed(
        settings.index_dir / "chinese-video" / "visual.npz",
        schema_version=np.asarray([2], dtype=np.int16),
        embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        start_times=np.asarray([0], dtype=np.float32),
        end_times=np.asarray([5], dtype=np.float32),
        thumbnails=np.asarray(["visual_000000.jpg"]),
        visual_model=np.asarray(["chinese-clip-vit-b16"]),
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
    assert calls == ["siglip2-so400m-384", "chinese-clip-vit-b16"]
