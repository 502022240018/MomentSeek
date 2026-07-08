from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.settings import Settings


def _frame(width: int = 20, height: int = 10) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _fake_thumbnail(_image, path: str | Path, **_kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"thumb")


def test_visual_index_writes_frame_offsets_and_no_per_segment_payload(tmp_path, monkeypatch):
    from app.indexing import visual

    frames = [(1.0, _frame()), (6.0, _frame()), (16.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(visual, "save_thumbnail", _fake_thumbnail)

    class FakeEncoder:
        device = "cpu"
        model_key = "siglip2-so400m-384"
        model_label = "SigLIP2"
        backend = "hf"
        model_id = "google/siglip2-so400m-patch14-384"

        def encode_frames(self, batch):
            vectors = {
                1: np.asarray([1.0, 0.0], dtype=np.float32),
                2: np.asarray([0.0, 1.0], dtype=np.float32),
            }
            return np.stack([vectors.get(len(batch), np.asarray([0.5, 0.5], dtype=np.float32)) for _ in batch])

    result = visual.build_visual_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "visual.npz"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        model_name="ViT-B-32",
        pretrained="openai",
        sample_fps=1.0,
        segment_seconds=5.0,
        batch_size=2,
        npu_enabled=False,
        npu_device_id=0,
        encoder=FakeEncoder(),
        duration_seconds=20.0,
    )

    with np.load(tmp_path / "visual.npz", allow_pickle=False) as data:
        assert set(data.files) == {"frame_embeddings", "frame_times_ms", "segment_frame_offsets"}
        assert data["frame_embeddings"].dtype == np.float16
        assert data["frame_times_ms"].tolist() == [1000, 6000, 16000]
        assert data["segment_frame_offsets"].tolist() == [0, 1, 2, 2, 3]
    assert result["schema_version"] == 3
    assert result["segments_total"] == 4
    assert result["segments_with_frames"] == 3
    assert result["empty_segments"] == 1
    assert result["decode_status"] == "partial"
    assert (tmp_path / "thumbs" / "visual_000003.jpg").exists()


def test_visual_index_can_write_optional_shot_segment_times(tmp_path, monkeypatch):
    from app.indexing import visual

    frames = [(1.0, _frame()), (3.0, _frame()), (8.0, _frame()), (12.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(visual, "save_thumbnail", _fake_thumbnail)
    monkeypatch.setattr(
        visual,
        "detect_shot_segments",
        lambda *_args, **_kwargs: [(0, 4000), (4000, 10000), (10000, 14000)],
        raising=False,
    )

    class FakeEncoder:
        device = "cpu"
        model_key = "siglip2-so400m-384"
        model_label = "SigLIP2"
        backend = "hf"
        model_id = "google/siglip2-so400m-patch14-384"

        def encode_frames(self, batch):
            return np.stack([np.asarray([1.0, 0.0], dtype=np.float32) for _ in batch])

    result = visual.build_visual_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "visual.npz"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        model_name="ViT-B-32",
        pretrained="openai",
        sample_fps=1.0,
        segment_seconds=5.0,
        batch_size=4,
        npu_enabled=False,
        npu_device_id=0,
        encoder=FakeEncoder(),
        duration_seconds=14.0,
        segment_strategy="shot",
        min_segment_seconds=0.8,
        max_segment_seconds=8.0,
    )

    with np.load(tmp_path / "visual.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "frame_embeddings",
            "frame_times_ms",
            "segment_frame_offsets",
            "segment_times_ms",
        }
        assert data["frame_times_ms"].tolist() == [1000, 3000, 8000, 12000]
        assert data["segment_frame_offsets"].tolist() == [0, 2, 3, 4]
        assert data["segment_times_ms"].tolist() == [[0, 4000], [4000, 10000], [10000, 14000]]
    assert result["segment_strategy"] == "shot"
    assert result["segments_total"] == 3
    assert result["segments_with_frames"] == 3
    assert result["empty_segments"] == 0
    assert result["decode_status"] == "complete"
    assert (tmp_path / "thumbs" / "visual_000002.jpg").exists()


def test_visual_index_can_use_pyscenedetect_shot_detector(tmp_path, monkeypatch):
    from app.indexing import visual

    frames = [(1.0, _frame()), (3.0, _frame()), (8.0, _frame()), (12.0, _frame())]
    calls: list[str] = []
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(visual, "save_thumbnail", _fake_thumbnail)
    monkeypatch.setattr(
        visual,
        "detect_shot_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("simple detector should not run")),
    )

    def fake_pyscenedetect_segments(*_args, detector: str, **_kwargs):
        calls.append(detector)
        return [(0, 4000), (4000, 10000), (10000, 14000)]

    monkeypatch.setattr(visual, "detect_pyscenedetect_segments", fake_pyscenedetect_segments, raising=False)

    class FakeEncoder:
        device = "cpu"
        model_key = "siglip2-so400m-384"
        model_label = "SigLIP2"
        backend = "hf"
        model_id = "google/siglip2-so400m-patch14-384"

        def encode_frames(self, batch):
            return np.stack([np.asarray([1.0, 0.0], dtype=np.float32) for _ in batch])

    result = visual.build_visual_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "visual.npz"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        model_name="ViT-B-32",
        pretrained="openai",
        sample_fps=1.0,
        segment_seconds=5.0,
        batch_size=4,
        npu_enabled=False,
        npu_device_id=0,
        encoder=FakeEncoder(),
        duration_seconds=14.0,
        segment_strategy="shot",
        min_segment_seconds=0.8,
        max_segment_seconds=8.0,
        shot_detector="pyscenedetect_adaptive",
    )

    with np.load(tmp_path / "visual.npz", allow_pickle=False) as data:
        assert data["segment_times_ms"].tolist() == [[0, 4000], [4000, 10000], [10000, 14000]]
    assert calls == ["pyscenedetect_adaptive"]
    assert result["segment_strategy"] == "shot"
    assert result["shot_detector"] == "pyscenedetect_adaptive"


def test_asr_index_writes_chunks_and_sparse_semantic_arrays(tmp_path, monkeypatch):
    from app.indexing import asr

    sidecar = tmp_path / "asr.json"
    sidecar.write_text(
        json.dumps([
            {"start_time": 1.0, "end_time": 2.5, "text": "hello world"},
            {"start_time": 5.0, "end_time": 7.0, "text": "green field"},
        ]),
        encoding="utf-8",
    )

    def fake_semantic_arrays(**_kwargs):
        return {
            "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([1], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = asr.build_asr_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="whisper",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        sidecar_path=str(sidecar),
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "asr.npz", allow_pickle=False) as data:
        assert set(data.files) == {"chunk_times_ms", "texts", "embeddings", "embedding_chunk_indices"}
        assert data["chunk_times_ms"].tolist() == [[1000, 2500], [5000, 7000]]
        assert data["texts"].tolist() == ["hello world", "green field"]
        assert data["embeddings"].dtype == np.float16
        assert data["embedding_chunk_indices"].tolist() == [1]
    assert result["chunks"] == 2
    assert result["semantic_chunks"] == 1
    assert result["decode_status"] == "complete"


def test_ocr_index_writes_box_level_arrays_and_chunk_semantics(tmp_path, monkeypatch):
    from app.indexing import ocr

    class Output:
        txts = ["FIFA", "WORLD CUP"]
        scores = [0.95, 0.9]
        boxes = np.asarray([
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[10, 5], [20, 5], [20, 10], [10, 10]],
        ], dtype=np.float32)

    class FakeOcr:
        def __call__(self, _frame):
            return Output()

    def fake_semantic_arrays(**_kwargs):
        return {
            "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(ocr, "_load_ocr", lambda *_args, **_kwargs: (FakeOcr(), {"rec": ["CPUExecutionProvider"]}))
    monkeypatch.setattr(ocr, "read_frames", lambda *_args, **_kwargs: iter([(5.0, _frame())]))
    monkeypatch.setattr(ocr, "save_thumbnail", _fake_thumbnail)
    monkeypatch.setattr(ocr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = ocr.build_ocr_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "ocr.npz"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        working_dir=str(tmp_path / "work"),
        sample_fps=1.0,
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "ocr.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "chunk_times_ms",
            "embeddings",
            "embedding_chunk_indices",
            "box_chunk_indices",
            "box_texts",
            "box_scores",
            "boxes",
        }
        assert data["chunk_times_ms"].tolist() == [[5000, 6000, 5000]]
        assert data["box_chunk_indices"].tolist() == [0, 0]
        assert data["box_texts"].tolist() == ["FIFA", "WORLD CUP"]
        assert np.allclose(data["boxes"][1], [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]])
        assert data["embedding_chunk_indices"].tolist() == [0]
    assert result["chunks"] == 1
    assert result["semantic_chunks"] == 1
    assert (tmp_path / "thumbs" / "ocr_000000.jpg").exists()


def test_face_index_writes_track_times_and_row_thumbnails(tmp_path, monkeypatch):
    from app.indexing import faces

    class Face:
        def __init__(self, score, bbox):
            self.det_score = score
            self.bbox = np.asarray(bbox, dtype=np.float32)
            self.normed_embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    class FakeEncoder:
        provider = "cpu"

        def detect(self, _frame):
            return [Face(0.9, [2, 1, 8, 7])]

    monkeypatch.setattr(faces, "read_frames", lambda *_args, **_kwargs: iter([(0.0, _frame()), (1.0, _frame())]))
    monkeypatch.setattr(faces, "save_thumbnail", _fake_thumbnail)

    result = faces.build_face_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "face.npz"),
        thumbnail_dir=str(tmp_path / "thumbs"),
        model_name="buffalo_l",
        sample_fps=1.0,
        provider="cpu",
        device_id=0,
        encoder=FakeEncoder(),
    )

    with np.load(tmp_path / "face.npz", allow_pickle=False) as data:
        assert set(data.files) == {"embeddings", "track_times_ms"}
        assert data["embeddings"].shape == (1, 2)
        assert data["track_times_ms"].tolist() == [[0, 2000, 0]]
    assert result["schema_version"] == 3
    assert result["tracks"] == 1
    assert result["decode_status"] == "complete"
    assert (tmp_path / "thumbs" / "face_000000.jpg").exists()


def test_write_stage_manifest_preserves_channels_and_records_small_metadata(tmp_path):
    from app.indexing.pipeline_manifest import write_stage_manifest

    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    video = {"id": "video-1", "name": "video.mp4", "duration": 12.0}
    index_dir = tmp_path / "runtime" / "indexes" / "video-1"

    write_stage_manifest(
        "visual",
        index_dir=index_dir,
        video=video,
        options={"visual_sample_fps": 5.0, "visual_segment_seconds": 5.0},
        settings=settings,
        result={"visual_model": "siglip2-so400m-384", "decode_status": "partial"},
    )
    write_stage_manifest(
        "asr",
        index_dir=index_dir,
        video=video,
        options={},
        settings=settings,
        result={
            "engine": "whisper",
            "model": "small",
            "task": "transcribe",
            "requested_language": "auto",
            "detected_language": "zh",
            "decode_status": "complete",
            "semantic_status": "complete",
            "postprocess_stats": {"raw_chunks": 2, "processed_chunks": 2},
            "text_profile": {"chunks": 2, "cjk_chars": 8},
        },
    )

    payload = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["video_id"] == "video-1"
    assert payload["duration_ms"] == 12000
    assert payload["segment_ms"] == 5000
    assert payload["channels"]["visual"] == {
        "file": "visual.npz",
        "model_key": "siglip2-so400m-384",
        "embedding_space": "siglip2-image-text",
        "sample_fps": 5.0,
        "decode_status": "partial",
    }
    assert payload["channels"]["asr"]["file"] == "asr.npz"
    assert payload["channels"]["asr"]["task"] == "transcribe"
    assert payload["channels"]["asr"]["requested_language"] == "auto"
    assert payload["channels"]["asr"]["detected_language"] == "zh"
    assert payload["channels"]["asr"]["language"] == "zh"
    assert payload["channels"]["asr"]["semantic_model_key"] == settings.asr_semantic_model
    assert payload["channels"]["asr"]["semantic_status"] == "complete"
    assert payload["channels"]["asr"]["postprocess_stats"]["processed_chunks"] == 2
    assert payload["channels"]["asr"]["text_profile"]["cjk_chars"] == 8


def test_write_stage_manifest_records_optional_visual_shot_metadata(tmp_path):
    from app.indexing.pipeline_manifest import write_stage_manifest

    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    video = {"id": "video-1", "name": "video.mp4", "duration": 12.0}
    index_dir = tmp_path / "runtime" / "indexes" / "video-1"

    write_stage_manifest(
        "visual",
        index_dir=index_dir,
        video=video,
        options={
            "visual_sample_fps": 5.0,
            "visual_segment_seconds": 5.0,
            "visual_segment_strategy": "shot",
            "visual_min_segment_seconds": 0.8,
            "visual_max_segment_seconds": 8.0,
            "visual_shot_detector": "pyscenedetect_adaptive",
            "visual_shot_threshold": 0.18,
        },
        settings=settings,
        result={
            "visual_model": "siglip2-so400m-384",
            "decode_status": "complete",
            "segment_strategy": "shot",
            "segment_times": "explicit",
            "shot_detector": "pyscenedetect_adaptive",
        },
    )

    payload = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert payload["segment_ms"] == 5000
    assert payload["channels"]["visual"]["segment_strategy"] == "shot"
    assert payload["channels"]["visual"]["segment_times"] == "explicit"
    assert payload["channels"]["visual"]["min_segment_ms"] == 800
    assert payload["channels"]["visual"]["max_segment_ms"] == 8000
    assert payload["channels"]["visual"]["shot_detector"] == "pyscenedetect_adaptive"
    assert payload["channels"]["visual"]["shot_threshold"] == 0.18


def test_index_request_accepts_visual_shot_segment_options():
    from pydantic import ValidationError

    from app.schemas import IndexRequest

    request = IndexRequest(
        visual_segment_strategy="shot",
        visual_min_segment_seconds=0.8,
        visual_max_segment_seconds=8.0,
        visual_shot_detector="pyscenedetect_adaptive",
        visual_shot_threshold=0.18,
    )

    assert request.visual_segment_strategy == "shot"
    assert request.visual_min_segment_seconds == 0.8
    assert request.visual_max_segment_seconds == 8.0
    assert request.visual_shot_detector == "pyscenedetect_adaptive"
    assert request.visual_shot_threshold == 0.18

    try:
        IndexRequest(visual_segment_strategy="scene")
    except ValidationError as exc:
        assert "visual_segment_strategy" in str(exc)
    else:
        raise AssertionError("invalid visual_segment_strategy should fail validation")

    try:
        IndexRequest(visual_shot_detector="histogram")
    except ValidationError as exc:
        assert "visual_shot_detector" in str(exc)
    else:
        raise AssertionError("invalid visual_shot_detector should fail validation")

    try:
        IndexRequest(visual_shot_threshold=1.2)
    except ValidationError as exc:
        assert "visual_shot_threshold" in str(exc)
    else:
        raise AssertionError("invalid visual_shot_threshold should fail validation")
