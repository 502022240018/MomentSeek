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
            "language": "zh",
            "decode_status": "complete",
            "semantic_status": "complete",
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
    assert payload["channels"]["asr"]["semantic_model_key"] == settings.asr_semantic_model
    assert payload["channels"]["asr"]["semantic_status"] == "complete"
