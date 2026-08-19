from __future__ import annotations

import json

import numpy as np
import pytest

from app.core.settings import Settings


def _frame(width: int = 20, height: int = 10) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _capture_visual_milvus_write(monkeypatch):
    from app.vector_store.milvus import milvus_indexer

    captured: dict = {}

    def fake_write(ctx, modality, arrays, **_kwargs):
        captured["ctx"] = ctx
        captured["modality"] = modality
        captured["arrays"] = arrays
        return len(arrays["embeddings"])

    monkeypatch.setattr(milvus_indexer, "write_modality_from_memory", fake_write)
    return captured, object()


def _capture_milvus_write(monkeypatch):
    from app.vector_store.milvus import milvus_indexer

    captured: dict = {}

    def fake_write(ctx, modality, arrays, **_kwargs):
        captured["ctx"] = ctx
        captured["modality"] = modality
        captured["arrays"] = arrays
        row_key = {
            "asr": "chunk_times_ms",
            "ocr": "frame_times_ms",
            "face": "embeddings",
        }[modality]
        return len(arrays[row_key])

    monkeypatch.setattr(milvus_indexer, "write_modality_from_memory", fake_write)
    return captured, object()


def test_non_visual_modality_builders_require_milvus_context(tmp_path):
    from app.indexing.modalities.asr import asr
    from app.indexing.modalities.face import faces
    from app.indexing.modalities.ocr import ocr

    with pytest.raises(ValueError, match="MilvusWriteContext"):
        asr.build_asr_index(
            video_path="video.mp4",
            working_dir=str(tmp_path / "asr-work"),
            engine="sidecar",
            model_name="small",
            device="cpu",
            model_dir=str(tmp_path / "models"),
            milvus_ctx=None,
        )
    with pytest.raises(ValueError, match="MilvusWriteContext"):
        faces.build_face_index(
            video_path="video.mp4",
            model_name="buffalo_l",
            sample_fps=1.0,
            provider="cpu",
            device_id=0,
            milvus_ctx=None,
        )
    with pytest.raises(ValueError, match="MilvusWriteContext"):
        ocr.build_ocr_index(
            video_path="video.mp4",
            working_dir=str(tmp_path / "ocr-work"),
            milvus_ctx=None,
            duration_seconds=10.0,
        )


def test_visual_index_writes_explicit_fixed_bounds_to_milvus_without_npz(tmp_path, monkeypatch):
    from app.indexing.modalities.visual import visual

    frames = [(1.0, _frame()), (6.0, _frame()), (16.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    captured, milvus_ctx = _capture_visual_milvus_write(monkeypatch)

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
        model_name="ViT-B-32",
        pretrained="openai",
        sample_fps=1.0,
        segment_seconds=5.0,
        batch_size=2,
        npu_enabled=False,
        npu_device_id=0,
        encoder=FakeEncoder(),
        duration_seconds=18.5,
        milvus_ctx=milvus_ctx,
    )

    arrays = captured["arrays"]
    assert captured["modality"] == "visual"
    assert arrays["embeddings"].dtype == np.float16
    assert arrays["frame_times_ms"].tolist() == [1000, 6000, 16000]
    assert arrays["segment_frame_offsets"].tolist() == [0, 1, 2, 2, 3]
    assert arrays["segment_times_ms"].tolist() == [
        [0, 5000],
        [5000, 10000],
        [10000, 15000],
        [15000, 18500],
    ]
    assert arrays["duration_ms"] == 18500
    assert not (tmp_path / "visual.npz").exists()
    assert result["schema_version"] == 3
    assert result["segments_total"] == 4
    assert result["segments_with_frames"] == 3
    assert result["empty_segments"] == 1
    assert result["decode_status"] == "partial"
    assert result["segment_times"] == "explicit"
    assert result["milvus_rows"] == 3


def test_visual_index_writes_shot_segment_times_to_milvus_without_npz(tmp_path, monkeypatch):
    from app.indexing.modalities.visual import visual

    frames = [(1.0, _frame()), (3.0, _frame()), (8.0, _frame()), (12.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(
        visual,
        "detect_shot_segments",
        lambda *_args, **_kwargs: [(0, 4000), (4000, 10000), (10000, 14000)],
        raising=False,
    )
    captured, milvus_ctx = _capture_visual_milvus_write(monkeypatch)

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
        milvus_ctx=milvus_ctx,
    )

    arrays = captured["arrays"]
    assert arrays["frame_times_ms"].tolist() == [1000, 3000, 8000, 12000]
    assert arrays["segment_frame_offsets"].tolist() == [0, 2, 3, 4]
    assert arrays["segment_times_ms"].tolist() == [[0, 4000], [4000, 10000], [10000, 14000]]
    assert arrays["duration_ms"] == 14000
    assert not (tmp_path / "visual.npz").exists()
    assert result["segment_strategy"] == "shot"
    assert result["segments_total"] == 3
    assert result["segments_with_frames"] == 3
    assert result["empty_segments"] == 0
    assert result["decode_status"] == "complete"


def test_visual_index_can_use_pyscenedetect_shot_detector(tmp_path, monkeypatch):
    from app.indexing.modalities.visual import visual

    frames = [(1.0, _frame()), (3.0, _frame()), (8.0, _frame()), (12.0, _frame())]
    calls: list[str] = []
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(
        visual,
        "detect_shot_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("simple detector should not run")),
    )

    def fake_pyscenedetect_segments(*_args, detector: str, **_kwargs):
        calls.append(detector)
        return [(0, 4000), (4000, 10000), (10000, 14000)]

    monkeypatch.setattr(visual, "detect_pyscenedetect_segments", fake_pyscenedetect_segments, raising=False)
    captured, milvus_ctx = _capture_visual_milvus_write(monkeypatch)

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
        milvus_ctx=milvus_ctx,
    )

    assert captured["arrays"]["segment_times_ms"].tolist() == [
        [0, 4000],
        [4000, 10000],
        [10000, 14000],
    ]
    assert not (tmp_path / "visual.npz").exists()
    assert calls == ["pyscenedetect_adaptive"]
    assert result["segment_strategy"] == "shot"
    assert result["shot_detector"] == "pyscenedetect_adaptive"


def test_asr_index_writes_chunks_and_sparse_semantic_arrays(tmp_path, monkeypatch):
    from app.indexing.modalities.asr import asr

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
    captured, milvus_ctx = _capture_milvus_write(monkeypatch)

    result = asr.build_asr_index(
        video_path="video.mp4",
        working_dir=str(tmp_path / "work"),
        engine="whisper",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        milvus_ctx=milvus_ctx,
        sidecar_path=str(sidecar),
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    arrays = captured["arrays"]
    assert captured["modality"] == "asr"
    assert arrays["chunk_times_ms"].tolist() == [[1000, 2500], [5000, 7000]]
    assert arrays["texts"] == ["hello world", "green field"]
    assert arrays["embeddings"].dtype == np.float16
    assert arrays["embedding_chunk_indices"].tolist() == [1]
    assert not (tmp_path / "asr.npz").exists()
    assert result["chunks"] == 2
    assert result["semantic_chunks"] == 1
    assert result["decode_status"] == "complete"
    assert result["milvus_rows"] == 2


def test_ocr_index_writes_box_level_arrays_and_chunk_semantics(tmp_path, monkeypatch):
    from app.indexing.modalities.ocr import ocr

    class Output:
        txts = ["FIFA", "WORLD CUP"]
        scores = [0.95, 0.9]
        boxes = np.asarray([
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[10, 5], [20, 5], [20, 10], [10, 10]],
        ], dtype=np.float32)

    class FakeOcr:
        engine = "fake"
        device = "cpu"
        providers = {"rec": ["CPUExecutionProvider"]}

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
    monkeypatch.setattr(ocr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)
    captured, milvus_ctx = _capture_milvus_write(monkeypatch)

    result = ocr.build_ocr_index(
        video_path="video.mp4",
        working_dir=str(tmp_path / "work"),
        milvus_ctx=milvus_ctx,
        duration_seconds=10.0,
        sample_fps=1.0,
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    arrays = captured["arrays"]
    assert captured["modality"] == "ocr"
    assert arrays["frame_times_ms"].tolist() == [5000]
    assert arrays["frame_windows_ms"].tolist() == [[5000, 6000]]
    assert arrays["box_frame_indices"].tolist() == [0, 0]
    assert arrays["box_texts"] == ["FIFA", "WORLD CUP"]
    assert arrays["embedding_frame_indices"].tolist() == [0]
    assert not (tmp_path / "ocr.npz").exists()
    assert result["schema_version"] == 3
    assert result["chunks"] == 1
    assert result["semantic_chunks"] == 1
    assert result["milvus_rows"] == 1
    assert result["ocr_rec_resized_inputs"] == 0
    assert result["ocr_rec_max_input_width"] == 0
    assert result["backend_init_elapsed_seconds"] >= 0
    assert result["frame_loop_elapsed_seconds"] >= result["ocr_elapsed_seconds"]
    assert result["decode_postprocess_elapsed_seconds"] >= 0
    assert result["semantic_elapsed_seconds"] >= 0
    assert result["milvus_write_elapsed_seconds"] >= 0
    assert result["total_elapsed_seconds"] >= result["frame_loop_elapsed_seconds"]


def test_ocr_index_clamps_last_window_to_video_duration(tmp_path, monkeypatch):
    from app.indexing.modalities.ocr import ocr

    class Output:
        txts = ["END"]
        scores = [0.99]
        boxes = np.asarray(
            [[[0, 0], [30, 0], [30, 12], [0, 12]]],
            dtype=np.float32,
        )

    class FakeOcr:
        engine = "fake"
        device = "cpu"
        providers = {"rec": ["CPUExecutionProvider"]}

        def __call__(self, _frame):
            return Output()

    monkeypatch.setattr(ocr, "read_frames", lambda *_args, **_kwargs: iter([(16.0, _frame())]))
    monkeypatch.setattr(
        ocr,
        "build_text_semantic_arrays",
        lambda **_kwargs: {
            "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        },
        raising=False,
    )
    captured, milvus_ctx = _capture_milvus_write(monkeypatch)

    ocr.build_ocr_index(
        video_path="video.mp4",
        working_dir=str(tmp_path / "work"),
        milvus_ctx=milvus_ctx,
        duration_seconds=17.0,
        sample_fps=0.5,
        semantic_enabled=True,
        semantic_model="fake-semantic",
        backend=FakeOcr(),
    )

    assert captured["arrays"]["frame_times_ms"].tolist() == [16000]
    assert captured["arrays"]["frame_windows_ms"].tolist() == [[16000, 17000]]


def test_face_index_writes_track_times_without_precomputed_thumbnails(tmp_path, monkeypatch):
    from app.indexing.modalities.face import faces

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
    captured, milvus_ctx = _capture_milvus_write(monkeypatch)

    result = faces.build_face_index(
        video_path="video.mp4",
        model_name="buffalo_l",
        sample_fps=1.0,
        provider="cpu",
        device_id=0,
        milvus_ctx=milvus_ctx,
        encoder=FakeEncoder(),
    )

    arrays = captured["arrays"]
    assert captured["modality"] == "face"
    assert arrays["embeddings"].shape == (1, 2)
    assert arrays["track_times_ms"].tolist() == [[0, 2000, 0]]
    assert not (tmp_path / "face.npz").exists()
    assert result["schema_version"] == 3
    assert result["tracks"] == 1
    assert result["decode_status"] == "complete"
    assert result["milvus_rows"] == 1


def test_channel_metadata_records_compact_online_publication_metadata(tmp_path):
    from app.indexing.publication import channel_metadata

    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    visual = channel_metadata(
        "visual",
        result={"visual_model": "siglip2-so400m-384", "decode_status": "partial"},
        options={"visual_sample_fps": 5.0, "visual_segment_seconds": 5.0},
        settings=settings,
    )
    asr = channel_metadata(
        "asr",
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
            "chunk_builder_stats": {"raw_items": 2, "retrieval_chunks": 2},
            "text_profile": {"chunks": 2, "cjk_chars": 8},
            "tag_source": "sensevoice",
        },
    )
    ocr = channel_metadata(
        "ocr",
        options={"ocr_sample_fps": 1.0},
        settings=settings,
        result={"ocr_version": "PP-OCRv6", "schema_version": 3, "decode_status": "complete"},
    )

    assert visual == {
        "model_key": "siglip2-so400m-384",
        "embedding_space": "siglip2-image-text",
        "sample_fps": 5.0,
        "decode_status": "partial",
        "segment_strategy": "fixed",
        "segment_times": "explicit",
    }
    assert ocr["schema_version"] == 3
    assert ocr["model_key"] == "PP-OCRv6"
    assert asr["task"] == "transcribe"
    assert asr["requested_language"] == "auto"
    assert asr["detected_language"] == "zh"
    assert asr["language"] == "zh"
    assert asr["semantic_model_key"] == settings.asr_semantic_model
    assert asr["semantic_status"] == "complete"
    assert asr["chunk_builder_stats"]["retrieval_chunks"] == 2
    assert "postprocess_strategy" not in asr
    assert "postprocess_stats" not in asr
    assert asr["text_profile"]["cjk_chars"] == 8
    assert asr["tag_source"] == "sensevoice"


def test_channel_metadata_records_optional_visual_shot_metadata(tmp_path):
    from app.indexing.publication import channel_metadata

    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")

    payload = channel_metadata(
        "visual",
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

    assert payload["segment_strategy"] == "shot"
    assert payload["segment_times"] == "explicit"
    assert payload["min_segment_ms"] == 800
    assert payload["max_segment_ms"] == 8000
    assert payload["shot_detector"] == "pyscenedetect_adaptive"
    assert payload["shot_threshold"] == 0.18


def test_index_request_accepts_visual_shot_segment_options():
    from pydantic import ValidationError

    from app.api.schemas import IndexRequest

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


def test_index_request_accepts_asr_engine_override():
    from pydantic import ValidationError

    from app.api.schemas import IndexRequest

    request = IndexRequest(asr_engine="faster_whisper", asr_language="auto", asr_model="turbo")

    assert request.asr_engine == "faster-whisper"
    assert request.asr_language == "auto"
    assert request.asr_model == "turbo"

    try:
        IndexRequest(asr_engine="sensevoice")
    except ValidationError as exc:
        assert "asr_engine" in str(exc)
    else:
        raise AssertionError("invalid asr_engine should fail validation")
