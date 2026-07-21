from __future__ import annotations

import json

import numpy as np

from app.settings import Settings


def _frame(width: int = 20, height: int = 10) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _decode_np_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def test_visual_index_writes_frame_offsets_and_no_per_segment_payload(tmp_path, monkeypatch):
    from app.indexing import visual

    frames = [(1.0, _frame()), (6.0, _frame()), (16.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))

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


def test_visual_index_can_write_optional_shot_segment_times(tmp_path, monkeypatch):
    from app.indexing import visual

    frames = [(1.0, _frame()), (3.0, _frame()), (8.0, _frame()), (12.0, _frame())]
    monkeypatch.setattr(visual, "read_frames", lambda *_args, **_kwargs: iter(frames))
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


def test_visual_index_can_use_pyscenedetect_shot_detector(tmp_path, monkeypatch):
    from app.indexing import visual

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
        assert set(data.files) == {
            "chunk_times_ms",
            "texts",
            "chunk_emotions",
            "chunk_audio_events",
            "embeddings",
            "embedding_chunk_indices",
        }
        assert data["chunk_times_ms"].tolist() == [[1000, 2500], [5000, 7000]]
        assert data["texts"].tolist() == ["hello world", "green field"]
        assert _decode_np_strings(data["chunk_emotions"]) == ["", ""]
        assert _decode_np_strings(data["chunk_audio_events"]) == ["", ""]
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
    monkeypatch.setattr(ocr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = ocr.build_ocr_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "ocr.npz"),
        working_dir=str(tmp_path / "work"),
        sample_fps=1.0,
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "ocr.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "frame_times_ms",
            "frame_windows_ms",
            "embeddings",
            "embedding_frame_indices",
            "box_frame_indices",
            "box_texts",
            "box_scores",
            "boxes",
        }
        assert data["frame_times_ms"].tolist() == [5000]
        assert data["frame_windows_ms"].tolist() == [[5000, 6000]]
        assert data["box_frame_indices"].tolist() == [0, 0]
        assert data["box_texts"].tolist() == ["FIFA", "WORLD CUP"]
        assert np.allclose(data["boxes"][1], [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]])
        assert data["embedding_frame_indices"].tolist() == [0]
    assert result["schema_version"] == 3
    assert result["chunks"] == 1
    assert result["semantic_chunks"] == 1
    assert result["ocr_rec_resized_inputs"] == 0
    assert result["ocr_rec_max_input_width"] == 0
    assert result["backend_init_elapsed_seconds"] >= 0
    assert result["frame_loop_elapsed_seconds"] >= result["ocr_elapsed_seconds"]
    assert result["decode_postprocess_elapsed_seconds"] >= 0
    assert result["semantic_elapsed_seconds"] >= 0
    assert result["index_save_elapsed_seconds"] >= 0
    assert result["total_elapsed_seconds"] >= result["frame_loop_elapsed_seconds"]


def test_face_index_writes_track_times_without_precomputed_thumbnails(tmp_path, monkeypatch):
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

    result = faces.build_face_index(
        video_path="video.mp4",
        output_path=str(tmp_path / "face.npz"),
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
            "chunk_builder_stats": {"raw_items": 2, "retrieval_chunks": 2},
            "text_profile": {"chunks": 2, "cjk_chars": 8},
            "tag_source": "sensevoice",
        },
    )
    write_stage_manifest(
        "ocr",
        index_dir=index_dir,
        video=video,
        options={"ocr_sample_fps": 1.0},
        settings=settings,
        result={"ocr_version": "PP-OCRv6", "schema_version": 3, "decode_status": "complete"},
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
    assert payload["channels"]["ocr"]["schema_version"] == 3
    assert payload["channels"]["ocr"]["model_key"] == "PP-OCRv6"
    assert payload["channels"]["asr"]["file"] == "asr.npz"
    assert payload["channels"]["asr"]["task"] == "transcribe"
    assert payload["channels"]["asr"]["requested_language"] == "auto"
    assert payload["channels"]["asr"]["detected_language"] == "zh"
    assert payload["channels"]["asr"]["language"] == "zh"
    assert payload["channels"]["asr"]["semantic_model_key"] == settings.asr_semantic_model
    assert payload["channels"]["asr"]["semantic_status"] == "complete"
    assert payload["channels"]["asr"]["chunk_builder_stats"]["retrieval_chunks"] == 2
    assert "postprocess_strategy" not in payload["channels"]["asr"]
    assert "postprocess_stats" not in payload["channels"]["asr"]
    assert payload["channels"]["asr"]["text_profile"]["cjk_chars"] == 8
    assert payload["channels"]["asr"]["tag_source"] == "sensevoice"


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


def test_index_request_accepts_asr_engine_override():
    from pydantic import ValidationError

    from app.schemas import IndexRequest

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
