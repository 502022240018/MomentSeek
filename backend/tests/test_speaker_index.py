from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.indexing.modalities.speaker.speaker import (
    _adaptive_turn_units,
    _asr_source_from_milvus,
    _track_count,
)


class _AsrCollection:
    def __init__(self, rows):
        self.rows = rows
        self.expression = None

    def query(self, *, expr, output_fields, limit, timeout):
        self.expression = expr
        assert output_fields == ["segment_idx", "start_ms", "end_ms", "text"]
        assert limit == 16_384
        assert timeout > 0
        return self.rows


def _context(rows):
    collection = _AsrCollection(rows)
    client = SimpleNamespace(collection_for=lambda modality: collection)
    return SimpleNamespace(video_id="video-1", client=client), collection


def test_speaker_source_is_version_pinned_milvus_asr():
    context, collection = _context([
        {"segment_idx": 1, "start_ms": 1000, "end_ms": 2000, "text": "second"},
        {"segment_idx": 0, "start_ms": 0, "end_ms": 1000, "text": "first"},
    ])

    times, texts = _asr_source_from_milvus(context, "asr-v7")

    assert times.tolist() == [[0, 1000], [1000, 2000]]
    assert texts == ["first", "second"]
    assert 'video_id == "video-1"' in collection.expression
    assert 'asset_version == "asr-v7"' in collection.expression


def test_speaker_source_rejects_sparse_asr_coverage():
    context, _ = _context([
        {"segment_idx": 1, "start_ms": 1000, "end_ms": 2000, "text": "second"},
    ])
    with pytest.raises(RuntimeError, match="coverage is sparse"):
        _asr_source_from_milvus(context, "asr-v7")


def test_track_count_ignores_unassigned_utterances():
    assert _track_count(np.asarray([-1, 0, 2, 2], dtype=np.int32)) == 3
    assert _track_count(np.asarray([-1], dtype=np.int32)) == 0


def test_speaker_build_writes_normalized_arrays_directly_to_milvus(tmp_path):
    from app.indexing.modalities.speaker.speaker import build_speaker_index

    context, _ = _context([
        {"segment_idx": 0, "start_ms": 0, "end_ms": 1000, "text": "hello"},
    ])

    class _Pipeline:
        fs = 16_000

        def do_vad(self, waveform):
            return [[0.0, 1.0]]

        def chunk(self, start, end):
            return [[start, end]]

        def cluster(self, embeddings):
            return np.asarray([0], dtype=np.int32)

    pipeline = _Pipeline()
    module = SimpleNamespace(
        Diarization3Dspeaker=lambda **kwargs: pipeline,
        load_audio=lambda *args: np.zeros((16_000,), dtype=np.float32),
    )
    raw_embedding = np.arange(1, 193, dtype=np.float32)[None, :]
    with (
        patch("app.indexing.modalities.speaker.speaker._load_3dspeaker", return_value=module),
        patch("app.indexing.modalities.speaker.speaker._extract_wav"),
        patch(
            "app.indexing.modalities.speaker.speaker._extract_embeddings",
            side_effect=[raw_embedding, raw_embedding],
        ),
        patch(
            "app.vector_store.milvus.milvus_indexer.write_modality_from_memory",
            return_value=1,
        ) as write,
    ):
        result = build_speaker_index(
            video_path=str(tmp_path / "video.mp4"),
            working_dir=str(tmp_path / "work"),
            model_repo="models/3d-speaker",
            model_cache_dir="models/cache",
            device="cpu",
            milvus_ctx=context,
            asr_asset_version="asr-v7",
        )

    assert result["utterances"] == 1
    assert result["tracks"] == 1
    assert result["milvus_rows"] == 1
    assert list(tmp_path.rglob("*.npz")) == []
    arrays = write.call_args.args[2]
    assert arrays["utterance_times_ms"].tolist() == [[0, 1000]]
    assert arrays["utterance_refs"].tolist() == [[0, 0]]
    np.testing.assert_allclose(
        np.linalg.norm(arrays["utterance_embeddings"], axis=1),
        np.ones((1,), dtype=np.float32),
    )


def test_adaptive_turns_follow_speaker_and_asr_boundaries():
    times, refs, tracks = _adaptive_turn_units(
        [[0.0, 1.5], [0.75, 2.25], [1.5, 3.0], [2.25, 3.75]],
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0, 1800], [1800, 4000]], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )

    assert times.tolist() == [[0, 1800], [1875, 3750]]
    assert refs.tolist() == [0, 1]
    assert tracks.tolist() == [0, 1]
