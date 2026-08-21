from types import SimpleNamespace
from unittest.mock import patch


class _EmptyAsrCollection:
    def query(self, *, expr, output_fields, limit, timeout):
        assert 'asset_version == "asr-v1"' in expr
        assert output_fields == ["segment_idx", "start_ms", "end_ms", "text"]
        assert limit == 16_384
        assert timeout > 0
        return []


def test_speaker_publishes_empty_milvus_version_without_npz(tmp_path):
    from app.indexing.modalities.speaker.speaker import build_speaker_index

    collection = _EmptyAsrCollection()
    context = SimpleNamespace(
        video_id="video-1",
        client=SimpleNamespace(collection_for=lambda modality: collection),
    )
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"")

    with patch(
        "app.vector_store.milvus.milvus_indexer.write_modality_from_memory",
        return_value=0,
    ) as write:
        result = build_speaker_index(
            video_path=str(video_path),
            working_dir=str(tmp_path / "work"),
            model_repo="dummy",
            model_cache_dir="dummy",
            device="cpu",
            milvus_ctx=context,
            asr_asset_version="asr-v1",
        )

    assert result["utterances"] == 0
    assert result["tracks"] == 0
    assert result["milvus_rows"] == 0
    assert list(tmp_path.rglob("*.npz")) == []
    arrays = write.call_args.args[2]
    assert arrays["utterance_embeddings"].shape == (0, 192)
    assert arrays["utterance_times_ms"].shape == (0, 2)
    assert arrays["utterance_refs"].shape == (0, 2)
