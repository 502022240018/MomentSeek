import numpy as np

from app.indexing import asr
from app.indexing.asr import build_asr_index


def test_sidecar_asr_index_postprocesses_ascii_fragments_and_preserves_schema(tmp_path, monkeypatch):
    sidecar = tmp_path / "demo.srt"
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\ntoday\n\n"
        "2\n00:00:00,800 --> 00:00:01,200\nwe discuss books\n\n"
        "3\n00:00:03,200 --> 00:00:03,700\nnext part\n",
        encoding="utf-8",
    )

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        return {
            "embeddings": np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float16),
            "embedding_chunk_indices": np.asarray(
                [index for index, chunk in enumerate(chunks) if chunk.get("semantic_eligible", True)],
                dtype=np.int32,
            ),
            "semantic_chunks": len(chunks),
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="sidecar",
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
        assert data["chunk_times_ms"].tolist() == [[0, 1200], [3200, 3700]]
        assert data["texts"].tolist() == ["today we discuss books", "next part"]
        assert data["embedding_chunk_indices"].tolist() == [0, 1]
    assert result["raw_chunks"] == 3
    assert result["chunks"] == 2
    assert result["chunk_builder_stats"]["merged_items"] == 1
