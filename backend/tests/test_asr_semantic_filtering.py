import numpy as np

from app.indexing import text_semantic


class FakeTextEmbeddingEncoder:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, batch_size=32):
        return np.asarray([[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)], dtype=np.float32)


def test_build_text_semantic_arrays_skips_semantic_ineligible_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(text_semantic, "TextEmbeddingEncoder", FakeTextEmbeddingEncoder)
    chunks = [
        {"text": "嗯", "semantic_eligible": False},
        {"text": "足球场上有人射门", "semantic_eligible": True},
        {"text": "好的", "semantic_eligible": False},
    ]

    result = text_semantic.build_text_semantic_arrays(
        chunks=chunks,
        model_name="fake-semantic",
        model_dir=tmp_path,
        device="cpu",
    )

    assert result["embeddings"].shape == (1, 3)
    assert result["embedding_chunk_indices"].tolist() == [1]
    assert result["semantic_chunks"] == 1


def test_build_text_semantic_arrays_keeps_backwards_compatible_default(tmp_path, monkeypatch):
    monkeypatch.setattr(text_semantic, "TextEmbeddingEncoder", FakeTextEmbeddingEncoder)
    chunks = [{"text": "足球场"}, {"text": "烤包子"}]

    result = text_semantic.build_text_semantic_arrays(
        chunks=chunks,
        model_name="fake-semantic",
        model_dir=tmp_path,
        device="cpu",
    )

    assert result["embeddings"].shape == (2, 3)
    assert result["embedding_chunk_indices"].tolist() == [0, 1]
