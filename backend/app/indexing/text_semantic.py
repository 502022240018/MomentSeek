from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.indexing.common import atomic_save_npz


def resolve_text_embedding_device(device: str, cuda_enabled: bool = False) -> str:
    """Resolve the device for sentence-transformer style text embeddings.

    Keep the default conservative: ASR semantic indexing is cheap compared with
    Whisper/CLIP, and sentence-transformers on Ascend NPU is not guaranteed to be
    supported. Use CUDA when explicitly enabled; otherwise CPU.
    """
    if device and device != "auto":
        return device
    if cuda_enabled:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


class TextEmbeddingEncoder:
    def __init__(
        self,
        model_name: str,
        model_dir: str | Path,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        if local_files_only:
            # Some transformers versions still make metadata calls while loading a
            # cached tokenizer. Force offline mode so shared-server jobs fail fast
            # to lexical fallback instead of hanging on Hugging Face networking.
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(model_dir),
            device=device,
            local_files_only=local_files_only,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_text_semantic_index(
    chunks: list[dict],
    output_path: str | Path,
    model_name: str,
    model_dir: str | Path,
    device: str,
    batch_size: int = 32,
    local_files_only: bool = True,
) -> dict:
    indexed = [
        (index, str(chunk.get("text", "")).strip())
        for index, chunk in enumerate(chunks)
        if str(chunk.get("text", "")).strip()
    ]
    if not indexed:
        atomic_save_npz(
            output_path,
            schema_version=np.asarray([1], dtype=np.int16),
            embeddings=np.empty((0, 0), dtype=np.float32),
            chunk_indices=np.empty((0,), dtype=np.int32),
            model=np.asarray([model_name]),
            device=np.asarray([device]),
        )
        return {"semantic_chunks": 0, "semantic_model": model_name, "semantic_device": device}

    chunk_indices = np.asarray([item[0] for item in indexed], dtype=np.int32)
    texts = [item[1] for item in indexed]
    encoder = TextEmbeddingEncoder(model_name, model_dir, device, local_files_only=local_files_only)
    embeddings = encoder.encode(texts, batch_size=batch_size)
    atomic_save_npz(
        output_path,
        schema_version=np.asarray([1], dtype=np.int16),
        embeddings=embeddings,
        chunk_indices=chunk_indices,
        model=np.asarray([model_name]),
        device=np.asarray([device]),
    )
    return {
        "semantic_chunks": len(texts),
        "semantic_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "semantic_model": model_name,
        "semantic_device": device,
    }
