from __future__ import annotations

from pathlib import Path

import numpy as np

from app.encoders.text import TextEmbeddingEncoder
from app.indexing.common import atomic_save_npz


def build_text_semantic_index(
    chunks: list[dict],
    output_path: str | Path,
    model_name: str,
    model_dir: str | Path,
    device: str,
    batch_size: int = 32,
    local_files_only: bool = True,
) -> dict:
    result = build_text_semantic_arrays(
        chunks=chunks,
        model_name=model_name,
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
        dtype=np.float32,
    )
    atomic_save_npz(
        output_path,
        schema_version=np.asarray([1], dtype=np.int16),
        embeddings=result["embeddings"],
        chunk_indices=result["embedding_chunk_indices"],
        model=np.asarray([model_name]),
        device=np.asarray([device]),
    )
    return {
        "semantic_chunks": result["semantic_chunks"],
        "semantic_dim": result["semantic_dim"],
        "semantic_model": model_name,
        "semantic_device": device,
    }


def build_text_semantic_arrays(
    chunks: list[dict],
    model_name: str,
    model_dir: str | Path,
    device: str,
    batch_size: int = 32,
    local_files_only: bool = True,
    dtype=np.float16,
) -> dict:
    indexed = [
        (index, str(chunk.get("text", "")).strip())
        for index, chunk in enumerate(chunks)
        if chunk.get("semantic_eligible", True) and str(chunk.get("text", "")).strip()
    ]
    if not indexed:
        return {
            "embeddings": np.empty((0, 0), dtype=dtype),
            "embedding_chunk_indices": np.empty((0,), dtype=np.int32),
            "semantic_chunks": 0,
            "semantic_dim": 0,
            "semantic_model": model_name,
            "semantic_device": device,
            "semantic_status": "empty",
        }

    chunk_indices = np.asarray([item[0] for item in indexed], dtype=np.int32)
    texts = [item[1] for item in indexed]
    encoder = TextEmbeddingEncoder(model_name, model_dir, device, local_files_only=local_files_only)
    embeddings = encoder.encode(texts, batch_size=batch_size)
    return {
        "embeddings": embeddings.astype(dtype),
        "embedding_chunk_indices": chunk_indices,
        "semantic_chunks": len(texts),
        "semantic_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "semantic_model": model_name,
        "semantic_device": device,
        "semantic_status": "complete",
    }
