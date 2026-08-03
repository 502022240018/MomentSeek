"""Sentence-transformers text embedding encoder shared by indexing and retrieval."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.model_sources import (
    hf_cached_snapshot_path,
    offline_env,
    resolve_hf_model_source,
)


def _hf_cached_snapshot_path(model_dir: str | Path, model_name: str) -> Path | None:
    return hf_cached_snapshot_path(model_dir, model_name)


def _resolve_model_source(model_name: str, model_dir: str | Path, local_files_only: bool) -> tuple[str, bool]:
    return resolve_hf_model_source(model_dir, model_name, local_files_only=local_files_only)


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
        model_source, resolved_local_only = _resolve_model_source(model_name, model_dir, local_files_only)
        from sentence_transformers import SentenceTransformer

        with offline_env(resolved_local_only):
            self.model = SentenceTransformer(
                model_source,
                cache_folder=str(model_dir),
                device=device,
                local_files_only=resolved_local_only,
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

