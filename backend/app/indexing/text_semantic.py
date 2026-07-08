from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.indexing.common import atomic_save_npz


def _hf_cached_snapshot_path(model_dir: str | Path, model_name: str) -> Path | None:
    root = Path(model_dir)
    repo_name = f"models--{model_name.replace('/', '--')}"
    for repo_dir in (root / "hub" / repo_name, root / repo_name):
        snapshots = repo_dir / "snapshots"
        if not repo_dir.exists() or not snapshots.exists():
            continue
        ref = repo_dir / "refs" / "main"
        if ref.exists():
            snapshot = snapshots / ref.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return snapshot
        complete_snapshots = []
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            has_config = (snapshot / "config.json").exists() or (snapshot / "modules.json").exists()
            has_weights = (
                (snapshot / "model.safetensors").exists()
                or (snapshot / "pytorch_model.bin").exists()
                or any(snapshot.glob("**/model.safetensors"))
                or any(snapshot.glob("**/pytorch_model.bin"))
            )
            if has_config and has_weights:
                complete_snapshots.append(snapshot)
        if complete_snapshots:
            return sorted(complete_snapshots, key=lambda path: path.name)[0]
        for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.name):
            if snapshot.is_dir():
                return snapshot
    return None


def _resolve_model_source(model_name: str, model_dir: str | Path, local_files_only: bool) -> tuple[str, bool]:
    if local_files_only:
        snapshot = _hf_cached_snapshot_path(model_dir, model_name)
        if snapshot is not None:
            return str(snapshot), True
    return model_name, local_files_only


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
        offline_env = {}
        if resolved_local_only:
            # Some transformers versions still make metadata calls while loading a
            # cached tokenizer. Force offline mode so shared-server jobs fail fast
            # to lexical fallback instead of hanging on Hugging Face networking.
            for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
                offline_env[name] = os.environ.get(name)
                os.environ[name] = "1"
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(
                model_source,
                cache_folder=str(model_dir),
                device=device,
                local_files_only=resolved_local_only,
            )
        finally:
            for name, old_value in offline_env.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value

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
