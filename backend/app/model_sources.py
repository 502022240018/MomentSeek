from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


HF_CONFIG_FILES = {
    "config.json",
    "open_clip_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "modules.json",
}
HF_WEIGHT_SUFFIXES = {".bin", ".safetensors"}
MODELSCOPE_WEIGHT_SUFFIXES = {".bin", ".onnx", ".pt", ".safetensors"}

FASTER_WHISPER_ALIASES = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

MODELSCOPE_ALIASES = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _repo_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _has_hf_assets(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False
    has_config = any((snapshot / name).is_file() for name in HF_CONFIG_FILES)
    has_weights = any(path.is_file() and path.suffix.lower() in HF_WEIGHT_SUFFIXES for path in snapshot.rglob("*"))
    return has_config and has_weights


def hf_cached_snapshot_path(model_cache_dir: str | Path | None, model_id: str) -> Path | None:
    if not model_cache_dir:
        return None
    cache_dir = Path(model_cache_dir)
    repo_name = _repo_cache_name(model_id)
    for repo_dir in (cache_dir / "hub" / repo_name, cache_dir / repo_name):
        snapshots = repo_dir / "snapshots"
        if not repo_dir.exists() or not snapshots.exists():
            continue
        ref = repo_dir / "refs" / "main"
        if ref.exists():
            snapshot = snapshots / ref.read_text(encoding="utf-8").strip()
            if _has_hf_assets(snapshot):
                return snapshot
        complete = [snapshot for snapshot in snapshots.iterdir() if _has_hf_assets(snapshot)]
        if complete:
            return sorted(complete, key=lambda path: path.name)[0]
    return None


def resolve_hf_model_source(
    model_cache_dir: str | Path | None,
    model_id: str,
    local_files_only: bool = True,
) -> tuple[str, bool]:
    explicit_path = Path(model_id)
    if explicit_path.exists():
        return str(explicit_path), True
    snapshot = hf_cached_snapshot_path(model_cache_dir, model_id)
    if snapshot is not None:
        return str(snapshot), True
    if local_files_only:
        raise FileNotFoundError(
            f"本地 Hugging Face 模型缺失: {model_id}; cache_dir={model_cache_dir}"
        )
    return model_id, False


def resolve_faster_whisper_model_source(
    model_root: str | Path,
    model_name: str,
    local_files_only: bool = True,
) -> str:
    explicit_path = Path(model_name)
    if explicit_path.exists():
        return str(explicit_path)
    model_id = FASTER_WHISPER_ALIASES.get(model_name.casefold(), model_name)
    snapshot = hf_cached_snapshot_path(model_root, model_id)
    if snapshot is not None:
        return str(snapshot)
    if local_files_only:
        raise FileNotFoundError(
            f"本地 faster-whisper 模型缺失: {model_name}; model_root={model_root}"
        )
    return model_name


def _modelscope_candidates(model_root: str | Path | None, model_id: str) -> list[Path]:
    parts = model_id.split("/")
    candidates: list[Path] = []
    roots = []
    if model_root:
        roots.append(Path(model_root))
    env_cache = os.environ.get("MODELSCOPE_CACHE")
    if env_cache:
        roots.append(Path(env_cache))
    roots.append(Path.home() / ".cache" / "modelscope")
    roots.append(Path("/root/.cache/modelscope"))
    for root in roots:
        candidates.append(root.joinpath(*parts))
        candidates.append(root / model_id.replace("/", "--"))
        candidates.append(root / "models" / model_id.replace("/", "--"))
    return candidates


def _has_modelscope_assets(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file() and item.suffix.lower() in MODELSCOPE_WEIGHT_SUFFIXES
        for item in path.rglob("*")
    )


def _modelscope_snapshot_path(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), key=lambda item: item.name):
        if _has_modelscope_assets(snapshot):
            return snapshot
    return None


def resolve_modelscope_model_source(
    model_root: str | Path | None,
    model_name: str,
    local_files_only: bool = True,
) -> str:
    explicit_path = Path(model_name)
    if explicit_path.exists():
        return str(explicit_path)
    model_id = MODELSCOPE_ALIASES.get(model_name.casefold(), model_name)
    for candidate in _modelscope_candidates(model_root, model_id):
        snapshot = _modelscope_snapshot_path(candidate)
        if snapshot is not None:
            return str(snapshot)
        if _has_modelscope_assets(candidate):
            return str(candidate)
    if local_files_only:
        raise FileNotFoundError(
            f"本地 ModelScope/FunASR 模型缺失: {model_name}; model_root={model_root}"
        )
    return model_name


@contextmanager
def offline_env(enabled: bool = True) -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    old_values = {name: os.environ.get(name) for name in names}
    if enabled:
        for name in names:
            os.environ[name] = "1"
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
