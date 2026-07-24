import os
from pathlib import Path

import pytest

from app.model_sources import (
    hf_cached_snapshot_path,
    offline_env,
    resolve_faster_whisper_model_source,
    resolve_hf_model_source,
    resolve_modelscope_model_source,
)


def test_hf_cached_snapshot_path_uses_ref_main(tmp_path):
    model_id = "google/siglip2-so400m-patch14-384"
    repo = tmp_path / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert hf_cached_snapshot_path(tmp_path, model_id) == snapshot
    assert resolve_hf_model_source(tmp_path, model_id) == (str(snapshot), True)


def test_hf_cached_snapshot_path_accepts_hub_layout_without_ref(tmp_path):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / "def456"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("{}", encoding="utf-8")
    (snapshot / "0").mkdir()
    (snapshot / "0" / "model.safetensors").write_bytes(b"weights")

    assert hf_cached_snapshot_path(tmp_path, model_id) == snapshot


def test_resolve_hf_model_source_raises_when_local_only_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="本地 Hugging Face 模型缺失"):
        resolve_hf_model_source(tmp_path, "missing/repo", local_files_only=True)


def test_resolve_faster_whisper_alias_uses_cached_snapshot(tmp_path):
    snapshot = (
        tmp_path
        / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
        / "snapshots"
        / "turbo123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.bin").write_bytes(b"weights")

    assert resolve_faster_whisper_model_source(tmp_path, "turbo") == str(snapshot)


def test_resolve_faster_whisper_model_source_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="本地 faster-whisper 模型缺失"):
        resolve_faster_whisper_model_source(tmp_path, "small", local_files_only=True)


def test_resolve_modelscope_model_source_accepts_cache_layout(tmp_path):
    model_dir = tmp_path / "iic" / "SenseVoiceSmall"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pt").write_bytes(b"weights")

    assert resolve_modelscope_model_source(tmp_path, "iic/SenseVoiceSmall") == str(model_dir)


def test_resolve_modelscope_model_source_accepts_modelscope_snapshot_layout(tmp_path):
    snapshot = tmp_path / "models" / "iic--SenseVoiceSmall" / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "configuration.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.pt").write_bytes(b"weights")

    assert resolve_modelscope_model_source(tmp_path, "iic/SenseVoiceSmall") == str(snapshot)


def test_offline_env_restores_previous_values(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "old")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with offline_env():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"

    assert os.environ["HF_HUB_OFFLINE"] == "old"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
