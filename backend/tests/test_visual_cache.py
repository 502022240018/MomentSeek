from pathlib import Path
import sys

from app.indexing.visual import _hf_cached_snapshot_path, resolve_device


def test_hf_cached_snapshot_path_accepts_root_cache_layout(tmp_path):
    model_id = "OFA-Sys/chinese-clip-vit-base-patch16"
    snapshot = (
        tmp_path
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"weights")

    assert _hf_cached_snapshot_path(tmp_path, model_id) == snapshot


def test_visual_resolve_device_returns_indexed_cuda_device(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    assert resolve_device(npu_enabled=False, npu_device_id=0, cuda_enabled=True) == "cuda:0"
