"""Unit tests for the offline OCR NPZ migration command."""
from __future__ import annotations

from pathlib import Path

from app.vector_store.milvus.rebuild_ocr_from_npz import discover_ocr_npz_assets


def test_discover_ocr_npz_assets_only_returns_immediate_video_indexes(tmp_path: Path):
    first = tmp_path / "video-a"
    second = tmp_path / "video-b"
    ignored = tmp_path / "video-c"
    first.mkdir()
    second.mkdir()
    ignored.mkdir()
    (first / "ocr.npz").touch()
    (second / "ocr.npz").touch()
    (ignored / "asr.npz").touch()
    nested = first / "nested"
    nested.mkdir()
    (nested / "ocr.npz").touch()

    assets = discover_ocr_npz_assets(tmp_path)

    assert [(asset.video_id, asset.npz_path.name) for asset in assets] == [
        ("video-a", "ocr.npz"),
        ("video-b", "ocr.npz"),
    ]
