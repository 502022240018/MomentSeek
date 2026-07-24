from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.backfill_milvus import (
    _expected_count,
    _load_progress,
    _progress_key,
    _record_progress,
)


def test_progress_is_scoped_by_asset_version(tmp_path):
    progress = tmp_path / "state" / "backfill.jsonl"
    _record_progress(progress, "video-1", "visual", "1", "done", "count=2")
    _record_progress(progress, "video-1", "visual", "2", "fail", "boom")

    assert _load_progress(progress) == {
        _progress_key("video-1", "visual", "1"),
    }


def test_legacy_unversioned_progress_is_not_resumed(tmp_path):
    progress = tmp_path / "backfill.jsonl"
    progress.write_text(
        json.dumps({
            "video_id": "video-1",
            "modality": "visual",
            "status": "done",
        }),
        encoding="utf-8",
    )

    assert _load_progress(progress) == set()


def test_expected_count_matches_each_modality_row_axis(tmp_path):
    cases = {
        "visual": ("frame_embeddings", np.zeros((3, 2), dtype=np.float32)),
        "asr": ("chunk_times_ms", np.zeros((4, 2), dtype=np.int32)),
        "ocr": ("frame_times_ms", np.zeros(5, dtype=np.int32)),
        "face": ("embeddings", np.zeros((6, 2), dtype=np.float32)),
        "speaker": ("utterance_embeddings", np.zeros((7, 2), dtype=np.float32)),
    }
    for modality, (field, value) in cases.items():
        path = tmp_path / f"{modality}.npz"
        np.savez_compressed(path, **{field: value})
        assert _expected_count(path, modality) == len(value)


def test_expected_count_rejects_malformed_npz(tmp_path):
    path = tmp_path / "asr.npz"
    np.savez_compressed(path, texts=np.asarray(["hello"]))

    with pytest.raises(ValueError, match="chunk_times_ms"):
        _expected_count(path, "asr")
