from __future__ import annotations

import numpy as np
import pytest

from scripts.copy_asr_hybrid_collection import (
    _add_rows,
    _empty_summary,
    _row_digest,
    _serializable_summary,
)


def _row(pk: str, *, video_id: str = "video-1", version: str = "1") -> dict:
    embedding = np.zeros(384, dtype=np.float32)
    embedding[int(pk[-1]) % 384] = 1.0
    return {
        "pk": pk,
        "video_id": video_id,
        "asset_version": version,
        "model_version": "asr-v1",
        "segment_idx": int(pk[-1]),
        "start_ms": 1_000,
        "end_ms": 2_000,
        "text": "测试文本",
        "has_embedding": True,
        "embedding": embedding,
    }


def test_row_digest_accepts_equivalent_float32_payloads():
    row = _row("pk-1")
    as_list = {**row, "embedding": row["embedding"].tolist()}

    assert _row_digest(row) == _row_digest(as_list)


def test_summary_is_order_independent_and_counts_versions():
    first = _empty_summary()
    second = _empty_summary()
    rows = [_row("pk-1"), _row("pk-2", version="2")]

    _add_rows(first, rows)
    _add_rows(second, list(reversed(rows)))

    assert _serializable_summary(first) == _serializable_summary(second)
    assert _serializable_summary(first)["version_counts"] == {
        "video-1\u00001": 1,
        "video-1\u00002": 1,
    }


def test_row_digest_rejects_missing_or_invalid_embedding():
    row = _row("pk-1")
    with pytest.raises(RuntimeError, match="missing fields"):
        _row_digest({key: value for key, value in row.items() if key != "text"})
    with pytest.raises(RuntimeError, match="embedding shape"):
        _row_digest({**row, "embedding": [0.0] * 8})
    broken = row["embedding"].copy()
    broken[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _row_digest({**row, "embedding": broken})
