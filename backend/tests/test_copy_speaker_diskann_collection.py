from __future__ import annotations

import numpy as np
import pytest

from scripts.copy_speaker_diskann_collection import (
    _add_rows,
    _empty_summary,
    _row_digest,
    _serializable_summary,
)


def _row(pk: str, *, video_id: str = "video-1", version: str = "1") -> dict:
    embedding = np.zeros(192, dtype=np.float32)
    embedding[int(pk[-1]) % 192] = 1.0
    return {
        "pk": pk,
        "video_id": video_id,
        "asset_version": version,
        "model_version": "speaker-v1",
        "utterance_idx": int(pk[-1]),
        "start_ms": 1_000,
        "end_ms": 2_000,
        "asr_chunk_idx": int(pk[-1]),
        "track_id": 0,
        "embedding": embedding,
    }


def test_speaker_digest_accepts_equivalent_float32_payloads():
    row = _row("pk-1")

    assert _row_digest(row) == _row_digest(
        {**row, "embedding": row["embedding"].tolist()}
    )


def test_speaker_summary_is_order_independent_and_counts_versions():
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


def test_speaker_digest_rejects_missing_or_invalid_embedding():
    row = _row("pk-1")
    with pytest.raises(RuntimeError, match="missing fields"):
        _row_digest({key: value for key, value in row.items() if key != "track_id"})
    with pytest.raises(RuntimeError, match="embedding shape"):
        _row_digest({**row, "embedding": [0.0] * 8})
    broken = row["embedding"].copy()
    broken[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _row_digest({**row, "embedding": broken})
