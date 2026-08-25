from __future__ import annotations

from types import SimpleNamespace

from app.catalog.db import Catalog
from app.maintenance import promote_verified_publications as promotion


class _Iterator:
    def __init__(self, rows):
        self.rows = list(rows)
        self.done = False

    def next(self):
        if self.done:
            return []
        self.done = True
        return self.rows

    def close(self):
        pass


class _Collection:
    def __init__(self, rows, visual=False):
        self.rows = rows
        fields = {"video_id", "asset_version"}
        if visual:
            fields.update(promotion._scan_visual_health.__globals__["_VISUAL_TIME_FIELDS"])
        self.schema = SimpleNamespace(
            fields=[SimpleNamespace(name=name) for name in fields]
        )

    def query_iterator(self, *, expr, output_fields, batch_size, timeout):
        del expr, batch_size, timeout
        return _Iterator(
            [
                {key: row[key] for key in output_fields if key in row}
                for row in self.rows
            ]
        )


class _Client:
    def __init__(self, rows):
        self.rows = rows

    def collection_for(self, modality):
        return _Collection(self.rows.get(modality, []), visual=modality == "visual")


def _catalog(path, video_id="v1"):
    catalog = Catalog(path)
    catalog.create_video(
        {
            "id": video_id,
            "name": "video.mp4",
            "file_path": "/tmp/video.mp4",
            "duration": 10.0,
            "fps": 1.0,
            "width": 100,
            "height": 100,
            "status": "ready",
        }
    )
    return catalog


def _visual_row(version="new"):
    return {
        "video_id": "v1",
        "asset_version": version,
        "frame_idx": 0,
        "timestamp_ms": 0,
        "segment_id": 0,
        "segment_start_ms": 0,
        "segment_end_ms": 5000,
    }


def test_promote_verified_publications_dry_run_and_execute(tmp_path):
    source = _catalog(tmp_path / "source.sqlite3")
    target = _catalog(tmp_path / "target.sqlite3")
    source.publish_modality(
        "v1", "visual", asset_version="new", row_count=1, metadata={"x": 1}
    )
    client = _Client({"visual": [_visual_row()]})

    dry = promotion.promote(
        source_catalog=source,
        target_catalog=target,
        client=client,
        target_index_root=tmp_path / "index",
    )
    assert len(dry["dry_run_ready"]) == 1
    assert target.get_modality_publication("v1", "visual") is None

    applied = promotion.promote(
        source_catalog=source,
        target_catalog=target,
        client=client,
        target_index_root=tmp_path / "index",
        execute=True,
    )
    assert len(applied["promoted"]) == 1
    assert target.get_modality_publication("v1", "visual")["asset_version"] == "new"


def test_promotion_rejects_bad_visual_bounds(tmp_path):
    source = _catalog(tmp_path / "source.sqlite3")
    target = _catalog(tmp_path / "target.sqlite3")
    source.publish_modality("v1", "visual", asset_version="new", row_count=1)
    bad = _visual_row()
    bad["segment_end_ms"] = 0

    try:
        promotion.promote(
            source_catalog=source,
            target_catalog=target,
            client=_Client({"visual": [bad]}),
            target_index_root=tmp_path / "index",
            execute=True,
        )
    except RuntimeError as exc:
        assert "invalid visual health" in str(exc)
    else:
        raise AssertionError("bad visual bounds must fail closed")
    assert target.get_modality_publication("v1", "visual") is None


def test_promotion_rejects_target_conflict(tmp_path):
    source = _catalog(tmp_path / "source.sqlite3")
    target = _catalog(tmp_path / "target.sqlite3")
    source.publish_modality("v1", "visual", asset_version="new", row_count=1)
    target.publish_modality("v1", "visual", asset_version="other", row_count=1)

    report = promotion.promote(
        source_catalog=source,
        target_catalog=target,
        client=_Client({"visual": [_visual_row()]}),
        target_index_root=tmp_path / "index",
        execute=True,
    )
    assert len(report["errors"]) == 1
    assert target.get_modality_publication("v1", "visual")["asset_version"] == "other"


def test_promotion_requires_matching_speaker_source(tmp_path):
    source = _catalog(tmp_path / "source.sqlite3")
    target = _catalog(tmp_path / "target.sqlite3")
    source.publish_modalities(
        "v1",
        [
            {"modality": "asr", "asset_version": "a1", "row_count": 1},
            {
                "modality": "speaker",
                "asset_version": "s1",
                "row_count": 1,
                "metadata": {"source_asr_asset_version": "wrong"},
            },
        ],
    )

    try:
        promotion.promote(
            source_catalog=source,
            target_catalog=target,
            client=_Client(
                {
                    "asr": [{"video_id": "v1", "asset_version": "a1"}],
                    "speaker": [{"video_id": "v1", "asset_version": "s1"}],
                }
            ),
            target_index_root=tmp_path / "index",
            execute=True,
        )
    except RuntimeError as exc:
        assert "source ASR version" in str(exc)
    else:
        raise AssertionError("speaker/ASR mismatch must fail closed")
