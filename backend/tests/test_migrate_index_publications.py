from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.catalog.db import Catalog
from app.core.settings import Settings
from app.maintenance import migrate_index_publications as migration


class _Iterator:
    def __init__(self, rows: list[dict], batch_size: int):
        self._rows = rows
        self._batch_size = batch_size
        self._offset = 0
        self.closed = False

    def next(self):
        if self._offset >= len(self._rows):
            return []
        page = self._rows[self._offset : self._offset + self._batch_size]
        self._offset += len(page)
        return page

    def close(self):
        self.closed = True


class _Collection:
    def __init__(self, rows: list[dict], fields: set[str]):
        self.rows = rows
        self.schema = SimpleNamespace(
            fields=[SimpleNamespace(name=name) for name in sorted(fields)]
        )

    @staticmethod
    def _video_ids(expr: str) -> set[str]:
        marker = "video_id in "
        if marker not in expr:
            return set()
        return set(json.loads(expr.split(marker, 1)[1]))

    def query_iterator(self, *, batch_size, expr, output_fields, timeout):
        del timeout
        video_ids = self._video_ids(expr)
        rows = [
            {field: row[field] for field in output_fields if field in row}
            for row in self.rows
            if row.get("video_id") in video_ids
        ]
        return _Iterator(rows, batch_size)


class _Client:
    def __init__(self, rows_by_modality: dict[str, list[dict]]):
        self.rows_by_modality = {
            modality: list(rows) for modality, rows in rows_by_modality.items()
        }

    def collection_for(self, modality: str):
        rows = self.rows_by_modality.get(modality, [])
        fields = {"video_id", "asset_version"}
        if modality == "visual":
            fields.update(migration._VISUAL_TIME_FIELDS)
        return _Collection(rows, fields)

    def count_video_modality_version(
        self, video_id: str, modality: str, asset_version: str
    ) -> int:
        return sum(
            row.get("video_id") == video_id
            and row.get("asset_version") == asset_version
            for row in self.rows_by_modality.get(modality, [])
        )


def _setup(
    monkeypatch,
    tmp_path,
    *,
    channels: dict,
    rows_by_modality: dict[str, list[dict]],
):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        milvus_enabled=True,
        milvus_write_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video-1.mp4"
    video_path.write_bytes(b"video")
    catalog.create_video(
        {
            "id": "video-1",
            "name": "video-1.mp4",
            "file_path": str(video_path),
            "duration": 10,
            "fps": 25,
            "width": 1280,
            "height": 720,
            "status": "ready",
        }
    )
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_manifest.json").write_text(
        json.dumps({"channels": channels}),
        encoding="utf-8",
    )
    client = _Client(rows_by_modality)
    monkeypatch.setattr(migration, "get_settings", lambda: settings)
    monkeypatch.setattr(migration, "get_milvus_client", lambda: client)
    return catalog


def test_single_discovered_version_is_dry_run_then_applied(monkeypatch, tmp_path):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={"asr": {"file": "asr.npz", "engine": "funasr"}},
        rows_by_modality={
            "asr": [
                {"video_id": "video-1", "asset_version": "server-v7"},
                {"video_id": "video-1", "asset_version": "server-v7"},
            ]
        },
    )

    dry_run = migration.migrate(apply=False)

    assert dry_run["errors"] == []
    assert dry_run["migrated"] == [
        {
            "video_id": "video-1",
            "modality": "asr",
            "asset_version": "server-v7",
            "row_count": 2,
            "version_source": "milvus_scan",
            "status": "would_publish",
        }
    ]
    assert catalog.get_modality_publication("video-1", "asr") is None

    applied = migration.migrate(apply=True)

    assert applied["errors"] == []
    assert applied["migrated"][0]["status"] == "published"
    publication = catalog.get_modality_publication("video-1", "asr")
    assert publication["asset_version"] == "server-v7"
    assert publication["row_count"] == 2
    assert publication["engine"] == "funasr"


def test_declared_empty_channel_uses_explicit_legacy_empty_version(
    monkeypatch, tmp_path
):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={"ocr": {"file": "ocr.npz"}},
        rows_by_modality={},
    )

    report = migration.migrate(apply=True)

    assert report["errors"] == []
    assert report["migrated"][0]["version_source"] == "legacy_empty"
    publication = catalog.get_modality_publication("video-1", "ocr")
    assert publication["asset_version"] == migration.LEGACY_EMPTY_VERSION
    assert publication["row_count"] == 0


def test_multiple_versions_without_manifest_pointer_fail_closed(
    monkeypatch, tmp_path
):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={"face": {"file": "face.npz"}},
        rows_by_modality={
            "face": [
                {"video_id": "video-1", "asset_version": "v1"},
                {"video_id": "video-1", "asset_version": "v2"},
            ]
        },
    )

    report = migration.migrate(apply=True)

    assert report["migrated"] == []
    assert "multiple Milvus asset versions" in report["errors"][0]["error"]
    assert catalog.get_modality_publication("video-1", "face") is None


def test_explicit_manifest_version_is_verified_and_disambiguates(
    monkeypatch, tmp_path
):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={
            "asr": {
                "milvus_asset_version": "asr-v4",
                "milvus_row_count": 1,
            },
            "speaker": {
                "milvus_asset_version": "v2",
                "milvus_row_count": 1,
            }
        },
        rows_by_modality={
            "asr": [
                {"video_id": "video-1", "asset_version": "asr-v4"},
            ],
            "speaker": [
                {"video_id": "video-1", "asset_version": "v1"},
                {"video_id": "video-1", "asset_version": "v2"},
            ]
        },
    )

    report = migration.migrate(apply=True)

    assert report["errors"] == []
    assert all(item["version_source"] == "manifest" for item in report["migrated"])
    publication = catalog.get_modality_publication("video-1", "speaker")
    assert publication["asset_version"] == "v2"
    assert publication["row_count"] == 1
    assert publication["source_asr_asset_version"] == "asr-v4"


def test_speaker_migration_without_unambiguous_asr_fails_closed(
    monkeypatch, tmp_path
):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={
            "speaker": {
                "milvus_asset_version": "speaker-v2",
                "milvus_row_count": 1,
            }
        },
        rows_by_modality={
            "speaker": [
                {"video_id": "video-1", "asset_version": "speaker-v2"},
            ]
        },
    )

    report = migration.migrate(apply=True)

    assert report["migrated"] == []
    assert "no ready or declared ASR" in report["errors"][0]["error"]
    assert catalog.get_modality_publication("video-1", "speaker") is None


def test_explicit_manifest_row_mismatch_fails_closed(monkeypatch, tmp_path):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={
            "asr": {
                "milvus_asset_version": "v3",
                "milvus_row_count": 2,
            }
        },
        rows_by_modality={
            "asr": [{"video_id": "video-1", "asset_version": "v3"}]
        },
    )

    report = migration.migrate(apply=True)

    assert report["migrated"] == []
    assert "manifest rows=2, Milvus rows=1" in report["errors"][0]["error"]
    assert catalog.get_modality_publication("video-1", "asr") is None


def test_visual_invalid_time_bounds_are_pending_then_published_disabled(
    monkeypatch, tmp_path
):
    catalog = _setup(
        monkeypatch,
        tmp_path,
        channels={"visual": {"file": "visual.npz"}},
        rows_by_modality={
            "visual": [
                {
                    "video_id": "video-1",
                    "asset_version": "legacy-visual",
                    "frame_idx": 0,
                    "timestamp_ms": 0,
                    "segment_id": 0,
                    "segment_start_ms": 0,
                    "segment_end_ms": 0,
                }
            ]
        },
    )
    catalog.publish_modality(
        "video-1",
        "visual",
        asset_version="previous-ready-visual",
        row_count=1,
    )

    dry_run = migration.migrate(apply=False)

    assert dry_run["errors"] == []
    assert dry_run["migrated"] == []
    assert dry_run["requires_rebuild"][0]["status"] == "pending"
    assert catalog.get_modality_publication("video-1", "visual")["status"] == "ready"
    assert "visual" in catalog.get_video("video-1")["indexed_modalities"]

    report = migration.migrate(apply=True)

    assert report["errors"] == []
    assert report["migrated"] == []
    rebuild = report["requires_rebuild"][0]
    assert rebuild["asset_version"] == "legacy-visual"
    assert rebuild["status"] == "disabled"
    assert "invalid=1" in rebuild["reason"]

    publication = catalog.get_modality_publication("video-1", "visual")
    assert publication["asset_version"] == "legacy-visual"
    assert publication["row_count"] == 1
    assert publication["status"] == "disabled"
    assert publication["migration_state"] == "requires_rebuild"
    assert publication["reason"] == rebuild["reason"]
    video = catalog.get_video("video-1")
    assert "visual" not in video["indexed_modalities"]

    from app.retrieval.search import _channel_publication_for

    with pytest.raises(ValueError, match="visual 索引尚未发布"):
        _channel_publication_for(video, "visual")


def test_unknown_requested_video_is_an_error(monkeypatch, tmp_path):
    _setup(
        monkeypatch,
        tmp_path,
        channels={"asr": {"file": "asr.npz"}},
        rows_by_modality={},
    )

    report = migration.migrate(apply=False, video_ids={"missing-video"})

    assert report["migrated"] == []
    assert report["errors"] == [
        {
            "video_id": "missing-video",
            "error": "video does not exist in Catalog",
        }
    ]


def test_scan_and_publish_happen_under_normal_publish_lock(
    monkeypatch, tmp_path
):
    _setup(
        monkeypatch,
        tmp_path,
        channels={"asr": {"file": "asr.npz"}},
        rows_by_modality={
            "asr": [{"video_id": "video-1", "asset_version": "v1"}]
        },
    )
    from contextlib import contextmanager
    from app.vector_store.milvus import milvus_stage_lock

    state = {"locked": False, "scanned": False, "published": False}

    @contextmanager
    def observed_lock(index_dir, *, video_id, stage):
        del index_dir
        assert video_id == "video-1"
        assert stage == "publish"
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    original_scan = migration._scan_version_counts
    original_publish = migration.Catalog.publish_modality

    def observed_scan(*args, **kwargs):
        assert state["locked"] is True
        state["scanned"] = True
        return original_scan(*args, **kwargs)

    def observed_publish(self, *args, **kwargs):
        assert state["locked"] is True
        state["published"] = True
        return original_publish(self, *args, **kwargs)

    monkeypatch.setattr(milvus_stage_lock, "video_stage_lock", observed_lock)
    monkeypatch.setattr(migration, "_scan_version_counts", observed_scan)
    monkeypatch.setattr(migration.Catalog, "publish_modality", observed_publish)

    report = migration.migrate(apply=True)

    assert report["errors"] == []
    assert state == {"locked": False, "scanned": True, "published": True}
