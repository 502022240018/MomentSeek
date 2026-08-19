from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.catalog.db import Catalog
from app.core.settings import Settings
from app.maintenance import reindex_milvus_only as migration


def _catalog_with_video(tmp_path, *, source_exists: bool = True):
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
    if source_exists:
        video_path.write_bytes(b"video")
    catalog.create_video(
        {
            "id": "video-1",
            "name": "demo.mp4",
            "file_path": str(video_path),
            "duration": 10.0,
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "status": "ready",
        }
    )
    return settings, catalog


def test_normalize_modalities_orders_stages_and_avoids_duplicate_speaker():
    assert migration.normalize_modalities(None) == ("visual", "face", "asr", "ocr")
    assert migration.normalize_modalities("speaker,asr,visual") == ("visual", "asr")
    assert migration.normalize_modalities("speaker") == ("speaker",)
    with pytest.raises(ValueError, match="不支持"):
        migration.normalize_modalities("visual,unknown")


def test_migration_targets_reports_unknown_requested_video(tmp_path):
    settings, catalog = _catalog_with_video(tmp_path)

    with pytest.raises(ValueError, match="不存在 video_id"):
        migration.migration_targets(catalog, settings, ["missing"])


def test_run_migration_updates_catalog_and_tracks_nested_speaker(monkeypatch, tmp_path):
    settings, catalog = _catalog_with_video(tmp_path)
    targets = migration.migration_targets(catalog, settings)
    calls: list[tuple[str, dict]] = []

    def fake_execute(stage, video, options, passed_settings):
        assert video["id"] == "video-1"
        assert passed_settings is settings
        calls.append((stage, options))
        catalog.publish_modality(
            video["id"],
            stage,
            asset_version=f"test-{stage}",
            row_count=2,
            metadata={"model_key": f"test-{stage}"},
        )
        result = {
            "milvus_asset_version": f"test-{stage}",
            "milvus_row_count": 2,
        }
        if stage == "asr":
            catalog.publish_modality(
                video["id"],
                "speaker",
                asset_version="test-speaker",
                row_count=1,
                metadata={"model_key": "test-speaker"},
            )
            result["speaker"] = {
                "milvus_asset_version": "test-speaker",
                "milvus_row_count": 1,
            }
        return result

    monkeypatch.setattr(migration, "execute_stage", fake_execute)
    results, failed = migration.run_migration(
        catalog, settings, targets, migration.normalize_modalities(None)
    )

    assert failed == 0
    assert results[0]["status"] == "completed"
    assert calls == [
        ("visual", {}),
        ("face", {}),
        ("asr", {"asr_speaker_enabled": True}),
        ("ocr", {}),
    ]
    video = catalog.get_video("video-1")
    assert video["status"] == "ready"
    assert video["indexed_modalities"] == [
        "asr",
        "face",
        "ocr",
        "speaker",
        "visual",
    ]


def test_run_migration_skips_missing_source_without_mutating_catalog(tmp_path):
    settings, catalog = _catalog_with_video(tmp_path, source_exists=False)
    targets = migration.migration_targets(catalog, settings)

    results, failed = migration.run_migration(catalog, settings, targets, ("visual",))

    assert failed == 1
    assert results[0]["status"] == "skipped"
    assert catalog.get_video("video-1")["status"] == "ready"


def test_run_migration_failure_preserves_previous_catalog_availability(
    monkeypatch, tmp_path
):
    settings, catalog = _catalog_with_video(tmp_path)
    catalog.publish_modality(
        "video-1",
        "visual",
        asset_version="existing-visual",
        row_count=4,
    )
    targets = migration.migration_targets(catalog, settings)
    monkeypatch.setattr(
        migration,
        "execute_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Milvus unavailable")
        ),
    )

    results, failed = migration.run_migration(catalog, settings, targets, ("visual",))

    assert failed == 1
    assert results[0]["status"] == "failed"
    assert catalog.get_video("video-1")["status"] == "ready"
    assert catalog.get_video("video-1")["indexed_modalities"] == ["visual"]


def test_verify_published_versions_checks_every_requested_channel(
    monkeypatch, tmp_path
):
    settings, catalog = _catalog_with_video(tmp_path)
    targets = migration.migration_targets(catalog, settings)
    for modality, version, rows in (
        ("visual", "2", 11),
        ("asr", "3", 2),
        ("speaker", "3", 3),
    ):
        catalog.publish_modality(
            "video-1",
            modality,
            asset_version=version,
            row_count=rows,
            metadata={"model_key": f"test-{modality}"},
        )
    client = SimpleNamespace(
        count_video_modality_version=lambda video_id, modality, version: {
            "visual": 11,
            "asr": 2,
            "speaker": 3,
        }[modality]
    )
    import app.vector_store.milvus.milvus_client as milvus_client

    monkeypatch.setattr(milvus_client, "get_milvus_client", lambda: client)

    results, failed = migration.verify_published_versions(
        catalog, settings, targets, ("visual", "asr")
    )

    assert failed == 0
    assert results[0]["status"] == "completed"
    assert set(results[0]["channels"]) == {"visual", "asr", "speaker"}
