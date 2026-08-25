from __future__ import annotations

import numpy as np

from app.catalog.db import Catalog
from app.maintenance.migrate_entity_voice_samples import migrate


def _sample(path, *, sample_id="sample-1", blob=None):
    return {
        "id": sample_id,
        "entity_id": "entity-1",
        "source_type": "video_utterance",
        "source_video_id": "video-1",
        "source_utterance_index": 7,
        "audio_path": None,
        "embedding_path": str(path),
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
        "voice_embedding": blob,
    }


def _catalog(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_entity(
        {
            "id": "entity-1",
            "name": "Alice",
            "reference_path": "",
            "embedding_path": "",
        }
    )
    return catalog


def _private_row(catalog, sample_id="sample-1"):
    with catalog.connect() as connection:
        return dict(
            connection.execute(
                "SELECT * FROM voice_samples WHERE id=?", (sample_id,)
            ).fetchone()
        )


def test_dry_run_is_read_only(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.ones(192, dtype=np.float32))
    catalog.create_voice_sample(_sample(path))

    report = migrate(catalog, entity_ids={"entity-1"})

    assert report["migrated"][0]["status"] == "would_migrate"
    row = _private_row(catalog)
    assert row["voice_embedding"] is None
    assert row["embedding_path"] == str(path)


def test_apply_writes_normalized_blob_and_keeps_backup_file(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.full(192, 2.0, dtype=np.float32))
    catalog.create_voice_sample(_sample(path))

    report = migrate(catalog, apply=True, sample_ids={"sample-1"})

    assert report["migrated"][0]["status"] == "migrated"
    row = _private_row(catalog)
    assert row["embedding_path"] == ""
    vector = np.frombuffer(row["voice_embedding"], dtype=np.float32)
    assert vector.shape == (192,)
    assert np.linalg.norm(vector) == np.float32(1.0)
    assert path.exists()


def test_apply_is_idempotent(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.ones(192, dtype=np.float32))
    catalog.create_voice_sample(_sample(path))

    first = migrate(catalog, apply=True)
    second = migrate(catalog, apply=True)

    assert first["errors"] == []
    assert second["skipped"] == [
        {"sample_id": "sample-1", "status": "already_migrated"}
    ]


def test_corrupt_legacy_file_fails_without_clearing_pointer(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.zeros(192, dtype=np.float32))
    catalog.create_voice_sample(_sample(path))

    report = migrate(catalog, apply=True)

    assert "零向量" in report["errors"][0]["error"]
    row = _private_row(catalog)
    assert row["voice_embedding"] is None
    assert row["embedding_path"] == str(path)


def test_corrupt_existing_blob_is_not_accepted_or_cleared(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.ones(192, dtype=np.float32))
    catalog.create_voice_sample(
        _sample(path, blob=np.zeros(192, dtype=np.float32).tobytes())
    )

    report = migrate(catalog, apply=True)

    assert "零向量" in report["errors"][0]["error"]
    assert _private_row(catalog)["embedding_path"] == str(path)


def test_requested_missing_sample_fails_closed(tmp_path):
    catalog = _catalog(tmp_path)

    report = migrate(catalog, sample_ids={"missing"})

    assert report["errors"] == [
        {"sample_id": "missing", "error": "未找到声音样本"}
    ]


def test_combined_filters_do_not_misreport_existing_entity(tmp_path):
    catalog = _catalog(tmp_path)
    path = tmp_path / "legacy.npz"
    np.savez(path, embedding=np.ones(192, dtype=np.float32))
    catalog.create_voice_sample(_sample(path))

    report = migrate(
        catalog,
        entity_ids={"entity-1"},
        sample_ids={"missing"},
    )

    assert report["errors"] == [
        {"sample_id": "missing", "error": "未找到声音样本"}
    ]
