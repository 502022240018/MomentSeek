from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from app.maintenance import migrate_entity_face_samples as migration
from app.vector_store.milvus.milvus_schema import entity_face_sample_pk


class _DeleteResult:
    def __init__(self, count: int):
        self.delete_count = count


class _Collection:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = list(rows or [])
        self.upsert_calls: list[list[dict]] = []
        self.delete_calls: list[str] = []
        self.fail_upsert = False
        self.upsert_embedding_override = None
        self.partial_delete = False
        self.fail_delete = False

    @staticmethod
    def _entity_id(expr: str) -> str:
        return json.loads(expr.split("==", 1)[1].strip())

    def query(self, *, expr, output_fields, limit, timeout):
        del limit, timeout
        entity_id = self._entity_id(expr)
        return [
            {field: row[field] for field in output_fields if field in row}
            for row in self.rows
            if row["entity_id"] == entity_id
        ]

    def upsert(self, rows):
        if self.fail_upsert:
            raise RuntimeError("injected upsert failure")
        copied = [dict(row) for row in rows]
        if self.upsert_embedding_override is not None:
            for row in copied:
                row["embedding"] = list(self.upsert_embedding_override)
        self.upsert_calls.append(copied)
        for row in copied:
            self.rows = [item for item in self.rows if item["pk"] != row["pk"]]
            self.rows.append(row)

    def delete(self, expr):
        self.delete_calls.append(expr)
        if self.fail_delete:
            raise RuntimeError("injected delete failure")
        if " in " in expr:
            pks = set(json.loads(expr.split(" in ", 1)[1]))
        else:
            pks = {json.loads(expr.split("==", 1)[1].strip())}
        if self.partial_delete and len(pks) > 1:
            pks = {sorted(pks)[0]}
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["pk"] not in pks]
        return _DeleteResult(before - len(self.rows))

    def flush(self):
        return None


class _Client:
    def __init__(self, collection: _Collection):
        self.face_samples = collection

    def collection(self, name: str):
        assert name == "entity_face_samples"
        return self.face_samples


class _Catalog:
    def __init__(self, entities: list[dict]):
        self.entities = [dict(entity) for entity in entities]
        self.update_calls: list[tuple[str, str]] = []

    def list_entities(self):
        return [dict(entity) for entity in self.entities]

    def update_entity_embedding(self, entity_id: str, embedding_path: str):
        self.update_calls.append((entity_id, embedding_path))
        for entity in self.entities:
            if entity["id"] == entity_id:
                entity["embedding_path"] = embedding_path
                return
        raise KeyError(entity_id)


class _Encoder:
    def __init__(self, vector=None, error: Exception | None = None):
        self.vector = vector
        self.error = error
        self.calls: list[str] = []

    def encode_reference(self, path: str):
        self.calls.append(path)
        if self.error:
            raise self.error
        return self.vector


def _settings(tmp_path):
    return SimpleNamespace(
        db_path=tmp_path / "catalog.sqlite3",
        milvus_query_timeout_seconds=3.0,
        face_model="buffalo_l",
        face_provider="cpu",
        npu_device_id=0,
        app_model_dir=tmp_path / "models",
        face_ort_intra_op_threads=2,
        face_ort_inter_op_threads=1,
    )


def _entity(reference_path, *, embedding_path="legacy-face.npz"):
    return {
        "id": "entity-1",
        "name": "Alice",
        "reference_path": str(reference_path),
        "embedding_path": embedding_path,
    }


def _old_sample(pk="old-pk"):
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return {
        "pk": pk,
        "entity_id": "entity-1",
        "sample_id": "old-sample",
        "source_video_id": "",
        "source_asset_version": "",
        "source_group_idx": -1,
        "quality": 1.0,
        "embedding": embedding.tolist(),
    }


def test_dry_run_never_encodes_writes_or_clears_legacy_path(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection()
    encoder = _Encoder(np.ones(512, dtype=np.float32))

    report = migration.migrate(
        apply=False,
        catalog=catalog,
        client=_Client(collection),
        encoder=encoder,
        settings=_settings(tmp_path),
    )

    assert report["errors"] == []
    assert report["migrated"][0]["status"] == "would_migrate"
    assert report["migrated"][0]["would_clear_embedding_path"] is True
    assert encoder.calls == []
    assert collection.upsert_calls == []
    assert catalog.update_calls == []


def test_apply_reencodes_reference_normalizes_and_clears_path(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"not-an-npz")
    catalog = _Catalog([_entity(reference, embedding_path="missing-old.npz")])
    collection = _Collection()
    raw = np.zeros(512, dtype=np.float32)
    raw[0], raw[1] = 3.0, 4.0
    encoder = _Encoder(raw)

    report = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=encoder,
        settings=_settings(tmp_path),
    )

    assert report["errors"] == []
    assert report["migrated"][0]["status"] == "migrated"
    assert catalog.update_calls == [("entity-1", "")]
    assert encoder.calls == [str(reference)]
    row = collection.rows[0]
    sample_id = migration._reference_sample_id("entity-1")
    assert row["pk"] == entity_face_sample_pk("entity-1", sample_id)
    assert row["source_video_id"] == ""
    assert row["source_asset_version"] == ""
    assert row["source_group_idx"] == -1
    assert row["quality"] == 1.0
    np.testing.assert_allclose(np.linalg.norm(row["embedding"]), 1.0)
    np.testing.assert_allclose(row["embedding"][:2], [0.6, 0.8])


def test_existing_sample_skips_encoder_but_can_detach_legacy_path(tmp_path):
    catalog = _Catalog([_entity(tmp_path / "does-not-need-to-exist.jpg")])
    collection = _Collection([_old_sample()])
    encoder = _Encoder(error=AssertionError("must not encode"))

    first = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=encoder,
        settings=_settings(tmp_path),
    )
    second = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=encoder,
        settings=_settings(tmp_path),
    )

    assert first["errors"] == second["errors"] == []
    assert first["skipped"][0]["embedding_path_cleared"] is True
    assert second["skipped"][0]["embedding_path_cleared"] is False
    assert encoder.calls == []
    assert collection.upsert_calls == []
    assert catalog.update_calls == [("entity-1", "")]


def test_invalid_existing_samples_do_not_clear_path_without_replace(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    invalid = _old_sample()
    invalid["embedding"] = [0.0] * 512
    collection = _Collection([invalid])

    report = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(error=AssertionError("must require replace")),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert report["skipped"] == []
    assert "none contains a usable normalized 512-d embedding" in report["errors"][0]["error"]
    assert "--replace" in report["errors"][0]["error"]
    assert catalog.update_calls == []
    assert collection.upsert_calls == []


def test_encoding_failure_keeps_legacy_path(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection()

    report = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(error=ValueError("no face")),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert report["errors"][0]["error"] == "no face"
    assert catalog.update_calls == []
    assert collection.rows == []


def test_replace_writes_and_verifies_before_deleting_initial_samples(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection([_old_sample()])

    report = migration.migrate(
        apply=True,
        replace=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(np.ones(512, dtype=np.float32)),
        settings=_settings(tmp_path),
    )

    assert report["errors"] == []
    assert report["migrated"][0]["status"] == "replaced"
    assert report["migrated"][0]["removed_sample_count"] == 1
    assert [row["sample_id"] for row in collection.rows] == [
        migration._reference_sample_id("entity-1")
    ]
    assert catalog.update_calls == [("entity-1", "")]


def test_replace_upsert_failure_preserves_old_sample_and_path(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection([_old_sample()])
    collection.fail_upsert = True

    report = migration.migrate(
        apply=True,
        replace=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(np.ones(512, dtype=np.float32)),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert report["errors"][0]["error"] == "injected upsert failure"
    assert [row["pk"] for row in collection.rows] == ["old-pk"]
    assert len(collection.delete_calls) == 1
    assert "pk ==" in collection.delete_calls[0]
    assert catalog.update_calls == []


def test_corrupted_writeback_embedding_is_rejected_and_path_is_preserved(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection()
    wrong = np.zeros(512, dtype=np.float32)
    wrong[1] = 1.0
    collection.upsert_embedding_override = wrong
    expected = np.zeros(512, dtype=np.float32)
    expected[0] = 1.0

    report = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(expected),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert "does not match encoded value" in report["errors"][0]["error"]
    assert catalog.update_calls == []


def test_failed_target_cleanup_cannot_be_blindly_accepted_on_retry(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection()
    wrong = np.zeros(512, dtype=np.float32)
    wrong[1] = 1.0
    expected = np.zeros(512, dtype=np.float32)
    expected[0] = 1.0
    collection.upsert_embedding_override = wrong
    collection.fail_delete = True

    first = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(expected),
        settings=_settings(tmp_path),
    )

    assert "cleanup of unverified target failed" in first["errors"][0]["error"]
    assert len(collection.rows) == 1
    assert catalog.update_calls == []

    collection.fail_delete = False
    collection.upsert_embedding_override = None
    second = migration.migrate(
        apply=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(expected),
        settings=_settings(tmp_path),
    )

    assert second["migrated"] == []
    assert "does not match encoded value" in second["errors"][0]["error"]
    assert catalog.update_calls == []


def test_replace_partial_delete_is_error_and_keeps_legacy_path(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])
    collection = _Collection([_old_sample("old-a"), _old_sample("old-b")])
    collection.partial_delete = True

    report = migration.migrate(
        apply=True,
        replace=True,
        catalog=catalog,
        client=_Client(collection),
        encoder=_Encoder(np.ones(512, dtype=np.float32)),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert "delete_count does not match" in report["errors"][0]["error"]
    assert catalog.update_calls == []
    assert any(row["pk"] in {"old-a", "old-b"} for row in collection.rows)


def test_unknown_requested_entity_is_an_error(tmp_path):
    reference = tmp_path / "alice.jpg"
    reference.write_bytes(b"image")
    catalog = _Catalog([_entity(reference)])

    report = migration.migrate(
        apply=False,
        entity_ids={"missing-entity"},
        catalog=catalog,
        client=_Client(_Collection()),
        encoder=_Encoder(error=AssertionError("dry-run must not encode")),
        settings=_settings(tmp_path),
    )

    assert report["migrated"] == []
    assert report["errors"] == [
        {
            "entity_id": "missing-entity",
            "error": "entity does not exist in Catalog",
        }
    ]
