from types import SimpleNamespace

import pytest

from app.identity.face_gallery_service import (
    FaceGroupMigrationRequired,
    _query_all,
    attach_group_to_entity,
    published_face_generation,
    video_face_groups,
)


class FakeIterator:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.closed = False

    def next(self):
        return next(self.pages)

    def close(self):
        self.closed = True


class FakeCollection:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.iterator = None

    def query_iterator(self, **kwargs):
        self.calls.append(kwargs)
        self.iterator = FakeIterator(self.pages)
        return self.iterator


def test_query_all_reads_every_face_page_and_closes_iterator():
    collection = FakeCollection([
        [{"track_idx": 0}, {"track_idx": 1}],
        [{"track_idx": 2}],
    ])

    rows = _query_all(
        collection,
        expr='video_id == "video-1"',
        output_fields=["track_idx"],
    )

    assert [row["track_idx"] for row in rows] == [0, 1, 2]
    assert collection.calls[0]["batch_size"] == 2000
    assert collection.iterator.closed is True


class FakeCatalog:
    def __init__(self, publication):
        self.publication = publication

    def get_modality_publication(self, _video_id, _modality):
        return self.publication

    def face_identity_bindings(self, _video_id, _asset_version, _group_version):
        return {}


def test_missing_group_pointer_requires_offline_migration():
    catalog = FakeCatalog({
        "status": "ready",
        "asset_version": "face-v1",
        "row_count": 3,
    })
    with pytest.raises(FaceGroupMigrationRequired, match="一次性迁移"):
        published_face_generation(catalog, "video-1")


def test_gallery_is_read_only_validates_count_and_caps_major_candidates(monkeypatch):
    rows = [
        {
            "group_idx": 0,
            "start_ms": 0,
            "end_ms": 10_000,
            "best_ms": 1_000,
            "duration_ms": 10_000,
            "occurrence_count": 10,
            "importance_score": 4.0,
        },
        {
            "group_idx": 1,
            "start_ms": 11_000,
            "end_ms": 11_400,
            "best_ms": 11_100,
            "duration_ms": 400,
            "occurrence_count": 1,
            "importance_score": 2.0,
        },
        {
            "group_idx": 2,
            "start_ms": 12_000,
            "end_ms": 12_500,
            "best_ms": 12_100,
            "duration_ms": 500,
            "occurrence_count": 4,
            "importance_score": 3.0,
        },
    ]
    collection = SimpleNamespace()
    client = SimpleNamespace(collection=lambda _name: collection)
    monkeypatch.setattr(
        "app.identity.face_gallery_service.get_milvus_client",
        lambda: client,
    )
    monkeypatch.setattr(
        "app.identity.face_gallery_service._query_all",
        lambda *_args, **_kwargs: [dict(row) for row in rows],
    )
    catalog = FakeCatalog({
        "status": "ready",
        "asset_version": "face-v1",
        "group_version": "major-people-v2:cosine=0.520",
        "group_row_count": 3,
    })

    result = video_face_groups(
        catalog,
        "video-1",
        limit=1,
        min_duration_ms=3_000,
        min_occurrence_count=3,
    )

    assert result["total_group_count"] == 3
    assert result["eligible_group_count"] == 2
    assert result["displayed_group_count"] == 1
    assert [row["group_idx"] for row in result["groups"]] == [0]
    assert "group_version=" in result["groups"][0]["thumbnail_url"]


def test_gallery_refuses_partial_published_generation(monkeypatch):
    monkeypatch.setattr(
        "app.identity.face_gallery_service.get_milvus_client",
        lambda: SimpleNamespace(collection=lambda _name: object()),
    )
    monkeypatch.setattr(
        "app.identity.face_gallery_service._query_all",
        lambda *_args, **_kwargs: [{"group_idx": 0}],
    )
    catalog = FakeCatalog({
        "status": "ready",
        "asset_version": "face-v1",
        "group_version": "groups-v2",
        "group_row_count": 2,
    })
    with pytest.raises(RuntimeError, match="publication mismatch"):
        video_face_groups(
            catalog,
            "video-1",
            limit=24,
            min_duration_ms=3_000,
            min_occurrence_count=3,
        )


def test_binding_replaces_only_the_same_group_generation(monkeypatch):
    class SampleCollection:
        def __init__(self):
            self.deleted_expr = None
            self.rows = []

        def delete(self, expr):
            self.deleted_expr = expr

        def upsert(self, rows):
            self.rows.extend(rows)

        def flush(self):
            pass

    class BindingCatalog:
        def __init__(self):
            self.bound = None

        def get_entity(self, entity_id):
            return {"id": entity_id, "name": "Alice"}

        def bind_face_identity(self, *args):
            self.bound = args

    collection = SampleCollection()
    monkeypatch.setattr(
        "app.identity.face_gallery_service.get_milvus_client",
        lambda: SimpleNamespace(collection=lambda _name: collection),
    )
    monkeypatch.setattr(
        "app.identity.face_gallery_service.get_face_group",
        lambda *_args, **_kwargs: {
            "representative_quality": 0.9,
            "embedding": [1.0, *([0.0] * 511)],
        },
    )
    catalog = BindingCatalog()

    result = attach_group_to_entity(
        catalog,
        "video-1",
        "face-v1",
        "groups-v2",
        7,
        "entity-1",
    )

    assert collection.deleted_expr == f'sample_id == "{result["sample_id"]}"'
    assert collection.rows[0]["sample_id"] == result["sample_id"]
    assert catalog.bound == (
        "video-1",
        "face-v1",
        "groups-v2",
        7,
        "entity-1",
    )
