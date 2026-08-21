from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.catalog.db import Catalog
from app.core.settings import Settings
from app.identity.face_gallery import (
    cluster_face_tracks,
    face_group_model_version,
)
from app.maintenance.migrate_face_groups import (
    migrate_face_groups_video,
    refine_group_representatives,
)
from app.vector_store.milvus import milvus_client as milvus_client_module


def _unit(*values: float) -> list[float]:
    vector = np.zeros(512, dtype=np.float32)
    vector[:len(values)] = values
    return (vector / np.linalg.norm(vector)).tolist()


class FakeIterator:
    def __init__(self, rows):
        self.pages = iter(([dict(row) for row in rows], []))

    def next(self):
        return next(self.pages)

    def close(self):
        pass


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def query_iterator(self, **_kwargs):
        return FakeIterator(self.rows)

    def delete(self, _expr):
        deleted = len(self.rows)
        self.rows.clear()
        return SimpleNamespace(delete_count=deleted)

    def flush(self):
        pass


class FakeClient:
    def __init__(self, track_rows):
        self.track_rows = track_rows
        self.group_rows = []

    def collection_for(self, modality):
        assert modality == "face"
        return FakeCollection(self.track_rows)

    def collection(self, name):
        assert name == "face_groups"
        return FakeCollection(self.group_rows)

    def count_video_modality_version(self, _video_id, modality, _asset_version):
        assert modality == "face"
        return len(self.track_rows)

    def count_face_groups_version(self, _video_id, _asset_version, _group_version):
        return len(self.group_rows)


def _catalog(tmp_path, *, publish=True):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    catalog.create_video({
        "id": "video-1",
        "name": "video.mp4",
        "file_path": str(video_path),
        "duration": 10,
        "fps": 25,
        "width": 640,
        "height": 480,
        "status": "ready",
    })
    if publish:
        catalog.publish_modality(
            "video-1",
            "face",
            asset_version="face-v1",
            row_count=2,
            metadata={"model_key": "buffalo_l", "provider": "cpu"},
        )
    return catalog


@pytest.fixture
def track_rows():
    return [
        {
            "track_idx": 0,
            "start_ms": 0,
            "end_ms": 4_000,
            "best_ms": 1_000,
            "embedding": _unit(1.0, 0.0),
        },
        {
            "track_idx": 1,
            "start_ms": 5_000,
            "end_ms": 8_000,
            "best_ms": 6_000,
            "embedding": _unit(0.0, 1.0),
        },
    ]


def _settings(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
    )
    settings.ensure_dirs()
    return settings


def _legacy_rows(track_rows, *, asset_version="3", model_version="legacy-face-v1"):
    return [
        {
            **row,
            "asset_version": asset_version,
            "model_version": model_version,
        }
        for row in track_rows
    ]


def _install_group_writer(monkeypatch, client):
    def fake_upsert(_ctx, **arrays):
        vectors = arrays["group_embeddings"]
        times = arrays["group_times_ms"]
        client.group_rows = [
            {
                "group_idx": index,
                "representative_track_idx": int(
                    arrays["group_track_indices"][index]
                ),
                "start_ms": int(times[index, 0]),
                "end_ms": int(times[index, 1]),
                "best_ms": int(times[index, 2]),
                "bbox_x1": float(arrays["group_bboxes"][index, 0]),
                "bbox_y1": float(arrays["group_bboxes"][index, 1]),
                "bbox_x2": float(arrays["group_bboxes"][index, 2]),
                "bbox_y2": float(arrays["group_bboxes"][index, 3]),
                "representative_quality": float(
                    arrays["group_qualities"][index]
                ),
                "duration_ms": int(arrays["group_durations_ms"][index]),
                "occurrence_count": int(
                    arrays["group_occurrence_counts"][index]
                ),
                "importance_score": float(
                    arrays["group_importance_scores"][index]
                ),
                "embedding": vectors[index].tolist(),
            }
            for index in range(len(vectors))
        ]
        return len(vectors)

    monkeypatch.setattr(
        "app.maintenance.migrate_face_groups.upsert_face_group_rows",
        fake_upsert,
    )


def test_existing_collections_client_isolated_to_explicit_collection_set(monkeypatch):
    events = []
    monkeypatch.setattr(
        milvus_client_module.connections,
        "connect",
        lambda **kwargs: events.append(("connect", kwargs)),
    )
    monkeypatch.setattr(
        milvus_client_module.connections,
        "disconnect",
        lambda alias: events.append(("disconnect", alias)),
    )
    checked = []
    monkeypatch.setattr(
        milvus_client_module.utility,
        "has_collection",
        lambda name, using: checked.append((name, using)) or True,
    )
    monkeypatch.setattr(
        milvus_client_module,
        "Collection",
        lambda name, using: (name, using),
    )

    client = milvus_client_module.ExistingMilvusCollectionsClient(
        ("face_embeddings", "face_groups")
    )

    assert [name for name, _alias in checked] == [
        "face_embeddings",
        "face_groups",
    ]
    assert client.collection_for("face")[0] == "face_embeddings"
    with pytest.raises(ValueError, match="not authorized"):
        client.collection("asr_embeddings")
    client.close()
    assert events[0][0] == "connect"
    assert events[-1][0] == "disconnect"


def test_bootstrap_is_explicit_and_dry_run_does_not_publish(tmp_path, track_rows):
    catalog = _catalog(tmp_path, publish=False)
    client = FakeClient(_legacy_rows(track_rows))
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="no ready Face publication"):
        migrate_face_groups_video(
            catalog=catalog,
            client=client,
            video=catalog.get_video("video-1"),
            settings=settings,
            apply=False,
        )

    result = migrate_face_groups_video(
        catalog=catalog,
        client=client,
        video=catalog.get_video("video-1"),
        settings=settings,
        apply=False,
        bootstrap_legacy_publication=True,
    )

    assert result["status"] == "dry_run_ready"
    assert result["asset_version"] == "3"
    assert result["publication_bootstrapped"] is True
    assert catalog.get_modality_publication("video-1", "face") is None


def test_bootstrap_apply_publishes_verified_legacy_generation(
    monkeypatch, tmp_path, track_rows
):
    catalog = _catalog(tmp_path, publish=False)
    client = FakeClient(_legacy_rows(track_rows))
    settings = _settings(tmp_path)
    _install_group_writer(monkeypatch, client)

    result = migrate_face_groups_video(
        catalog=catalog,
        client=client,
        video=catalog.get_video("video-1"),
        settings=settings,
        apply=True,
        bootstrap_legacy_publication=True,
        refine_representatives=False,
    )

    assert result["status"] == "migrated"
    publication = catalog.get_modality_publication("video-1", "face")
    assert publication["asset_version"] == "3"
    assert publication["row_count"] == 2
    assert publication["publication_bootstrapped"] is True
    assert publication["legacy_track_model_version"] == "legacy-face-v1"
    assert publication["model_key"] == settings.face_model


def test_bootstrap_rejects_ambiguous_legacy_asset_versions(tmp_path, track_rows):
    catalog = _catalog(tmp_path, publish=False)
    rows = _legacy_rows(track_rows)
    rows[1]["asset_version"] = "4"

    with pytest.raises(ValueError, match="asset version is ambiguous"):
        migrate_face_groups_video(
            catalog=catalog,
            client=FakeClient(rows),
            video=catalog.get_video("video-1"),
            settings=_settings(tmp_path),
            apply=False,
            bootstrap_legacy_publication=True,
        )


def test_dry_run_reads_tracks_without_writing_or_publishing(monkeypatch, tmp_path, track_rows):
    catalog = _catalog(tmp_path)
    client = FakeClient(track_rows)
    monkeypatch.setattr(
        "app.maintenance.migrate_face_groups.upsert_face_group_rows",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not write"),
    )

    result = migrate_face_groups_video(
        catalog=catalog,
        client=client,
        video=catalog.get_video("video-1"),
        settings=_settings(tmp_path),
        apply=False,
    )

    assert result["status"] == "dry_run_ready"
    assert result["group_row_count"] == 2
    assert catalog.get_modality_publication("video-1", "face").get("group_version") is None


def test_apply_validates_rows_then_publishes_group_pointer(monkeypatch, tmp_path, track_rows):
    catalog = _catalog(tmp_path)
    client = FakeClient(track_rows)
    _install_group_writer(monkeypatch, client)

    result = migrate_face_groups_video(
        catalog=catalog,
        client=client,
        video=catalog.get_video("video-1"),
        settings=_settings(tmp_path),
        apply=True,
        refine_representatives=False,
    )

    assert result["status"] == "migrated"
    publication = catalog.get_modality_publication("video-1", "face")
    assert publication["asset_version"] == "face-v1"
    assert publication["row_count"] == 2
    assert publication["group_row_count"] == 2
    assert publication["group_source"] == "legacy-face-tracks"
    assert publication["provider"] == "cpu"


def test_publication_change_after_write_refuses_publish(monkeypatch, tmp_path, track_rows):
    base = _catalog(tmp_path)
    client = FakeClient(track_rows)

    class RacingCatalog:
        def __init__(self):
            self.reads = 0
            self.published = False

        def get_modality_publication(self, *_args):
            self.reads += 1
            publication = base.get_modality_publication("video-1", "face")
            if self.reads > 1:
                return {**publication, "asset_version": "new-face-version"}
            return publication

        def publish_modality(self, *_args, **_kwargs):
            self.published = True

    racing = RacingCatalog()
    _install_group_writer(monkeypatch, client)
    with pytest.raises(RuntimeError, match="changed during migration"):
        migrate_face_groups_video(
            catalog=racing,
            client=client,
            video=base.get_video("video-1"),
            settings=_settings(tmp_path),
            apply=True,
            refine_representatives=False,
        )
    assert racing.published is False


def test_replace_refuses_to_mutate_published_immutable_generation(
    monkeypatch, tmp_path, track_rows
):
    catalog = _catalog(tmp_path)
    settings = _settings(tmp_path)
    group_version = face_group_model_version(
        settings.face_gallery_cosine_threshold
    )
    catalog.publish_modality(
        "video-1",
        "face",
        asset_version="face-v1",
        row_count=2,
        metadata={
            "group_version": group_version,
            "group_row_count": 2,
        },
    )
    monkeypatch.setattr(
        "app.maintenance.migrate_face_groups.upsert_face_group_rows",
        lambda *_args, **_kwargs: pytest.fail("published generation must not mutate"),
    )

    with pytest.raises(ValueError, match="published immutable"):
        migrate_face_groups_video(
            catalog=catalog,
            client=FakeClient(track_rows),
            video=catalog.get_video("video-1"),
            settings=settings,
            apply=True,
            replace_existing=True,
            refine_representatives=False,
        )


def test_replace_clears_only_unpublished_target_generation(
    monkeypatch, tmp_path, track_rows
):
    catalog = _catalog(tmp_path, publish=False)
    client = FakeClient(_legacy_rows(track_rows))
    client.group_rows = [{"group_idx": 99}]
    _install_group_writer(monkeypatch, client)

    result = migrate_face_groups_video(
        catalog=catalog,
        client=client,
        video=catalog.get_video("video-1"),
        settings=_settings(tmp_path),
        apply=True,
        replace_existing=True,
        bootstrap_legacy_publication=True,
        refine_representatives=False,
    )

    assert result["status"] == "migrated"
    assert result["replaced_group_rows"] == 1
    assert len(client.group_rows) == 2


def test_representative_refinement_uses_identity_match_and_normalized_bbox():
    groups = cluster_face_tracks(
        np.asarray([_unit(1.0, 0.0)], dtype=np.float32),
        np.asarray([[0, 5_000, 1_000]], dtype=np.int64),
    )
    matching_face = SimpleNamespace(
        normed_embedding=np.asarray(_unit(1.0, 0.0), dtype=np.float32),
        bbox=np.asarray([20, 10, 80, 90], dtype=np.float32),
        det_score=0.95,
    )
    other_face = SimpleNamespace(
        normed_embedding=np.asarray(_unit(0.0, 1.0), dtype=np.float32),
        bbox=np.asarray([0, 0, 20, 20], dtype=np.float32),
        det_score=0.99,
    )
    encoder = SimpleNamespace(detect=lambda _frame: [other_face, matching_face])

    refined, count = refine_group_representatives(
        groups,
        frame_at_ms=lambda _ms: np.zeros((100, 100, 3), dtype=np.uint8),
        encoder=encoder,
        identity_threshold=0.35,
        max_groups=24,
        min_duration_ms=3_000,
        min_occurrence_count=3,
    )

    assert count == 1
    assert refined[0].bbox == pytest.approx((0.2, 0.1, 0.8, 0.9))
    assert refined[0].quality > groups[0].quality
