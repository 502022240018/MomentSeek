from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

from app.maintenance import migrate_visual_time_bounds as migration
from app.vector_store.milvus.milvus_schema import EMBEDDING_DIMS


NEW_VERSION = "1" * 32


class FakeQueryIterator:
    def __init__(self, rows: list[dict], batch_size: int):
        self.rows = rows
        self.batch_size = batch_size
        self.position = 0
        self.closed = False

    def next(self) -> list[dict]:
        batch = self.rows[self.position : self.position + self.batch_size]
        self.position += len(batch)
        return batch

    def close(self) -> None:
        self.closed = True


class FakeCollection:
    def __init__(self, rows: list[dict], events: list[str]):
        self.rows = deepcopy(rows)
        self.events = events
        self.upsert_calls: list[list[dict]] = []
        self.flush_calls = 0
        self.query_timeouts: list[float] = []
        self.fail_after_upsert_calls: int | None = None

    def query_iterator(
        self,
        *,
        expr: str,
        output_fields: list[str],
        batch_size: int,
        timeout: float,
    ):
        self.query_timeouts.append(timeout)
        quoted = re.findall(r"==\s*(\"(?:\\.|[^\"])*\")", expr)
        video_id, asset_version = (json.loads(item) for item in quoted)
        selected = [
            {field: row.get(field) for field in output_fields}
            for row in self.rows
            if row["video_id"] == video_id and row["asset_version"] == asset_version
        ]
        return FakeQueryIterator(selected, batch_size)

    def upsert(self, rows: list[dict]) -> None:
        if (
            self.fail_after_upsert_calls is not None
            and len(self.upsert_calls) >= self.fail_after_upsert_calls
        ):
            raise RuntimeError("injected interrupted copy")
        copied = deepcopy(rows)
        self.upsert_calls.append(copied)
        by_pk = {row["pk"]: row for row in self.rows}
        by_pk.update({row["pk"]: row for row in copied})
        self.rows = list(by_pk.values())
        self.events.append("upsert")

    def flush(self) -> None:
        self.flush_calls += 1
        self.events.append("flush")


class FakeClient:
    def __init__(self, rows: list[dict], events: list[str]):
        self.collection = FakeCollection(rows, events)
        self.count_overrides: dict[str, int] = {}

    def collection_for(self, modality: str):
        assert modality == "visual"
        return self.collection

    def count_video_modality_version(
        self, video_id: str, modality: str, asset_version: str
    ) -> int:
        assert modality == "visual"
        if asset_version in self.count_overrides:
            return self.count_overrides[asset_version]
        return sum(
            row["video_id"] == video_id and row["asset_version"] == asset_version
            for row in self.collection.rows
        )


class FakeCatalog:
    def __init__(self, publication: dict, events: list[str]):
        self.publication = deepcopy(publication)
        self.events = events
        self.publish_calls: list[dict] = []

    def get_modality_publication(self, video_id: str, modality: str):
        assert video_id == self.publication["video_id"]
        assert modality == "visual"
        return deepcopy(self.publication)

    def publish_modality(self, video_id: str, modality: str, **kwargs):
        self.publish_calls.append(
            {"video_id": video_id, "modality": modality, **deepcopy(kwargs)}
        )
        self.events.append("publish")
        self.publication.update(
            asset_version=kwargs["asset_version"],
            row_count=kwargs["row_count"],
            status=kwargs.get("status", "ready"),
            metadata=deepcopy(kwargs["metadata"]),
        )


def _source_rows(*, mixed: bool = False) -> list[dict]:
    rows = []
    for frame_idx, (timestamp_ms, segment_id) in enumerate(
        ((0, 0), (6000, 1), (11999, 2))
    ):
        rows.append(
            {
                "pk": f"old-{frame_idx}",
                "video_id": "video-1",
                "asset_version": "old-version",
                "model_version": "siglip2-so400m-v1",
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "segment_id": segment_id,
                "segment_start_ms": -1,
                "segment_end_ms": -1,
                "embedding": [float(frame_idx)] * EMBEDDING_DIMS["visual"],
            }
        )
    if mixed:
        rows[0]["segment_start_ms"] = 0
        rows[0]["segment_end_ms"] = 5000
    return rows


def _publication() -> dict:
    return {
        "video_id": "video-1",
        "modality": "visual",
        "asset_version": "old-version",
        "row_count": 3,
        "status": "ready",
        "model_key": "siglip2-so400m-384",
        "embedding_space": "siglip2-image-text",
        "segment_strategy": "fixed",
        "metadata": {
            "model_key": "siglip2-so400m-384",
            "embedding_space": "siglip2-image-text",
            "segment_strategy": "fixed",
        },
    }


def _write_manifest(
    index_dir,
    *,
    strategy: str = "fixed",
    segment_ms=5000,
    include_publication_pointer: bool = True,
):
    index_dir.mkdir(parents=True, exist_ok=True)
    visual = {"segment_strategy": strategy}
    if include_publication_pointer:
        visual.update(
            milvus_asset_version="old-version",
            milvus_row_count=3,
        )
    (index_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "video_id": "video-1",
                "duration_ms": 12000,
                "segment_ms": segment_ms,
                "channels": {"visual": visual},
            }
        ),
        encoding="utf-8",
    )


def _dependencies(tmp_path, *, rows=None):
    events: list[str] = []
    client = FakeClient(rows or _source_rows(), events)
    catalog = FakeCatalog(_publication(), events)
    index_dir = tmp_path / "video-1"
    _write_manifest(index_dir)
    video = {"id": "video-1", "duration": 12.0, "name": "demo.mp4"}
    return catalog, client, index_dir, video, events


def test_default_dry_run_validates_every_row_without_writing(tmp_path):
    catalog, client, index_dir, video, events = _dependencies(tmp_path)

    result = migration.migrate_visual_video(
        catalog=catalog,
        client=client,
        video=video,
        index_dir=index_dir,
    )

    assert result["status"] == "dry_run_ready"
    assert result["row_count"] == 3
    assert client.collection.upsert_calls == []
    assert catalog.publish_calls == []
    assert events == []


def test_manifest_without_version_or_row_count_uses_catalog_and_milvus(tmp_path):
    catalog, client, index_dir, video, events = _dependencies(tmp_path)
    _write_manifest(index_dir, include_publication_pointer=False)

    result = migration.migrate_visual_video(
        catalog=catalog,
        client=client,
        video=video,
        index_dir=index_dir,
        execute=True,
        version_factory=lambda: NEW_VERSION,
    )

    assert result["status"] == "migrated"
    assert result["source_asset_version"] == "old-version"
    assert result["row_count"] == 3
    assert catalog.publish_calls[0]["asset_version"] == NEW_VERSION
    assert events[-1] == "publish"


def test_disabled_requires_rebuild_source_is_repaired_and_published_ready(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    catalog.publication["status"] = "disabled"
    catalog.publication["migration_state"] = "requires_rebuild"
    catalog.publication["metadata"]["migration_state"] = "requires_rebuild"

    result = migration.migrate_visual_video(
        catalog=catalog,
        client=client,
        video=video,
        index_dir=index_dir,
        execute=True,
        version_factory=lambda: NEW_VERSION,
    )

    assert result["status"] == "migrated"
    assert result["source_status"] == "disabled"
    assert catalog.publish_calls[0]["status"] == "ready"
    assert catalog.publish_calls[0]["metadata"]["migration_state"] == "completed"


def test_unmarked_disabled_source_is_rejected(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    catalog.publication["status"] = "disabled"

    with pytest.raises(ValueError, match="unsupported Catalog visual source"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
        )

    assert client.collection.upsert_calls == []
    assert catalog.publish_calls == []


def test_execute_reuses_embeddings_and_publishes_only_after_validation(tmp_path):
    catalog, client, index_dir, video, events = _dependencies(tmp_path)

    result = migration.migrate_visual_video(
        catalog=catalog,
        client=client,
        video=video,
        index_dir=index_dir,
        execute=True,
        batch_size=2,
        version_factory=lambda: NEW_VERSION,
    )

    assert result["status"] == "migrated"
    assert result["asset_version"] == NEW_VERSION
    migrated = sorted(
        (
            row
            for row in client.collection.rows
            if row["asset_version"] == NEW_VERSION
        ),
        key=lambda row: row["frame_idx"],
    )
    assert [
        (row["segment_start_ms"], row["segment_end_ms"]) for row in migrated
    ] == [(0, 5000), (5000, 10000), (10000, 12000)]
    assert migrated[1]["embedding"] == [1.0] * EMBEDDING_DIMS["visual"]
    assert events[-1] == "publish"
    assert events.index("publish") > events.index("flush")
    assert catalog.publish_calls[0]["metadata"]["segment_times"] == "explicit"
    assert (
        catalog.publish_calls[0]["metadata"]["migrated_from_asset_version"]
        == "old-version"
    )
    assert len([row for row in client.collection.rows if row["asset_version"] == "old-version"]) == 3


def test_mixed_source_bounds_fail_before_any_write(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(
        tmp_path, rows=_source_rows(mixed=True)
    )

    with pytest.raises(ValueError, match="mixed valid and invalid"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
        )

    assert client.collection.upsert_calls == []
    assert catalog.publish_calls == []


@pytest.mark.parametrize(
    ("strategy", "segment_ms", "match"),
    [
        ("shot", 5000, "only fixed-window"),
        ("fixed", None, "segment_ms must be a positive integer"),
        ("fixed", 0, "segment_ms must be a positive integer"),
    ],
)
def test_shot_or_missing_fixed_configuration_fails_closed(
    tmp_path, strategy, segment_ms, match
):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    _write_manifest(index_dir, strategy=strategy, segment_ms=segment_ms)

    with pytest.raises(ValueError, match=match):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
        )

    assert client.collection.upsert_calls == []
    assert catalog.publish_calls == []


def test_post_write_count_mismatch_never_publishes(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    client.count_overrides[NEW_VERSION] = 2

    with pytest.raises(RuntimeError, match="persisted rows=2"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
            version_factory=lambda: NEW_VERSION,
        )

    assert catalog.publish_calls == []
    assert catalog.publication["asset_version"] == "old-version"


def test_catalog_pointer_race_never_publishes(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    original_get = catalog.get_modality_publication
    calls = 0

    def changing_publication(video_id, modality):
        nonlocal calls
        calls += 1
        publication = original_get(video_id, modality)
        if calls >= 2:
            publication["asset_version"] = "concurrent-version"
        return publication

    catalog.get_modality_publication = changing_publication

    with pytest.raises(RuntimeError, match="pointer changed"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
            version_factory=lambda: NEW_VERSION,
        )

    assert catalog.publish_calls == []
    assert catalog.publication["asset_version"] == "old-version"


def test_catalog_status_race_never_publishes(tmp_path):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    original_get = catalog.get_modality_publication
    calls = 0

    def changing_publication(video_id, modality):
        nonlocal calls
        calls += 1
        publication = original_get(video_id, modality)
        if calls >= 2:
            publication["status"] = "disabled"
            publication["migration_state"] = "requires_rebuild"
            publication["metadata"]["migration_state"] = "requires_rebuild"
        return publication

    catalog.get_modality_publication = changing_publication

    with pytest.raises(RuntimeError, match="pointer changed"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
            version_factory=lambda: NEW_VERSION,
        )

    assert catalog.publish_calls == []


def test_invalid_segment_id_fails_before_copy(tmp_path):
    rows = _source_rows()
    rows[2]["segment_id"] = 1
    catalog, client, index_dir, video, _ = _dependencies(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="does not match"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
        )

    assert client.collection.upsert_calls == []
    assert catalog.publish_calls == []


def test_default_target_version_is_deterministic_and_retry_resumes_partial_copy(
    tmp_path,
):
    catalog, client, index_dir, video, _ = _dependencies(tmp_path)
    expected_version = migration.target_asset_version(
        video_id="video-1", source_version="old-version", segment_ms=5000
    )
    client.collection.fail_after_upsert_calls = 1

    with pytest.raises(RuntimeError, match="injected interrupted copy"):
        migration.migrate_visual_video(
            catalog=catalog,
            client=client,
            video=video,
            index_dir=index_dir,
            execute=True,
            batch_size=2,
            timeout=7.5,
        )

    partial = [
        row
        for row in client.collection.rows
        if row["asset_version"] == expected_version
    ]
    assert len(partial) == 2
    assert catalog.publication["asset_version"] == "old-version"

    client.collection.fail_after_upsert_calls = None
    result = migration.migrate_visual_video(
        catalog=catalog,
        client=client,
        video=video,
        index_dir=index_dir,
        execute=True,
        batch_size=2,
        timeout=7.5,
    )

    assert result["asset_version"] == expected_version
    assert catalog.publication["asset_version"] == expected_version
    target_versions = {
        row["asset_version"]
        for row in client.collection.rows
        if row["asset_version"] != "old-version"
    }
    assert target_versions == {expected_version}
    assert len(
        [
            row
            for row in client.collection.rows
            if row["asset_version"] == expected_version
        ]
    ) == 3
    assert client.collection.query_timeouts
    assert set(client.collection.query_timeouts) == {7.5}
