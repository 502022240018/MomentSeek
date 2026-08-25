import sqlite3

import pytest

from app.catalog.db import Catalog


def test_legacy_face_binding_schema_is_migrated_without_reusing_binding(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """CREATE TABLE videos (id TEXT PRIMARY KEY);
               CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT);
               INSERT INTO videos(id) VALUES('video-1');
               INSERT INTO entities(id,name) VALUES('entity-1','Alice');
               CREATE TABLE face_identity_bindings (
                 video_id TEXT NOT NULL,
                 asset_version TEXT NOT NULL,
                 group_idx INTEGER NOT NULL,
                 entity_id TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY(video_id,asset_version,group_idx)
               );
               INSERT INTO face_identity_bindings(
                 video_id,asset_version,group_idx,entity_id
               ) VALUES('video-1','face-v1',3,'entity-1');"""
        )

    catalog = Catalog(path)

    assert catalog.face_identity_bindings("video-1", "face-v1", "groups-v2") == {}
    with catalog.connect() as connection:
        legacy = connection.execute(
            "SELECT group_version,group_idx,entity_id FROM face_identity_bindings"
        ).fetchone()
    assert dict(legacy) == {
        "group_version": "",
        "group_idx": 3,
        "entity_id": "entity-1",
    }


def test_catalog_video_job_and_entity_roundtrip(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    video = catalog.create_video({
        "id": "video-1",
        "name": "demo.mp4",
        "file_path": str(tmp_path / "demo.mp4"),
        "duration": 12.5,
        "fps": 25,
        "width": 1280,
        "height": 720,
        "status": "uploaded",
    })
    assert video["indexed_modalities"] == []

    catalog.update_video("video-1", status="ready")
    catalog.publish_modality(
        "video-1", "visual", asset_version="visual-v1", row_count=2
    )
    catalog.publish_modality(
        "video-1", "asr", asset_version="asr-v1", row_count=3
    )
    assert catalog.get_video("video-1")["indexed_modalities"] == ["asr", "visual"]

    job = catalog.create_job({
        "id": "job-1",
        "video_id": "video-1",
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": ["visual"],
        "options": {"visual_sample_fps": 1},
    })
    assert job["modalities"] == ["visual"]
    assert catalog.claim_queued_job("job-1", worker_pid=123) is True
    assert catalog.claim_queued_job("job-1", worker_pid=456) is False
    catalog.update_job("job-1", status="completed", progress=1)
    assert catalog.get_job("job-1")["progress"] == 1

    catalog.create_entity({
        "id": "entity-1",
        "name": "Neymar",
        "reference_path": "neymar.jpg",
        "embedding_path": "neymar.npz",
    })
    assert catalog.find_entity_in_text("find Neymar on the field")["id"] == "entity-1"


def test_update_video_cannot_bypass_publication_control(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video(
        {
            "id": "video-1",
            "name": "demo.mp4",
            "file_path": str(tmp_path / "demo.mp4"),
            "duration": 1,
            "fps": 25,
            "width": 640,
            "height": 480,
            "status": "uploaded",
        }
    )

    with pytest.raises(ValueError, match="publish_modality"):
        catalog.update_video(
            "video-1", status="ready", indexed_modalities=["visual"]
        )

    video = catalog.get_video("video-1")
    assert video["status"] == "uploaded"
    assert video["indexed_modalities"] == []
    assert video["index_publications"] == {}


def test_rename_and_delete_video_removes_jobs(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    for video_id in ("video-1", "video-2"):
        catalog.create_video({
            "id": video_id, "name": f"{video_id}.mp4", "file_path": str(tmp_path / f"{video_id}.mp4"),
            "duration": 5, "fps": 25, "width": 640, "height": 480, "status": "uploaded",
        })
        catalog.create_job({
            "id": f"job-{video_id}", "video_id": video_id, "status": "completed",
            "stage": "completed", "progress": 1, "modalities": ["asr"], "options": {},
        })

    catalog.update_video("video-1", name="renamed.mp4")
    assert catalog.get_video("video-1")["name"] == "renamed.mp4"

    assert catalog.delete_video("video-1") is True
    assert catalog.get_video("video-1") is None
    assert catalog.get_job("job-video-1") is None
    # unrelated video and its job are untouched
    assert catalog.get_video("video-2") is not None
    assert catalog.get_job("job-video-2") is not None
    # deleting a missing video reports no row removed
    assert catalog.delete_video("video-1") is False


def test_video_folder_memberships_are_many_to_many_and_non_destructive(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    for video_id in ("video-1", "video-2"):
        catalog.create_video({"id": video_id, "name": f"{video_id}.mp4", "file_path": str(tmp_path / f"{video_id}.mp4"), "duration": 5, "fps": 25, "width": 640, "height": 480, "status": "uploaded"})
    campaign, interview = catalog.create_folder("Campaign"), catalog.create_folder("Interview")
    catalog.update_video_folders(["video-1"], [campaign["id"], interview["id"]], "replace")
    catalog.update_video_folders(["video-2"], [campaign["id"]], "add")
    assert {folder["name"] for folder in catalog.get_video("video-1")["folders"]} == {"Campaign", "Interview"}
    assert catalog.delete_folder(campaign["id"]) == 2
    assert catalog.get_video("video-1") is not None
    assert catalog.get_video("video-1")["folder_ids"] == [interview["id"]]
    catalog.update_video_folders(["video-1"], [], "replace")
    assert catalog.get_video("video-1")["folder_ids"] == []


def test_folder_scope_unions_explicit_assets_and_preserves_empty_folder(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    for video_id in ("video-1", "video-2", "video-3"):
        catalog.create_video({"id": video_id, "name": f"{video_id}.mp4", "file_path": str(tmp_path / f"{video_id}.mp4"), "duration": 5, "fps": 25, "width": 640, "height": 480, "status": "uploaded"})
    folder, empty = catalog.create_folder("Project"), catalog.create_folder("Empty")
    catalog.update_video_folders(["video-1", "video-2"], [folder["id"]], "add")
    assert set(catalog.resolve_video_scope(["video-3"], [folder["id"]]) or []) == {"video-1", "video-2", "video-3"}
    assert catalog.resolve_video_scope(None, [empty["id"]]) == []
    assert set(catalog.resolve_video_scope(None, [Catalog.DEFAULT_FOLDER_ID]) or []) == {"video-3"}


def test_milvus_cleanup_queue_is_durable_and_deduplicated(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.enqueue_milvus_cleanup("video-1", "connection refused")
    catalog.enqueue_milvus_cleanup("video-1", "timeout")

    pending = catalog.list_milvus_cleanup_queue()
    assert len(pending) == 1
    assert pending[0]["video_id"] == "video-1"
    assert pending[0]["last_error"] == "timeout"
    assert pending[0]["attempts"] == 2

    catalog.complete_milvus_cleanup("video-1")
    assert catalog.list_milvus_cleanup_queue() == []


def test_next_queued_job_queries_oldest_queued_record(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video({
        "id": "video-1", "name": "demo.mp4", "file_path": str(tmp_path / "demo.mp4"),
        "duration": 5, "fps": 25, "width": 640, "height": 480, "status": "uploaded",
    })
    for job_id, status in (("newer", "queued"), ("oldest", "queued"), ("ignored", "running")):
        catalog.create_job({
            "id": job_id, "video_id": "video-1", "status": status,
            "stage": status, "progress": 0, "modalities": ["asr"], "options": {},
        })
    with catalog.connect() as connection:
        connection.execute("UPDATE jobs SET created_at='2026-01-02' WHERE id='newer'")
        connection.execute("UPDATE jobs SET created_at='2026-01-01' WHERE id='oldest'")
        connection.execute("UPDATE jobs SET created_at='2025-01-01' WHERE id='ignored'")

    assert catalog.next_queued_job()["id"] == "oldest"

    catalog.update_job("oldest", status="completed")
    catalog.update_job("newer", status="completed")
    assert catalog.next_queued_job() is None
