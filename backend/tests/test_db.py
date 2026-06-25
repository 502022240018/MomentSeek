from app.db import Catalog


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

    catalog.update_video("video-1", status="ready", indexed_modalities=["visual", "asr"])
    assert catalog.get_video("video-1")["indexed_modalities"] == ["visual", "asr"]

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
    catalog.update_job("job-1", status="completed", progress=1)
    assert catalog.get_job("job-1")["progress"] == 1

    catalog.create_entity({
        "id": "entity-1",
        "name": "Neymar",
        "reference_path": "neymar.jpg",
        "embedding_path": "neymar.npz",
    })
    assert catalog.find_entity_in_text("find Neymar on the field")["id"] == "entity-1"


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

