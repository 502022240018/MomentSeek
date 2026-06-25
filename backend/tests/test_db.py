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

