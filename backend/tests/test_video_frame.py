from fastapi.testclient import TestClient

from app.db import Catalog
from app.settings import Settings


def test_video_frame_bounds_timestamp_and_reuses_cached_jpeg(monkeypatch, tmp_path):
    import app.main as main

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video-1.mp4"
    video_path.write_bytes(b"video")
    catalog.create_video({
        "id": "video-1",
        "name": "demo.mp4",
        "file_path": str(video_path),
        "duration": 10.0,
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "status": "ready",
    })
    calls: list[tuple[float, str]] = []

    def fake_extract(source, destination, timestamp):
        calls.append((float(timestamp), str(source)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg")
        return destination

    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "catalog", catalog)
    monkeypatch.setattr(main, "extract_video_frame", fake_extract)

    with TestClient(main.app) as client:
        first = client.get("/api/videos/video-1/frame?time=99")
        second = client.get("/api/videos/video-1/frame?time=99")

    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert first.content == b"jpeg"
    assert second.status_code == 200
    assert calls == [(10.0, str(video_path))]
